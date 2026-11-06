"""Editing API: `Design` = a mutable working copy of an assembly document + its Library.

Every edit (add / remove / swap / mate / unmate / wire / set, or a batch through `edit`) is atomic: the document is
validated afterwards (numbers rule included); on any error nothing changes. Each returns an `EditResult` with what changed
and the DRC delta (findings new / resolved / changed against the previous state). `undo` / `redo` walk a snapshot stack.

Snapping and suggestions are heuristics over library data only (port type / gender / symmetry, category in this assembly,
ratings on power ports); the score weights below are choices, not physical data.
"""

import copy
import difflib
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import assembly as A
from . import auto_devices as AD
from . import pins as PN
from . import wiring as W
from .assembly import Ctx
from .build import _mate_spec
from .build import build as _build
from .build import compute as _compute
from .drc import _default_parts, _World, joint_holding, placement_metrics
from .drc import check as _check
from .library import DesignError, Library, Part, genders_ok, types_ok
from .snap import (
    find_compatible_ports,
    incompatible,
    matrix_rpy,
    n_mate,
    pose_matrix,
    solve_tree,
    symmetry,
    thread_of,
)

# suggest / snap score weights (design choices)
W_EXACT, W_FAMILY = (
    100,
    50,
)  # identical port type / paired family (shaft+bore, tube+tube-clamp)
W_GENDER, W_NEUTRAL = 10, 5  # male+female / a neutral side
W_SYMMETRY = 5  # x min(sym) / max(sym): equal symmetry scores the most
W_CATEGORY, W_FAMILY_CAT = (
    30,
    15,
)  # candidate category seen on ports of this type in the assembly / same top-level family
W_STOCK = 20  # the part that currently sits on the port
W_RATING_BAD = -40  # per failed rating (voltage / current) on a power-port candidate
# placement scoring (geometry of the trial design, added to the port score above; see Design._place). The geometry terms add up
# to at most 45 < W_FAMILY (50): geometry decides between placements of one port pair and between near-equal pairs, but never
# outvotes the port-type fit. Interfering placements lose W_INTERFERE, so they rank below every clean one.
W_CLEAR = 15  # x min(gap, CLEAR_CAP) / CLEAR_CAP: bounding-box clearance to the nearest non-mated instance
CLEAR_CAP = 0.02  # m: clearance beyond 2 cm earns nothing more (a choice; a hand's width of slack is not "better")
W_UPRIGHT = 20  # x (z . up + 1) / 2 of the part's own +z axis; skipped for jointed mates (a moving part has no "up")
W_INBOARD = 10  # x (1 - offset / reach): horizontal distance of the part's CoM from the rest of the vehicle's CoM (free base only)
W_INTERFERE = 1000  # drc.INTERFERENCE_TOL (1 mm) overlap with a non-mated instance
AMBIG_BAND = 2.0  # another port pair within this many points of the best is reported as ambiguous
MAX_ROT = 12  # rotations tried per port pair (a 360-fold shaft is sampled every 30 deg)
MAX_PAIRS = 10  # port pairs evaluated geometrically (best port score first)
MECH, POWER, SIGNAL = "mech", "power", "signal"
MOUNT_FAMILIES = (
    "bolt-pattern.",
    "bolt-circle.",
)  # port types of hole patterns: a mount that fits or does not


def _split(ref: str):
    i, _, p = ref.partition(".")
    return i, (p or None)


def _num(e):
    v = e.get("value") if isinstance(e, dict) else None
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _fkey(f):
    return (f.code, tuple(f.where))


def finding_delta(before, after):
    """(new, resolved, changed) between two finding lists, matched on (code, where); changed = same key, other severity / text."""

    def index(fs):
        seen, out = {}, {}
        for f in fs:
            k = _fkey(f)
            n = seen[k] = seen.get(k, 0) + 1
            out[(k, n)] = f
        return out

    b, a = index(before), index(after)
    new = [f.as_dict() for k, f in a.items() if k not in b]
    gone = [f.as_dict() for k, f in b.items() if k not in a]
    chg = [
        {"before": b[k].as_dict(), "after": f.as_dict()}
        for k, f in a.items()
        if k in b and (b[k].severity, b[k].message) != (f.severity, f.message)
    ]
    return new, gone, chg


def _counts(fs):
    return {s: sum(f.severity == s for f in fs) for s in ("error", "warning", "info")}


def _note(ops) -> str:
    """One line for an undo-history entry: the first op ("add esc1", "wire pdb.out1 -> esc1.power_in") and how many more."""
    if not ops:
        return "edit"
    o = ops[0]
    what = o.get("id") or o.get("name") or o.get("instance") or o.get("part") or ""
    if o.get("op") in ("wire", "autowire", "pinwire") and (o.get("from") or o.get("a")):
        what = f"{o.get('from') or o.get('a')} -> {o.get('to') or o.get('b') or '...'}"
    elif o.get("op") in ("mate", "unmate") and o.get("a"):
        what = f"{o['a']} + {o.get('b', '')}".strip(" +")
    s = f"{o.get('op', 'edit')} {what}".strip()
    return s + (f"  (+{len(ops) - 1} more)" if len(ops) > 1 else "")


@dataclass
class EditResult:
    changes: list  # one dict per applied op: what changed (swap: kept / lost mates + wires, dangling references, ...)
    new: list = field(
        default_factory=list
    )  # DRC findings that did not exist before the edit
    resolved: list = field(default_factory=list)  # findings that no longer exist
    changed: list = field(
        default_factory=list
    )  # {"before", "after"} for the same finding with another severity / text
    before: dict = field(default_factory=dict)  # {"error": n, "warning": n, "info": n}
    after: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "changes": self.changes,
            "drc": {
                "before": self.before,
                "after": self.after,
                "new": self.new,
                "resolved": self.resolved,
                "changed": self.changed,
            },
        }


class Design:
    def __init__(self, asm: A.Assembly, lib: Library, path=None):
        self.asm, self.lib, self.path = asm, lib, path
        if asm.path is None and path is not None:
            asm.path = path  # sub-assembly files resolve against it
        PN.normalize_wires(asm.doc, lib)
        self._undo: list = []
        self._redo: list = []
        self._undo_notes: list = []  # one short line per undo / redo step (the viewer's HISTORY tab), kept in step with the stacks
        self._redo_notes: list = []
        self._findings = None
        self._saved_hash = self._raw_hash()

    def _raw_hash(self) -> str:
        return hashlib.sha1(
            json.dumps(self.asm.raw, sort_keys=True, default=str).encode()
        ).hexdigest()

    @property
    def dirty(self) -> bool:
        """True when the document differs from what was last loaded / saved (the viewer's per-document marker)."""
        return self._raw_hash() != self._saved_hash

    # ---- open / save -------------------------------------------------------------------------------------------------
    @classmethod
    def open(cls, path, parts=None) -> "Design":
        lib = Library(parts if parts is not None else _default_parts())
        return cls(A.load(path), lib, Path(path))

    @classmethod
    def new(
        cls, name, parts=None, base="free", description="", root_frame=None
    ) -> "Design":
        """An empty design (no instances): the first `add(part, root=True)` is the frame everything else mates to. `base` =
        'free' (a vehicle) or 'fixed' (welded to the world). `root_frame` = 'urdf_link' puts the world at the root part's URDF
        frame (needs base = 'fixed'; build.py). Nothing is checked until the root exists."""
        lib = Library(parts if parts is not None else _default_parts())
        a = {"name": name, "base": base}
        if description:
            a["description"] = description
        if root_frame:
            a["root_frame"] = root_frame
        return cls(A.Assembly({"assembly": a, "instances": []}), lib)

    @property
    def doc(self) -> dict:
        return self.asm.doc

    def save(self, path=None) -> Path:
        """Write the assembly TOML (deterministic; comments of a hand-written file are not kept). Checked by reading it back."""
        p = Path(path or self.path or "")
        if not str(p):
            raise DesignError(
                "save: no path given and the design was not opened from a file"
            )
        A.validate(
            self.asm.raw
        )  # the document as written; the flat one was validated by edit()
        text = A.dumps(self.asm)
        if A.loads(text, p) != self.asm:
            raise DesignError(
                "save: the TOML written does not read back equal (writer bug); nothing written"
            )
        p.write_text(text, encoding="utf-8")
        self.path = p
        self._saved_hash = self._raw_hash()
        if (
            self.asm.path != p
        ):  # saved elsewhere: relative sub-assembly files now resolve against the new place
            self.asm.path = p
            self.reload_children()
        return p

    def reload_children(self) -> bool:
        """Re-read every sub-assembly file from disk (a child edited and saved elsewhere) and re-flatten. -> True when the
        flattened document changed. Undo history is kept; raises DesignError (and keeps the old state) on a bad child."""
        if not self.asm.raw.get("subassemblies"):
            return False
        old = self.asm.doc
        self.asm.set_raw(self.asm.raw)
        self._findings = None
        return self.asm.doc != old

    # ---- checks / build / simulate -----------------------------------------------------------------------------------
    def check(self) -> list:
        # an empty design (Design.new, no root yet) has nothing to check
        self._findings = (
            _check(self.asm, self.lib)
            if any(i.get("root") for i in self.doc["instances"])
            else []
        )
        return self._findings

    def findings(self) -> list:
        return self._findings if self._findings is not None else self.check()

    def build(self, out=None) -> dict:
        """-> {"xml": Path, "report_path": Path, "report": dict} (the DRC findings are in report["findings"])."""
        if not out:  # one scratch dir per Design, reused by every build and removed with the Design
            if getattr(self, "_tmp", None) is None:
                self._tmp = tempfile.TemporaryDirectory(prefix="dvsim-design-")
            out = self._tmp.name
        out = Path(out)
        r = _build(self.asm, self.lib.dir, out, lib=self.lib)
        self._findings = None
        return r

    def simulate(self, kind: str = "hover", out=None, **kw) -> dict:
        """kind = "hover" (multirotor: built-in altitude + attitude controller on the ESC throttles) or "hold" (arm: servos
        torque-on at a pose). Builds first; needs the C++ kernel. Returns the numbers listed in simulate.py."""
        from . import simulate as S

        if kind not in S.KINDS:
            raise DesignError(f"simulate: kind {kind!r} not in {sorted(S.KINDS)}")
        r = self.build(out)
        return S.KINDS[kind](r["xml"], r["report"], **kw)

    # ---- lookups -----------------------------------------------------------------------------------------------------
    def _inst(self, doc, iid: str) -> dict:
        for d in doc["instances"]:
            if d["id"] == iid:
                return d
        raise DesignError(
            f"unknown instance '{iid}'"
            + _didyoumean(iid, [d["id"] for d in doc["instances"]])
        )

    def _part(self, pid: str) -> Part:
        try:
            return self.lib.get(pid)
        except DesignError as e:
            raise DesignError(
                f"{e}" + _didyoumean(pid, list(self.lib.parts), "part")
            ) from None

    def _ports_used(self, doc, seats=True) -> set:
        """{(instance, port)} on mech ports that some mate uses (a bare 'inst' end counts when its pair is unique);
        seats = True: plus the ports co-located with a used one (the same physical seat)."""
        used = set()
        by_id = {d["id"]: d for d in doc["instances"]}
        for m in doc.get("mates", []):
            (ia, pa), (ib, pb) = _split(m["a"]), _split(m["b"])
            if ia not in by_id or ib not in by_id:
                continue
            if not (pa and pb):
                try:
                    c = find_compatible_ports(
                        self._part(by_id[ia]["part"]),
                        self._part(by_id[ib]["part"]),
                        pa,
                        pb,
                    )
                except DesignError:
                    c = []
                if len(c) != 1:
                    continue
                pa, pb = c[0]
            used |= {(ia, pa), (ib, pb)}
        if seats:  # alternative patterns / adapters of one seat (co-located ports) are exclusive: the seat is taken
            for iid, pn in list(used):
                p0 = self._part(by_id[iid]["part"]).port(pn)
                for q in self._part(by_id[iid]["part"]).ports:
                    alt_fam = (
                        MOUNT_FAMILIES + ("shaft.", "bore.")
                    )  # only alternative patterns / adapters; other co-located ports are separate seats
                    if (
                        q.get("kind") == MECH
                        and str(q.get("type", "")).startswith(alt_fam)
                        and str(p0.get("type", "")).startswith(alt_fam)
                        and all(
                            np.allclose(
                                q.get(k, [0, 0, 0]), p0.get(k, [0, 0, 0]), atol=1e-9
                            )
                            for k in ("pos", "axis", "ref")
                        )
                    ):
                        used.add((iid, q["name"]))
        return used

    def _colocated(self, doc, iid, port) -> list:
        """Other unmated mech ports of instance `iid` at the same place and orientation as `port` (alternative hole patterns)."""
        part = self._part(self._inst(doc, iid)["part"])
        used = self._ports_used(doc, seats=False)
        same = lambda p, k: np.allclose(
            p.get(k, [0, 0, 0]), port.get(k, [0, 0, 0]), atol=1e-9
        )
        return [
            p
            for p in part.ports
            if p.get("kind") == MECH
            and p["name"] != port["name"]
            and (iid, p["name"]) not in used
            and all(same(p, k) for k in ("pos", "axis", "ref"))
        ]

    def _connected(self, doc) -> set:
        root = next(d["id"] for d in doc["instances"] if d.get("root"))
        adj = {}
        for m in doc.get("mates", []):
            a, b = _split(m["a"])[0], _split(m["b"])[0]
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        for d in doc["instances"]:  # free-placed (pos / rpy): welded to the root
            if ("pos" in d or "rpy" in d) and not d.get("root"):
                adj.setdefault(root, set()).add(d["id"])
        seen, todo = {root}, [root]
        while todo:
            for n in adj.get(todo.pop(), set()) - seen:
                seen.add(n)
                todo.append(n)
        return seen

    def poses(self, doc=None) -> dict:
        """{instance id: 4x4 pose in the root frame} of every on-board instance, from the mates and free placements."""
        doc = doc or self.doc
        ctx = Ctx(A.Assembly(doc), self.lib)
        onboard = [i for i, x in ctx.insts.items() if not x.offboard]
        root = next(i for i, x in ctx.insts.items() if x.root)
        specs = [_mate_spec(ctx, m) for m in doc.get("mates", [])]
        placed = {
            i: pose_matrix(x.pos, x.rpy)
            for i, x in ctx.insts.items()
            if x.placed and not x.offboard and not x.root
        }
        tree = solve_tree(root, onboard, specs, placed=placed)
        self._tree = tree
        return tree.T

    # ---- snapping ----------------------------------------------------------------------------------------------------
    @staticmethod
    def _pairs(ports_a, ports_b) -> list:
        """Compatible mech port pairs, best first: [{"a": port a, "b": port b, "score": s, "why": [...]}]."""
        out = []
        for i, pa in enumerate(ports_a):
            for j, pb in enumerate(ports_b):
                if (
                    pa.get("kind") != MECH
                    or pb.get("kind") != MECH
                    or incompatible(pa, pb)
                ):
                    continue
                exact = pa.get("type") == pb.get("type")
                s = W_EXACT if exact else W_FAMILY
                why = ["same port type" if exact else "paired family"]
                ga, gb = pa.get("gender"), pb.get("gender")
                if {ga, gb} == {"male", "female"}:
                    s += W_GENDER
                    why.append("male+female")
                else:
                    s += W_NEUTRAL
                sa, sb = symmetry(pa), symmetry(pb)
                s += W_SYMMETRY * min(sa, sb) / max(sa, sb)
                if sa != sb:
                    why.append(f"symmetry {sa} vs {sb}")
                # tie-break, not a score term: +1 when mating leaves the child right side up (its port faces -z where the
                # parent port faces +z, e.g. a board's `base`), -1 when it would be flipped (a `stack_top` onto a plate)
                up = pa.get("axis", [0, 0, 0])[2] * -pb.get("axis", [0, 0, 0])[2]
                out.append(
                    {"a": pa, "b": pb, "score": s, "why": why, "up": up, "_o": (i, j)}
                )
        out.sort(key=lambda r: (-r["score"], -r["up"], r["_o"]))
        return out

    # ---- placement search --------------------------------------------------------------------------------------------
    @staticmethod
    def _ks(n, k):
        """Rotations to try: the one given, else all of 0..n-1 (n > MAX_ROT, e.g. a 360-fold shaft: MAX_ROT evenly spaced)."""
        if k is not None:
            return [k]
        if n <= MAX_ROT:
            return list(range(n))
        return sorted({(i * n) // MAX_ROT for i in range(MAX_ROT)})

    def _geometry(self, doc2, before, jointed):
        """Build the trial design and measure the instances this edit just connected: -> dict, or {"error": why}."""
        moved = self._connected(doc2) - before
        try:
            xml, _ = _compute(
                A.Assembly(doc2), self.lib.dir, self.lib.root, self.lib, route=False
            )
            world = _World(xml, self.lib.root)
        except (DesignError, KeyError, ValueError, StopIteration) as e:
            return {"error": str(e).splitlines()[0][:160]}
        mated = {
            frozenset((_split(m["a"])[0], _split(m["b"])[0]))
            for m in doc2.get("mates", [])
        }
        out = placement_metrics(world, moved, mated, CLEAR_CAP)
        bodies = [world.bodies[world.idx[i]] for i in moved if i in world.idx]
        if not bodies:
            return {"error": "the moved instances have no body"}
        out["upright"] = (
            None if jointed else float(np.mean([b["T"][2, 2] for b in bodies]))
        )
        out["com_offset"] = out["radius"] = None
        if doc2["assembly"].get("base", "free") == "free":
            rest = [b for b in world.bodies if b["name"] not in moved and b["mass"] > 0]
            mv = [b for b in bodies if b["mass"] > 0]
            if rest and mv:
                at = lambda bs: (
                    sum(b["mass"] * (b["T"] @ np.append(b["com"], 1.0))[:3] for b in bs)
                    / sum(b["mass"] for b in bs)
                )
                cr, cm = at(rest), at(mv)
                out["com_offset"] = float(np.linalg.norm((cm - cr)[:2]))
                out["radius"] = max(
                    0.05,
                    max(
                        float(
                            np.linalg.norm(
                                ((b["T"] @ np.append(b["com"], 1.0))[:3] - cr)[:2]
                            )
                        )
                        for b in rest
                    ),
                )
        return out

    def _place(self, doc, items, k, apply, what, *, force=False, jointed=False):
        """Rank every placement (compatible port pair x rotation k): the port score plus the geometry terms of the trial
        design built with that placement (see W_* at the top of this file). Refuses when there is a choice (>= 2 placements)
        and every one interferes (unless `force`); a single determined placement is made, the DRC reports its INTERFERENCE. `apply(doc2, item, k)` writes the placement into a copy of the document.
        -> (best item, best k, info)."""
        before = self._connected(doc)
        cands = []
        for it in items[:MAX_PAIRS]:
            for kk in [k(it)] if callable(k) else self._ks(n_mate(it["a"], it["b"]), k):
                d2 = copy.deepcopy(doc)
                apply(d2, it, kk)
                g = self._geometry(d2, before, jointed)
                s, why, up = it["score"], list(it["why"]), 0
                if "error" in g:
                    up = it["up"]
                    why.append(f"geometry not evaluated ({g['error']})")
                    hit = {}
                else:
                    hit = g["interference"]
                    if hit:
                        s -= W_INTERFERE
                        why += [
                            f"interferes with {o} by {d * 1000:.1f} mm"
                            for o, d in sorted(hit.items())
                        ]
                    else:
                        gap = (
                            CLEAR_CAP if g["gap"] is None else min(g["gap"], CLEAR_CAP)
                        )
                        s += W_CLEAR * gap / CLEAR_CAP
                        why.append(
                            "no other instance within " + f"{CLEAR_CAP * 1000:.0f} mm"
                            if g["gap"] is None
                            else f"clearance >= {g['gap'] * 1000:.1f} mm to the nearest instance"
                        )
                    if g["upright"] is not None:
                        s += W_UPRIGHT * (g["upright"] + 1) / 2
                        why.append(f"upright {g['upright']:+.2f} (part +z . world up)")
                    if g["com_offset"] is not None:
                        s += W_INBOARD * (1 - min(1.0, g["com_offset"] / g["radius"]))
                        why.append(
                            f"{g['com_offset'] * 1000:.0f} mm from the rest of the vehicle's centre of mass (reach {g['radius'] * 1000:.0f} mm)"
                        )
                cands.append(
                    {
                        "it": it,
                        "k": kk,
                        "score": s,
                        "why": why,
                        "hit": sorted(hit),
                        "up": up,
                        "geo": g,
                    }
                )
        cands.sort(
            key=lambda c: (-round(c["score"], 9), -c["up"], c["it"]["_o"], c["k"])
        )
        top = cands[0]
        evaluated = [c for c in cands if "error" not in c["geo"]]
        if (
            len(cands) > 1
            and evaluated
            and all(c["hit"] for c in evaluated)
            and not force
        ):
            raise DesignError(
                f"{what}: refused, all {len(cands)} placements interfere: "
                + "; ".join(
                    f"{c['it']['a']['name']}<->{c['it']['b']['name']} k={c['k']}: "
                    + ", ".join(w for w in c["why"] if w.startswith("interferes"))
                    for c in cands[:6]
                )
                + (" ..." if len(cands) > 6 else "")
                + ". Move the obstacle, pick another seat, or pass force=true"
            )
        pair = lambda c: (c["it"]["a"]["name"], c["it"]["b"]["name"])
        # alternative hole patterns / adapters of one seat (co-located ports on both parts) place the part identically: not a choice
        seat = lambda p, q: all(
            np.allclose(p.get(k, [0, 0, 0]), q.get(k, [0, 0, 0]), atol=1e-9)
            for k in ("pos", "axis", "ref")
        )
        tied = [
            c
            for c in cands
            if pair(c) != pair(top)
            and not (
                seat(c["it"]["a"], top["it"]["a"])
                and seat(c["it"]["b"], top["it"]["b"])
            )
            and abs(c["score"] - top["score"]) < AMBIG_BAND
            and c["up"] == top["up"]
        ]
        g = top["geo"]
        info = {
            "score": round(top["score"], 2),
            "why": top["why"],
            "ambiguous": bool(tied),
            "k": top["k"],
            "evaluated": len(cands),
            "interferes": top["hit"],
            "alternatives": [
                {
                    "a": c["it"]["a"]["name"],
                    "b": c["it"]["b"]["name"],
                    "k": c["k"],
                    "score": round(c["score"], 2),
                    "why": c["why"],
                    "interferes": c["hit"],
                }
                for c in cands[1:7]
            ],
        }
        if "error" not in g:
            info["clearance_mm"] = (
                None if g["gap"] is None else round(g["gap"] * 1000, 2)
            )
        return top["it"], top["k"], info

    def _pick(
        self,
        doc,
        parent_id,
        parent_port,
        child_part,
        child_port,
        what,
        k,
        apply,
        **kw,
    ) -> tuple:
        """Best open compatible pair parent.port <-> child.port and rotation -> (parent port, child port, k, info dict)."""
        pinst = self._inst(doc, parent_id)
        ppart = self._part(pinst["part"])
        used = self._ports_used(doc)
        pports = [
            p
            for p in ppart.ports
            if (parent_port is None or p.get("name") == parent_port)
            and p.get("kind") == MECH
            and (parent_id, p["name"]) not in used
        ]
        if parent_port is not None and not pports:
            ppart.port(parent_port)  # raises with the port list when it does not exist
            raise DesignError(
                f"{parent_id}.{parent_port} is not an open mech port (already mated, or not mechanical)"
            )
        cports = [
            p
            for p in child_part.ports
            if child_port is None or p.get("name") == child_port
        ]
        if child_port is not None and not cports:
            child_part.port(child_port)
        pairs = self._pairs(pports, cports)
        if not pairs:
            free = [
                p["name"]
                for p in ppart.ports
                if p.get("kind") == MECH and (parent_id, p["name"]) not in used
            ]
            raise DesignError(
                f"{what}: no compatible open mech port pair between {parent_id} ({ppart.id}) and {child_part.id}; "
                f"open ports of {parent_id}: {free or 'none'}. design suggest {parent_id}.<port> lists parts that fit"
            )
        it, k, info = self._place(
            doc,
            pairs,
            k,
            apply,
            what,
            **kw,
        )
        for alt in info[
            "alternatives"
        ]:  # add-style keys: where it would go, which child port
            alt["to"], alt["port"] = f"{parent_id}.{alt.pop('a')}", alt.pop("b")
        return it["a"]["name"], it["b"]["name"], k, info

    def _mate_dict(
        self,
        a,
        b,
        k,
        pa,
        pb,
        fasteners="auto",
        joint=None,
        name=None,
        nut=None,
        closure=False,
    ):
        m = {"a": a, "b": b}
        if name:
            m["name"] = name
        n = n_mate(pa, pb)
        k = 0 if k is None else k
        if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < n:
            raise DesignError(
                f"rotation k={k!r} out of range: ports {a} / {b} allow 0..{n - 1}"
            )
        m["rotation"] = k
        if fasteners == "auto":
            if thread_of(pa) or thread_of(pb):
                m["fasteners"] = "auto"
        elif fasteners:
            m["fasteners"] = fasteners
        if nut:
            m["nut"] = nut
        if joint:
            m["joint"] = joint
        if closure:
            m["closure"] = True
        return m

    # ---- the ops (each mutates `doc` and returns a change dict) ------------------------------------------------------
    def _op_add(self, doc, s) -> dict:
        part = self._part(s["part"])
        ids = {d["id"] for d in doc["instances"]}
        iid = s.get("id")
        if not iid:
            iid, n = part.id, 1
            while iid in ids:
                n += 1
                iid = f"{part.id}-{n}"
        if iid in ids:
            raise DesignError(f"instance id '{iid}' already exists")
        inst = {"id": iid, "part": part.id}
        for k in ("group", "offboard", "color"):
            if s.get(k) is not None:
                inst[k] = s[k]
        ch = {"op": "add", "id": iid, "part": part.id}
        has_root = any(d.get("root") for d in doc["instances"])
        if s.get("root"):
            if has_root or s.get("to") or s.get("offboard"):
                raise DesignError(
                    "add root: the design already has a root, or `to` / offboard was given (the root is mated to nothing and is on board)"
                )
            inst["root"] = True
            doc["instances"].append(inst)
            ch["note"] = "root instance: everything else mates to it"
            return ch
        if not has_root:
            raise DesignError(
                f"add {part.id}: the design has no root yet; add the frame first with root=true"
            )
        if s.get("to"):
            tid, tport = _split(s["to"])

            def apply(
                d2, it, kk
            ):  # the placement, written into a document (a trial copy or the real one)
                m = self._mate_dict(
                    f"{tid}.{it['a']['name']}",
                    f"{iid}.{it['b']['name']}",
                    kk,
                    it["a"],
                    it["b"],
                    fasteners=s.get("fasteners", "auto"),
                    joint=s.get("joint"),
                    nut=s.get("nut"),
                    name=s.get(
                        "mate"
                    ),  # mate name (device bindings refer to spin mates by it)
                )
                d2["instances"].append(copy.deepcopy(inst))
                d2.setdefault("mates", []).append(m)
                return m

            ppn, cpn, k, info = self._pick(
                doc,
                tid,
                tport,
                part,
                s.get("port"),
                f"add {part.id} to {s['to']}",
                s.get("k"),
                apply,
                force=bool(s.get("force")),
                jointed=bool(s.get("joint")),
            )
            if s.get("strict") and info["ambiguous"]:
                raise DesignError(
                    f"snap is ambiguous (strict): {[info['why']] + info['alternatives']}"
                )
            pa = self._part(self._inst(doc, tid)["part"]).port(ppn)
            m = apply(doc, {"a": pa, "b": part.port(cpn)}, k)
            ch.update(mate=m, snap=info, needs=self._needs(doc, part, {cpn}))
        else:
            if s.get("port") or s.get("k") is not None:
                raise DesignError("add: port / k only apply together with `to`")
            if s.get("pos") is not None or s.get("rpy") is not None:
                inst["pos"] = [float(x) for x in s.get("pos") or [0, 0, 0]]
                inst["rpy"] = [float(x) for x in s.get("rpy") or [0, 0, 0]]
                ch["note"] = "free-placed at pos / rpy (root frame), welded to the root"
            else:
                ch["note"] = (
                    "not mated: it is floating until a mate connects it (design suggest <inst.port> lists parts / ports)"
                )
            doc["instances"].append(inst)
        return ch

    def _drop_refs(self, doc, iid, mate_names, ch):
        """Devices bound to a removed instance / mate go with it."""
        keep = []
        for d in doc.get("devices", []):
            hit = [
                k
                for k in ("instance", "body", "site", "motor")
                if iid and d.get(k) == iid
            ] or [k for k in ("joint", "backlash_joint") if d.get(k) in mate_names]
            if hit:
                ch["devices_removed"].append(
                    {"name": d["name"], "type": d["type"], "because": hit}
                )
            else:
                keep.append(d)
        if "devices" in doc:
            doc["devices"] = keep
        if "devices" in doc and not doc["devices"]:
            del doc["devices"]

    def _op_remove(self, doc, s) -> dict:
        iid = s["id"]
        inst = self._inst(doc, iid)
        if inst.get("root"):
            raise DesignError(f"cannot remove the root instance '{iid}'")
        before = self._connected(doc)
        ch = {
            "op": "remove",
            "id": iid,
            "part": inst["part"],
            "mates_removed": [],
            "wires_removed": [],
            "devices_removed": [],
        }
        doc["instances"].remove(inst)
        names = set()
        keep = []
        for m in doc.get("mates", []):
            if iid in (_split(m["a"])[0], _split(m["b"])[0]):
                nm = m.get("name", f"{m['a']}--{m['b']}")
                names.add(nm)
                ch["mates_removed"].append(nm)
            else:
                keep.append(m)
        if "mates" in doc:
            doc["mates"] = keep
        wk = []
        for w in doc.get("wires", []):
            if (
                iid in (_split(w["from"])[0], _split(w["to"])[0])
                or w.get("included_in") == iid
            ):
                ch["wires_removed"].append(w.get("name", f"{w['from']}--{w['to']}"))
            else:
                wk.append(w)
        if "wires" in doc:
            doc["wires"] = wk
        self._drop_refs(doc, iid, names, ch)
        after = self._connected(doc)
        orphans = sorted(before - after - {iid})
        if orphans:
            ch["orphaned"] = (
                orphans  # still in the design, no longer connected to the root (DRC FLOATING_INSTANCE)
            )
        return ch

    def _op_unmate(self, doc, s) -> dict:
        mates = doc.get("mates", [])
        if s.get("name"):
            hit = [
                m for m in mates if m.get("name", f"{m['a']}--{m['b']}") == s["name"]
            ]
        else:
            if not (s.get("a") and s.get("b")):
                raise DesignError("unmate: give the mate `name`, or both ends a and b")

            def end(
                given, actual
            ):  # 'inst' matches any port of inst, 'inst.port' only that port
                return given == actual or (
                    _split(given)[1] is None and given == _split(actual)[0]
                )

            hit = [
                m
                for m in mates
                if (end(s["a"], m["a"]) and end(s["b"], m["b"]))
                or (end(s["a"], m["b"]) and end(s["b"], m["a"]))
            ]
        if not hit:
            raise DesignError(
                "unmate: no such mate; mates: "
                + ", ".join(m.get("name", f"{m['a']}--{m['b']}") for m in mates[:40])
            )
        if len(hit) > 1:
            raise DesignError(
                f"unmate: {len(hit)} mates match, give the mate name: {[m.get('name', m['a'] + '--' + m['b']) for m in hit]}"
            )
        m = hit[0]
        before = self._connected(doc)
        nm = m.get("name", f"{m['a']}--{m['b']}")
        mates.remove(m)
        ch = {"op": "unmate", "mate": nm, "dangling": self._dev_joints(doc, {nm})}
        orphans = sorted(before - self._connected(doc))
        if orphans:
            ch["orphaned"] = orphans
        return ch

    def _op_mate(self, doc, s) -> dict:
        (ia, pa), (ib, pb) = _split(s["a"]), _split(s["b"])
        A_, B_ = self._inst(doc, ia), self._inst(doc, ib)
        partA, partB = self._part(A_["part"]), self._part(B_["part"])
        k = s.get("k")
        conn = self._connected(doc)
        swapped = (
            ia not in conn and ib in conn
        )  # a = parent: the side already connected to the root
        if swapped:
            ia, pa, partA, ib, pb, partB = ib, pb, partB, ia, pa, partA
        closes = ia in conn and ib in conn
        if closes and not s.get("closure"):
            raise DesignError(
                f"mate {ia}.{pa or ''} / {ib}.{pb or ''}: both are already connected to the root, so this closes a loop; pass closure=true if it is a loop closure"
            )
        if pa and pb:  # both ports named: the one pair, whatever else is open
            porta, portb = partA.port(pa), partB.port(pb)
            if porta.get("kind") != MECH or portb.get("kind") != MECH:
                raise DesignError(
                    f"mate {ia}.{pa} / {ib}.{pb}: both ports must be mech ports"
                )
            why = incompatible(porta, portb)
            if why:
                raise DesignError(f"mate {ia}.{pa} / {ib}.{pb}: {why}")
            items = self._pairs([porta], [portb])
        else:
            used = self._ports_used(doc)
            items = self._pairs(
                [
                    p
                    for p in partA.ports
                    if (pa is None or p.get("name") == pa)
                    and (ia, p["name"]) not in used
                ],
                [
                    p
                    for p in partB.ports
                    if (pb is None or p.get("name") == pb)
                    and (ib, p["name"]) not in used
                ],
            )
            if not items:
                raise DesignError(
                    f"mate {s['a']} / {s['b']}: no compatible open mech port pair"
                )
        names = {m.get("name", f"{m['a']}--{m['b']}") for m in doc.get("mates", [])}

        def apply(d2, it, kk):
            m = self._mate_dict(
                f"{ia}.{it['a']['name']}",
                f"{ib}.{it['b']['name']}",
                kk,
                it["a"],
                it["b"],
                s.get("fasteners", "auto"),
                s.get("joint"),
                s.get("name"),
                s.get("nut"),
                s.get("closure"),
            )
            nm = m.get("name", f"{m['a']}--{m['b']}")
            if nm in names:
                raise DesignError(f"a mate named '{nm}' already exists (pass a name)")
            d2.setdefault("mates", []).append(m)
            return m

        info = None
        if (
            closes
        ):  # a loop closure is checked against the tree, there is nothing to search
            it, kk = items[0], (0 if k is None else k)
        else:
            if k is not None and swapped:  # k is in the orientation the caller wrote
                k0 = k
                k = lambda it: (-k0) % n_mate(it["a"], it["b"])
            it, kk, info = self._place(
                doc,
                items,
                k,
                apply,
                f"mate {s['a']} / {s['b']}",
                force=bool(s.get("force")),
                jointed=bool(s.get("joint")),
            )
            for alt in info["alternatives"]:
                alt["a"], alt["b"] = f"{ia}.{alt['a']}", f"{ib}.{alt['b']}"
        m = apply(doc, it, kk)
        ch = {"op": "mate", "mate": m}
        if info and not (pa and pb and info["evaluated"] == 1):
            ch["snap"] = info
        return ch

    def _op_wire(self, doc, s) -> dict:
        # `from` / `to` = "inst.port" or the pin shorthand "inst.CONN.PIN"; `from_pins` / `to_pins` = ["CONN.PIN", ...] (one per conductor)
        label = s.get("name") or f"{s['from']}--{s['to']}"
        s = {**s}
        PN.normalize_wires({"instances": doc["instances"], "wires": [s]}, self.lib)
        for end in ("from", "to"):
            iid, pn = _split(s[end])
            if pn is None:
                raise DesignError(
                    f"wire {end} '{s[end]}': name the port as 'inst.port' or the pin as 'inst.CONN.PIN'"
                )
            part = self._part(self._inst(doc, iid)["part"])
            p = part.port(pn)
            if p.get("kind") not in (POWER, SIGNAL):
                raise DesignError(
                    f"wire {end} '{s[end]}': port kind {p.get('kind')!r}, wires join power or signal ports"
                )
            for ref in s.get(end + "_pins") or []:
                pin = PN.find_pin(part, ref)
                if pin is None:
                    raise DesignError(
                        f"wire {end} '{s[end]}': {part.id} has no pin '{ref}' (connectors: {sorted(part.connectors)})"
                    )
                # a joinable connector (a splice point on a lead) may be tapped from any port of its part, as the assembly loader allows
                if p.get("pins") and ref not in p["pins"] and not pin.get("joinable"):
                    raise DesignError(
                        f"wire {end} '{s[end]}': pin {ref} is not on port {pn} (its pins: {p['pins']})"
                    )
        fa, fb = s.get("from_pins") or [], s.get("to_pins") or []
        if fa and fb and len(fa) != len(fb):
            raise DesignError(
                f"wire {label}: from_pins and to_pins need one pin per conductor ({len(fa)} vs {len(fb)})"
            )
        self._part(s["part"])
        w = {
            "name": label,
            "from": s["from"],
            "to": s["to"],
            "part": s["part"],
            "length": s["length"],
        }
        for k in ("from_pins", "to_pins"):
            if s.get(k):
                w[k] = list(s[k])
        if s.get("conductors") is not None:
            w["conductors"] = s["conductors"]
        if s.get("included_in"):
            w["included_in"] = s["included_in"]
        if s.get("connectors"):  # pin the end connector parts: {"from": id, "to": id}
            w["connectors"] = s["connectors"]
        if w["name"] in {x.get("name") for x in doc.get("wires", [])}:
            raise DesignError(
                f"a wire named '{w['name']}' already exists (pass a name)"
            )
        doc.setdefault("wires", []).append(w)
        return {"op": "wire", "wire": w}

    def _op_device_map(self, doc, s) -> dict:
        """device_map: name + map ({"motor.kv": "motor:kv", ...}): a [device_maps.<name>] table, replaced when it exists."""
        if not s.get("name") or not isinstance(s.get("map"), dict) or not s["map"]:
            raise DesignError("device_map: needs name and a non-empty map {field: ref}")
        doc.setdefault("device_maps", {})[s["name"]] = dict(s["map"])
        return {"op": "device_map", "name": s["name"], "map": dict(s["map"])}

    def _op_device(self, doc, s) -> dict:
        """device: device = {type, name, instance, port, map, set, ...}: one [[devices]] binding (see assemblies/README.md)."""
        dev = s.get("device")
        if not isinstance(dev, dict) or not dev.get("type") or not dev.get("name"):
            raise DesignError("device: needs device = {type, name, ...}")
        if dev["name"] in {x.get("name") for x in doc.get("devices", [])}:
            raise DesignError(f"a device named '{dev['name']}' already exists")
        doc.setdefault("devices", []).append(copy.deepcopy(dev))
        return {"op": "device", "device": dev}

    def _op_target(self, doc, s) -> dict:
        """target: name + value ('<inst>.<param>' or a provenance table) + groups=[...] | all=True: a mass target; or name +
        reach = [provenance table with value [x, y, z] m, ...] and / or workspace = one table with [x0, y0, z0, x1, y1, z1] m:
        points the arm must reach (DRC ARM_REACH)."""
        if not s.get("name") or (
            s.get("value") is None and not (s.get("reach") or s.get("workspace"))
        ):
            raise DesignError("target: needs name and value (or reach / workspace)")
        t = {"name": s["name"]}
        if s.get("reach") or s.get("workspace"):
            if s.get("value") is not None:
                raise DesignError("target: reach / workspace and value are exclusive")
            for k in ("reach", "workspace"):
                if s.get(k):
                    t[k] = s[k]
            if s["name"] in {x.get("name") for x in doc.get("targets", [])}:
                raise DesignError(f"a target named '{s['name']}' already exists")
            doc.setdefault("targets", []).append(t)
            return {"op": "target", "target": t}
        if s.get("all"):
            t["all"] = True
        else:
            t["groups"] = list(s.get("groups") or [])
        t["value"] = s["value"]
        if s["name"] in {x.get("name") for x in doc.get("targets", [])}:
            raise DesignError(f"a target named '{s['name']}' already exists")
        doc.setdefault("targets", []).append(t)
        return {"op": "target", "target": t}

    def device_map(self, name, map) -> EditResult:
        return self.edit([{"op": "device_map", "name": name, "map": map}])

    def bind(self, **device) -> EditResult:
        """Add a device binding: bind(type="esc", name="esc1", instance="esc", port="power_in", motor="mot1", ...)."""
        return self.edit([{"op": "device", "device": device}])

    def target(
        self, name, value=None, groups=None, all=False, reach=None, workspace=None
    ) -> EditResult:
        return self.edit(
            [
                {
                    "op": "target",
                    "name": name,
                    "value": value,
                    "groups": groups,
                    "all": all,
                    "reach": reach,
                    "workspace": workspace,
                }
            ]
        )

    def _op_unwire(self, doc, s) -> dict:
        wires = doc.get("wires", [])
        hit = [
            w for w in wires if w.get("name", f"{w['from']}--{w['to']}") == s["name"]
        ]
        if not hit:
            raise DesignError(
                f"unwire: no wire named '{s['name']}'; wires: "
                + ", ".join(
                    w.get("name", f"{w['from']}--{w['to']}") for w in wires[:40]
                )
            )
        wires.remove(hit[0])
        if not wires:
            del doc["wires"]
        return {"op": "unwire", "wire": hit[0]}

    def _op_set(self, doc, s) -> dict:
        inst = self._inst(doc, s["id"])
        part = self._part(inst["part"])
        param = s["param"]
        old = inst.get("params", {}).get(param) or part.params.get(param)
        if s.get("unset"):
            if param not in inst.get("params", {}):
                raise DesignError(f"set: '{s['id']}' has no override for '{param}'")
            del inst["params"][param]
            if not inst["params"]:
                del inst["params"]
            return {"op": "set", "id": s["id"], "param": param, "unset": True}
        v = s.get("value")
        if isinstance(v, dict):  # a full provenance table
            entry = dict(v)
        else:
            entry = {"value": v, "unit": s.get("unit"), "fidelity": s.get("fidelity")}
            for k in ("src", "quote", "derivation", "rationale"):
                if s.get(k):
                    entry[k] = s[k]
            if entry["unit"] is None or entry["fidelity"] is None:
                raise DesignError(
                    f"set {s['id']}.{param}: numbers rule: give unit and fidelity (+ src & quote | derivation | rationale), "
                    "or a provenance table as value; a bare number is not accepted"
                )
        A.check_provenance(entry, f"set {s['id']}.{param}")
        inst.setdefault("params", {})[param] = entry
        ch = {
            "op": "set",
            "id": s["id"],
            "param": param,
            "value": entry["value"],
            "unit": entry["unit"],
            "fidelity": entry["fidelity"],
        }
        if isinstance(old, dict) and "value" in old:
            ch["was"] = {
                "value": old["value"],
                "unit": old.get("unit"),
                "fidelity": old.get("fidelity"),
            }
            if old.get("unit") != entry["unit"]:
                ch["warning"] = (
                    f"unit changed from {old.get('unit')!r} to {entry['unit']!r}"
                )
        else:
            ch["note"] = f"part {part.id} has no param '{param}': a new override"
        return ch

    # ---- swap --------------------------------------------------------------------------------------------------------
    def _new_port(
        self,
        new: Part,
        old_port: dict,
        other_port: dict | None,
        other_part: Part,
        kind: str,
    ):
        """Port of `new` that takes over `old_port`: same name, else same type, else any compatible with the other end.
        -> (port, how, alternatives) or (None, "", [])."""
        if kind == MECH:

            def ok(q):
                if q.get("kind") != MECH:
                    return False
                if other_port is not None:
                    return incompatible(other_port, q) is None
                return any(
                    incompatible(o, q) is None
                    for o in other_part.ports
                    if o.get("kind") == MECH
                )

            cands = [q for q in new.ports if ok(q)]
        else:
            cands = [
                q
                for q in new.ports
                if q.get("kind") == kind
                and types_ok(old_port.get("type", ""), q.get("type", ""))
                and genders_ok(old_port.get("gender", ""), q.get("gender", ""))
            ]
        by_name = [q for q in cands if q.get("name") == old_port.get("name")]
        same_type = [q for q in cands if q.get("type") == old_port.get("type")]
        for tier, how in (
            (by_name, "same port name"),
            (same_type, "same port type"),
            (cands, "compatible port"),
        ):
            if tier:
                alts = [q["name"] for q in tier[1:]] if how != "same port name" else []
                return tier[0], how, alts
        if (
            kind != MECH
        ):  # power / signal: same name and kind but another type is kept, DRC will judge the connector
            same = [
                q
                for q in new.ports
                if q.get("name") == old_port.get("name") and q.get("kind") == kind
            ]
            if same:
                return same[0], "same port name (type changed)", []
        return None, "", []

    def _op_swap(self, doc, s) -> dict:
        iid = s["id"]
        inst = self._inst(doc, iid)
        old, new = self._part(inst["part"]), self._part(s["part"])
        ch = {
            "op": "swap",
            "id": iid,
            "from": old.id,
            "to": new.id,
            "mates": {"kept": [], "lost": []},
            "wires": {"kept": [], "lost": []},
            "devices": [],
            "overrides": {"kept": [], "dropped": []},
            "dangling": [],
            "notes": [],
        }
        by_id = {d["id"]: d for d in doc["instances"]}
        # mates
        for m in list(doc.get("mates", [])):
            for end in ("a", "b"):
                if _split(m[end])[0] != iid:
                    continue
                oend = "b" if end == "a" else "a"
                oi, opn = _split(m[oend])
                opart = self._part(by_id[oi]["part"])
                pn = _split(m[end])[1]
                nm = m.get("name", f"{m['a']}--{m['b']}")
                oport = opart.port(opn) if opn else None
                oldp = old.port(pn) if pn else None
                if oldp is None:  # bare 'inst' end: resolve through the other end
                    c = find_compatible_ports(old, opart, None, opn)
                    if len(c) == 1:
                        oldp = old.port(c[0][0])
                        oport = oport or opart.port(c[0][1])
                if oldp is None:
                    ch["mates"]["lost"].append(
                        {"mate": nm, "old_port": pn, "why": "old port not resolvable"}
                    )
                    m["__lost"] = True
                    continue
                q, how, alts = self._new_port(new, oldp, oport, opart, MECH)
                if q is None and oport is not None:
                    # the other part may offer the same seat in another hole pattern (a motor plate with 16x16 and 19x19 ports)
                    for ap in self._colocated(doc, oi, oport):
                        q, how, alts = self._new_port(new, oldp, ap, opart, MECH)
                        if q is not None:
                            m[oend] = f"{oi}.{ap['name']}"
                            oport = ap
                            break
                if q is None and oport is not None and s.get("refuse_mount", True):
                    if str(oport.get("type", "")).startswith(MOUNT_FAMILIES):
                        pats = [
                            f"{p['name']} {p['type']}"
                            for p in new.ports
                            if str(p.get("type", "")).startswith(MOUNT_FAMILIES)
                        ]
                        raise DesignError(
                            f"swap {iid} -> {new.id} refused: it cannot be mounted on {m[oend]} (seat pattern {oport['type']}); "
                            f"{new.id} mounts with {', '.join(pats) or 'no hole pattern'}. A different pattern is a different part "
                            f"(the physical hole spacing / thread must match)"
                        )
                if q is None:
                    ch["mates"]["lost"].append(
                        {
                            "mate": nm,
                            "old_port": oldp["name"],
                            "other_end": m[oend],
                            "why": f"no port of {new.id} is compatible with {m[oend]}"
                            + (
                                f" ({oport.get('type')}, {oport.get('gender')})"
                                if oport
                                else ""
                            ),
                        }
                    )
                    m["__lost"] = True
                    continue
                if "name" not in m:
                    m["name"] = (
                        nm  # the default name derives from the port names; keep it (device bindings use it)
                    )
                m[end] = f"{iid}.{q['name']}"
                rec = {
                    "mate": nm,
                    "old_port": oldp["name"],
                    "new_port": q["name"],
                    "how": how,
                }
                if alts:
                    rec["alternatives"] = alts
                if oport is not None:
                    n = n_mate(*(oport, q))
                    if m.get("rotation", 0) >= n:
                        m["rotation"] = 0
                        rec["rotation"] = "reset to 0 (new symmetry)"
                    f = m.get("fasteners")
                    ta, tb = thread_of(oport), thread_of(q)
                    th = ta or tb
                    if f is not None and ((ta and tb and ta != tb) or (th is None)):
                        rec["fasteners"] = (
                            f"was {f if f == 'auto' else f['part']}; reset to auto (thread {ta} vs {tb})"
                            if th
                            else "removed (no thread on the new ports)"
                        )
                        if th:
                            m["fasteners"] = "auto"
                        else:
                            m.pop("fasteners", None)
                            m.pop("nut", None)
                    elif (
                        isinstance(f, dict)
                        and th
                        and f.get("part")
                        and not self._part(f["part"]).has_port(f"thread.{th}", "male")
                    ):
                        m["fasteners"] = "auto"
                        rec["fasteners"] = (
                            f"was {f['part']}; reset to auto (not a {th} screw)"
                        )
                ch["mates"]["kept"].append(rec)
        if any(m.get("__lost") for m in doc.get("mates", [])):
            lost_names = [
                m.get("name", f"{m['a']}--{m['b']}")
                for m in doc["mates"]
                if m.get("__lost")
            ]
            doc["mates"] = [m for m in doc["mates"] if not m.get("__lost")]
            ch["dangling"] += self._dev_joints(doc, set(lost_names))
        # wires
        for w in list(doc.get("wires", [])):
            for end in ("from", "to"):
                if _split(w[end])[0] != iid:
                    continue
                pn = _split(w[end])[1]
                nm = w.get("name", f"{w['from']}--{w['to']}")
                oldp = old.port(pn)
                q, how, alts = self._new_port(new, oldp, None, old, oldp.get("kind"))
                if q is None:
                    ch["wires"]["lost"].append({"wire": nm, "old_port": pn})
                    w["__lost"] = True
                    continue
                if "name" not in w:
                    w["name"] = nm
                w[end] = f"{iid}.{q['name']}"
                rec = {"wire": nm, "old_port": pn, "new_port": q["name"], "how": how}
                if alts:
                    rec["alternatives"] = alts
                ch["wires"]["kept"].append(rec)
        if any(w.get("__lost") for w in doc.get("wires", [])):
            doc["wires"] = [w for w in doc["wires"] if not w.get("__lost")]
        # overrides: those that shadow a param of the old part were tuned for it; custom extras stay
        ov = inst.get("params", {})
        for k in list(ov):
            if k in old.params:
                ch["overrides"]["dropped"].append(k)
                del ov[k]
            else:
                ch["overrides"]["kept"].append(k)
        if not ov:
            inst.pop("params", None)
        inst["part"] = new.id
        # wires whose length names a param of this instance that the new part lacks (a motor's `cable_length`): the builder
        # would fail on them, so name them here; `rewire` drops them and lets autowire size them again (lead / routed length)
        have = set(new.params) | set(inst.get("params", {}))
        need = []
        for w in doc.get("wires", []):
            mm = re.fullmatch(
                rf"{re.escape(iid)}\.([A-Za-z0-9_]+)", str(w.get("length"))
            )
            if mm and mm.group(1) not in have:
                need.append((w, mm.group(1)))
        if need:
            names = [w.get("name", f"{w['from']}--{w['to']}") for w, _ in need]
            if s.get("rewire"):
                ch["wires"]["rewired"] = []
                for (w, _), nm in zip(need, names):
                    doc["wires"].remove(w)
                    spec = {"a": w["from"], "b": w["to"], "part": w.get("part")}
                    if w.get("included_in"):
                        spec["included_in"] = w["included_in"]
                    r = W.autowire(self, doc, spec)
                    for nw in doc[
                        "wires"
                    ]:  # keep the name (the assembly and its devices know the wire by it)
                        if nw.get("name") == r["wire"]["name"]:
                            nw["name"] = nm
                    ch["wires"]["rewired"].append(
                        {
                            "wire": nm,
                            "length": r["wire"].get("length"),
                            "reasons": r["reasons"].get("length"),
                        }
                    )
            else:
                ch["wires"]["need_length"] = [
                    {
                        "wire": nm,
                        "length": w["length"],
                        "missing_param": p,
                        "ends": [w["from"], w["to"]],
                    }
                    for (w, p), nm in zip(need, names)
                ]
                ch["autowire_offer"] = {
                    "op": "swap",
                    "rewire": True,
                    "note": f"{new.id} has no param(s) {sorted({p for _, p in need})} that wire(s) {names} take their length from: "
                    "swap again with rewire = True (drops them and re-runs autowire: a port `lead` length, else the routed run), "
                    f"or `set` the param on '{iid}' with provenance",
                }
        # device bindings: port-valued fields on this instance
        for d in doc.get("devices", []):
            if d.get("instance") == iid:
                for key in ("port", "a", "b", "site_port"):
                    pn = d.get(key)
                    if (
                        pn is None
                        or pn == "ground"
                        or any(p.get("name") == pn for p in new.ports)
                    ):
                        continue
                    try:
                        oldp = old.port(pn)
                    except DesignError:
                        continue
                    q, how, _ = self._new_port(new, oldp, None, old, oldp.get("kind"))
                    if q:
                        d[key] = q["name"]
                        ch["devices"].append(
                            {"device": d["name"], key: f"{pn} -> {q['name']} ({how})"}
                        )
                    else:
                        ch["dangling"].append(
                            f"device '{d['name']}' {key} = '{pn}': {new.id} has no such port"
                        )
        ch["dangling"] += self._dangling(doc, iid, new)
        # device bindings whose numbers came from the part that was replaced
        ch["stale_devices"] = AD.stale(doc, iid)
        if s.get(
            "auto"
        ):  # regenerate the generated bindings in this same edit (explicit ones only with overwrite)
            ow = (
                [x["device"] for x in ch["stale_devices"]]
                if s.get("overwrite")
                else False
            )
            ch["auto_devices"] = AD.auto_devices(
                self, doc, {"overwrite": ow, "fill": s.get("fill")}
            )
        elif ch["stale_devices"] or any(d.get("auto") for d in doc.get("devices", [])):
            ch["auto_devices_offer"] = {
                "op": "auto_devices",
                "note": "re-derives the generated bindings from the new part (hand-written ones are kept; "
                "overwrite = [device names] replaces them); unset fields are reported, never invented",
            }
        return ch

    @staticmethod
    def _dev_joints(doc, names) -> list:
        """Device bindings that name a joint (mate) that no longer exists: kept, so re-creating the mate restores them."""
        return [
            f"device '{d['name']}' {k} = '{d[k]}': that mate no longer exists (mate it again with name '{d[k]}')"
            for d in doc.get("devices", [])
            for k in ("joint", "backlash_joint")
            if d.get(k) in names
        ]

    def _dangling(self, doc, iid, new: Part) -> list:
        """References the swapped-in part cannot satisfy: '<iid>.<param>' strings, device-map params, @table.column."""
        out, params = [], {*new.params}
        inst = self._inst(doc, iid)
        have = params | set(inst.get("params", {}))
        tables = new.raw.get("tables", {})

        def ref(s, where):
            m = (
                re.fullmatch(rf"{re.escape(iid)}\.([A-Za-z0-9_]+)", s)
                if isinstance(s, str)
                else None
            )
            if m and m.group(1) not in have:
                out.append(
                    f"{where}: '{s}' no longer resolves ({new.id} has no param '{m.group(1)}')"
                )

        def walk(x, where):
            if isinstance(x, str):
                ref(x, where)
            elif isinstance(x, dict):
                for k, v in x.items():
                    walk(v, f"{where}.{k}")
            elif isinstance(x, list):
                for i, v in enumerate(x):
                    walk(v, f"{where}[{i}]")

        for sec, key in (("mates", "joint"), ("wires", "length"), ("targets", "value")):
            for i, it in enumerate(
                doc.get(sec, [])
            ):  # numeric fields only: mate / wire ends are port refs, not params
                walk(it.get(key), f"{sec}[{i}].{key}")
        for d in doc.get("devices", []):
            walk(d.get("set", {}), f"device '{d['name']}' set")
            maps = doc.get("device_maps", {}).get(d.get("map"), {})
            for fld, spec in maps.items():
                src, _, name = spec.rpartition(":") if ":" in spec else ("", "", spec)
                holder = d.get(src) if src else d.get("instance")
                if holder != iid:
                    continue
                if name.startswith("@"):
                    tb, _, col = name[1:].partition(".")
                    t = tables.get(tb)
                    if not t or col not in t.get("columns", []):
                        out.append(
                            f"device '{d['name']}' map '{d['map']}' {fld}: {new.id} has no table column {name}"
                        )
                elif name not in have:
                    out.append(
                        f"device '{d['name']}' map '{d['map']}' {fld}: {new.id} has no param '{name}' (set it on '{iid}' with provenance, or pick a part that has it)"
                    )
        return out

    # ---- transaction -------------------------------------------------------------------------------------------------
    # ---- placement and cosmetics (UI: gizmo, colour, duplicate, mirror) ------------------------------------------------------
    @staticmethod
    def _vec3(v, what):
        if not (isinstance(v, (list, tuple)) and len(v) == 3):
            raise DesignError(f"{what}: three numbers [x, y, z]")
        return np.array([float(x) for x in v])

    def _op_place(self, doc, s) -> dict:
        """Free-place instance `id` (pose in the ROOT frame, metres / radians): `pos` / `rpy` absolute, or `delta` (added to the
        position) and `rot` (3x3 matrix, premultiplied about `about` = the instance origin by default). A mated instance
        must say `unmate = true`: the mate that holds it to the root side is dropped (its own children stay mated to it)."""
        iid = s["id"]
        inst = self._inst(doc, iid)
        if inst.get("root"):
            raise DesignError(
                f"place {iid}: the root defines the frame; move the other instances"
            )
        if inst.get("offboard"):
            raise DesignError(f"place {iid}: offboard instances have no pose")
        T = self.poses(doc)[iid]
        ch = {"op": "place", "id": iid, "unmated": []}
        if "pos" not in inst and "rpy" not in inst:  # held by a mate
            held = self._tree.parent[iid][1].name
            if not s.get("unmate"):
                raise DesignError(
                    f"place {iid}: it is mated (mate '{held}'); pass unmate=true to detach it and place it freely"
                )
            self._op_unmate(doc, {"name": held})
            ch["unmated"].append(held)
        R = T[:3, :3]
        p = T[:3, 3]
        if s.get("rot") is not None:
            Rd = np.array(s["rot"], float).reshape(3, 3)
            about = self._vec3(s["about"], "about") if s.get("about") else p
            p = about + Rd @ (p - about)
            R = Rd @ R
        rpy = None
        if s.get("rpy") is not None:
            from .snap import rpy_matrix

            rpy = self._vec3(s["rpy"], "rpy")
            R = rpy_matrix(rpy)
        if s.get("pos") is not None:
            p = self._vec3(s["pos"], "pos")
        if s.get("delta") is not None:
            p = p + self._vec3(s["delta"], "delta")
        inst["pos"] = [round(float(x), 6) for x in p]
        # an rpy the caller gave is stored as written (a half turn about y is [0, pi, 0], not matrix_rpy's equivalent [pi, 0, pi])
        inst["rpy"] = [
            round(float(x), 6)
            for x in (
                rpy if rpy is not None and s.get("rot") is None else matrix_rpy(R)
            )
        ]
        ch.update(pos=inst["pos"], rpy=inst["rpy"])
        return ch

    def _op_assembly(self, doc, s) -> dict:
        """assembly: the header of the design: base = 'free' (a vehicle) | 'fixed' (welded to the world), timestep (s), description."""
        a, ch = doc["assembly"], {"op": "assembly"}
        if s.get("base") is not None:
            if s["base"] not in ("free", "fixed"):
                raise DesignError("assembly: base is 'free' or 'fixed'")
            a["base"] = ch["base"] = s["base"]
        if s.get("timestep") is not None:
            if not 0 < float(s["timestep"]) <= 0.01:
                raise DesignError("assembly: timestep is in (0, 0.01] s")
            a["timestep"] = ch["timestep"] = float(s["timestep"])
        if s.get("seed") is not None:  # the devices' noise seed
            a["seed"] = ch["seed"] = int(s["seed"])
        if s.get("description"):
            a["description"] = ch["description"] = str(s["description"])
        if len(ch) == 1:
            raise DesignError("assembly: give base, timestep, seed or description")
        return ch

    def _op_color(self, doc, s) -> dict:
        """Cosmetic colour override of instance `id`: rgba = [r, g, b(, a)] in 0..1, or None / [] to go back to the part's own."""
        inst = self._inst(doc, s["id"])
        c = s.get("rgba")
        if c:
            inst["color"] = [round(float(x), 4) for x in c]
        else:
            inst.pop("color", None)
        return {"op": "color", "id": inst["id"], "color": inst.get("color")}

    def _copy_of(self, doc, src, new_id, T, ch):
        ids = {d["id"] for d in doc["instances"]}
        if src.get("root"):
            raise DesignError(f"the root '{src['id']}' cannot be copied")
        iid = new_id or f"{src['id']}-2"
        n = 2
        while iid in ids:
            n += 1
            iid = f"{new_id or src['id']}-{n}"
        c = {
            k: copy.deepcopy(v)
            for k, v in src.items()
            if k not in ("id", "pos", "rpy", "root")
        }
        c.update(
            id=iid,
            pos=[round(float(x), 6) for x in T[:3, 3]],
            rpy=[round(float(x), 6) for x in matrix_rpy(T[:3, :3])],
        )
        doc["instances"].append(c)
        ch["created"].append(iid)

    def _op_duplicate(self, doc, s) -> dict:
        """Copy instance `id` (part, group, params, colour; no mates / wires / devices) free-placed at its pose + `delta`."""
        src = self._inst(doc, s["id"])
        T = self.poses(doc)[src["id"]].copy()
        T[:3, 3] += self._vec3(s.get("delta", [0.02, 0.02, 0.0]), "delta")
        ch = {"op": "duplicate", "id": src["id"], "created": []}
        self._copy_of(doc, src, s.get("new_id"), T, ch)
        return ch

    def _op_mirror(self, doc, s) -> dict:
        """Free-placed copies of instances `ids` mirrored across the plane axis = x|y|z at coordinate `at` (root frame). The
        position is reflected, the orientation is M R M (a proper rotation): the part itself is not chirality-flipped."""
        ax = "xyz".find(s.get("axis", "y"))
        if ax < 0:
            raise DesignError("mirror: axis is x, y or z")
        M = np.eye(3)
        M[ax, ax] = -1
        at = float(s.get("at", 0.0))
        ch = {"op": "mirror", "ids": list(s["ids"]), "created": []}
        poses = self.poses(doc)
        for iid in s["ids"]:
            src = self._inst(doc, iid)
            T = np.eye(4)
            p = poses[iid][:3, 3].copy()
            p[ax] = 2 * at - p[ax]
            T[:3, :3], T[:3, 3] = M @ poses[iid][:3, :3] @ M, p
            self._copy_of(doc, src, f"{iid}-m", T, ch)
        return ch

    # ---- sub-assemblies (subasm.py): ops edit the document as written; edit() re-flattens and validates ------------------
    @staticmethod
    def _sub(doc, sid):
        subs = doc.get("subassemblies", [])
        for s in subs:
            if s["id"] == sid:
                return s
        raise DesignError(
            f"unknown sub-assembly '{sid}'" + _didyoumean(sid, [s["id"] for s in subs])
        )

    def _op_sub_insert(self, doc, s) -> dict:
        """sub_insert: file (relative to this design's file, or absolute), id, then mate = {a: '<inst>.<port>', export: '<port
        export of the child>', name?} | pos / rpy (welded like a placed instance); optional mirror ('xz' | 'yz' | 'xy'), group, color."""
        if self.asm.path is None:
            raise DesignError(
                "sub.insert: save the design first (sub-assembly files are relative to it)"
            )
        sid = s.get("id")
        if not sid or not s.get("file"):
            raise DesignError("sub.insert: needs id and file")
        if sid in {x["id"] for x in doc.get("subassemblies", [])}:
            raise DesignError(f"a sub-assembly '{sid}' already exists")
        file = str(s["file"])
        if Path(file).is_absolute():
            file = os.path.relpath(file, Path(self.asm.path).resolve().parent)
        ent = {"id": sid, "file": file}
        for k in ("pos", "rpy", "mirror", "group", "color"):
            if s.get(k) is not None:
                ent[k] = s[k]
        doc.setdefault("subassemblies", []).append(ent)
        ch = {"op": "sub_insert", "id": sid, "file": file}
        if m := s.get("mate"):
            name = m.get("name") or f"{str(m['a']).partition('.')[0]}--{sid}"
            if name in {x.get("name") for x in doc.get("mates", [])}:
                raise DesignError(f"a mate named '{name}' already exists")
            doc.setdefault("mates", []).append(
                {"name": name, "a": m["a"], "b": f"{sid}:{m['export']}"}
            )
            ch["mate"] = name
        return ch

    def _op_sub_mirror(self, doc, s) -> dict:
        """sub_mirror: id, plane ('xz' | 'yz' | 'xy'). Default: a mirrored copy '<id>_m' (same file / params / group; welded at the
        reflected pos when the original has one; `to` = '<inst>.<port>' mates the copy's same export as the original's mate
        to that port). toggle = true: flip the mirror of `id` itself (set it, or clear it when already that plane)."""
        src, plane = self._sub(doc, s["id"]), s.get("plane")
        if plane not in ("xz", "yz", "xy"):
            raise DesignError("sub.mirror: plane is 'xz', 'yz' or 'xy'")
        if s.get("toggle"):
            if src.get("mirror") == plane:
                del src["mirror"]
            else:
                src["mirror"] = plane
            return {"op": "sub_mirror", "id": src["id"], "mirror": src.get("mirror")}
        nid = s.get("new_id") or f"{src['id']}_m"
        if nid in {x["id"] for x in doc["subassemblies"]}:
            raise DesignError(f"a sub-assembly '{nid}' already exists")
        if any(abs(float(x)) > 0 for x in src.get("rpy", ())):
            raise DesignError(
                "sub.mirror: a rotated (rpy) sub-assembly cannot be copied by reflection; mate the copy with `to`"
            )
        cp = copy.deepcopy(src)
        cp.update(id=nid, mirror=plane)
        if "pos" in cp:
            ax = {"yz": 0, "xz": 1, "xy": 2}[plane]
            cp["pos"][ax] = -cp["pos"][ax]
        doc["subassemblies"].append(cp)
        ch = {"op": "sub_mirror", "id": nid, "of": src["id"], "mirror": plane}
        if s.get("to"):
            mate = next(
                (
                    m
                    for m in doc.get("mates", [])
                    if str(m["b"]).startswith(src["id"] + ":")
                ),
                None,
            )
            if mate is None:
                raise DesignError(
                    f"sub.mirror: '{src['id']}' has no mate to copy; pass pos / use sub.insert"
                )
            name = f"{str(s['to']).partition('.')[0]}--{nid}"
            doc.setdefault("mates", []).append(
                {
                    "name": name,
                    "a": s["to"],
                    "b": nid + mate["b"][len(src["id"]) :],
                }
            )
            ch["mate"] = name
        return ch

    def _op_sub_override(self, doc, s) -> dict:
        """sub_override: id, name (an exported param of the child), value (a provenance table; None removes the override)."""
        sub = self._sub(doc, s["id"])
        if s.get("value") is None:
            sub.get("params", {}).pop(s["name"], None)
            if not sub.get("params"):
                sub.pop("params", None)
        else:
            sub.setdefault("params", {})[s["name"]] = s["value"]
        return {"op": "sub_override", "id": s["id"], "name": s["name"]}

    KINDS = {"port": "ports", "connector": "connectors", "param": "params"}

    def _op_sub_export(self, doc, s) -> dict:
        """sub_export: kind (port | connector | param), ref '<inst>.<x>' (or '<sub>:<export>'), name: adds it to [exports]."""
        k = self.KINDS.get(s.get("kind"))
        if k is None:
            raise DesignError("sub.export: kind is port, connector or param")
        name, ref = s.get("name"), s.get("ref", "")
        if not name or not str(name).isidentifier():
            raise DesignError("sub.export: name is a plain identifier")
        head = ref.partition(".")[0]
        if ":" not in head and head not in {i["id"] for i in doc["instances"]}:
            raise DesignError(
                f"sub.export: unknown instance '{head}'"
                + _didyoumean(head, [i["id"] for i in doc["instances"]])
            )
        tab = doc.setdefault("exports", {}).setdefault(k, {})
        if name in tab:
            raise DesignError(f"sub.export: {s['kind']} export '{name}' exists")
        tab[name] = ref
        return {"op": "sub_export", "kind": s["kind"], "name": name, "ref": ref}

    def _op_sub_unexport(self, doc, s) -> dict:
        k = self.KINDS.get(s.get("kind"))
        tab = doc.get("exports", {}).get(k or "", {})
        if k is None or s.get("name") not in tab:
            raise DesignError(
                f"sub.unexport: no {s.get('kind')} export '{s.get('name')}'"
            )
        del tab[s["name"]]
        if not tab:
            del doc["exports"][k]
        if not doc["exports"]:
            del doc["exports"]
        return {"op": "sub_unexport", "kind": s["kind"], "name": s["name"]}

    def _expand(self, ops):
        out = []
        for o in ops:
            if o.get("op") == "swap" and isinstance(o.get("id"), list):
                out += [{**o, "id": i} for i in o["id"]]
            else:
                out.append(o)
        return out

    def edit(self, ops: list) -> EditResult:
        """Apply a list of op dicts ({"op": add|remove|swap|mate|unmate|wire|unwire|set, ...}) atomically: all or none."""
        table = {
            "add": self._op_add,
            "remove": self._op_remove,
            "swap": self._op_swap,
            "mate": self._op_mate,
            "unmate": self._op_unmate,
            "wire": self._op_wire,
            "unwire": self._op_unwire,
            "set": self._op_set,
            "place": self._op_place,
            "color": self._op_color,
            "assembly": self._op_assembly,
            "duplicate": self._op_duplicate,
            "mirror": self._op_mirror,
            "device_map": self._op_device_map,
            "device": self._op_device,
            "target": self._op_target,
            "sub_insert": self._op_sub_insert,
            "sub_mirror": self._op_sub_mirror,
            "sub_override": self._op_sub_override,
            "sub_export": self._op_sub_export,
            "sub_unexport": self._op_sub_unexport,
            "autowire": lambda doc, s: W.autowire(self, doc, s),
            "autowire_all": lambda doc, s: W.autowire_all(self, doc, s),
            "pinwire": lambda doc, s: W.pinwire(self, doc, s),
            "auto_devices": lambda doc, s: AD.auto_devices(self, doc, s),
        }
        before = self.findings()
        raw = self.asm.raw  # ops edit the document as written (== the flat one without sub-assemblies), then it is re-flattened
        snap = copy.deepcopy(raw)
        changes = []
        try:
            for o in self._expand(ops):
                fn = table.get(o.get("op"))
                if fn is None:
                    raise DesignError(
                        f"unknown op {o.get('op')!r} (allowed: {sorted(table)})"
                    )
                changes.append(fn(raw, {k: v for k, v in o.items() if k != "op"}))
            A.validate(raw)
            if raw.get("subassemblies"):
                self.asm.set_raw(raw)
                A.validate(self.doc, self.asm.origins)
            elif self.asm.origins:  # the last sub-assembly was removed
                self.asm.set_raw(raw)
            for d in self.doc[
                "instances"
            ]:  # unknown part ids surface here, not at build time
                self._part(d["part"])
        except DesignError:
            self.asm.set_raw(snap)
            raise
        except (KeyError, TypeError, ValueError, StopIteration) as e:
            self.asm.set_raw(snap)
            raise DesignError(
                f"bad edit ({type(e).__name__}: {e}); nothing changed"
            ) from None
        self._undo.append(snap)
        self._redo.clear()
        self._undo_notes.append(_note(ops))
        del self._redo_notes[:]
        return self._result(changes, before)

    def _result(self, changes, before) -> EditResult:
        after = self.check()
        new, gone, chg = finding_delta(before, after)
        return EditResult(changes, new, gone, chg, _counts(before), _counts(after))

    # ---- public edit methods -----------------------------------------------------------------------------------------
    def add(self, part, id=None, to=None, port=None, k=None, **kw) -> EditResult:
        return self.edit(
            [
                {
                    "op": "add",
                    "part": part,
                    "id": id,
                    "to": to,
                    "port": port,
                    "k": k,
                    **kw,
                }
            ]
        )

    def snap(
        self, new_part, to, port=None, k=None, id=None, strict=False, **kw
    ) -> EditResult:
        """add(new_part) mated to the best open compatible port pair of `to` (instance or 'inst.port'); the change record
        has snap = {score, why, ambiguous, alternatives}; strict=True raises when the best pair is tied. Extra keywords
        (joint=..., fasteners=..., nut=..., group=...) go to the mate / instance like `mate` / `add` take them."""
        return self.add(new_part, id=id, to=to, port=port, k=k, strict=strict, **kw)

    def remove(self, id) -> EditResult:
        return self.edit([{"op": "remove", "id": id}])

    def swap(
        self, id, new_part, auto=False, overwrite=False, fill=None, rewire=False
    ) -> EditResult:
        """Replace the part of instance `id`. The change record lists `stale_devices` (bindings that name the instance).
        auto = True also regenerates the generated device bindings in the same edit (`changes[0].auto_devices`);
        overwrite = True lets that replace the stale hand-written ones too; fill = auto_devices' fill.
        A new part that cannot be mounted where the old one sat (another hole pattern / thread) is refused (DesignError).
        Wires whose length names a param the new part lacks are listed in `wires.need_length` with an `autowire_offer`;
        rewire = True drops and re-runs autowire on them in the same edit (`wires.rewired`)."""
        return self.edit(
            [
                {
                    "op": "swap",
                    "id": id,
                    "part": new_part,
                    "auto": auto,
                    "overwrite": overwrite,
                    "fill": fill,
                    "rewire": rewire,
                }
            ]
        )

    def auto_devices(
        self, scope=None, overwrite=False, fill=None, ocv=None
    ) -> EditResult:
        """Derive the device bindings the parts + topology imply (auto_devices.py). scope = instance id(s) (default: all);
        overwrite = True | [device names]: replace hand-written bindings too (generated ones are always regenerated);
        fill = {device-name glob: {"set": {field: provenance table}, binding field: value}} for what no part carries;
        ocv = the OCV curve part for LiPo packs. The change record lists created / refreshed / unchanged / kept / removed,
        and `unset`: the fields no part provides (never invented)."""
        return self.edit(
            [
                {
                    "op": "auto_devices",
                    "scope": scope,
                    "overwrite": overwrite,
                    "fill": fill,
                    "ocv": ocv,
                }
            ]
        )

    def mate(self, a, b, k=None, joint=None, fasteners="auto", **kw) -> EditResult:
        return self.edit(
            [
                {
                    "op": "mate",
                    "a": a,
                    "b": b,
                    "k": k,
                    "joint": joint,
                    "fasteners": fasteners,
                    **kw,
                }
            ]
        )

    def unmate(self, name=None, a=None, b=None) -> EditResult:
        return self.edit([{"op": "unmate", "name": name, "a": a, "b": b}])

    def unwire(self, name) -> EditResult:
        return self.edit([{"op": "unwire", "name": name}])

    def wire(self, from_, to, part, length, **kw) -> EditResult:
        return self.edit(
            [
                {
                    "op": "wire",
                    "from": from_,
                    "to": to,
                    "part": part,
                    "length": length,
                    **kw,
                }
            ]
        )

    def set(
        self,
        id,
        param,
        value=None,
        unit=None,
        fidelity=None,
        src=None,
        quote=None,
        derivation=None,
        rationale=None,
        unset=False,
    ) -> EditResult:
        return self.edit(
            [
                {
                    "op": "set",
                    "id": id,
                    "param": param,
                    "value": value,
                    "unit": unit,
                    "fidelity": fidelity,
                    "src": src,
                    "quote": quote,
                    "derivation": derivation,
                    "rationale": rationale,
                    "unset": unset,
                }
            ]
        )

    def autowire(
        self, a, b=None, kind=None, part=None, length=None, included_in=None
    ) -> EditResult:
        """Wire the open power / signal port `a` ('inst.port') to the open port that needs it (or to `b`): wire part = the
        lightest library wire rated for the worst-case current x WIRE_MARGIN, length = port distance x ROUTING_FACTOR
        (rounded up to a stocked length), mating connectors pinned. The change record has wire + reasons. Several equally
        good targets (not interchangeable outlets of one part) raise; name `b`. `part` / `length` override the choice.
        `included_in` = the instance whose datasheet mass already contains this lead (a motor's own 150 mm leads): the wire
        keeps its electrical model but adds no mass."""
        return self.edit(
            [
                {
                    "op": "autowire",
                    "a": a,
                    "b": b,
                    "kind": kind,
                    "part": part,
                    "length": length,
                    "included_in": included_in,
                }
            ]
        )

    def autowire_all(self, kind=None, included_in=None) -> EditResult:
        """Auto-wire every open power / signal port that has one sensible target; the change record lists `wired`,
        `ambiguous` (one entry per decision: {ports, candidates}, several equally good targets, not guessed) and
        `unmatched` (a floating power port with no target). `included_in` = {"inst.port": instance}: a wire that ends on that
        port is a lead already inside the instance's datasheet mass (the pack's 120 mm XT60 lead): no mass counted twice."""
        return self.edit(
            [{"op": "autowire_all", "kind": kind, "included_in": included_in}]
        )

    def undo(self) -> EditResult:
        return self._travel(self._undo, self._redo, "undo")

    def redo(self) -> EditResult:
        return self._travel(self._redo, self._undo, "redo")

    def _travel(self, src, dst, what) -> EditResult:
        if not src:
            raise DesignError(f"nothing to {what}")
        before = self.findings()
        cur = copy.deepcopy(self.asm.raw)
        self.asm.set_raw(src[-1])
        src.pop()  # only after the snapshot re-flattened (a child gone from disk keeps the stacks intact)
        dst.append(cur)
        a, b = (
            (self._undo_notes, self._redo_notes)
            if what == "undo"
            else (self._redo_notes, self._undo_notes)
        )
        if a:
            b.append(a.pop())
        return self._result([{"op": what}], before)

    # ---- suggestions -------------------------------------------------------------------------------------------------
    def suggest(self, target: str, limit: int = 10, category=None) -> list:
        """target = 'inst.port': library parts (with the port they would use) that mate / connect that port, best first.
        target = 'inst': replacement candidates for that instance, best fit first (what swap would keep / lose).
        category = a dotted category prefix ('esc', 'motor.bldc', 'pcb'): only parts of that category, applied before `limit`
        (a flat-mount seat is offered every board in the library; the filter is how a caller asks for "an ESC for this seat").
        target = a joint (a mate that carries `joint`, e.g. 'shoulder_lift'): servos ranked by holding-torque margin at that
        joint's worst static pose (drc.joint_holding; stall torque at the bus voltage / the gravity torque of the SERVO_TORQUE
        rule; rows carry margin, stall_Nm, tau_worst_Nm, mates_fit)."""
        if any(
            m.get("joint") and m.get("name") == target
            for m in self.doc.get("mates", [])
        ):
            return joint_holding(self.asm, self.lib, target)["rows"][:limit]
        iid, pn = _split(target)
        inst = self._inst(self.doc, iid)
        part = self._part(inst["part"])
        return (
            self._suggest_port(iid, part, part.port(pn), limit, category)
            if pn
            else self._suggest_swap(inst, part, limit)
        )

    def _suggest_port(self, iid, part: Part, port: dict, limit, category=None) -> list:
        by_id = {d["id"]: d for d in self.doc["instances"]}
        kind = port.get("kind")
        # categories / parts seen on ports of this type in the assembly (the occupant of this very port is the stock part)
        cats, stock = set(), set()
        for m in self.doc.get("mates", []):
            for e, o in (("a", "b"), ("b", "a")):
                ei, ep = _split(m[e])
                oi, op = _split(m[o])
                if ei not in by_id or oi not in by_id:
                    continue
                pp = self._part(by_id[ei]["part"])
                try:
                    q = pp.port(ep) if ep else None
                except DesignError:
                    q = None
                if (
                    q
                    and pp.category == part.category
                    and types_ok(q.get("type", ""), port.get("type", ""))
                ):
                    other = self._part(by_id[oi]["part"])
                    cats.add(other.category)
                    if (ei, ep) == (iid, port["name"]):
                        stock.add(other.id)
        top = {c.split(".")[0] for c in cats}
        rows = []
        for pid, cand in sorted(self.lib.parts.items()):
            if category and not (
                cand.category == category or cand.category.startswith(category + ".")
            ):
                continue
            for q in cand.ports:
                if q.get("kind") != kind:
                    continue
                row = self._score_port(port, q, kind)
                if row is None:
                    continue
                s, why = row
                if cand.category in cats:
                    s += W_CATEGORY
                    why.append(
                        f"category {cand.category} is used on {port.get('type')} ports here"
                    )
                elif cand.category.split(".")[0] in top:
                    s += W_FAMILY_CAT
                    why.append("same part family as the neighbours on this port type")
                if cand.category == part.category:
                    s -= W_CATEGORY  # a peer of the part that owns the port is rarely what plugs into it
                    why.append(f"same category as {part.id}")
                if pid in stock:
                    s += W_STOCK
                    why.append("stock part (currently on this port)")
                rows.append(
                    {
                        "part": pid,
                        "port": q["name"],
                        "category": cand.category,
                        "score": round(s, 2),
                        "why": why,
                        "mass": cand.value("mass"),
                        "name": cand.raw.get("name", ""),
                    }
                )
        rows.sort(key=lambda r: (-r["score"], r["part"], r["port"]))
        rows = rows[:limit]
        for (
            r
        ) in rows:  # what the part would need next, from the kinds of its other ports
            r["needs"] = self._needs(self.doc, self.lib.get(r["part"]), {r["port"]})
        return rows

    def _needs(self, doc, part, skip=()):
        return W.needs(self, doc, part, skip)

    def _score_port(self, port, q, kind):
        if kind == MECH:
            r = self._pairs([port], [q])
            return (r[0]["score"], list(r[0]["why"])) if r else None
        if not (
            types_ok(port.get("type", ""), q.get("type", ""))
            and genders_ok(port.get("gender", ""), q.get("gender", ""))
        ):
            return None
        ra, rb = port.get("role"), q.get("role")
        if ra == rb and ra in ("source", "sink"):
            return None  # source-source / sink-sink cannot be one net
        s = (W_EXACT if port.get("type") == q.get("type") else W_FAMILY) + W_NEUTRAL
        why = [
            "same port type" if port.get("type") == q.get("type") else "paired family"
        ]
        # ratings: the sink must tolerate the source's voltage; the weaker current rating limits the path
        src, snk = (port, q) if ra == "source" or rb == "sink" else (q, port)
        vmax_src, vmax_snk = _num(src.get("v_max")), _num(snk.get("v_max"))
        if vmax_src and vmax_snk and vmax_snk < vmax_src:
            s += W_RATING_BAD
            why.append(f"v_max {vmax_snk} V below the source's {vmax_src} V")
        im_t, im_q = _num(port.get("i_max")), _num(q.get("i_max"))
        if im_t and im_q and im_q < im_t:
            s += W_RATING_BAD
            why.append(f"i_max {im_q} A below this port's {im_t} A")
        return s, why

    def _suggest_swap(self, inst, old: Part, limit) -> list:
        pool = [
            p
            for p in self.lib.parts.values()
            if p.id != old.id and p.category == old.category
        ]
        fam = old.category.split(".")[0]
        pool += [
            p
            for p in self.lib.parts.values()
            if p.id != old.id
            and p.category != old.category
            and p.category.split(".")[0] == fam
        ]
        rows = []
        for cand in pool:
            doc = copy.deepcopy(self.doc)
            try:
                ch = self._op_swap(
                    doc, {"id": inst["id"], "part": cand.id, "refuse_mount": False}
                )
            except DesignError:
                continue
            md = None
            if cand.value("mass") is not None and old.value("mass") is not None:
                md = round(cand.value("mass") - old.value("mass"), 6)
            deficits = []
            for k in ch["mates"]["kept"] + ch["wires"]["kept"]:
                try:
                    po, pn_ = old.port(k["old_port"]), cand.port(k["new_port"])
                except DesignError:
                    continue
                for key in ("v_max", "i_max", "i_burst"):
                    a, b = _num(po.get(key)), _num(pn_.get(key))
                    if a and b is not None and b < a:
                        deficits.append(f"{k['new_port']}.{key} {b} < {a}")
            lost = len(ch["mates"]["lost"]) + len(ch["wires"]["lost"])
            same_cat = cand.category == old.category
            tables = set(old.raw.get("tables", {})) - set(cand.raw.get("tables", {}))
            rows.append(
                {
                    "part": cand.id,
                    "category": cand.category,
                    "name": cand.raw.get("name", ""),
                    "mates_kept": len(ch["mates"]["kept"]),
                    "mates_lost": [m["mate"] for m in ch["mates"]["lost"]],
                    "wires_kept": len(ch["wires"]["kept"]),
                    "wires_lost": [w["wire"] for w in ch["wires"]["lost"]],
                    "dangling": ch["dangling"],
                    "rating_deficits": deficits,
                    "missing_tables": sorted(tables),
                    "mass_delta": md,
                    "_k": (
                        not same_cat,
                        lost,
                        len(ch["dangling"]),
                        len(deficits),
                        len(tables),
                        abs(md) if md is not None else 1e9,
                        cand.id,
                    ),
                }
            )
        rows.sort(key=lambda r: r["_k"])
        for r in rows:
            del r["_k"]
        return rows[:limit]


def _didyoumean(x, names, what="") -> str:
    m = difflib.get_close_matches(x, names, n=3)
    return f" (did you mean {', '.join(m)}?)" if m else ""
