"""Assembly files (assemblies/<name>.toml): load + schema + numbers rule, save, and the resolution context.

NUMBERS RULE: every numeric leaf is a reference string "<inst>.<param>" or a provenance table
{value, unit, fidelity, + src & quote | derivation | rationale}. Bare numbers only in [assembly] and for the keys
rotation / count / conductors / dir.
"""

import json
import math
import re
import tomllib
from pathlib import Path

from . import pins, subasm
from .library import DesignError, Library, Part

FIDELITY = {"datasheet", "literature", "measured", "derived", "estimated"}
SOURCED = {"datasheet", "literature", "measured"}
BARE_OK = {
    "rotation",
    "count",
    "conductors",
    "dir",
    "timeout_mode",  # qdd / foot_contact: enum selectors (0 / 1), not physical numbers
    "mode",
    "shape",  # foot_contact: geometry shape index
    "pos",
    "rpy",
    "color",
}  # pos / rpy / color: instance placement + cosmetics
TOP = [
    "assembly",
    "exports",
    "subassemblies",
    "instances",
    "mates",
    "wires",
    "device_maps",
    "devices",
    "targets",
]
ALLOWED = {
    "subassemblies": {"id", "file", "pos", "rpy", "mirror", "params", "group", "color"},
    "assembly": {
        "name",
        "description",
        "base",
        "timestep",
        "seed",
        "mesh_inertia",
        "root_frame",
    },
    "instances": {
        "id",
        "part",
        "root",
        "group",
        "offboard",
        "params",
        "pos",
        "rpy",
        "color",
        "_mirror",  # set by subasm.expand on the instances of a mirrored sub-assembly; never written
    },
    "mates": {"name", "a", "b", "rotation", "fasteners", "nut", "joint", "closure"},
    "wires": {
        "name",
        "from",
        "to",
        "part",
        "length",
        "conductors",
        "included_in",
        "connectors",
        "from_pins",
        "to_pins",
    },
    "targets": {"name", "groups", "all", "value", "reach", "workspace"},
}
JOINT_KEYS = {"type", "range", "reverse", "spin"}
DEVICE_TYPES = {
    "battery",
    "psu",
    "wire",
    "load",
    "esc",
    "rotor",
    "servo",
    "imu",
    "baro",
    "mag",
    "gnss",
    "environment",
    "body_drag",
    "launcher",
    "camera",
    "solid_motor",
    "airframe",
    "sea",
    "vessel",
    "rangefinder",
    "jet_vane",
    "qdd",
    "foot_contact",
}
UNITS = {
    "m": {"m": 1.0},
    "kg": {"kg": 1.0},
    "rad": {"rad": 1.0, "deg": math.pi / 180},
}


class Assembly:
    """A validated assembly document (plain dict, as parsed). `doc` is the flattened document everything downstream reads;
    `raw` is the unexpanded one as written (what `dumps` saves; the same dict as `doc` when there are no sub-assemblies);
    `origins[flat_id] = {sub, file, id}` for every flattened item (subasm.py)."""

    def __init__(self, doc: dict, path=None, raw=None, origins=None):
        self._same = raw is None
        self._doc = doc
        self.raw = doc if raw is None else raw
        self.origins = origins or {}
        self.path = path

    @property
    def doc(self) -> dict:
        return self._doc

    @doc.setter
    def doc(self, v: dict):
        self._doc = v
        if (
            self._same
        ):  # no sub-assemblies: undo snapshots etc. replace the one document
            self.raw = v

    def set_raw(self, raw: dict):
        """Replace the unexpanded document and re-expand it (re-reads the child files). Raises DesignError, state unchanged."""
        flat, origins = subasm.expand(raw, self.path)
        self.raw, self._doc, self.origins, self._same = raw, flat, origins, flat is raw

    def __eq__(self, other):
        return isinstance(other, Assembly) and self.doc == other.doc

    @property
    def name(self) -> str:
        return self.doc["assembly"]["name"]


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def check_provenance(t: dict, where: str):
    f = t.get("fidelity")
    v = t.get("value")
    if f not in FIDELITY:
        raise DesignError(f"{where}: fidelity {f!r} not in {sorted(FIDELITY)}")
    if not (_is_num(v) or (isinstance(v, list) and v and all(_is_num(x) for x in v))):
        raise DesignError(
            f"{where}: provenance value must be a number or a list of numbers"
        )
    if not isinstance(t.get("unit"), str):
        raise DesignError(f"{where}: provenance needs a string 'unit'")
    need = {"derived": ["derivation"], "estimated": ["rationale"]}.get(
        f, ["src", "quote"]
    )
    for k in need:
        if not t.get(k):
            raise DesignError(f"{where}: {f} provenance needs '{k}'")


def _walk(x, path: list):
    where = ".".join(str(p) for p in path)
    if _is_num(x):
        keys = [p for p in path if isinstance(p, str)]
        if keys[0] == "assembly" or keys[-1] in BARE_OK:
            return
        raise DesignError(
            f"bare number at '{where}': use a '<inst>.<param>' reference or a provenance table "
            "{value, unit, fidelity, src+quote | derivation | rationale}"
        )
    if isinstance(x, dict):
        if {"value", "unit", "fidelity"} <= x.keys():
            return check_provenance(x, where)
        for k, v in x.items():
            _walk(v, path + [k])
    elif isinstance(x, list):
        for i, v in enumerate(x):
            _walk(v, path + [i])


def _need(d: dict, keys, where: str):
    for k in keys:
        if k not in d:
            raise DesignError(f"{where}: missing '{k}'")


def validate(doc: dict, origins: dict | None = None):
    """`origins` (subasm.expand): flattened ids may contain ':' (and are not re-checked against the written-id regex)."""
    origins = origins or {}
    for k in doc:
        if k not in TOP:
            raise DesignError(f"unknown top-level section '{k}' (allowed: {TOP})")
    _need(doc, ["assembly", "instances"], "assembly file")
    for sec, allowed in ALLOWED.items():
        items = doc.get(sec, [])
        for i, it in enumerate(items if isinstance(items, list) else [items]):
            if not isinstance(it, dict):
                raise DesignError(f"[{sec}] entry {i} is not a table")
            bad = set(it) - allowed
            if bad:
                raise DesignError(
                    f"[{sec}] entry {i}: unknown keys {sorted(bad)} (allowed: {sorted(allowed)})"
                )
    a = doc["assembly"]
    _need(a, ["name"], "[assembly]")
    if a.get("base", "free") not in ("free", "fixed"):
        raise DesignError("[assembly] base must be 'free' or 'fixed'")
    ids, roots = set(), []
    for i, it in enumerate(doc["instances"]):
        _need(it, ["id", "part"], f"[[instances]] {i}")
        if not isinstance(it["id"], str) or not (
            it["id"] in origins or re.fullmatch(r"[A-Za-z0-9_-]+", it["id"])
        ):
            raise DesignError(
                f"instance id {it['id']!r}: use letters, digits, '_' and '-' only"
            )
        if it["id"] in ids:
            raise DesignError(f"duplicate instance id '{it['id']}'")
        ids.add(it["id"])
        if it.get("root"):
            roots.append(it)
        for k, ok in (("pos", (3,)), ("rpy", (3,)), ("color", (3, 4))):
            v = it.get(k)
            if v is None:
                continue
            if (
                not isinstance(v, list)
                or len(v) not in ok
                or not all(_is_num(x) for x in v)
            ):
                raise DesignError(
                    f"instance '{it['id']}' {k}: a list of {' or '.join(map(str, ok))} numbers"
                )
            if k == "color" and not all(0 <= x <= 1 for x in v):
                raise DesignError(
                    f"instance '{it['id']}' color: rgb(a) components in 0..1"
                )
        if it.get("root") and ("pos" in it or "rpy" in it):
            raise DesignError(
                f"instance '{it['id']}': the root defines the frame, it has no pos / rpy"
            )
        for k, v in it.get("params", {}).items():
            if not isinstance(v, dict) or not {"value", "unit", "fidelity"} <= v.keys():
                raise DesignError(
                    f"instance '{it['id']}' params.{k}: an override must be a provenance table"
                )
    if len(roots) != 1:
        raise DesignError(
            f"exactly one instance must have root = true (found {len(roots)})"
        )
    if roots[0].get("offboard"):
        raise DesignError("the root instance cannot be offboard")
    for i, m in enumerate(doc.get("mates", [])):
        _need(m, ["a", "b"], f"[[mates]] {i}")
        f = m.get("fasteners")
        if (
            f is not None
            and f != "auto"
            and not (
                isinstance(f, dict)
                and "part" in f
                and set(f) <= {"part", "count", "nut"}
            )
        ):
            raise DesignError(
                f'[[mates]] {i}: fasteners must be "auto" or {{part=..., count=...}}'
            )
        j = m.get("joint")
        if j is not None:
            if (
                not isinstance(j, dict)
                or set(j) - JOINT_KEYS
                or j.get("type") not in ("hinge", "slide", "free")
            ):
                raise DesignError(
                    f"[[mates]] {i}: joint = {{type = hinge|slide|free, range, reverse, spin}}"
                )
            if m.get("closure") and (j["type"] != "hinge" or j.get("spin")):
                raise DesignError(
                    f"[[mates]] {i}: a closure mate can only have a limited hinge joint (type = hinge, no spin)"
                )
            if j.get("spin") and ("range" in j or j["type"] != "hinge"):
                raise DesignError(
                    f"[[mates]] {i}: spin = true is an unlimited hinge: no range, type must be hinge"
                )
    for i, w in enumerate(doc.get("wires", [])):
        _need(w, ["from", "to", "part", "length"], f"[[wires]] {i}")
    for i, d in enumerate(doc.get("devices", [])):
        _need(d, ["type", "name"], f"[[devices]] {i}")
        if d["type"] not in DEVICE_TYPES:
            raise DesignError(
                f"[[devices]] {i}: type {d['type']!r} not in {sorted(DEVICE_TYPES)}"
            )
        if d.get("map") and d["map"] not in doc.get("device_maps", {}):
            raise DesignError(
                f"[[devices]] {i}: unknown device_maps table '{d['map']}'"
            )
    for name, m in doc.get("device_maps", {}).items():
        if not isinstance(m, dict) or not all(isinstance(v, str) for v in m.values()):
            raise DesignError(
                f'device_maps.{name}: must map field paths to "<param>" or "<field>:<param>" strings'
            )
    for i, t in enumerate(doc.get("targets", [])):
        _need(t, ["name"], f"[[targets]] {i}")
        if ("reach" in t or "workspace" in t) and (
            "value" in t or "groups" in t or "all" in t
        ):
            raise DesignError(
                f"[[targets]] {i}: a reach / workspace target has no value / groups / all (a mass target does)"
            )
        if "reach" in t or "workspace" in t:
            if not isinstance(t.get("reach", []), list):
                raise DesignError(
                    f"[[targets]] {i}: reach = a list of provenance tables, value = [x, y, z] m"
                )
            continue
        _need(t, ["value"], f"[[targets]] {i}")
        if ("groups" in t) == bool(t.get("all")):
            raise DesignError(
                f"[[targets]] {i}: give exactly one of groups = [...] or all = true"
            )
    _walk(doc, [])


def loads(text: str, path=None) -> Assembly:
    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise DesignError(f"assembly TOML error: {e}") from None
    flat, origins = subasm.expand(doc, path)
    validate(flat, origins)
    return Assembly(flat, path, raw=None if flat is doc else doc, origins=origins)


def load(path) -> Assembly:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as e:
        raise DesignError(f"cannot read assembly '{path}': {e}") from None
    return loads(text, path)


# ---- minimal TOML writer for the subset above ----------------------------------------------------------------------
_BARE = re.compile(r"[A-Za-z0-9_-]+")


def _k(k: str) -> str:
    return k if _BARE.fullmatch(k) else json.dumps(k, ensure_ascii=False)


def _v(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise DesignError(f"cannot write non-finite number {v!r}")
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        return "[" + ", ".join(_v(x) for x in v) + "]"
    if isinstance(v, dict):
        return (
            "{ " + ", ".join(f"{_k(k)} = {_v(x)}" for k, x in v.items()) + " }"
            if v
            else "{}"
        )
    raise DesignError(f"cannot write {type(v).__name__} to TOML")


def dumps(asm: Assembly) -> str:
    out = []
    for sec in TOP:
        if sec not in asm.raw:
            continue
        v = asm.raw[sec]
        if sec in ("assembly", "exports"):
            out += [f"[{sec}]"] + [f"{_k(k)} = {_v(x)}" for k, x in v.items()] + [""]
        elif sec == "device_maps":
            for name, m in v.items():
                out += (
                    [f"[device_maps.{_k(name)}]"]
                    + [f"{_k(k)} = {_v(x)}" for k, x in m.items()]
                    + [""]
                )
        else:
            for it in v:
                out += (
                    [f"[[{sec}]]"]
                    + [f"{_k(k)} = {_v(x)}" for k, x in it.items()]
                    + [""]
                )
    return "\n".join(out)


def save(asm: Assembly, path) -> None:
    Path(path).write_text(dumps(asm), encoding="utf-8")


# ---- resolution context --------------------------------------------------------------------------------------------
class Inst:
    def __init__(self, d: dict, part: Part):
        self.id, self.part = d["id"], part
        self.orig_part = part  # as written; `part` is its opposite-hand part when the instance is mirrored
        self.mirror = (
            None  # {plane, sub, frame} (subasm.py) when inside a mirrored sub-assembly
        )
        self.group = d.get("group", "")
        self.offboard = bool(d.get("offboard"))
        self.root = bool(d.get("root"))
        self.overrides = d.get("params", {})
        self.color = d.get("color")  # cosmetic rgb(a) override, never physics
        # free-placed: welded to the root at pos / rpy (root frame; rpy = URDF fixed-axis roll, pitch, yaw), no mate needed
        self.placed = "pos" in d or "rpy" in d
        self.pos, self.rpy = d.get("pos", [0, 0, 0]), d.get("rpy", [0, 0, 0])

    def entry(self, name: str):
        return self.overrides.get(name, self.part.params.get(name))

    def value(self, name: str):
        e = self.entry(name)
        v = e.get("value") if isinstance(e, dict) else None
        return v if _is_num(v) else None


class Ctx:
    """Library + instances + the collectors that end up in the report."""

    def __init__(self, asm: Assembly, lib: Library):
        self.asm, self.doc, self.lib = asm, asm.doc, lib
        pins.normalize_wires(
            self.doc, lib
        )  # `from = "inst.CONN.PIN"` -> port + from_pins
        self.insts: dict[str, Inst] = {}
        for d in self.doc["instances"]:
            inst = Inst(d, lib.get(d["part"]))
            inst.mirror = d.get("_mirror")
            if inst.mirror and (opp := inst.orig_part.mirror.get("opposite")):
                if opp not in lib.parts:
                    raise DesignError(
                        f"mirror: part {inst.orig_part.id} names opposite '{opp}', which is not in the library"
                    )
                inst.part = lib.get(
                    opp
                )  # the opposite-hand part stands in for the mirrored copy
            self.insts[d["id"]] = inst
        self.estimated: list = []
        self.warnings: list = []
        self.unresolved: list = []
        self._seen: set = set()

    def inst(self, iid: str, where: str = "") -> Inst:
        if iid not in self.insts:
            raise DesignError(f"{where}: unknown instance '{iid}'")
        return self.insts[iid]

    def warn(self, code: str, message: str):
        self.warnings.append({"code": code, "message": message})

    def note(self, where: str, key: str, entry: dict):
        if entry.get("fidelity") == "estimated" and (where, key) not in self._seen:
            self._seen.add((where, key))
            self.estimated.append(
                {
                    "where": where,
                    "key": key,
                    "value": entry.get("value"),
                    "unit": entry.get("unit"),
                    "rationale": entry.get("rationale", ""),
                }
            )

    def entry(self, x, where: str) -> dict:
        """Reference string '<inst>.<param>' or provenance table -> the provenance entry."""
        if isinstance(x, str):
            iid, _, pname = x.partition(".")
            if iid not in self.insts or not pname:
                raise DesignError(
                    f"{where}: {x!r} is not an '<inst>.<param>' reference to a known instance"
                )
            e = self.insts[iid].entry(pname)
            if not isinstance(e, dict) or "value" not in e:
                raise DesignError(
                    f"{where}: instance '{iid}' (part {self.insts[iid].part.id}) has no param '{pname}'"
                )
            return e
        if isinstance(x, dict) and {"value", "unit", "fidelity"} <= x.keys():
            return x
        raise DesignError(
            f"{where}: expected '<inst>.<param>' or a provenance table, got {x!r}"
        )

    def num(self, x, where: str, units: dict | None = None, key: str = ""):
        """Resolve a numeric leaf -> (float, entry). `units` = accepted unit -> factor; a wrong unit is an error."""
        e = self.entry(x, where)
        v = e["value"]
        if not _is_num(v):
            raise DesignError(f"{where}: value {v!r} is not a number")
        f = 1.0
        if units is not None:
            if e.get("unit") not in units:
                raise DesignError(
                    f"{where}: unit {e.get('unit')!r}, expected one of {sorted(units)}"
                )
            f = units[e["unit"]]
        self.note(where, key or (x if isinstance(x, str) else "value"), e)
        return float(v) * f, e
