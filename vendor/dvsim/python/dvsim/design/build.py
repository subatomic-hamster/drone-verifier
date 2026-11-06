"""Assembly -> MJCF + report. `compute` does everything in memory, `build` writes <out>/<name>.xml and .report.json."""

import json
import math
import os
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from . import visuals
from .assembly import UNITS, Assembly, Ctx, Inst, load
from .bom import bom
from .devices import VEHICLE_GROUP, build_devices, resolve_wires, wire_points
from .library import DesignError, Library
from .mass import mesh_props, printed_mass, shape_volume
from .snap import (
    MateSpec,
    find_compatible_ports,
    incompatible,
    inv_T,
    n_mate,
    nut_height,
    port_frame,
    pose_matrix,
    rot_angle,
    select_fasteners,
    solve_tree,
)
from .subasm import PLANES
from .wireroute import world_geoms
from .wiretubes import wire_tubes

LANDING_GROUP = 3  # MJCF geom group of a landing surface (shape key `landing = true`): ray-visible, unlike VEHICLE_GROUP
POINT_MASS_RADIUS = 1e-3  # m: size of the placeholder sphere that carries the mass of a part without geometry (numerical, not physical)
ESTIMATED_TOL = (
    0.5  # default relative mass tolerance of an `estimated` mass with no mass_tol
)
SIZE_N = {"box": 3, "cylinder": 2, "sphere": 1, "capsule": 2}


# ---- formatting ---------------------------------------------------------------------------------------------------
CLOSURE_SPAN = (
    0.012  # m, half distance between the two connect anchors on a closure hinge axis
)
CLOSURE_SOLREF = "0.004 1"  # [est] 4 x the 1 ms timestep


def f(x) -> str:
    x = float(x)
    return (
        "0" if abs(x) < 1e-14 else repr(x)
    )  # shortest round-trip repr; |x| < 1e-14 is numerical noise


def vec(v) -> str:
    return " ".join(f(x) for x in v)


def q(s) -> str:
    return '"' + escape(str(s), {'"': "&quot;"}) + '"'


def quat_of(R: np.ndarray) -> np.ndarray:
    """[w, x, y, z] from a rotation matrix, w >= 0 (first non-zero component positive when w == 0)."""
    t = np.trace(R)
    if t > 0:
        s = 2 * math.sqrt(t + 1)
        w, x, y, z = (
            s / 4,
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
        )
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * math.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        v = [0.0] * 3
        v[i], v[j], v[k] = s / 4, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s
        w = (R[k, j] - R[j, k]) / s
        x, y, z = v
    out = np.array([w, x, y, z])
    out /= np.linalg.norm(out)
    if out[0] < 0 or (out[0] == 0 and next((c for c in out if c != 0), 1) < 0):
        out = -out
    return out


def pose_attrs(T: np.ndarray) -> str:
    s = f' pos="{vec(T[:3, 3])}"'
    if not np.allclose(T[:3, :3], np.eye(3), atol=1e-14, rtol=0):
        s += f' quat="{vec(quat_of(T[:3, :3]))}"'
    return s


def shape_frame(s: dict, where: str):
    pos = np.array(s.get("pos", [0, 0, 0]), dtype=float)
    if pos.shape != (3,):
        raise DesignError(f"{where}: pos must be 3 numbers")
    if "quat" in s and "zaxis" in s:
        raise DesignError(f"{where}: give quat or zaxis, not both")
    if "quat" in s:
        w, x, y, z = np.array(s["quat"], dtype=float) / np.linalg.norm(
            np.array(s["quat"], dtype=float)
        )
        R = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
    elif (
        "zaxis" in s
    ):  # MuJoCo semantics: shortest-arc rotation taking +z to the given axis
        v = np.array(s["zaxis"], dtype=float)
        v /= np.linalg.norm(v)
        a = np.cross([0.0, 0.0, 1.0], v)
        c, sn = v[2], np.linalg.norm(a)
        if sn < 1e-12:
            R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
        else:
            K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
            R = np.eye(3) + K + K @ K * ((1 - c) / sn**2)
    else:
        R = np.eye(3)
    return pos, R


# ---- instances: shapes, mass --------------------------------------------------------------------------------------
def part_shapes(lib: Library, part) -> list:
    g = part.geometry
    raw = g.get("shapes") or ([g] if "shape" in g else [])
    out = []
    for k, s in enumerate(raw):
        where = f"part {part.id} geometry shape {k}"
        kind = s.get("shape")
        if (
            "offset" in s
        ):  # was silently ignored (shape sat at the part origin); the key is `pos`
            raise DesignError(
                f"{where}: unknown key 'offset' (shape placement is `pos`)"
            )
        d = {
            "kind": kind,
            "mass_param": s.get("mass_param"),
            "mesh": None,
            "path": None,
            "size": s.get("size"),
            "friction": s.get(
                "friction"
            ),  # optional [sliding(, torsional, rolling)]: wheel / caster contact (see geom_xml)
            "solref": s.get(
                "solref"
            ),  # optional [timeconst s, damping ratio]: a soft contact (landing-leg foot)
            "landing": bool(
                s.get("landing")
            ),  # optional: a landing surface (deck / pad), see geom_xml
        }
        if kind == "mesh":
            p = lib.root / str(s.get("mesh", ""))
            if not s.get("mesh") or not p.is_file():
                raise DesignError(f"{where}: mesh file not found: {s.get('mesh')!r}")
            d["mesh"], d["path"] = s["mesh"], str(p)
        elif kind in SIZE_N:
            if not isinstance(d["size"], list) or len(d["size"]) != SIZE_N[kind]:
                raise DesignError(
                    f"{where}: {kind} needs size with {SIZE_N[kind]} numbers"
                )
        else:
            raise DesignError(f"{where}: unknown shape {kind!r}")
        d["pos"], d["R"] = shape_frame(s, where)
        out.append(d)
    return out


# Cosmetic only (renderer colour, never physics): material part `[appearance] rgba`, else this table by the longest
# matching category prefix, else grey. A pure function of the library, so builds stay deterministic.
CATEGORY_RGBA = {
    "motor.servo": (0.07, 0.07, 0.08),
    "motor": (0.30, 0.31, 0.34),
    "prop": (0.12, 0.12, 0.13),
    "esc": (0.10, 0.12, 0.30),
    "pcb.fc": (0.08, 0.08, 0.09),
    "pcb": (0.05, 0.38, 0.20),
    "battery.lipo": (0.08, 0.10, 0.22),
    "wire": (0.72, 0.42, 0.22),
    "connector": (0.90, 0.75, 0.10),
    "fastener": (0.58, 0.60, 0.62),
    "sensor": (0.09, 0.09, 0.10),
    "gps": (0.85, 0.85, 0.83),
}


def appearance_rgba(lib: Library, part) -> tuple:
    mid = part.geometry.get("material")
    if mid in lib.parts:
        c = lib.parts[mid].raw.get("appearance", {}).get("rgba")
        if isinstance(c, list) and len(c) in (3, 4):
            return tuple(float(x) for x in c[:3]) + (1.0,)
    best = max(
        (
            k
            for k in CATEGORY_RGBA
            if part.category == k or part.category.startswith(k + ".")
        ),
        key=len,
        default=None,
    )
    return (*CATEGORY_RGBA.get(best, (0.62, 0.63, 0.65)), 1.0)


def mass_tol(
    inst, mass: float, fid: str
) -> float:  # inst: Inst or Part (anything with .value)
    if inst.value("mass_tol") is not None:
        return inst.value("mass_tol")
    return ESTIMATED_TOL * mass if fid == "estimated" else 0.0


def instance_mass(ctx: Ctx, inst: Inst, shapes: list):
    """-> (mass, fidelity, tol, note, per-shape masses)."""
    part, where = inst.part, f"instance {inst.id}"
    g = part.geometry
    sm = None
    if inst.value("mass") is not None:
        e = inst.entry("mass")
        ctx.note(where, "mass", e)
        mass, fid, note = inst.value("mass"), e["fidelity"], "params.mass"
        tol = mass_tol(inst, mass, fid)
    elif (
        g.get("material")
        and len(shapes) == 1
        and shapes[0]["kind"] == "mesh"
        and all(
            inst.value(k) is not None
            for k in ("wall_count", "infill_fraction", "nozzle_diameter")
        )
    ):
        mat = ctx.lib.get(g["material"])
        if mat.value("density") is None:
            raise DesignError(f"{where}: material {mat.id} has no density")
        vol, area, _ = mesh_props(shapes[0]["path"])
        w, infill, noz, rho = (
            inst.value("wall_count"),
            inst.value("infill_fraction"),
            inst.value("nozzle_diameter"),
            mat.value("density"),
        )
        for k in ("wall_count", "infill_fraction", "nozzle_diameter"):
            ctx.note(where, k, inst.entry(k))
        ctx.note(where, f"material {mat.id} density", mat.params["density"])
        mass = printed_mass(rho, vol, area, w, noz, infill)
        fid, note = (
            "derived",
            f"printed {mat.id}: V={vol:.6g} m3, A={area:.6g} m2, wall_count={w:g}, infill={infill:g}",
        )
        if inst.value("mass_tol") is not None:
            tol = inst.value("mass_tol")
        else:
            dev = [
                abs(printed_mass(rho, vol, area, w + dw, noz, infill) - mass)
                for dw in (1, -1)
                if w + dw >= 1
            ]
            tol = max(dev) if dev else 0.0
    elif shapes and all(s["mass_param"] for s in shapes):
        ents = []
        for s in shapes:
            if inst.value(s["mass_param"]) is None:
                raise DesignError(
                    f"{where}: shape mass_param '{s['mass_param']}' is not a param of {part.id}"
                )
            ents.append(inst.entry(s["mass_param"]))
            ctx.note(where, s["mass_param"], ents[-1])
        sm = [inst.value(s["mass_param"]) for s in shapes]
        mass = sum(sm)
        fid = (
            "estimated"
            if any(e["fidelity"] == "estimated" for e in ents)
            else ents[0]["fidelity"]
        )
        note = "sum of shape mass_params"
        tol = mass_tol(inst, mass, fid)
    else:
        raise DesignError(
            f"no mass for part {part.id} (instance '{inst.id}'): needs params.mass, a printed-part recipe or shape mass_params"
        )
    # per-shape split: shapes with mass_param get it, the rest share the remainder by volume
    if sm is None and shapes:
        fixed = [
            inst.value(s["mass_param"]) if s["mass_param"] else None for s in shapes
        ]
        for s, v in zip(shapes, fixed):
            if s["mass_param"] and v is None:
                raise DesignError(
                    f"{where}: shape mass_param '{s['mass_param']}' is not a param of {part.id}"
                )
        free = [i for i, v in enumerate(fixed) if v is None]
        if free:
            rem = mass - sum(v for v in fixed if v is not None)
            if rem <= 0:
                raise DesignError(
                    f"{where}: shape mass_params use up the whole mass {mass} kg"
                )
            vols = {
                i: shape_volume(shapes[i]["kind"], shapes[i]["size"], shapes[i]["path"])
                for i in free
            }
            tot = sum(vols.values())
            sm = [
                v if v is not None else rem * vols[i] / tot for i, v in enumerate(fixed)
            ]
        else:
            sm = fixed
            if abs(sum(fixed) - mass) > 1e-9 * max(mass, 1e-12):
                ctx.warn(
                    "MASS_SPLIT",
                    f"{where}: shape mass_params sum to {sum(fixed)} kg, params.mass is {mass} kg; using the sum",
                )
                mass = sum(fixed)
    return mass, fid, tol, note, sm


def geom_min_z(T: np.ndarray, g: dict) -> float:
    Rw = T[:3, :3] @ g["R"]
    cz = float((T[:3, :3] @ g["pos"] + T[:3, 3])[2])
    r2 = Rw[2]
    k = g["kind"]
    if k == "box":
        return cz - float(np.abs(r2) @ np.array(g["size"]))
    if k == "sphere":
        return cz - g["size"][0]
    if k == "cylinder":
        return cz - (
            abs(r2[2]) * g["size"][1]
            + g["size"][0] * math.sqrt(max(0.0, 1 - r2[2] ** 2))
        )
    if k == "capsule":
        return cz - (abs(r2[2]) * g["size"][1] + g["size"][0])
    return cz + float((mesh_props(g["path"])[2] @ r2).min())


# ---- mates --------------------------------------------------------------------------------------------------------
def _mate_spec(ctx: Ctx, m: dict, left: bool = False) -> MateSpec:
    """`left` = the mate as the unmirrored sub-assembly would have it (original parts, rotation as written). A mate whose two ends
    are in the same mirrored sub-assembly uses the opposite-hand parts and the negated rotation index (a reflection reverses the
    sense of a turn about the mating axis)."""
    ends = []
    for key in ("a", "b"):
        iid, _, pn = m[key].partition(".")
        inst = ctx.inst(iid, f"mate {m[key]!r}")
        if inst.offboard:
            raise DesignError(f"mate {m[key]!r}: instance '{iid}' is offboard")
        ends.append((inst, pn or None))
    (A, pna), (B, pnb) = ends
    name = m.get("name", f"{m['a']}--{m['b']}")
    if A.id == B.id:
        raise DesignError(f"mate '{name}': both ends are instance '{A.id}'")
    pa_, pb_ = (A.orig_part, B.orig_part) if left else (A.part, B.part)
    for inst, pn in ends:
        if pn and (pa_ if inst is A else pb_).port(pn).get("kind") != "mech":
            raise DesignError(f"mate '{name}': port {inst.id}.{pn} is not a mech port")
    cands = find_compatible_ports(pa_, pb_, pna, pnb)
    if len(cands) != 1:
        if cands:
            lst = "; ".join(f"{A.id}.{x} <-> {B.id}.{y}" for x, y in cands)
            raise DesignError(
                f"mate '{name}': ambiguous, {len(cands)} compatible port pairs: {lst}; name the ports"
            )
        why = ""
        if pna and pnb and (r := incompatible(pa_.port(pna), pb_.port(pnb))):
            why = f" ({r})"
        raise DesignError(
            f"mate '{name}': no compatible mech ports between {A.id} ({pa_.id}) and {B.id} ({pb_.id}){why}"
        )
    na, nb = cands[0]
    rot = m.get("rotation", 0)
    if not left and A.mirror and B.mirror and A.mirror["sub"] == B.mirror["sub"]:
        rot = (-rot) % n_mate(pa_.port(na), pb_.port(nb))
    return MateSpec(
        name,
        A.id,
        B.id,
        pa_.port(na),
        pb_.port(nb),
        rot,
        bool(m.get("closure")),
        "joint" in m,
    )


def _joint_axis(u: MateSpec, j: dict):
    axis = -np.array(port_frame(u.pb)[:3, 2])
    return -axis if j.get("reverse") else axis


def _mirror_pass(ctx: Ctx, doc: dict, tree, joints: dict) -> dict:
    """Check every mirrored sub-assembly against its unmirrored form and return {joint: +-1} (report["mirror_signs"]).
    Solving the mirrored instances with their opposite-hand parts and negated rotations must give exactly the reflection of the
    unmirrored sub-assembly: T'_i = M T_i S_i (M: the mirror plane in the declaring document's root frame, S_i: the part's local
    reflection). A mismatch means the parts' [mirror] data or the parent mate disagrees with the plane: an error, never silent.
    sign = qdot(mirrored) / qdot(unmirrored) of the mirror-image motion: -1 for a hinge whose axis the reflection leaves
    (a reflection reverses the sense of a turn), +1 for a slide."""
    groups = {}
    for iid, inst in ctx.insts.items():
        if inst.mirror and not inst.offboard:
            groups.setdefault(inst.mirror["sub"], []).append(iid)
    signs = {}
    for sid, ids in groups.items():
        idset = set(ids)
        for iid in ids:
            inst = ctx.insts[iid]
            if inst.orig_part.mirror_plane() not in PLANES:
                raise DesignError(
                    f"mirror: part {inst.orig_part.id} (instance {iid}) has no [mirror] symmetry / opposite; "
                    "it cannot be used in a mirrored sub-assembly"
                )
        entries = [c for c in ids if tree.parent[c][0] not in idset]
        if len(entries) != 1:
            raise DesignError(
                f"mirror: sub-assembly '{sid}' must hang on the rest of the design by exactly one mate (found {len(entries)}: {entries})"
            )
        entry = entries[0]
        frame = ctx.insts[entry].mirror["frame"]
        if frame not in tree.T:
            raise DesignError(f"mirror: sub-assembly '{sid}': no root frame '{frame}'")
        Tf = tree.T[frame]
        M3 = Tf[:3, :3] @ PLANES[ctx.insts[entry].mirror["plane"]] @ Tf[:3, :3].T
        M4 = np.eye(4)
        M4[:3, :3], M4[:3, 3] = M3, Tf[:3, 3] - M3 @ Tf[:3, 3]

        def S4(iid):
            S = np.eye(4)
            S[:3, :3] = PLANES[ctx.insts[iid].orig_part.mirror_plane()]
            return S

        lspecs = [
            _mate_spec(ctx, m, left=True)
            for m in doc.get("mates", [])
            if m["a"].partition(".")[0] in idset and m["b"].partition(".")[0] in idset
        ]
        ltree = solve_tree(entry, ids, lspecs)
        T0 = M4 @ tree.T[entry] @ S4(entry)  # the unmirrored sub-assembly's entry pose
        for iid in ids:
            Tl = T0 @ ltree.T[iid]
            want = M4 @ Tl @ S4(iid)
            got = tree.T[iid]
            ep = float(np.linalg.norm(want[:3, 3] - got[:3, 3]))
            er = rot_angle(want[:3, :3].T @ got[:3, :3])
            if ep > 1e-9 or er > 1e-9:
                raise DesignError(
                    f"mirror: '{iid}' of sub-assembly '{sid}' is not the mirror image of its unmirrored pose "
                    f"(off by {ep:.3g} m, {er:.3g} rad): check the parts' [mirror] data (opposite-hand ports must be the reflection "
                    "of the original's, a symmetric part's ports must lie on its symmetry plane) and that the parent mate "
                    "puts the sub-assembly on the mirrored mount"
                )
        lused = {u.name: u for _, u in ltree.parent.values()}
        for m in doc.get("mates", []):
            if "joint" not in m or m.get("name") not in lused:
                continue
            ul = lused[m["name"]]
            if ul.b not in idset:
                continue
            ar = tree.T[ul.b][:3, :3] @ joints[ul.b]["axis"]
            al = M3 @ ((T0 @ ltree.T[ul.b])[:3, :3] @ _joint_axis(ul, m["joint"]))
            aligned = float(ar @ al) > 0
            slide = m["joint"]["type"] == "slide"
            signs[m["name"]] = 1 if aligned == slide else -1
    return signs


def _range(ctx: Ctx, r, jtype: str, where: str):
    units = UNITS["rad"] if jtype == "hinge" else UNITS["m"]
    if isinstance(r, str):  # "<inst>.<param>" whose value is [lo, hi]
        r = ctx.entry(r, where + " range")
    if (
        isinstance(r, dict)
        and isinstance(r.get("value"), list)
        and len(r["value"]) == 2
    ):
        if r.get("unit") not in units:
            raise DesignError(
                f"{where}: range unit {r.get('unit')!r}, expected one of {sorted(units)}"
            )
        ctx.note(where, "range", r)
        lo, hi = (float(v) * units[r["unit"]] for v in r["value"])
    elif isinstance(r, list) and len(r) == 2:
        lo, hi = (ctx.num(x, where + " range", units, "range")[0] for x in r)
    else:
        raise DesignError(
            f"{where}: range must be a provenance table with value = [lo, hi] or a list of two references"
        )
    if not lo < hi:
        raise DesignError(f"{where}: range lo must be < hi")
    return lo, hi


# ---- the whole computation ----------------------------------------------------------------------------------------
def compute(asm, parts_dir, out_dir, lib=None, visual=True, route=True):
    """-> (mjcf text, report dict). `asm` is an Assembly or a path; `lib` = an already loaded Library of `parts_dir`.
    `visual` = emit the parts' cosmetic visual layer (STL meshes are written to <out_dir>/visuals); physics is identical.
    `route` = plan the wire tubes around the part geometry (wireroute.py; cosmetic, the DRC passes False: it never reads tubes)."""
    if not isinstance(asm, Assembly):
        asm = load(asm)
    lib = lib or Library(parts_dir)
    ctx = Ctx(asm, lib)
    doc = ctx.doc
    a = doc["assembly"]
    base = a.get("base", "free")
    onboard = [i for i, x in ctx.insts.items() if not x.offboard]
    root = next((i for i, x in ctx.insts.items() if x.root), None)
    if (
        root is None
    ):  # an empty design (the viewer asks for its state before the first part: a StopIteration took the window's design mode down)
        raise DesignError(
            "the design has no root instance yet: add the frame first (add with root = true)"
        )

    specs = [_mate_spec(ctx, m) for m in doc.get("mates", [])]
    seen = set()
    for s in specs:
        if s.name in seen:
            raise DesignError(f"duplicate mate name '{s.name}'")
        seen.add(s.name)
    placed = {
        i: pose_matrix(x.pos, x.rpy)
        for i, x in ctx.insts.items()
        if x.placed and not x.offboard and not x.root
    }
    tree = solve_tree(root, onboard, specs, placed=placed)
    used = {
        u.name: u for _, u in tree.parent.values()
    }  # mate name -> MateSpec as placed (a = parent)
    for s, _, _ in tree.closures:
        used[s.name] = s
    children = {i: [] for i in onboard}
    for c in tree.order[1:]:
        children[tree.parent[c][0]].append(c)

    # joints (in the child body, named by the mate)
    joint_of, joints, cut_hinges = {}, {}, []
    for m in doc.get("mates", []):
        if "joint" not in m:
            continue
        name = m.get("name", f"{m['a']}--{m['b']}")
        u, j, where = used[name], m["joint"], f"mate '{name}' joint"
        jt = j["type"]
        if u.closure:  # passive loop-cut hinge: two connect equalities, no MJCF joint
            if "range" in j:
                _range(ctx, j["range"], jt, where)
            cut_hinges.append(u)
            continue
        axis = -np.array(port_frame(u.pb)[:3, 2])
        if j.get("reverse"):
            axis = -axis
        rng = None
        if jt != "free":
            if "range" in j:
                rng = _range(ctx, j["range"], jt, where)
        elif "range" in j:
            raise DesignError(f"{where}: a free joint has no range")
        joints[u.b] = {
            "name": name,
            "type": jt,
            "axis": axis,
            "pos": port_frame(u.pb)[:3, 3],
            "range": rng,
        }
        joint_of[name] = name
    mirror_signs = _mirror_pass(ctx, doc, tree, joints)

    # mass items: instances
    items, inst_rows, geoms = [], [], {}
    body_geoms = {i: [] for i in onboard}  # list of MJCF geom dicts per body
    shapes_of = {}
    for iid, inst in ctx.insts.items():
        if inst.offboard:
            inst_rows.append(
                {
                    "id": iid,
                    "part": inst.part.id,
                    "group": inst.group,
                    "body": None,
                    "mass": 0.0,
                    "mass_fidelity": None,
                    "mass_tol": 0.0,
                    "mass_note": "offboard: excluded from mass totals",
                    "offboard": True,
                }
            )
            continue
        shapes = part_shapes(lib, inst.part)
        shapes_of[iid] = shapes
        hide = visual and visuals.applies(
            inst.part
        )  # the plain shape stays for collision / mass, only its drawing is hidden
        mass, fid, tol, note, sm = instance_mass(ctx, inst, shapes)
        inst_rows.append(
            {
                "id": iid,
                "part": inst.part.id,
                "group": inst.group,
                "body": iid,
                "mass": mass,
                "mass_fidelity": fid,
                "mass_tol": tol,
                "mass_note": note,
            }
        )
        items.append(
            {"key": iid, "group": inst.group, "mass": mass, "fid": fid, "tol": tol}
        )
        if not shapes:
            body_geoms[iid].append(
                {
                    "name": iid,
                    "kind": "sphere",
                    "size": [POINT_MASS_RADIUS],
                    "pos": np.zeros(3),
                    "R": np.eye(3),
                    "mass": mass,
                    "collide": False,
                    "part": inst.part.id,
                    "hide": hide,
                }
            )
        mesh_n = 0
        for k, (s, m_) in enumerate(zip(shapes, sm or [])):
            asset = None
            if s["kind"] == "mesh":
                asset = inst.part.id if mesh_n == 0 else f"{inst.part.id}:{k}"
                mesh_n += 1
            body_geoms[iid].append(
                {
                    **s,
                    "name": iid if len(shapes) == 1 else f"{iid}:{k}",
                    "mass": m_,
                    "collide": True,
                    "asset": asset,
                    "part": inst.part.id,
                    "hide": hide,
                }
            )

    # fasteners
    fasteners = []
    for m in doc.get("mates", []):
        spec = m.get("fasteners")
        if spec is None:
            continue
        name = m.get("name", f"{m['a']}--{m['b']}")
        u = used[name]
        nut_id = m.get("nut") or (spec.get("nut") if isinstance(spec, dict) else None)
        rec, screw, nut, pos, warns = select_fasteners(
            lib, name, u.pa, u.pb, spec, nut_id
        )
        for code, msg in warns:
            ctx.warn(code, msg)
        child = ctx.insts[u.b]
        d = float(rec["thread"][1:]) / 1000
        F = tree.T[u.a] @ port_frame(u.pa)
        Tc = tree.T[u.b]
        tol, fids = 0.0, []
        for part in (screw, nut):
            if part is None:
                continue
            e = part.params["mass"]
            ctx.note(f"mate {name}", f"{part.id}.mass", e)
            fids.append(e["fidelity"])
            tol += rec["count"] * mass_tol(part, part.value("mass"), e["fidelity"])
        for kk, (x, y) in enumerate(pos):
            o = F @ np.array([x, y, 0.0, 1.0])
            local = inv_T(Tc) @ o
            Rl = Tc[:3, :3].T @ F[:3, :3]
            for kind, part, h in (
                ("screw", screw, screw.value("length")),
                ("nut", nut, nut_height(nut) if nut else None),
            ):
                if part is None:
                    continue
                body_geoms[u.b].append(
                    {
                        "name": f"{name}:{kind}:{kk}",
                        "kind": "cylinder",
                        "size": [d / 2, h / 2],
                        "pos": local[:3],
                        "R": Rl,
                        "mass": part.value("mass"),
                        "collide": False,
                        "part": part.id,
                    }
                )
        rec.update(
            mass_fidelity="estimated" if "estimated" in fids else fids[0],
            mass_tol=tol,
            group=child.group,
            closure=used[name].closure,
        )
        fasteners.append(rec)
        items.append(
            {
                "key": f"{name}:fasteners",
                "group": child.group,
                "mass": rec["mass"],
                "fid": rec["mass_fidelity"],
                "tol": tol,
            }
        )

    # wires and connectors
    wires = resolve_wires(ctx)
    for w in wires:
        pts = wire_points(w, tree.T)
        w["counted"] = False
        if pts is None or w["included_in"] or w["mass"] is None:
            body = None
        else:
            body = w["_ia"].id if not w["_ia"].offboard else w["_ib"].id
            w["counted"] = True
            Tb = inv_T(tree.T[body])
            p0, p1 = (Tb[:3, :3] @ p + Tb[:3, 3] for p in pts)
            cd = w["_part"].value("conductor_diameter")
            r = cd / 2 if cd else POINT_MASS_RADIUS
            g = {
                "name": f"wire:{w['index']}",
                "mass": w["mass"],
                "collide": False,
                "part": w["part"],
                "pos": np.zeros(3),
                "R": np.eye(3),
            }
            if np.linalg.norm(p1 - p0) < 1e-9:
                g.update(kind="sphere", size=[r], pos=p0)
            else:
                g.update(kind="capsule", size=[r], fromto=(p0, p1))
            body_geoms[body].append(g)
            items.append(
                {
                    "key": g["name"],
                    "group": w["_ia"].group,
                    "mass": w["mass"],
                    "fid": w["mass_fidelity"],
                    "tol": w["mass_tol"],
                }
            )
        for k, c in enumerate(w["connectors"]):
            ci = ctx.insts[c["instance"]]
            c["counted"] = bool(
                c["mass"] is not None and not ci.offboard and not w["included_in"]
            )
            if c["counted"]:
                port = ci.part.port(c["port"])
                gname = f"wire:{w['index']}:conn:{k}"
                body_geoms[ci.id].append(
                    {
                        "name": gname,
                        "kind": "sphere",
                        "size": [POINT_MASS_RADIUS],
                        "R": np.eye(3),
                        "collide": False,
                        "pos": np.array(port.get("pos", [0, 0, 0]), dtype=float),
                        "mass": c["mass"],
                        "part": c["part"],
                    }
                )
                items.append(
                    {
                        "key": gname,
                        "group": ci.group,
                        "mass": c["mass"],
                        "fid": c["mass_fidelity"],
                        "tol": c["mass_tol"],
                    }
                )

    # devices
    devices, dev_report, sites = build_devices(ctx, wires, joint_of)
    obstacles = None
    if (
        route and wires
    ):  # the tubes avoid what is drawn too: the parts' cosmetic meshes count as obstacles
        shown = {i: list(body_geoms[i]) for i in onboard}
        if visual:
            for i in onboard:
                if visuals.applies(ctx.insts[i].part):
                    for k, p in enumerate(
                        visuals.pieces(
                            ctx.insts[i].part, Path(out_dir) / "visuals", lib.root
                        )
                    ):
                        shown[i].append(
                            {
                                "name": f"{i}:v{k}",
                                "kind": "mesh",
                                "path": str(p.path),
                                "R": p.R,
                                "pos": p.pos,
                            }
                        )
        obstacles = world_geoms(shown, tree.T)
    tendon_lines = wire_tubes(
        wires, tree.T, sites, obstacles
    )  # cables as visual-only tendons (the mass capsules above are hidden)

    # placement: free base sits on the floor (lowest collision point at z = 0)
    zmin = 0.0
    if base == "free":
        zs = [
            geom_min_z(tree.T[i], g)
            for i in onboard
            for g in body_geoms[i]
            if g["collide"]
        ]
        zmin = min(zs) if zs else 0.0
    T_root = np.eye(4)
    T_root[2, 3] = -zmin if base == "free" else 0.0
    rf = a.get("root_frame")
    if (
        rf is not None
    ):  # world = the root part's URDF link frame (its mesh sits at [geometry] origin_xyz / origin_rpy)
        g = ctx.insts[root].part.geometry
        if rf != "urdf_link" or base != "fixed":
            raise DesignError(
                "[assembly] root_frame must be 'urdf_link' and needs base = 'fixed'"
            )
        if "origin_xyz" not in g or "origin_rpy" not in g:
            raise DesignError(
                f"root_frame: part {ctx.insts[root].part.id} has no geometry origin_xyz / origin_rpy"
            )
        r, p, y = g["origin_rpy"]  # URDF fixed-axis rpy: R = Rz(y) Ry(p) Rx(r)
        cr, sr, cp, sp, cy, sy = (
            math.cos(r),
            math.sin(r),
            math.cos(p),
            math.sin(p),
            math.cos(y),
            math.sin(y),
        )
        T_root[:3, :3] = (
            np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        )
        T_root[:3, 3] = g["origin_xyz"]

    # MJCF
    out_dir = Path(out_dir).resolve()
    assets, lines = {}, []

    def geom_xml(g):
        s = f"<geom name={q(g['name'])}"
        if g["kind"] == "mesh":
            rel = os.path.relpath(lib.root / g["mesh"], out_dir).replace(os.sep, "/")
            assets.setdefault(g["asset"], rel)
            s += f' type="mesh" mesh={q(g["asset"])}'
        elif "fromto" in g:
            s += f' type="capsule" fromto="{vec(g["fromto"][0])} {vec(g["fromto"][1])}" size="{vec(g["size"])}"'
        else:
            s += f' type="{g["kind"]}" size="{vec(g["size"])}"'
        if "fromto" not in g:
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = g["R"], g["pos"]
            s += pose_attrs(T)
        rgba = appearance_rgba(lib, lib.get(g["part"]))
        owner = ctx.insts.get(g["name"]) or ctx.insts.get(g["name"].rpartition(":")[0])
        if (
            owner is not None and owner.color
        ):  # instance colour override: cosmetic, the part's own geoms only
            rgba = (*owner.color[:3], owner.color[3] if len(owner.color) > 3 else 1.0)
        hidden = (
            g.get("hide")
            or (g["name"].startswith("wire:") and ":conn:" not in g["name"])
        )  # wire mass capsules: drawn as a tube instead (wiretubes.py); hidden = alpha 0 (renderer skips it)
        s += f' rgba="{vec(tuple(rgba[:3]) + (0.0,) if hidden else rgba)}"'
        if g.get(
            "friction"
        ):  # priority 1: this geom's friction replaces the floor's (MuJoCo default is the max of the two)
            s += f' friction="{vec(g["friction"])}" priority="1"'
        if g.get("solref"):
            s += f' solref="{vec(g["solref"])}"'
        if g.get(
            "landing"
        ):  # a landing surface: other vehicles' geoms (contype 1) land on it (conaffinity 1); group 3, so rangefinders and rotor rays see it (they skip VEHICLE_GROUP)
            s += f' mass="{f(g["mass"])}" contype="0" conaffinity="1" group="{LANDING_GROUP}"/>'
        else:
            s += f' mass="{f(g["mass"])}" contype="{1 if g["collide"] else 0}" conaffinity="0" group="{VEHICLE_GROUP}"/>'
        geoms[g["name"]] = g["part"]
        return s

    vis_meshes, vis_mats = {}, {}

    def visual_xml(iid):
        """Cosmetic geoms of the instance's part: zero density, no contacts, same geom group as the plain shape."""
        if not visual or not visuals.applies(ctx.insts[iid].part):
            return []
        out = []
        for k, p in enumerate(
            visuals.pieces(ctx.insts[iid].part, out_dir / "visuals", lib.root)
        ):
            vis_meshes[f"v_{p.name}"] = f"visuals/{p.path.name}"
            m = p.material
            mname = f"m_{m['name']}"
            vis_mats[mname] = m
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = p.R, p.pos
            c = ctx.insts[
                iid
            ].color  # instance colour override: every piece of the part takes it
            rgba = (*c[:3], c[3] if len(c) > 3 else 1.0) if c else m["rgba"]
            out.append(
                f'<geom name={q(f"{iid}:v{k}")} type="mesh" mesh="v_{p.name}" material={q(mname)} rgba="{vec(rgba)}"'
                f'{pose_attrs(T)} mass="0" contype="0" conaffinity="0" group="{VEHICLE_GROUP}"/>'
            )
        return out

    def body_xml(iid, depth):
        ind = "  " * depth
        T = T_root if iid == root else inv_T(tree.T[tree.parent[iid][0]]) @ tree.T[iid]
        out = [f"{ind}<body name={q(iid)}{pose_attrs(T)}>"]
        if iid == root and base == "free":
            out.append(f"{ind}  <freejoint name={q(iid + ':free')}/>")
        j = joints.get(iid)
        if j:
            if j["type"] == "free":
                out.append(f"{ind}  <freejoint name={q(j['name'])}/>")
            else:
                r = f' range="{vec(j["range"])}"' if j["range"] else ""
                out.append(
                    f'{ind}  <joint name={q(j["name"])} type="{j["type"]}" axis="{vec(j["axis"])}" pos="{vec(j["pos"])}"{r}/>'
                )
        out += [f"{ind}  {geom_xml(g)}" for g in body_geoms[iid]]
        out += [f"{ind}  {x}" for x in visual_xml(iid)]
        for sname, (p, R) in sorted(sites.get(iid, {}).items()):
            T_s = np.eye(4)
            T_s[:3, :3], T_s[:3, 3] = R, p
            out.append(f"{ind}  <site name={q(sname)}{pose_attrs(T_s)}/>")
        for c in children[iid]:
            out += body_xml(c, depth + 1)
        out.append(f"{ind}</body>")
        return out

    body_lines = body_xml(root, 2)
    head = [
        f"<mujoco model={q(a['name'])}>",
        '  <compiler angle="radian"/>',
        f'  <option timestep="{f(a.get("timestep", 0.001))}"/>',
    ]
    if assets or vis_meshes:
        head.append("  <asset>")
        inertia = a.get("mesh_inertia", "exact")
        head += [
            f'    <mesh name={q(n)} file={q(p)} inertia="{inertia}"/>'
            for n, p in assets.items()
        ]
        head += [
            f'    <mesh name={q(n)} file={q(p)} inertia="convex"/>'
            for n, p in sorted(vis_meshes.items())
        ]
        head += [
            f'    <material name={q(n)} rgba="{vec(m["rgba"])}" specular="{f(m["specular"])}" shininess="{f(m["shininess"])}"'
            f' metallic="{f(m["metallic"])}" roughness="{f(m["roughness"])}"/>'
            for n, m in sorted(vis_mats.items())
        ]
        head.append("  </asset>")
    head += [
        "  <worldbody>",
        '    <geom name="floor" type="plane" size="0 0 1" contype="1" conaffinity="1" group="0" rgba="0.80 0.82 0.84 1"/>',
    ]
    tail = ["  </worldbody>"]
    if cut_hinges:  # two anchors on the hinge axis, CLOSURE_SPAN either side of the port origin: 5 constraints, spin free
        tail.append("  <equality>")
        for u in cut_hinges:
            Fb = port_frame(u.pb)
            for k, s in enumerate((-CLOSURE_SPAN, CLOSURE_SPAN)):
                w = tree.T[u.b] @ np.append(Fb[:3, 3] + s * Fb[:3, 2], 1.0)
                anchor = (inv_T(tree.T[u.a]) @ w)[:3]
                tail.append(
                    f'    <connect name={q(f"{u.name}:c{k}")} body1={q(u.a)} body2={q(u.b)} anchor="{vec(anchor)}"'
                    f' solref="{CLOSURE_SOLREF}"/>'
                )
        tail.append("  </equality>")
    if tendon_lines:
        tail += ["  <tendon>", *tendon_lines, "  </tendon>"]
    if devices:
        data = escape(
            json.dumps(devices, sort_keys=True, separators=(",", ":")), {"'": "&apos;"}
        )
        tail += [
            "  <custom>",
            f"    <text name=\"dvsim.devices\" data='{data}'/>",
            "  </custom>",
        ]
    tail.append("</mujoco>")
    xml = "\n".join(head + body_lines + tail) + "\n"

    # report
    by_group, by_item = {}, {}
    for it in items:
        by_group[it["group"] or "(none)"] = (
            by_group.get(it["group"] or "(none)", 0.0) + it["mass"]
        )
        by_item[it["key"]] = it["mass"]
    total = math.fsum(by_item.values())
    targets = []
    for t in doc.get("targets", []):
        if (
            "value" not in t
        ):  # a reach / workspace target: checked by the DRC (ARM_REACH)
            continue
        where = f"target {t['name']}"
        tv, _ = ctx.num(t["value"], where, UNITS["kg"], "value")
        scope = "all" if t.get("all") else sorted(t["groups"])
        sel = [
            it for it in items if scope == "all" or (it["group"] or "(none)") in scope
        ]
        if not sel:
            ctx.warn("TARGET_EMPTY", f"{where}: no mass items in scope {scope}")
        model = math.fsum(it["mass"] for it in sel)
        tol = math.fsum(it["tol"] for it in sel)
        est = math.fsum(it["mass"] for it in sel if it["fid"] == "estimated")
        gap = model - tv
        targets.append(
            {
                "name": t["name"],
                "scope": scope,
                "model": model,
                "target": tv,
                "gap": gap,
                "gap_pct": 100 * gap / tv if tv else None,
                "tolerance": tol,
                "within": abs(gap) <= tol,
                "estimated_share": est / model if model else 0.0,
            }
        )
    report = {
        "assembly": a["name"],
        "instances": inst_rows,
        "fasteners": fasteners,
        "wires": [
            {
                k: v
                for k, v in w.items()
                if not k.startswith("_") and k not in ("connectors",)
            }
            for w in wires
        ],
        "connectors": [
            dict(c, group=ctx.insts[c["instance"]].group)
            for w in wires
            for c in w["connectors"]
        ],
        "geoms": dict(sorted(geoms.items())),
        "devices": dev_report,
        "estimated": ctx.estimated,
        "unresolved": ctx.unresolved,
        "warnings": ctx.warnings,
        "closures": [
            {"mate": m.name, "pos_err": ep, "rot_err": er}
            | ({"equalities": [f"{m.name}:c0", f"{m.name}:c1"]} if m.has_joint else {})
            for m, ep, er in tree.closures
        ],
        "mass": {
            "total": total,
            "by_group": dict(sorted(by_group.items())),
            "by_instance": by_item,
        },
        "targets": targets,
        "mirror_signs": mirror_signs,
    }
    report["bom"] = bom(report, lib, asm.origins)
    if lib.errors:
        report["library_errors"] = len(lib.errors)
    return xml, report


def build(assembly_path, parts_dir, out_dir, lib=None) -> dict:
    """Write <out_dir>/<name>.xml and <out_dir>/<name>.report.json. Returns {"xml", "report_path", "report"}.
    The report gets `findings`: the design-rule check (drc.py) of the same assembly; findings never make the build fail."""
    from .drc import check

    asm = assembly_path if isinstance(assembly_path, Assembly) else load(assembly_path)
    lib = lib or Library(parts_dir)
    xml, report = compute(asm, parts_dir, out_dir, lib)
    report["findings"] = [
        f.as_dict()
        for f in check(asm, lib, model=(xml, report), mesh_base=Path(out_dir))
    ]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = report["assembly"]
    xp, rp = out / f"{name}.xml", out / f"{name}.report.json"
    xp.write_text(xml, encoding="utf-8", newline="\n")
    rp.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {"xml": xp, "report_path": rp, "report": report}


def format_budget(report: dict) -> str:
    """Mass budget as text: per-instance table, group totals, targets."""
    L = [f"assembly {report['assembly']}: total {report['mass']['total']:.4f} kg"]
    L.append(f"{'item':36s} {'group':14s} {'mass kg':>10s} {'tol kg':>9s}  fidelity")
    for r in report["instances"]:
        L.append(
            f"{r['id']:36s} {r['group'] or '-':14s} {r['mass']:10.4f} {r['mass_tol']:9.4f}  {r['mass_fidelity'] or '-'}"
        )
    for r in report["fasteners"]:
        L.append(
            f"{r['mate'] + ' fasteners':36s} {r['group'] or '-':14s} {r['mass']:10.4f} {r['mass_tol']:9.4f}  {r['count']}x {r['part']}"
        )
    for r in report["wires"]:
        if r.get("counted"):
            L.append(
                f"{'wire ' + r['name']:36s} {r['group'] or '-':14s} {r['mass']:10.4f} {r['mass_tol']:9.4f}  {r['mass_fidelity']}"
            )
    L.append(
        "groups: "
        + ", ".join(f"{k} {v:.4f}" for k, v in report["mass"]["by_group"].items())
    )
    for t in report["targets"]:
        pct = "n/a" if t["gap_pct"] is None else f"{t['gap_pct']:+.1f}%"
        L.append(
            f"target {t['name']} ({t['scope']}): model {t['model']:.4f} vs {t['target']:.4f} kg, gap {t['gap']:+.4f} kg ({pct}), "
            f"tolerance {t['tolerance']:.4f} kg, {'within' if t['within'] else 'OUTSIDE'}, estimated share {t['estimated_share']:.2f}"
        )
    if (b := report.get("bom")) and b["total"]["cost"]:
        L.append(
            f"cost: {b['total']['cost']:.2f} USD (estimated share {b['total']['estimated_share']:.2f})"
            + (f", no price: {', '.join(b['missing'])}" if b["missing"] else "")
        )
    for u in report["unresolved"]:
        L.append(f"unresolved: {u['where']}: {u['what']}")
    for w in report["warnings"]:
        L.append(f"warning {w['code']}: {w['message']}")
    return "\n".join(L)
