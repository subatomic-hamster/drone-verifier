"""Cosmetic visual layer: `[visual]` of a part file -> closed triangle meshes + materials (parts/README.md "Visual layer").
Never physics: the builder emits these as zero-density, non-colliding geoms and hides the plain `[geometry]` shape.

    pieces(part, cache_dir) -> [Piece]        generated STL files (content-addressed, written once) + pose + material
    has_visual(part)                          explicit [visual] or a category default family
    validate(part, root) -> [str]             problems for tools/check_parts.py
"""

import hashlib
import json
import math
import os
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .materials import MATERIALS, resolve_material
from .mesh import Mesh, write_stl

GENERATORS: dict = {}  # name -> (fn, {slot: default material name})
# Generators drawn in a frame with the underside at z = 0, centred in x, y: placed on the [geometry] bounds' underside unless
# the element gives `pos`. Every other generator draws in the part frame (motor base / prop hub / gearmotor face at the origin).
BOTTOM_ANCHORED = {
    "lipo_pack", "cell_18650", "board_generic", "board_esc", "board_fc", "board_power", "board_sbc", "gnss_puck",
    "camera_module", "sensor_module", "box_device", "sbc_board", "sensor_ic",
}  # fmt: skip


def gen(name, **slots):
    def deco(fn):
        GENERATORS[name] = (fn, slots)
        return fn

    return deco


class Ctx:
    """What a generator knows about its part: numeric params (SI), geometry bounds, ports, a per-part RNG seed."""

    def __init__(self, part):
        self.part = part
        self.root = Path(
            "."
        )  # repo root (mesh paths are relative to it); build_meshes sets it
        self.seed = zlib.crc32(part.id.encode())

    def p(self, *names, default=None):
        for n in names:
            v = self.part.value(n)
            if v is not None:
                return float(v)
        return default

    def text(self, name, default=""):
        e = self.part.params.get(name)
        return str(e.get("value", default)) if isinstance(e, dict) else default

    def rng(self):
        return np.random.default_rng(self.seed)

    def bounds(self):
        """(lo, hi) AABB of the [geometry] shapes in the part frame (mesh shapes unknown -> skipped), or None."""
        g = self.part.geometry
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for s in g.get("shapes") or ([g] if "shape" in g else []):
            sz, pos, k = (
                s.get("size"),
                np.array(s.get("pos", [0, 0, 0]), float),
                s.get("shape"),
            )
            if sz is None:
                continue
            if k == "box":
                h = np.array(sz, float)
            elif k == "cylinder":
                h = np.array([sz[0], sz[0], sz[1]], float)
            elif k == "sphere":
                h = np.full(3, sz[0], float)
            elif k == "capsule":
                h = np.array(
                    [sz[0], sz[0], sz[0] + (sz[1] if len(sz) > 1 else 0)], float
                )
            else:
                continue
            if "zaxis" in s or "quat" in s:
                h = np.full(3, np.linalg.norm(h))  # rotated: conservative cube
            lo, hi = np.minimum(lo, pos - h), np.maximum(hi, pos + h)
        return (lo, hi) if np.all(np.isfinite(lo)) else None

    def size(self, *names, axis=None, default=None):
        """A dimension: first named param, else the geometry AABB extent along `axis` (0, 1, 2), else default."""
        v = self.p(*names)
        if v is not None:
            return v
        b = self.bounds()
        if b is not None and axis is not None:
            return float(b[1][axis] - b[0][axis])
        return default

    def ports(self, prefix):
        return [p for p in self.part.ports if p.get("type", "").startswith(prefix)]


# Category -> default family when the part has no [visual] (geometry-driven, so every part of the category is covered).
DEFAULTS = {
    "motor.bldc": "bldc_outrunner",
    "prop": "propeller",
    "battery.lipo": "lipo_pack",
    "battery.liion": "lipo_pack",
    "battery.cell": "cell_18650",
    "esc": "board_esc",
    "pcb.fc": "board_fc",
    "pcb.pm": "board_power",
    "pcb.pdb": "board_power",
    "pcb.bec": "board_power",
    "pcb.gps": "gnss_puck",
    "pcb.servo_driver": "board_generic",
    "pcb.motor_controller": "board_generic",
    "computer.sbc": "sbc_board",
    "sensor.imu": "sensor_ic",
    "sensor.baro": "sensor_ic",
    "sensor.mag": "sensor_ic",
    "sensor.gnss": "sensor_ic",
    "driver": "board_generic",
    "motor.servo": "servo_case",
    "motor.dc.gearmotor": "gearmotor",
    "motor.stepper": "stepper_nema",
    "actuator.bldc.qdd": "qdd_actuator",
    "wheel": "wheel_tyre",
    "frame.tube": "cf_tube",
    "structure.tube": "cf_tube",
    "frame.plate": "plate_outline",
    "structure.plate": "plate_outline",
    "plate": "plate_outline",
    "structure.bearing": "bearing",
    "structure.standoff": "standoff",
    "fastener": "screw",
    "fastener.insert": "heat_insert",
    "ammo.pellet": "diabolo_pellet",
    "sensor.camera": "camera_module",
    "sensor.rangefinder": "sensor_module",
    "sensor.flow": "sensor_module",
    "rc.receiver": "board_generic",
    "radio.telemetry": "board_generic",
    "psu": "box_device",
    "psu.dcdc": "box_device",
    "connector": "connector",
    "structure.pulley": "pulley",
    "structure.shaft-collar": "shaft_collar",
    "structure.hub": "hub_disc",
    "shaft-hub": "pulley",
    "frame": "shapes",
    "frame.printed": "shapes",
    "frame.arm": "shapes",
    "frame.connector": "shapes",
    "frame.hanger": "shapes",
    "frame.hardware": "shapes",
    "frame.landing_gear": "shapes",
    "frame.mount": "shapes",
    "bracket": "shapes",
    "structure.bracket": "bracket_plate",
    "structure.extrusion": "extrusion_profile",
    "structure.tnut": "t_nut",
    "structure.belt": "belt_loop",
    "structure.nut": "flanged_nut",
    "effector.launcher": "pcp_launcher",
    "caster": "shapes",
}


# Per-part finishes for a category default: (regex on the part id, {slot: material}); first match wins.
TINTS = [
    (r"sg9[02]", {"case": "plastic-blue"}),
    (r"pololu-(a4988|drv8833|tb6612)", {"pcb": "soldermask-red"}),
    (r"pololu-drv8825", {"pcb": "soldermask-purple"}),
    (r"pololu-(vl53|minimu|altimu|lps)", {"pcb": "soldermask-black"}),
    (r"amass-ec5", {"body": "plastic-blue"}),
    (r"amass-mr(30|60)", {"body": "plastic-black"}),
    (r"jst-(gh|sh)", {"body": "plastic-black"}),
    (r"jst-(ph|xh)", {"body": "plastic-white"}),
    (r"wurth-wr-phd|same-sky", {"body": "plastic-black"}),
]


def default_colors(part) -> dict:
    import re

    return next((c for rx, c in TINTS if re.search(rx, part.id)), {})


def default_family(part):
    cat = part.category
    if cat.startswith("motor.servo"):  # servo style follows the maker
        v = str(part.raw.get("vendor", "")).lower()
        if (
            part.value("envelope_length") is not None
        ):  # micro analog servo (Feetech FS), not a bus servo
            return "hobby_servo"
        return (
            "dxl_servo"
            if "robotis" in v
            else "bus_servo"
            if "feetech" in v
            else "hobby_servo"
        )
    best = max(
        (k for k in DEFAULTS if cat == k or cat.startswith(k + ".")),
        key=len,
        default=None,
    )
    return DEFAULTS.get(best)


def elements(part) -> list:
    """The part's `[[visual.elements]]` or one synthesised from the category default; [] = none."""
    v = part.raw.get("visual")
    if v is not None:
        return list(v.get("elements", []))
    fam = default_family(part)
    g = part.geometry
    if any(
        s.get("shape") == "mesh" for s in g.get("shapes") or [g]
    ):  # vendor / printed mesh: redraw it with a finish
        cat = part.category
        return [
            {
                "gen": "shapes",
                "colors": {
                    "*": "plastic-black" if cat.startswith("motor") else "pla-matte"
                },
            }
        ]
    if not fam:
        return []
    col = default_colors(part)
    return [{"gen": fam, "colors": col}] if col else [{"gen": fam}]


def has_visual(part) -> bool:
    return bool(elements(part))


def applies(part) -> bool:
    """Whether a build draws the visual in place of the plain shape: an explicit [visual], or a category default. A part
    without [geometry] is drawn at the instance frame from its own params (the generators' dimension fallbacks), with the
    zero-mass point body it already has; `shapes` needs [geometry], so there it draws nothing."""
    if part.raw.get("visual") is not None:
        return True
    if part.geometry.get("shapes") or "shape" in part.geometry:
        return has_visual(part)
    return has_visual(part) and default_family(part) != "shapes"


@dataclass
class Piece:
    name: str  # content hash: STL file stem and MJCF mesh asset name
    path: Path
    pos: np.ndarray
    R: np.ndarray
    material: dict  # {name, rgba, specular, shininess, metallic, roughness}


def _frame(e):
    pos = np.array(e.get("pos", [0, 0, 0]), float)
    R = np.eye(3)
    if "rpy" in e:  # fixed-axis roll-pitch-yaw, radians
        r, p, y = e["rpy"]
        cr, sr, cp, sp, cy, sy = (
            math.cos(r),
            math.sin(r),
            math.cos(p),
            math.sin(p),
            math.cos(y),
            math.sin(y),
        )
        R = (
            np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        )
    elif "zaxis" in e:
        from .mesh import rot_to

        R = rot_to(e["zaxis"])
    return pos, R


def _params(c: Ctx, e):
    out = {}
    for k, v in (e.get("params") or {}).items():
        if isinstance(v, str) and v.startswith(
            "param:"
        ):  # value of one of the part's own params
            v = c.p(v[6:])
        out[k] = v
    return out


def build_meshes(part, e, root) -> dict:
    """{slot: Mesh} of one element (a generator, or a vendor mesh file as the single slot 'body')."""
    if "mesh" in e:
        from .stl import read_stl

        return {"body": read_stl(Path(root) / e["mesh"])}
    c = Ctx(part)
    c.root = Path(root)
    return GENERATORS[e["gen"]][0](c, **_params(c, e))


def _code_hash() -> str:
    """Changes when any generator source does, so cached STLs never outlive the code that made them."""
    h = hashlib.sha1()
    for f in sorted(Path(__file__).parent.glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:8]


CODE = _code_hash()


def pieces(part, cache_dir, root) -> list:
    """Meshes of every element/slot (written into `cache_dir` unless already there) with pose and material."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    out = []
    for e in elements(part):
        slots = (
            GENERATORS[e["gen"]][1]
            if "gen" in e
            else {"body": e.get("material", "plastic-black")}
        )
        raw = {k: part.raw.get(k) for k in ("params", "geometry", "ports")}
        key = CODE + json.dumps(
            [part.id, e.get("gen"), e.get("mesh"), e.get("params"), raw],
            default=str,
            sort_keys=True,
        )
        h = hashlib.sha1(key.encode()).hexdigest()[:12]
        index = (
            cache / f"{h}.json"
        )  # which slots this element produced (a slot can be empty)
        if not index.exists():
            meshes = build_meshes(part, e, root)
            names = []
            for slot, m in meshes.items():
                if m.tri_count():
                    write_stl(cache / f"{h}_{slot}.stl", m)
                    names.append(slot)
            tmp = index.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(names))
            os.replace(
                tmp, index
            )  # atomic: a parallel reader never sees an empty index
        pos, R = _frame(e)
        if (
            "pos" not in e
            and e.get("gen") in BOTTOM_ANCHORED
            and (b := Ctx(part).bounds()) is not None
        ):
            pos = np.array(
                [(b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2, b[0][2]]
            )  # sit on the geometry's underside
        for slot in json.loads(index.read_text()):
            col = e.get("colors") or {}
            mat = (
                col.get(slot)
                or col.get("*")
                or slots.get(slot)
                or slots.get("*", "plastic-black")
            )
            out.append(
                Piece(
                    f"{h}_{slot}",
                    cache / f"{h}_{slot}.stl",
                    pos,
                    R,
                    resolve_material(mat),
                )
            )
    return out


def validate(part, root: Path, sources: set) -> list:
    """Problems with the part's [visual] (empty list = fine). Category defaults need nothing."""
    v = part.raw.get("visual")
    if v is None:
        return []
    bad = []
    if "src" in v and v["src"] not in sources:
        bad.append(f"visual.src '{v['src']}' not in [[sources]]")
    elif "src" not in v and not v.get("rationale"):
        bad.append("visual needs src (a [[sources]] id) or rationale")
    els = v.get("elements")
    if not isinstance(els, list) or not els:
        return bad + ["visual needs [[visual.elements]]"]
    for i, e in enumerate(els):
        w = f"visual.elements[{i}]"
        if ("gen" in e) == ("mesh" in e):
            bad.append(f"{w}: exactly one of gen / mesh")
        elif "gen" in e and e["gen"] not in GENERATORS:
            bad.append(f"{w}: unknown generator '{e['gen']}'")
        elif "mesh" in e and not (root / e["mesh"]).is_file():
            bad.append(f"{w}: mesh '{e['mesh']}' not found")
        for k in ("pos", "zaxis", "rpy"):
            if k in e and not (
                isinstance(e[k], list)
                and len(e[k]) == 3
                and all(isinstance(x, (int, float)) for x in e[k])
            ):
                bad.append(f"{w}: {k} must be 3 numbers")
        for slot, col in (e.get("colors") or {}).items():
            try:
                resolve_material(col)
            except Exception as ex:
                bad.append(f"{w}: colors.{slot}: {ex}")
        if "gen" in e and e["gen"] in GENERATORS:
            for k, val in (e.get("params") or {}).items():
                if (
                    isinstance(val, str)
                    and val.startswith("param:")
                    and val[6:] not in part.params
                ):
                    bad.append(
                        f"{w}: params.{k} names param '{val[6:]}', which the part lacks"
                    )
    return bad


from . import (  # noqa: F401  (register generators)
    gens_board,
    gens_conn,
    gens_drone,
    gens_edf,
    gens_marine,
    gens_mech,
    gens_rocket,
    gens_sense,
    gens_servo,
    gens_struct,
)
