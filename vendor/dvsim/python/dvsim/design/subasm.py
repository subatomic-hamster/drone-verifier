"""Sub-assemblies: `[[subassemblies]]` + `[exports]`, flattened at load (docs/superpowers/specs/2026-10-16-quadruped-design.md s3).

`expand(doc, path)` is a pure function doc -> (flat doc, origins). Child instances / mates / wires / devices / device_maps /
targets are copied with ids prefixed "<sub>:" (any nesting depth: "fl:knee:motor"); every reference in them is rewritten with the
same prefix. The parent reaches a child only through its exports ("fl:hip_mount", "fl:power"). Everything downstream of
`assembly.load` sees a flat assembly. Order: the parent's own items first, then each sub-assembly's in `[[subassemblies]]` order.

Limitations (ponytail): a child's instances placed by pos / rpy (other than its root) cannot be included in another assembly
(the weld frame would silently change); mate them instead. Mirroring lands in a later task (`mirror` raises until then).
"""

import copy
import re
import tomllib
from pathlib import Path

import numpy as np

from .library import DesignError

PLANES = {  # reflection across the plane named by its axes (normal = the missing axis)
    "yz": np.diag([-1.0, 1, 1]),
    "xz": np.diag([1.0, -1, 1]),
    "xy": np.diag([1.0, 1, -1]),
}
SUB_ID = re.compile(r"[A-Za-z0-9_-]+")
EXPORT_KINDS = ("ports", "connectors", "params")

# ---- device reference fields (Review Focus 1) -----------------------------------------------------------------------------------
# Kind of every binding field that holds a reference (devices.py::build_devices): an instance id, a mate (joint) name, a
# device_maps table name, or a device name.
FIELD_KIND = {
    "instance": "inst",
    "site": "inst",
    "body": "inst",
    "ref_body": "inst",
    "stator_body": "inst",
    "motor": "inst",
    "duct": "inst",
    "pellet": "inst",
    "joint": "mate",
    "backlash_joint": "mate",
    "map": "map",
    "rotor": "device",
}
_COMMON = (
    "instance",
    "site",
    "body",
    "ref_body",
    "stator_body",
    "joint",
    "backlash_joint",
    "map",
)
_EXTRA = {
    "esc": ("motor",),
    "servo": ("motor",),
    "rotor": ("duct",),
    "launcher": ("pellet",),
    "jet_vane": ("rotor",),
}
_TYPES = [
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
]
REF_FIELDS = {t: _COMMON + _EXTRA.get(t, ()) for t in _TYPES}
# string-valued binding fields that are NOT references (port names, library ids, verbatim text); `name` / `set` are handled apart
NON_REF = {
    "type",
    "name",
    "port",
    "a",
    "b",
    "site_port",
    "ocv",
    "cells",
    "projection",
    "set",
}


def _leaf(x, p):
    """A numeric leaf: '<inst>.<param>' string -> prefixed; provenance table / number untouched; lists elementwise."""
    if isinstance(x, str):
        return p + x
    if isinstance(x, list):
        return [_leaf(v, p) for v in x]
    return copy.deepcopy(x)


def _device(b: dict, p: str, sid: str, file: str) -> dict:
    t = b.get("type")
    fields = REF_FIELDS.get(t)
    if fields is None:
        raise DesignError(f"sub-assembly '{sid}' ({file}): device type {t!r} unknown")
    out = {}
    for k, v in b.items():
        if k == "name":
            out[k] = p + v
        elif k == "set":
            out[k] = {kk: _leaf(vv, p) for kk, vv in v.items()}
        elif k in fields and isinstance(v, str):
            out[k] = p + v
        elif isinstance(v, str) and k not in NON_REF:
            raise DesignError(
                f"sub-assembly '{sid}' ({file}): device '{b.get('name')}' ({t}) field '{k}' is a string that is not in "
                f"subasm.REF_FIELDS['{t}'] or NON_REF; add it, or the prefix would miss a reference"
            )
        else:
            out[k] = copy.deepcopy(v)
    return out


def _check_schema(doc: dict, where: str):
    from .assembly import ALLOWED

    ex = doc.get("exports", {})
    if not isinstance(ex, dict) or set(ex) - set(EXPORT_KINDS):
        raise DesignError(f"{where}: [exports] keys must be {list(EXPORT_KINDS)}")
    for k, t in ex.items():
        if not isinstance(t, dict) or not all(
            isinstance(n, str) and isinstance(v, str) and n.isidentifier()
            for n, v in t.items()
        ):
            raise DesignError(
                f'{where}: exports.{k} must map names to "<inst>.<x>" strings'
            )
    for i, s in enumerate(doc.get("subassemblies", [])):
        if not isinstance(s, dict):
            raise DesignError(f"{where}: [[subassemblies]] entry {i} is not a table")
        bad = set(s) - ALLOWED["subassemblies"]
        if bad:
            raise DesignError(
                f"{where}: [[subassemblies]] entry {i}: unknown keys {sorted(bad)} "
                f"(allowed: {sorted(ALLOWED['subassemblies'])})"
            )
        for k in ("id", "file"):
            if not isinstance(s.get(k), str):
                raise DesignError(
                    f"{where}: [[subassemblies]] entry {i}: missing '{k}'"
                )
        if not SUB_ID.fullmatch(s["id"]):
            raise DesignError(
                f"sub-assembly id {s['id']!r}: use letters, digits, '_' and '-' only"
            )


def expand(doc: dict, path, lib_ids=None, _stack=()):
    """-> (flat_doc, origins). No sub-assemblies: (doc itself, {}). `origins[flat_id] = {"sub", "file", "id"}`."""
    _check_schema(doc, "assembly file")
    subs = doc.get("subassemblies")
    if not subs:
        return doc, {}
    if path is None:
        raise DesignError(
            "[[subassemblies]] needs the parent file's path (relative 'file' paths resolve against it): use load(path), "
            "or pass path= to loads"
        )
    path = Path(path).resolve()
    stack = _stack or (path,)
    out = {k: copy.deepcopy(v) for k, v in doc.items() if k != "subassemblies"}
    for sec in ("instances", "mates", "wires", "devices", "targets"):
        out.setdefault(sec, [])
    out.setdefault("device_maps", {})
    n_mates, n_wires = len(out["mates"]), len(out["wires"])
    own_root = next((i["id"] for i in doc.get("instances", []) if i.get("root")), None)
    origins: dict = {}
    exp: dict = {}  # sub id -> {"file", "ports": {}, "connectors": {}, "params": {}} (values already prefixed)
    seen = set()
    for s in subs:
        sid, file = s["id"], s["file"]
        p = sid + ":"
        if sid in seen:
            raise DesignError(f"duplicate sub-assembly id '{sid}'")
        seen.add(sid)
        if s.get("mirror") and s["mirror"] not in PLANES:
            raise DesignError(
                f"sub-assembly '{sid}': mirror is one of {sorted(PLANES)}, got {s['mirror']!r}"
            )
        cpath = (path.parent / file).resolve()
        if cpath in stack:
            names = [q.name for q in stack[stack.index(cpath) :]] + [cpath.name]
            raise DesignError("cycle: " + " -> ".join(names))
        try:
            cdoc = tomllib.loads(cpath.read_text(encoding="utf-8"))
        except OSError as e:
            raise DesignError(
                f"sub-assembly '{sid}': cannot read '{file}': {e}"
            ) from None
        except tomllib.TOMLDecodeError as e:
            raise DesignError(
                f"sub-assembly '{sid}' ({file}): TOML error: {e}"
            ) from None
        flat, corig = expand(cdoc, cpath, lib_ids, stack + (cpath,))

        def gmap(g, s=s, sid=sid):
            return (
                s["group"] if s.get("group") is not None else f"{sid}:{g}" if g else sid
            )

        roots = [i for i in flat.get("instances", []) if i.get("root")]
        for i in flat.get("instances", []):
            if not i.get("root") and ("pos" in i or "rpy" in i):
                raise DesignError(
                    f"sub-assembly '{sid}' ({file}): instance '{i['id']}' is placed by pos / rpy; mate it instead "
                    "(placed instances of a child would change frame)"
                )
        if len(roots) != 1:
            raise DesignError(
                f"sub-assembly '{sid}' ({file}): needs exactly one root instance (found {len(roots)})"
            )
        if lib_ids is not None:
            for i in flat["instances"]:
                if i["part"] not in lib_ids:
                    raise DesignError(
                        f"sub-assembly '{sid}' ({file}): instance '{i['id']}' uses unknown part '{i['part']}'"
                    )

        def note(key, sid=sid, file=file, corig=corig):
            o = corig.get(key)
            origins[p + key] = (
                {"sub": f"{sid}:{o['sub']}", "file": o["file"], "id": o["id"]}
                if o
                else {"sub": sid, "file": file, "id": key}
            )

        for i in flat.get("instances", []):
            n = copy.deepcopy(i)
            n["id"] = p + i["id"]
            if n.pop("root", False):
                for k in ("pos", "rpy"):
                    if k in s:
                        n[k] = copy.deepcopy(s[k])
            n["group"] = gmap(i.get("group", ""))
            if (
                "_mirror" in i
            ):  # declared inside the child: its plane is in the child's root frame
                m = i["_mirror"]
                n["_mirror"] = {**m, "sub": p + m["sub"], "frame": p + m["frame"]}
            if s.get("mirror"):
                if "_mirror" in i:
                    raise DesignError(
                        f"sub-assembly '{sid}' ({file}) is mirrored and contains a mirrored sub-assembly: not supported"
                    )
                n["_mirror"] = {"plane": s["mirror"], "sub": sid, "frame": own_root}
            if "color" in s and "color" not in n:
                n["color"] = copy.deepcopy(s["color"])
            out["instances"].append(n)
            note(i["id"])
        for m in flat.get("mates", []):
            n = copy.deepcopy(m)
            for k in ("name", "a", "b"):
                if k in n:
                    n[k] = p + n[k]
            if isinstance(n.get("joint"), dict) and isinstance(
                n["joint"].get("range"), str
            ):  # a joint range read from a part param ("thigh.knee_range")
                n["joint"]["range"] = p + n["joint"]["range"]
            out["mates"].append(n)
            if "name" in m:
                note(m["name"])
        for w in flat.get("wires", []):
            n = copy.deepcopy(w)
            for k in ("name", "from", "to", "included_in"):
                if k in n:
                    n[k] = p + n[k]
            n["length"] = _leaf(w["length"], p)
            out["wires"].append(n)
            if "name" in w:
                note(w["name"])
        for name, mp in flat.get("device_maps", {}).items():
            out["device_maps"][p + name] = copy.deepcopy(mp)
            note(name)
        for b in flat.get("devices", []):
            out["devices"].append(_device(b, p, sid, file))
            note(b["name"])
        groups = sorted({gmap(i.get("group", "")) for i in flat.get("instances", [])})
        for t in flat.get("targets", []):
            n = copy.deepcopy(t)
            n["name"] = p + t["name"]
            if "value" in t:
                n["value"] = _leaf(t["value"], p)
            if t.get("all"):
                n.pop("all")
                n["groups"] = groups
            elif "groups" in t:
                n["groups"] = [gmap(g) for g in t["groups"]]
            out["targets"].append(n)
            note(t["name"])
        e = flat.get("exports", {})
        exp[sid] = {"file": file} | {
            k: {n: p + v for n, v in e.get(k, {}).items()} for k in EXPORT_KINDS
        }

    def resolve(ref, kinds, where):
        """'<sub>:<export>[.<pin>]' -> the flat reference; a plain reference is returned as is."""
        if not isinstance(ref, str) or ":" not in ref:
            return ref
        sid, _, rest = ref.partition(":")
        if sid not in exp:
            raise DesignError(f"{where}: {ref!r}: no sub-assembly '{sid}'")
        name, dot, pin = rest.partition(".")
        for k in kinds:
            if name in exp[sid][k]:
                return exp[sid][k][name] + dot + pin
        have = sorted(n for k in kinds for n in exp[sid][k])
        f = exp[sid]["file"]
        if dot:
            raise DesignError(
                f"{where}: {ref!r} reaches inside sub-assembly '{sid}'; export it in {f} [exports] "
                f"(then reference '{sid}:<name>'; exported: {', '.join(have) or 'none'})"
            )
        raise DesignError(
            f"{where}: {ref!r} is not an export of {f} (exports: {', '.join(have) or 'none'})"
        )

    for i, m in enumerate(
        out["mates"][:n_mates]
    ):  # the document's own; children's are done
        for k in ("a", "b"):
            m[k] = resolve(m[k], ("ports",), f"[[mates]] {i}")
    for i, w in enumerate(out["wires"][:n_wires]):
        for k in ("from", "to"):
            w[k] = resolve(w[k], ("connectors", "ports"), f"[[wires]] {i}")
    ex = out.get("exports", {})
    for kind, t in ex.items():
        for n, v in t.items():
            t[n] = resolve(v, (kind,), f"exports.{kind}.{n}")
    # exported-param overrides -> params entry of the target instance
    byid = {i["id"]: i for i in out["instances"]}
    for s in subs:
        sid = s["id"]
        for name, val in s.get("params", {}).items():
            tgt = exp[sid]["params"].get(name)
            if tgt is None:
                raise DesignError(
                    f"sub-assembly '{sid}': param override '{name}' is not exported by {exp[sid]['file']} "
                    f"(exports: {', '.join(sorted(exp[sid]['params'])) or 'none'})"
                )
            iid, _, pname = tgt.partition(".")
            byid[iid].setdefault("params", {})[pname] = copy.deepcopy(val)
    for sec in ("device_maps", "devices", "targets", "wires", "mates"):
        if not out[sec]:
            del out[sec]
    return out, origins
