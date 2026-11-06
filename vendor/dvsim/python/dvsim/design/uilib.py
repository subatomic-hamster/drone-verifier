"""Library page back end: fuzzy search with param filters, category tree, part detail, part bounding boxes and the ghost
placements of a drag. Pure functions of a `Library` (+ the world poses of a built design for the placement ones); the viewer's
library page, the compact BUILD sidebar and scripted agents all go through `dvsim.design.viewer.Backend`, which calls these."""

import difflib
import json
import re
from pathlib import Path

import numpy as np

from .build import part_shapes
from .library import DesignError, Library, Part
from .snap import mate_transform, port_frame

# filter name -> the part params it reads (first one present wins); any other name is read as a param of that name.
# Values are SI base units (V, kg, m, A ...), as stored in the part files.
ALIASES = {
    "voltage": ("v_nominal", "voltage_rated", "v_full", "v_operating_max"),
    "capacity": ("capacity_ah",),
    "current": ("i_max_continuous", "current_rated", "i_peak"),
    "diameter": ("diameter", "outer_diameter"),
}
FIELDS = (  # (field, weight of a substring hit, weight of a word-prefix hit)
    ("id", 60, 80),
    ("name", 50, 65),
    ("vendor", 40, 55),
    (
        "category",
        40,
        85,
    ),  # "motor" lists the motors before parts whose id merely mentions a motor
    ("part_number", 35, 45),
)


def _words(s: str) -> list:
    return [w for w in re.split(r"[^a-z0-9.]+", s.lower()) if w]


def num_params(part: Part) -> dict:
    """{param: (value, unit)} of the numeric params."""
    out = {}
    for k, e in part.params.items():
        v = part.value(k)
        if v is not None:
            out[k] = (v, e.get("unit", ""))
    return out


def _param_of(part: Part, name: str):
    for k in ALIASES.get(name, (name,)):
        v = part.value(k)
        if v is not None:
            return v
    return None


def thumbs_dir(lib: Library) -> Path:
    return lib.root / "build" / "thumbs"


def thumb_index(lib: Library) -> dict:
    """build/thumbs/index.json {id: {name, category, vendor, thumb}} (tools/thumbs.py), {} when it was not generated."""
    try:
        return json.loads((thumbs_dir(lib) / "index.json").read_text())
    except (OSError, ValueError):
        return {}


def thumb_path(lib: Library, pid: str, index: dict | None = None):
    index = thumb_index(lib) if index is None else index
    t = (index.get(pid) or {}).get("thumb")
    for p in ([Path(t), thumbs_dir(lib) / t, lib.root / t] if t else []) + [
        thumbs_dir(lib) / f"{pid}.png"
    ]:
        if p.is_file():
            return str(p.resolve())
    return None


def _score(part: Part, tokens: list) -> float:
    """Sum over the query tokens of the best field hit; 0 when any token hits nothing (all tokens must match)."""
    total = 0.0
    nums = num_params(part)
    for t in tokens:
        best = 0.0
        for field, w_sub, w_pre in FIELDS:
            val = (
                part.id
                if field == "id"
                else part.category
                if field == "category"
                else str(part.raw.get(field, ""))
            ).lower()
            if not val:
                continue
            if val == t:
                best = max(best, 100)
            elif any(w.startswith(t) for w in _words(val)):
                best = max(best, w_pre)
            elif t in val:
                best = max(best, w_sub)
            elif len(t) >= 4:  # typo tolerance: close to a whole word
                r = max(
                    (difflib.SequenceMatcher(None, t, w).ratio() for w in _words(val)),
                    default=0,
                )
                if r >= 0.8:
                    best = max(best, 30 * r)
        if (
            best == 0
        ):  # a param name ("kv", "mass") or a number that is a param value ("920")
            if t in nums:
                best = 20
            else:
                try:
                    x = float(t)
                    best = (
                        15
                        if any(
                            abs(v - x) <= 1e-9 + 1e-6 * abs(x) for v, _ in nums.values()
                        )
                        else 0
                    )
                except ValueError:
                    pass
        if best == 0:
            return 0.0
        total += best
    return total


def search(
    lib: Library,
    text: str = "",
    filters=None,
    category: str = "",
    limit: int = 60,
    offset: int = 0,
) -> dict:
    """Fuzzy search. `text`: whitespace-separated tokens, all must hit id / name / vendor / category / part number / a param name or
    value (prefix > substring > typo). `category`: dotted prefix. `filters`: {param or alias: [lo, hi]} (null = open end, a bare
    number = equal); a part without that param is out. Returns {total, parts: [{id, name, vendor, category, mass, score}]}."""
    tokens = _words(text)
    index = thumb_index(lib)
    rows = []
    for p in lib.parts.values():
        if category and not (
            p.category == category or p.category.startswith(category + ".")
        ):
            continue
        ok = True
        for k, rng in (filters or {}).items():
            v = _param_of(p, k)
            lo, hi = (
                (rng, rng)
                if isinstance(rng, (int, float))
                else (list(rng) + [None, None])[:2]
            )
            tol = 1e-6 * abs(rng) if isinstance(rng, (int, float)) else 0
            if (
                v is None
                or (lo is not None and v < lo - tol)
                or (hi is not None and v > hi + tol)
            ):
                ok = False
                break
        if not ok:
            continue
        s = _score(p, tokens) if tokens else 1.0
        if s > 0:
            rows.append((-s, p.id, p))
    rows.sort(key=lambda r: r[:2])
    return {
        "total": len(rows),
        "parts": [
            {
                "id": p.id,
                "name": p.raw.get("name", p.id),
                "vendor": p.raw.get("vendor", ""),
                "category": p.category,
                "mass": p.value("mass"),
                "score": -s,
                "thumb": thumb_path(lib, p.id, index),
            }
            for s, _, p in rows[offset : offset + limit]
        ],
    }


def categories(lib: Library) -> list:
    """Category tree as a flat list [{path, count, depth}] (count includes sub-categories), sorted by path."""
    cnt = {}
    for p in lib.parts.values():
        segs = p.category.split(".") if p.category else ["(none)"]
        for i in range(len(segs)):
            k = ".".join(segs[: i + 1])
            cnt[k] = cnt.get(k, 0) + 1
    return [
        {"path": k, "count": n, "depth": k.count(".")} for k, n in sorted(cnt.items())
    ]


def param_ranges(lib: Library, category: str = "") -> dict:
    """{param: [min, max, unit]} over the parts of `category` for the params that at least 3 of them carry (filter sliders)."""
    seen = {}
    for p in lib.parts.values():
        if category and not (
            p.category == category or p.category.startswith(category + ".")
        ):
            continue
        for k, (v, u) in num_params(p).items():
            seen.setdefault(k, []).append((v, u))
    out = {
        k: [min(v for v, _ in xs), max(v for v, _ in xs), xs[0][1]]
        for k, xs in seen.items()
        if len(xs) >= 3
    }
    for alias in ALIASES:  # aliases show as one filter over the first param present
        vals = [x for p in lib.parts.values() if (x := _param_of(p, alias)) is not None]
        if len(vals) >= 3:
            out[alias] = [min(vals), max(vals), "V" if alias == "voltage" else ""]
    return out


# ---- detail -------------------------------------------------------------------------------------------------------------
def part_boxes(lib: Library, part: Part) -> list:
    """Bounding boxes of the part's shapes in the part frame: [{c: centre, R: 9 floats row-major, h: half extents}]."""
    from .drc import _mesh_info

    out = []
    for s in part_shapes(lib, part):
        k, sz = s["kind"], s["size"]
        if k == "box":
            c, h = np.zeros(3), np.array(sz[:3], float)
        elif k == "sphere":
            c, h = np.zeros(3), np.full(3, sz[0])
        elif k == "cylinder":
            c, h = np.zeros(3), np.array([sz[0], sz[0], sz[1]])
        elif k == "capsule":
            c, h = np.zeros(3), np.array([sz[0], sz[0], sz[1] + sz[0]])
        else:
            _, lo, hi = _mesh_info(s["path"])
            c, h = (lo + hi) / 2, (hi - lo) / 2
        R = np.asarray(s["R"], float)
        out.append(
            {
                "c": (R @ c + np.asarray(s["pos"], float)).tolist(),
                "R": R.flatten().tolist(),
                "h": h.tolist(),
            }
        )
    return out


def fnum(x):
    return None if x is None else float(x)


def info(lib: Library, pid: str) -> dict:
    p = lib.get(pid)
    src = {s.get("id"): s for s in p.raw.get("sources", [])}
    params = []
    for k, e in p.params.items():
        if not isinstance(e, dict):
            continue
        s = src.get(e.get("src"), {})
        params.append(
            {
                "name": k,
                "value": e.get("value")
                if not isinstance(e.get("value"), (dict, list))
                or len(str(e.get("value"))) < 80
                else "(table)",
                "unit": e.get("unit", ""),
                "fidelity": e.get("fidelity", ""),
                "src": e.get("src", ""),
                "url": s.get("url", ""),
                "quote": e.get("quote")
                or s.get("quote", "")
                or e.get("rationale", "")
                or e.get("derivation", ""),
            }
        )
    ports = []
    for q in p.ports:
        ports.append(
            {
                "name": q.get("name"),
                "kind": q.get("kind", ""),
                "type": q.get("type", ""),
                "gender": q.get("gender", ""),
                "role": q.get("role", ""),
                "pos": q.get("pos", [0, 0, 0]),
                "axis": q.get("axis", [0, 0, 1]),
            }
        )
    try:
        boxes = part_boxes(lib, p)
    except DesignError:
        boxes = []
    return {
        "id": p.id,
        "name": p.raw.get("name", p.id),
        "vendor": p.raw.get("vendor", ""),
        "category": p.category,
        "part_number": p.raw.get("part_number", ""),
        "url": p.raw.get("url", ""),
        "params": params,
        "ports": ports,
        "boxes": boxes,
        "thumb": thumb_path(lib, p.id),
    }


# ---- ghost placements -----------------------------------------------------------------------------------------------------
def ghost_poses(
    design, geo: dict, used: set, part: Part, only: str | None = None
) -> list:
    """Every open mech port of the design that `part` can mate to, with the pose the part would take (default rotation k = 0,
    best child port by the snap's own port score): [{to, port, score, pos, R, port_pos, port_axis}] best first. `geo` =
    Backend's per-instance world poses {iid: {pos, R, ...}}. The real drop re-ranks the rotations (Design.snap); this is the
    cheap, build-free preview a cursor can ask for every few frames."""
    out = []
    for iid, g in geo.items():
        inst = next((d for d in design.doc["instances"] if d["id"] == iid), None)
        if inst is None or (only and only != iid):
            continue
        parent = design._part(inst["part"])
        Ta = np.eye(4)
        Ta[:3, :3], Ta[:3, 3] = np.array(g["R"]).reshape(3, 3), g["pos"]
        for pa in parent.ports:
            if pa.get("kind") != "mech" or (iid, pa["name"]) in used:
                continue
            pairs = design._pairs([pa], part.ports)
            if not pairs:
                continue
            best = pairs[0]
            Tb = mate_transform(Ta, best["a"], best["b"], 0)
            Pa = Ta @ port_frame(pa)
            out.append(
                {
                    "to": f"{iid}.{pa['name']}",
                    "port": best["b"]["name"],
                    "score": best["score"],
                    "pos": Tb[:3, 3].tolist(),
                    "R": Tb[:3, :3].flatten().tolist(),
                    "port_pos": Pa[:3, 3].tolist(),
                    "port_axis": Pa[:3, 2].tolist(),
                }
            )
    out.sort(key=lambda r: (-r["score"], r["to"]))
    return out


def my_parts(root: Path, lib: Library | None = None) -> list:
    """MY PARTS: the workspace assemblies (`root/*.toml`, `root/assemblies/*.toml`) that declare an `[exports]` table, as
    [{name, path, exports: {ports|connectors|params: [names]}, mass, cost}] sorted by name. `mass` (kg) and `cost` (USD) come from
    a build when `lib` is given (None otherwise, or when the file does not build; then `error` says why); cost needs the BOM
    (report["bom"]) and is None until the parts carry prices."""
    import tempfile
    import tomllib

    from .build import compute

    out = []
    for p in sorted(
        [*Path(root).glob("*.toml"), *Path(root).glob("assemblies/*.toml")]
    ):
        try:
            doc = tomllib.loads(p.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            continue
        ex = {k: sorted(v) for k, v in (doc.get("exports") or {}).items() if v}
        if not ex:
            continue
        row = {
            "name": doc.get("assembly", {}).get("name", p.stem),
            "path": str(p),
            "exports": ex,
            "mass": None,
            "cost": None,
        }
        if lib is not None:
            try:
                with tempfile.TemporaryDirectory(prefix="dvsim-mypart-") as t:
                    _, rep = compute(p, lib.dir, t, lib, visual=False, route=False)
                row["mass"] = rep["mass"]["total"]
                row["cost"] = rep.get("bom", {}).get("total", {}).get("cost")
            except DesignError as e:
                row["error"] = str(e)
        out.append(row)
    return sorted(out, key=lambda r: r["name"])
