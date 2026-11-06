#!/usr/bin/env python3
"""Validate parts/**/*.toml against parts/README.md. Exit 1 on any problem.

Usage: python3 tools/check_parts.py [parts_dir]
"""

import collections
import importlib.util
import pathlib
import sys
import tomllib
import types

FIDELITY = {"datasheet", "literature", "measured", "derived", "estimated"}
SOURCED = {"datasheet", "literature", "measured"}


def _pins_module():
    """python/dvsim/design/pins.py loaded by path (stdlib only; importing the package would need numpy)."""
    f = pathlib.Path(__file__).resolve().parents[1] / "python/dvsim/design/pins.py"
    spec = importlib.util.spec_from_file_location("dvsim_pins_schema", f)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def check(path: pathlib.Path, problems: list[str], tally: collections.Counter) -> None:
    def bad(msg: str) -> None:
        problems.append(f"{path}: {msg}")

    try:
        doc = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as e:
        return bad(f"TOML parse error: {e}")
    for key in ("id", "name", "category", "vendor"):
        if not doc.get(key):
            bad(f"missing '{key}'")
    if doc.get("id") != path.stem:
        bad(f"id '{doc.get('id')}' != file name '{path.stem}'")
    sources = {}
    for s in doc.get("sources", []):
        for key in ("id", "url", "title", "accessed", "quote"):
            if not s.get(key):
                bad(f"source {s.get('id', '?')} missing '{key}'")
        sources[s.get("id")] = s

    def check_entry(name: str, e: dict, need_unit: bool) -> None:
        fid = e.get("fidelity")
        tally[fid] += 1
        if fid not in FIDELITY:
            return bad(f"{name}: fidelity '{fid}' not in {sorted(FIDELITY)}")
        if need_unit and ("value" not in e or "unit" not in e):
            bad(f"{name}: needs 'value' and 'unit'")
        if fid in SOURCED and e.get("src") not in sources:
            bad(f"{name}: src '{e.get('src')}' not in [[sources]]")
        if fid == "derived" and not e.get("derivation"):
            bad(f"{name}: derived without 'derivation'")
        if fid == "estimated" and not e.get("rationale"):
            bad(f"{name}: estimated without 'rationale'")
        if "src" in e and e["src"] not in sources:
            bad(f"{name}: src '{e['src']}' not in [[sources]]")

    params = doc.get("params", {})
    if not params:
        bad("no [params]")
    for name, e in params.items():
        if not isinstance(e, dict):
            bad(f"param {name}: must be a table")
            continue
        check_entry(f"param {name}", e, need_unit=True)
        if name == "price" and e.get("unit") != "USD":
            bad(
                f"param price: unit must be 'USD' (the BOM sums it), got {e.get('unit')!r}"
            )
    mir = doc.get("mirror")
    if mir is not None:
        planes = ("xy", "yz", "xz")
        if not isinstance(mir, dict) or set(mir) - {"symmetry", "opposite", "plane"}:
            bad("[mirror]: keys are symmetry, opposite, plane")
        else:
            if mir.get("symmetry") not in (None, *planes):
                bad(f"[mirror] symmetry must be one of {planes}")
            if mir.get("plane", "xz") not in planes:
                bad(f"[mirror] plane must be one of {planes}")
            if not (mir.get("symmetry") or mir.get("opposite")):
                bad("[mirror]: give symmetry or opposite")
    for name, t in doc.get("tables", {}).items():
        check_entry(f"table {name}", t, need_unit=False)
        cols = t.get("columns", [])
        for i, row in enumerate(t.get("rows", [])):
            if len(row) != len(cols):
                bad(f"table {name} row {i}: {len(row)} values for {len(cols)} columns")
    names = set()
    for i, p in enumerate(doc.get("ports", [])):
        label = f"port {p.get('name', i)}"
        if not p.get("name") or p["name"] in names:
            bad(f"{label}: missing or duplicate name")
        names.add(p.get("name"))
        if p.get("kind") not in ("mech", "power", "signal"):
            bad(f"{label}: kind '{p.get('kind')}' not mech|power|signal")
        if not p.get("type"):
            bad(f"{label}: missing type")
        if p.get("gender", "neutral") not in ("male", "female", "neutral"):
            bad(f"{label}: gender '{p.get('gender')}'")
        if p.get("kind") == "mech":
            for key in ("pos", "axis"):
                v = p.get(key)
                if not (isinstance(v, list) and len(v) == 3):
                    bad(f"{label}: '{key}' must be 3 numbers")
            axis = p.get("axis")
            if (
                isinstance(axis, list)
                and len(axis) == 3
                and abs(sum(a * a for a in axis) - 1) > 1e-6
            ):
                bad(f"{label}: axis not unit length")
            # every mech port needs ref (unit, perpendicular to axis) and symmetry, or snapping to it fails
            ref = p.get("ref")
            if not (isinstance(ref, list) and len(ref) == 3):
                bad(f"{label}: 'ref' must be 3 numbers")
            elif isinstance(axis, list) and len(axis) == 3:
                if abs(sum(r * r for r in ref) - 1) > 1e-6:
                    bad(f"{label}: ref not unit length")
                elif abs(sum(r * a for r, a in zip(ref, axis))) > 1e-6:
                    bad(f"{label}: ref not perpendicular to axis")
            sym = p.get("symmetry")
            if isinstance(sym, bool) or not isinstance(sym, int) or sym < 1:
                bad(f"{label}: 'symmetry' must be an integer >= 1")
        lead = p.get("lead")
        if (
            lead is not None
        ):  # parts/README.md "Lead flag": length / awg = a param name of this part, or a provenance table
            if p.get("kind") not in ("power", "signal") or not isinstance(lead, dict):
                bad(f"{label}: lead must be a table on a power / signal port")
            for lk in ("length", "awg"):
                v = (lead if isinstance(lead, dict) else {}).get(lk)
                if isinstance(v, str):
                    if not isinstance(params.get(v), dict):
                        bad(
                            f"{label}.lead.{lk}: names param '{v}', which the part lacks"
                        )
                elif isinstance(v, dict):
                    check_entry(f"{label}.lead.{lk}", v, need_unit=True)
                elif v is not None:
                    bad(f"{label}.lead.{lk}: param name or provenance table")
        for key, e in p.items():
            if isinstance(e, dict) and key != "lead":
                check_entry(f"{label}.{key}", e, need_unit=True)
    for msg in _pins_module().validate(doc):  # parts/README.md "Connectors and pins"
        bad(msg)
    mesh = doc.get("geometry", {}).get("mesh")
    if mesh and not (path.parents[2] / mesh).exists():
        bad(f"mesh '{mesh}' not found")
    if (
        "visual" in doc
    ):  # cosmetic layer (parts/README.md "Visual layer"): no fidelity tags, only structure + sources
        try:
            sys.path.insert(
                0, str(pathlib.Path(__file__).resolve().parents[1] / "python")
            )
            from dvsim.design import visuals
        except ImportError:  # numpy missing: the generator names cannot be checked here (pytest covers them)
            return
        part = types.SimpleNamespace(raw=doc, params=params)
        for msg in visuals.validate(part, path.parents[2], set(sources)):
            bad(msg)


def main() -> int:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "parts")
    files = sorted(root.glob("[!_]*/*.toml"))
    problems: list[str] = []
    tally: collections.Counter = collections.Counter()
    for f in files:
        check(f, problems, tally)
    ids = {f.stem for f in files}
    for f in sorted((root / "_prices").glob("*.toml")):  # parts/README.md "Price files"
        try:
            prices = tomllib.loads(f.read_text()).get("prices", {})
        except tomllib.TOMLDecodeError as e:
            problems.append(f"{f}: TOML parse error: {e}")
            continue
        for pid, e in prices.items():
            bad = lambda m: problems.append(f"{f}: {pid}: {m}")
            if pid not in ids:
                bad("no such part id")
            if not isinstance(e, dict):
                bad("must be a table")
                continue
            if e.get("not_for_sale") is True:
                if not e.get("rationale"):
                    bad("not_for_sale without 'rationale'")
                continue
            ks = [k for k in ("value", "price_per_m", "price_per_kg") if k in e]
            if (
                len(ks) != 1
                or isinstance(e[ks[0]], bool)
                or not isinstance(e[ks[0]], (int, float))
                or e[ks[0]] <= 0
            ):
                bad(
                    "needs exactly one positive value / price_per_m / price_per_kg (or not_for_sale = true)"
                )
            if e.get("fidelity") not in FIDELITY:
                bad(f"fidelity '{e.get('fidelity')}' not in {sorted(FIDELITY)}")
            elif e["fidelity"] == "estimated" and not e.get("rationale"):
                bad("estimated without 'rationale'")
    try:  # coverage: needs numpy (the BOM's printed-cost function), like the visual check above
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
        from dvsim.design import bom, library

        lib = library.Library(root)
        unpriced = [
            p.id
            for p in lib.parts.values()
            if not any(
                k in p.params
                for k in ("price", "price_per_m", "price_per_kg", "not_for_sale")
            )
            and bom.print_cost(p, lib) is None
        ]
        print(
            f"price coverage: {len(lib.parts) - len(unpriced)}/{len(lib.parts)} parts; no price: {', '.join(unpriced)}"
        )
    except ImportError:
        print("price coverage: skipped (numpy missing)")
    for p in problems:
        print(p)
    print(
        f"{len(files)} parts, {sum(tally.values())} values "
        f"({', '.join(f'{k}={v}' for k, v in sorted(tally.items(), key=str))}), "
        f"{len(problems)} problem(s)"
    )
    return 1 if problems or not files else 0


if __name__ == "__main__":
    sys.exit(main())
