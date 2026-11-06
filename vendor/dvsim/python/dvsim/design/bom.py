"""Bill of materials: a pure function of the build report + library (+ the sub-assembly origins of the flat document).

Lines (`kind`): placed instances (part), fasteners of the mates (screws and nuts: fastener), custom printed / cut parts
(printed: cost from material and volume, `print_cost`), wires (wire) and the connectors on wire ends (connector).
A part's unit price is its `price` param, in USD (tools/check_parts.py enforces the unit; parts/_prices/*.toml are merged into
it by Library). An instance without `price` is priced, in this order: custom part -> `print_cost`; `price_per_kg` x its mass
(plates); `price_per_m` x its length = mass / mass_per_m (cut stock: tubes, extrusions); `not_for_sale` -> $0, listed in
`not_for_sale`. Anything else is listed in `missing` and adds nothing, so `total.cost` is a lower bound while `missing` is not empty.

Wires: length x conductors x price_per_m when the wire part is ONE conductor (an AWG wire; `price_per_m` is per metre of one
conductor), length x price_per_m when the part has a `conductors` param (a whole multi-conductor cable, the same rule the
wire mass uses in devices.py). Skipped: wires with `included_in` (priced inside that part) and wires with both ends offboard.
Connectors: one unit price of the connector part per `report["connectors"]` entry, skipping those of an `included_in` wire,
offboard ends, and cable-own end connectors (the connector part is the wire part: inside the per-metre price). Crimp terminals
are not modelled.

    {"lines": [{part, kind, count, unit_price, fidelity, cost, ...}], "by_sub": {sub: {mass, cost, missing}}, "by_kind": {kind: cost},
     "total": {mass, cost, estimated_share, parent_mass, parent_cost}, "missing": [part ids], "not_for_sale": [part ids]}

A printed line also has material, volume_cm3, grams (per unit) and note; a wire line counts metres of conductor
(`unit` "m", count x unit_price = cost). `by_sub` is keyed by the outermost sub-assembly id ("fl" also holds "fl:knee:motor").
`total.mass` is the build report's total (wires included); `parent_mass` / `parent_cost` are the lines that belong to no
sub-assembly, so by_sub + parent = everything counted.
"""

# Printed / cut part cost knobs. All ESTIMATES: calibrate on the bench (a real print's grams and hours) and edit here.
PRINT_WASTE = 1.10  # support + purge + failed-print allowance on filament mass
PRINT_CM3_PER_H = 10.0  # machine throughput, cm3 of material per hour
MACHINE_USD_PER_H = 0.50  # printer wear + electricity per hour
INFILL_EQUIVALENT = 0.40  # material fraction of the envelope volume (3 walls + 20 % infill), when the part does not say
CUT_WASTE = (
    1.5  # stock offcuts for cut aluminium / carbon (buy 1.5x the part's material)
)
CUT_MATERIALS = ("aluminium", "carbon", "stainless")  # id prefixes: cut, not printed
DEFAULT_MATERIAL = "pla"  # a custom part that names no material
CUSTOM_VENDOR = ("printed", "own design", "custom")


def _unit_price(lib, pid, key="price"):
    """-> (value, fidelity) or None (key "price" must be unit USD)."""
    if pid not in lib.parts:
        return None
    e = lib.get(pid).params.get(key)
    v = e.get("value") if isinstance(e, dict) else None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if key == "price" and e.get("unit") != "USD":
        return None
    return float(v), e.get("fidelity", "")


def print_cost(part, lib) -> dict | None:
    """Cost of ONE custom printed / cut part from material and volume, or None (not custom, or no material price / density /
    volume). Material per geometry shape (default pla, said in `note`); material volume = `volume` (or `volume_solid`) x
    `infill_equivalent` (default INFILL_EQUIVALENT), else mass / density, else sum of shape volumes x INFILL_EQUIVALENT
    (cut parts: solid, factor 1). Printed: V x density x price_per_kg x PRINT_WASTE + V_cm3 / PRINT_CM3_PER_H x MACHINE_USD_PER_H.
    Cut: V x density x price_per_kg x CUT_WASTE."""
    from .build import part_shapes
    from .library import DesignError
    from .mass import shape_volume

    g, vendor = part.geometry, str(part.raw.get("vendor", "")).lower()
    if not (
        part.category == "frame.printed" or any(w in vendor for w in CUSTOM_VENDOR)
    ):
        return None
    raw = g.get("shapes") or ([g] if "shape" in g else [])
    mats = [s.get("material") or g.get("material") for s in raw] or [g.get("material")]
    note = "" if all(mats) else f"material not named: default {DEFAULT_MATERIAL}"
    mats = [m or DEFAULT_MATERIAL for m in mats]
    try:
        sv = [
            shape_volume(s["kind"], s["size"], s["path"])
            for s in part_shapes(lib, part)
        ]
    except (DesignError, OSError, ValueError, TypeError):
        sv = []
    geo = sum(sv) if len(sv) == len(mats) else 0.0  # shape volumes weight the materials
    vols = sv if geo > 0 else [1.0] * len(mats)
    w = [v / sum(vols) for v in vols]
    env = part.value("volume") or part.value("volume_solid")
    cost = m3 = kg = 0.0
    names = []
    for mid, wi in zip(mats, w):
        mat = lib.parts.get(mid)
        rho = mat and (
            mat.value("density")
            or next((mat.value(k) for k in mat.params if k.startswith("density")), None)
        )
        pk = mat and mat.value("price_per_kg")
        if not (rho and pk):
            return None
        cut = mid.startswith(CUT_MATERIALS)
        if env:
            v = (
                env
                * wi
                * (1.0 if cut else part.value("infill_equivalent") or INFILL_EQUIVALENT)
            )
        elif part.value("mass"):
            v = part.value("mass") * wi / rho
        elif geo:
            v = geo * wi * (1.0 if cut else INFILL_EQUIVALENT)
        else:
            return None
        cost += (
            v * rho * pk * CUT_WASTE
            if cut
            else v * rho * pk * PRINT_WASTE
            + v * 1e6 / PRINT_CM3_PER_H * MACHINE_USD_PER_H
        )
        m3, kg = m3 + v, kg + v * rho
        names.append(mid)
    return {
        "unit_price": cost,
        "fidelity": "derived",
        "material": "+".join(dict.fromkeys(names)),
        "volume_cm3": m3 * 1e6,
        "grams": kg * 1000,
        "note": note,
    }


def _part_price(lib, pid, mass):
    """Priced instance -> {cost, fidelity, extra, kind}; "free" for not_for_sale; None when unpriced."""
    if pid not in lib.parts:
        return None
    part = lib.get(pid)
    if up := _unit_price(lib, pid):
        return {"cost": up[0], "fidelity": up[1], "kind": "part"}
    if pc := print_cost(part, lib):
        return {"cost": pc.pop("unit_price"), "kind": "printed", **pc}
    if pk := _unit_price(lib, pid, "price_per_kg"):
        return {"cost": mass * pk[0], "fidelity": pk[1], "kind": "part"}
    ppm, mpm = _unit_price(lib, pid, "price_per_m"), part.value("mass_per_m")
    if ppm and mpm:
        return {"cost": mass / mpm * ppm[0], "fidelity": ppm[1], "kind": "part"}
    if "not_for_sale" in part.params:
        return "free"
    return None


def bom(report: dict, lib, origins: dict | None = None) -> dict:
    origins = origins or {}
    sub_of = lambda iid: origins.get(iid, {}).get("sub", "")
    off = {r["id"] for r in report["instances"] if r.get("offboard")}
    rows = []  # (part id, kind, count, mass, sub or "", price {cost, fidelity, ...} / "free" / None)
    for r in report["instances"]:
        if not r.get("offboard"):
            rows.append(
                (
                    r["part"],
                    "part",
                    1,
                    r["mass"],
                    sub_of(r["id"]),
                    _part_price(lib, r["part"], r["mass"]),
                )
            )
    for f in report.get("fasteners", []):
        sub = origins.get(f["mate"], {}).get("sub", "")
        for pid, mass in ((f["part"], f["mass"]), (f.get("nut"), 0.0)):
            # the screws carry the mate's fastener mass (screw + nut)
            if pid:
                p = _unit_price(lib, pid)
                rows.append(
                    (
                        pid,
                        "fastener",
                        f["count"],
                        mass,
                        sub,
                        p and {"cost": p[0], "fidelity": p[1]},
                    )
                )
    for w in report.get("wires", []):
        if (
            w.get("included_in")
            or {w["from"].split(".")[0], w["to"].split(".")[0]} <= off
        ):
            continue
        part = lib.parts.get(w["part"])
        per_cable = part is not None and part.value("conductors") is not None
        n = w["length"] * (1 if per_cable else w["conductors"])
        p = _unit_price(lib, w["part"], "price_per_m")
        if (
            p is None and part is not None
        ):  # sold as a fixed-length cable: price / its length
            u = _unit_price(lib, w["part"])
            sold = part.value("length_default") or part.value("length")
            p = (u[0] / sold, u[1]) if u and sold else None
        rows.append(
            (
                w["part"],
                "wire",
                n,
                0.0,
                sub_of(w["from"].split(".")[0]),
                p and {"cost": p[0], "fidelity": p[1]},
            )
        )
    wires = report.get("wires", [])
    for c in report.get("connectors", []):
        w = wires[c["wire"]] if c["wire"] < len(wires) else {}
        if (
            not c["part"]
            or w.get("included_in")
            or c["instance"] in off
            or c["part"] == w.get("part")
        ):
            continue
        p = _unit_price(lib, c["part"])
        rows.append(
            (
                c["part"],
                "connector",
                1,
                0.0,
                sub_of(c["instance"]),
                p and {"cost": p[0], "fidelity": p[1]},
            )
        )
    lines, by_sub, by_kind, missing, free = {}, {}, {}, [], []
    est_cost = cost = 0.0
    parent = {"mass": 0.0, "cost": 0.0}
    for pid, kind, n, mass, sub, price in rows:
        top = sub.split(":")[0]
        tgt = (
            by_sub.setdefault(top, {"mass": 0.0, "cost": 0.0, "missing": []})
            if top
            else parent
        )
        tgt["mass"] += mass
        if isinstance(price, dict):
            kind = price.get("kind", kind)
        line = lines.setdefault(
            (pid, kind),
            {
                "part": pid,
                "kind": kind,
                "count": 0,
                "unit_price": None,
                "fidelity": None,
                "cost": 0.0,
            },
        )
        line["count"] += n
        if kind == "wire":
            line["unit"] = "m"
        if price == "free":
            line["unit_price"] = 0.0
            if pid not in free:
                free.append(pid)
            continue
        if price is None:
            if pid not in missing:
                missing.append(pid)
            if top and pid not in tgt["missing"]:
                tgt["missing"].append(pid)
            continue
        c = price["cost"] * (n if kind in ("fastener", "wire", "connector") else 1)
        line["cost"] += c
        line["fidelity"] = price["fidelity"]
        line.update(
            {
                k: price[k]
                for k in ("material", "volume_cm3", "grams", "note")
                if k in price
            }
        )
        tgt["cost"] += c
        cost += c
        by_kind[kind] = by_kind.get(kind, 0.0) + c
        if price["fidelity"] == "estimated" or kind == "printed":
            est_cost += c
    for line in lines.values():
        if line["fidelity"] is not None:
            line["unit_price"] = line["cost"] / line["count"] if line["count"] else None
    return {
        "lines": sorted(lines.values(), key=lambda x: (x["part"], x["kind"])),
        "by_sub": dict(sorted(by_sub.items())),
        "by_kind": by_kind,
        "total": {
            "mass": report["mass"]["total"],
            "cost": cost,
            "estimated_share": est_cost / cost if cost else 0.0,
            "parent_mass": parent["mass"],
            "parent_cost": parent["cost"],
        },
        "missing": sorted(missing),
        "not_for_sale": sorted(free),
    }
