"""The agent's design: slot choices applied to a real dvsim Design built from assemblies/hopper.toml, so mates, wires and devices are real and dvsim's DRC and BOM are the truth.

Prices live in dv/task.toml and are written into the in-memory part library (dvsim's BOM reads `price` from there). The GNSS chip is an off-board data instance whose module the
harness carries, so its price is added to the BOM total and its mass set on the harness, as assemblies/hopper.toml does for its RTK receiver."""

import math
import os
import re
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from dvsim.design import Design, DesignError, Library, compute
from dvsim.design import bom as _bom
from dvsim.design.assembly import loads

HERE = Path(__file__).resolve().parent
DVSIM = Path(os.environ["DVSIM_ROOT"])  # the only way to the dvsim tree (hf.sh sets it)
TASK = tomllib.loads((HERE / "task.toml").read_text())
SLOTS = TASK["slots"]
LIMITS = TASK["limits"]
ASSEMBLY = DVSIM / "assemblies" / "hopper.toml"
EST = "drone-verifier catalog value, see dv/task.toml"
_lib = None


def library():
    """The parts library with the catalog's prices. ponytail: mutated in memory, dvsim's files are never touched."""
    global _lib
    if _lib is None:
        _lib = Library(DVSIM / "parts")
        for part, price in TASK[
            "kit"
        ].items():  # only where dvsim's own part has no price
            if _lib.get(part).value("price") is None:
                _price(_lib, part, price)
        for slot in SLOTS.values():
            for o in slot["options"]:
                _price(_lib, o["part"], o["price"], o.get("price_src"))
                pt = _lib.get(o["part"])
                if o.get(
                    "joinable"
                ):  # one VOUT pad feeds the wires to the servos, flight computer and lidar: a solder joint
                    for c in pt.raw.get("connectors", []):
                        c["joinable"] = True
                for port in pt.ports:  # an 8S board on a 6S pack: the bus it passes on is the pack's, not its rating
                    if "bus_v" in o and port.get("kind") == "power" and "v_max" in port:
                        port["v_max"] = dict(port["v_max"], value=o["bus_v"])
        # the printed deck also takes the smaller regulator modules: same seat, other hole pattern (dvsim's swap looks for a co-located port)
        frame = _lib.get("hopper-deck-frame-edf80")
        for name in ("deck_reg1", "deck_reg2"):
            p = dict(next(x for x in frame.ports if x["name"] == name))
            frame.ports.append(
                {**p, "name": name + "_b", "type": "bolt-pattern.13.46x13.46.m2"}
            )
            frame.ports.append({**p, "name": name + "_c", "type": "board.18x13-header"})
    return _lib


def _price(lib, part, usd, src=None):
    """`src` = where a real price came from; without one the price is marked estimated (task.toml, dvsim part files carry almost no prices)."""
    lib.get(part).params["price"] = dict(
        value=float(usd),
        unit="USD",
        fidelity="literature" if src else "estimated",
        **({"derivation": src} if src else {"rationale": EST}),
    )


def option(slot, part):
    for o in SLOTS[slot]["options"]:
        if o["part"] == part:
            return o
    return None


def _value(part, key):
    e = part.params.get(key)
    return e.get("value") if isinstance(e, dict) else None


def catalog(slot=None, query=None):
    """Options of one slot (or all) with price and specs. Never says which is the reference choice."""
    if slot is not None and slot not in SLOTS:
        raise DesignError(f"unknown slot '{slot}'. Slots: {', '.join(SLOTS)}")
    out = {}
    for name in [slot] if slot else SLOTS:
        s = SLOTS[name]
        opts = []
        for o in s["options"]:
            pt = library().get(o["part"])
            spec = dict(mass_kg=_value(pt, "mass"))
            spec.update(
                {
                    k: _value(pt, k)
                    for k in s.get("spec", [])
                    if _value(pt, k) is not None
                }
            )
            spec.update(o.get("spec", {}))
            row = dict(
                part=o["part"],
                name=o.get("name") or pt.raw.get("name", o["part"]),
                price_usd=o["price"],
                spec=spec,
            )
            if query and query.lower() not in (row["part"] + " " + row["name"]).lower():
                continue
            opts.append(row)
        out[name] = dict(what=s["what"], options=opts)
    return out


class Built:
    """One applied design. Fields: design (dvsim Design), choices, xml (None when it cannot build), report, findings, errors, warnings, cost, mass, envelope_mm, props, build_error."""


def _set(d, iid, values):
    for k, (v, unit) in values.items():
        d.set(
            iid, k, value=dict(value=v, unit=unit, fidelity="estimated", rationale=EST)
        )


def apply(choices):
    """choices: {slot: part, battery_x_mm: float, battery_y_mm: float}. -> Built. Raises DesignError with a useful message on an invalid choice."""
    lib = library()
    bad = [
        k
        for k in choices
        if k not in SLOTS and k not in ("battery_x_mm", "battery_y_mm")
    ]
    if bad:
        raise DesignError(
            f"unknown key(s) {bad}. Keys: {', '.join(SLOTS)}, battery_x_mm, battery_y_mm"
        )
    for slot in SLOTS:
        part = choices.get(slot)
        if part is not None and option(slot, part) is None:
            names = ", ".join(o["part"] for o in SLOTS[slot]["options"])
            raise DesignError(
                f"'{part}' is not in the {slot} catalog. Options: {names}"
            )
    for k in ("battery_x_mm", "battery_y_mm"):
        if k in choices and not (
            isinstance(choices[k], (int, float)) and abs(choices[k]) < 200
        ):
            raise DesignError(f"{k} must be a number of mm within +-200")
    d = Design(loads(ASSEMBLY.read_text()), lib, None)
    for slot, s in SLOTS.items():
        part = choices.get(slot)
        fn = globals().get(f"_slot_{slot}")
        if fn:
            fn(d, part, option(slot, part) if part else None, choices)
        elif part is None:
            for iid in s["instances"]:
                d.remove(iid)
        else:
            for iid in s["instances"]:
                d.swap(iid, part)
    b = Built()
    b.design, b.choices = d, dict(choices)
    b.findings = [f.as_dict() for f in d.check()]
    b.errors = [f for f in b.findings if f["severity"] == "error"]
    b.warnings = [f for f in b.findings if f["severity"] == "warning"]
    missing = [s for s in SLOTS if not choices.get(s)]
    b.missing = missing
    b.xml, b.report, b.build_error = None, None, None
    try:
        b.xml, b.report = compute(
            d.asm, lib.dir, lib.root, lib, visual=False, route=False
        )
    except (DesignError, KeyError, ValueError, IndexError) as e:
        b.build_error = str(e)
    gnss_price = (
        option("gnss", choices["gnss"])["price"] if choices.get("gnss") else 0.0
    )
    if b.report:
        b.xml = re.sub(
            r'file="(?!/)', f'file="{lib.root}/', b.xml
        )  # the MJCF is flown from other directories
        bm = _bom.bom(b.report, lib)
        b.cost = bm["total"]["cost"] + gnss_price
        b.cost_by_kind = _cost_by_kind(bm, gnss_price)
        b.mass = b.report["mass"]["total"]
        b.bom = bm
        b.props = mass_props(b.xml)
        b.envelope_mm = b.props["envelope_mm"]
    else:  # a partial design that does not build yet: sum the instances
        insts = [i for i in d.doc["instances"] if not i.get("offboard")]
        b.cost = (
            sum((lib.get(i["part"]).value("price") or 0.0) for i in insts) + gnss_price
        )
        b.mass = sum((lib.get(i["part"]).value("mass") or 0.0) for i in insts)
        b.bom, b.props, b.envelope_mm, b.cost_by_kind = None, None, [0, 0, 0], {}
    return b


def _cost_by_kind(bm, gnss_price):
    """USD by what kind of line it is. The GNSS chip is off-board in the assembly, so dvsim's BOM has no line for it."""
    kinds = {}
    for ln in bm["lines"]:
        k = "estimated" if ln.get("fidelity") == "estimated" else "sourced"
        kinds[k] = round(kinds.get(k, 0.0) + (ln["unit_price"] or 0.0) * ln["count"], 2)
    if gnss_price:
        kinds["gnss_overlay"] = gnss_price
    if bm["missing"]:
        kinds["unpriced_lines"] = len(bm["missing"])
    return kinds


# ---- slots that need more than a swap -----------------------------------------------------------------------------------
def _slot_battery(d, part, o, choices):
    if part is None:
        d.remove("batt")
        return
    d.swap("batt", part)
    pt = library().get(part)
    inst = next(i for i in d.doc["instances"] if i["id"] == "batt")
    x = choices.get("battery_x_mm", 62.0) / 1000.0
    y = choices.get("battery_y_mm", 14.8) / 1000.0
    inst["pos"] = [
        x,
        y,
        _value(pt, "height") / 2.0,
    ]  # rests on the deck, whatever its height


def _slot_esc(d, part, o, choices):
    if part is None:
        d.remove("esc")
        return
    if (
        "place" in o
    ):  # a board with solder pads and no mounting holes: taped to the deck where the mate would hold it
        d.edit([dict(op="swap", id="esc", part=part, refuse_mount=False)])
        inst = next(i for i in d.doc["instances"] if i["id"] == "esc")
        inst["pos"], inst["rpy"] = o["place"], [0.0, 0.0, 1.5707963267948966]
    else:
        d.swap("esc", part)
    if "set" in o:
        _set(d, "esc", {k: tuple(v) for k, v in o["set"].items()})


def _slot_servo(d, part, o, choices):
    if part is None:
        for iid in SLOTS["servo"]["instances"]:
            d.remove(iid)
        return
    for iid in SLOTS["servo"]["instances"]:
        d.swap(iid, part)
        if "set" in o:
            _set(d, iid, {k: tuple(v) for k, v in o["set"].items()})


def _slot_gnss(d, part, o, choices):
    if part is None:
        d.remove("gnss_chip")
        return
    d.swap("gnss_chip", o["chip"])
    _set(d, "harness", dict(mass=(o["harness_mass"], "kg")))
    dev = next(x for x in d.doc["devices"] if x["name"] == "gnss")
    for k in ("horiz_std", "vert_std"):
        dev["set"][k]["value"] = o[k]
    dev["set"]["corr_time_s"]["value"] = o["corr_time_s"]
    dev["set"]["corr_frac"]["value"] = o["corr_frac"]


def _slot_range(d, part, o, choices):
    if part is None:
        d.remove("lidar")
        return
    d.swap("lidar", part)
    if "to_pins" in o:
        next(w for w in d.doc["wires"] if w["name"] == "reg1_lidar")["to_pins"] = o[
            "to_pins"
        ]
    dev = next(x for x in d.doc["devices"] if x["name"] == "lidar")
    dev.pop("map", None)
    new = dict(dev["set"])
    for k, unit in (
        ("min_range", "m"),
        ("max_range", "m"),
        ("resolution", "m"),
        ("rate_hz", "Hz"),
        ("sigma0", "m"),
        ("sigma1", "1"),
    ):
        new[k] = dict(value=o[k], unit=unit, fidelity="estimated", rationale=EST)
    for k in ("fov_h", "fov_v"):
        new[k] = dict(value=o["fov"], unit="rad", fidelity="estimated", rationale=EST)
    dev["set"] = new


# ---- mass properties, measured on the built MJCF ---------------------------------------------------------------------------
def _quat_mat(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _floats(s, default=None):
    return [float(v) for v in s.split()] if s else default


def _geom_frame(g):
    """Rotation of a geom relative to its body: quat, or the z axis from fromto / zaxis."""
    if g.get("quat"):
        return _quat_mat(_floats(g.get("quat")))
    z = None
    if g.get("zaxis"):
        z = np.array(_floats(g.get("zaxis")))
    if g.get("fromto"):
        f = _floats(g.get("fromto"))
        z = np.array(f[3:]) - np.array(f[:3])
    if z is None or np.linalg.norm(z) < 1e-12:
        return np.eye(3)
    z = z / np.linalg.norm(z)
    a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(a, z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def mass_props(xml):
    """Mass, centre of mass, inertia about it, sites, envelope of the vehicle in the deck frame (a small port of the walk in dvsim's tests/python/test_hopper.py, with shape inertias)."""
    root = ET.fromstring(xml)
    frame = next(
        b for b in root.find("worldbody").findall("body") if b.get("name") == "frame"
    )
    z0 = float(frame.get("pos", "0 0 0").split()[2])
    items, sites, pts = [], {}, []

    def walk(el, p0, R0):
        for ch in el:
            if ch.tag == "body":
                p = p0 + R0 @ np.array(_floats(ch.get("pos"), [0, 0, 0]))
                R = R0 @ _quat_mat(_floats(ch.get("quat"), [1, 0, 0, 0]))
                walk(ch, p, R)
            elif ch.tag == "site":
                sites[ch.get("name")] = p0 + R0 @ np.array(
                    _floats(ch.get("pos"), [0, 0, 0])
                )
            elif ch.tag == "geom":
                kind = ch.get("type", "sphere")
                size = _floats(ch.get("size"), [0.0])
                ft = _floats(ch.get("fromto"))
                c = (
                    np.array(ft[:3]) / 2 + np.array(ft[3:]) / 2
                    if ft
                    else np.array(_floats(ch.get("pos"), [0, 0, 0]))
                )
                Rg = R0 @ _geom_frame(ch)
                pos = p0 + R0 @ c
                half = (
                    np.linalg.norm(np.array(ft[3:]) - np.array(ft[:3])) / 2
                    if ft
                    else (size[1] if len(size) > 1 else 0.0)
                )
                m = float(ch.get("mass", 0.0))
                if kind == "box":
                    e = np.array(size[:3])
                    Ig = (
                        m
                        / 3
                        * np.array(
                            [
                                e[1] ** 2 + e[2] ** 2,
                                e[0] ** 2 + e[2] ** 2,
                                e[0] ** 2 + e[1] ** 2,
                            ]
                        )
                    )
                    if ch.get("contype") != "0" or m > 0:
                        pts.extend(
                            pos + Rg @ (np.array([sx, sy, sz]) * e)
                            for sx in (-1, 1)
                            for sy in (-1, 1)
                            for sz in (-1, 1)
                        )
                elif kind in ("cylinder", "capsule"):
                    r = size[0]
                    Ig = np.array(
                        [
                            m * (3 * r * r + 4 * half * half) / 12,
                            m * (3 * r * r + 4 * half * half) / 12,
                            m * r * r / 2,
                        ]
                    )
                    if m > 0:
                        for sgn in (-1, 1):
                            q = pos + Rg @ np.array([0, 0, sgn * half])
                            pts.extend([q - r, q + r])
                elif kind == "sphere":
                    r = size[0]
                    Ig = np.full(3, 0.4 * m * r * r)
                    pts.extend([pos - r, pos + r])
                else:
                    Ig = np.zeros(3)
                if m > 0:
                    items.append((m, pos, Rg @ np.diag(Ig) @ Rg.T))

    walk(frame, np.zeros(3), np.eye(3))
    M = sum(i[0] for i in items)
    cg = sum(i[0] * i[1] for i in items) / M
    In = np.zeros((3, 3))
    for m, p, Ig in items:
        r = p - cg
        In += Ig + m * ((r @ r) * np.eye(3) - np.outer(r, r))
    pts = np.array(pts)
    env = (pts.max(axis=0) - pts.min(axis=0)) * 1000.0
    return dict(
        mass=float(M),
        cg=[float(v) for v in cg],
        inertia=[float(In[0, 0]), float(In[1, 1]), float(In[2, 2])],
        h_stand=float(z0 + cg[2]),
        frame_z0=z0,
        imu_rel=[float(v) for v in sites["site:fc"] - cg]
        if "site:fc" in sites
        else [0.0, 0.0, 0.0],
        lidar_rel=[float(v) for v in sites["site:lidar"] - cg]
        if "site:lidar" in sites
        else [0.0, 0.0, 0.0],
        vane_site_rel=[
            [float(v) for v in sites[f"site:vane{k}:aero"] - cg] for k in range(1, 5)
        ]
        if "site:vane1:aero" in sites
        else [],
        envelope_mm=[round(float(v), 1) for v in env],
    )


def static_violations(b):
    """Every static constraint the design breaks, as {name,value,limit,ok,message}; ok ones included so the caller can see restores."""
    L = LIMITS
    out = []
    add = lambda name, v, lim, ok, msg: out.append(
        dict(name=name, value=v, limit=lim, ok=bool(ok), message=msg)
    )
    add(
        "cost",
        round(b.cost, 2),
        L["cost_max_usd"],
        b.cost <= L["cost_max_usd"],
        f"cost ${b.cost:.2f} against the ${L['cost_max_usd']:.0f} limit",
    )
    add(
        "mass",
        round(b.mass, 4),
        L["mass_max_kg"],
        b.mass <= L["mass_max_kg"],
        f"mass {b.mass:.3f} kg against the {L['mass_max_kg']} kg limit",
    )
    env_ok = all(
        e <= m for e, m in zip(sorted(b.envelope_mm), sorted(L["envelope_mm"]))
    )
    add(
        "envelope",
        b.envelope_mm,
        L["envelope_mm"],
        env_ok,
        f"envelope {b.envelope_mm} mm against {L['envelope_mm']} mm",
    )
    add(
        "drc",
        len(b.errors),
        0,
        not b.errors,
        f"{len(b.errors)} design rule errors"
        + (
            f", first: {b.errors[0]['code']}: {b.errors[0]['message']}"
            if b.errors
            else ""
        ),
    )
    add(
        "slots",
        b.missing,
        [],
        not b.missing,
        "unchosen slots: " + ", ".join(b.missing) if b.missing else "all slots chosen",
    )
    return out


def controller_cfg(b, static=None):
    """What the controller gets as cfg: the design's numbers (never gains). Lists of numbers arrive as numpy arrays."""
    import sys

    sys.path.insert(0, str(DVSIM / "examples" / "hopper"))
    import vehicle as veh

    lib, c = library(), b.choices
    p = b.props
    esc = lib.get(c["esc"])
    o = option("esc", c["esc"])
    r_esc = (
        dict((k, v[0]) for k, v in o.get("set", {}).items()).get("r_on_total")
        or _value(esc, "r_on_total")
        or 0.002
    )
    i_lim = dict((k, v[0]) for k, v in o.get("set", {}).items()).get(
        "current_continuous"
    ) or _value(esc, "current_continuous")
    batt = lib.get(c["battery"])
    gain = 0.5 * veh.JET_FACTOR**2 / veh.A_EXIT * veh.VANE_AREA * veh.CN_ALPHA

    def volts(T):
        n = math.sqrt(T / (veh.CT0 * veh.RHO0 * veh.D_FAN**4))
        return veh.KE * 2 * math.pi * n + (
            veh.CQ0 * veh.RHO0 * n * n * veh.D_FAN**5 / veh.KE + veh.I0
        ) * (veh.R_MOTOR + r_esc)

    return dict(
        g=veh.G,
        mass=p["mass"],
        cg=p["cg"],
        inertia=p["inertia"],
        h_stand=p["h_stand"],
        thrust_max_n=(static or {}).get("thrust_n"),
        dt=0.004,
        t_cal=TASK["mission"]["t_cal_s"],
        mission={
            k: TASK["mission"][k]
            for k in (
                "alt_m",
                "alt_tol_m",
                "hold_s",
                "land_radius_m",
                "touchdown_m_s",
                "tilt_max_deg",
                "mission_s",
            )
        },
        fan=dict(
            thrust_motor_volts=[
                [float(T), round(volts(T) * (1 + veh.SW_LOSS), 3)]
                for T in range(2, 42, 2)
            ],
            spin_inertia=veh.J_FAN,
            ct=veh.CT0,
            cq=veh.CQ0,
            diameter=veh.D_FAN,
            rho=veh.RHO0,
            yaw_torque_per_newton=veh.Q_UNRECOVERED * veh.CQ0 / veh.CT0 * veh.D_FAN,
            current_limit_a=i_lim,
        ),
        vanes=dict(
            site_rel=p["vane_site_rel"],
            force_dir=veh.VANE_DIR.tolist(),
            force_per_rad_per_newton=gain,
            area=veh.VANE_AREA,
            exit_area=veh.A_EXIT,
            limit_rad=veh.VANE_LIMIT,
        ),
        battery=dict(
            cells=_value(batt, "cells_series"),
            capacity_ah=_value(batt, "capacity_ah"),
            r_internal_ohm=_value(batt, "r_internal_pack"),
        ),
        range_sensor=dict(
            min_m=option("range", c["range"])["min_range"],
            max_m=option("range", c["range"])["max_range"],
        ),
    )


def flight_inputs(b):
    """The design numbers a flight needs besides the MJCF."""
    c = b.choices
    g = option("gnss", c["gnss"])
    r = option("range", c["range"])
    return dict(
        props=b.props,
        batt_mass=_value(library().get(c["battery"]), "mass"),
        gnss=dict(horiz_std=g["horiz_std"], vert_std=g["vert_std"]),
        range=dict(min_range=r["min_range"], max_range=r["max_range"]),
        servo=dict(
            p=2000, d=6000
        ),  # XL330 runtime gains as examples/hopper/vehicle.py: the default P 400 sags under load
    )
