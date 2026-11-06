"""Wires (mass + electrical) and device bindings -> the dvsim.devices JSON (schema/devices.fbs)."""

import numpy as np

from . import pins as _pins
from .assembly import UNITS, Ctx
from .library import DesignError
from .reynolds import ref_scale, reynolds_shape
from .snap import incompatible, port_frame

OHM, AMP = {"ohm": 1.0}, {"A": 1.0}
# Expected unit of every device field the builder fills from params (last path segment). unit -> factor to the schema unit.
FIELD_UNITS = {
    "kv": {"rpm/V": 1.0},
    "r20": OHM,
    "r0": OHM,
    "r1": OHM,
    "resistance": OHM,
    "r_series": OHM,
    "r_on": OHM,
    "sw_loss": {"1": 1.0},
    "i_q": AMP,
    "i0": AMP,
    "i_limit": AMP,
    "rotor_inertia": {"kg*m^2": 1.0},
    "c_th": {"J/K": 1.0},
    "r_th": {"K/W": 1.0},
    "c1": {"F": 1.0},
    "capacity_ah": {"Ah": 1.0, "mAh": 1e-3, "C": 1 / 3600},
    "voltage": {"V": 1.0},
    "diameter": UNITS["m"],
    "length_m": UNITS["m"],
    "gear_ratio": {"1": 1.0},
    "efficiency": {"1": 1.0},
    "encoder_cpr": {"1": 1.0, "counts/rev": 1.0},
    "g": {"S": 1.0},
    "area": {"m^2": 1.0},
    "cd": {"1": 1.0},
    "noise_density": {  # imu gyro | accel (both spellings of m/s^2 are in use in the parts / tests)
        "rad/s/sqrt(Hz)": 1.0,
        "m/s2/sqrt(Hz)": 1.0,
        "m/s^2/sqrt(Hz)": 1.0,
    },
    "full_scale": {"rad/s": 1.0, "m/s2": 1.0, "m/s^2": 1.0},  # imu gyro | accel
    "bits": {"bit": 1.0},
    "noise_std": {"Pa": 1.0, "T": 1.0, "N": 1.0},  # baro | mag | foot_contact
    "vel_std": {"m/s": 1.0},
    "rate_hz": {"Hz": 1.0},
    "latency_s": {"s": 1.0},
    "corr_time_s": {"s": 1.0},  # gnss
    "corr_frac": {"1": 1.0},  # gnss
    "alt0": UNITS["m"],
    "origin_alt": UNITS["m"],
    "w20": {"m/s": 1.0},
    "w_s": {"rad/s": 1.0},
    "firmware_hz": {"Hz": 1.0},
    "p": {"count": 1.0},
    "i": {"count": 1.0},
    "d": {"count": 1.0},
    "pid_scale": {"1": 1.0},
    "d_scale": {"1": 1.0},
    "deadband": {"count": 1.0},
    "max_torque": {"0.1 %": 1.0},
    # servo protection registers (initial values; the part's register-unit params -> schema units)
    "min_voltage": {"V": 1.0},
    "max_voltage": {"V": 1.0},
    "max_temp": {"degC": 1.0},
    "overload_load": {"%": 0.01},
    "overload_time": {"10 ms": 0.01, "s": 1.0},
    "unload_mask": {"bitmask": 1.0},
    "i0_visc": {"1": 1.0},
    "i0_quad": {"1": 1.0},
    "i0_w": {"rad/s": 1.0},
    # launcher / camera (counter-UAS)
    "muzzle_velocity": {"m/s": 1.0},
    "pellet_mass": {"kg": 1.0},
    "v_std": {"m/s": 1.0},
    "spread_std": {"rad": 1.0},
    "cl_k": {"1": 1.0},  # cuas2 Magnus slope
    "spin_tau": {"s": 1.0},  # cuas2 spin decay
    "max_time": {"s": 1.0},
    "fov_v": {"rad": 1.0},
    "width": {"px": 1.0},
    "height": {"px": 1.0},
    # solid motor / airframe / sea / vessel / rangefinder (phase 5)
    "propellant_mass": {"kg": 1.0},
    "total_mass": {"kg": 1.0},
    "length": {"m": 1.0},
    "ignition_delay": {"s": 1.0},
    "time": {"s": 1.0},
    "thrust": {"N": 1.0},
    "s_ref": {"m^2": 1.0},
    "s_plan": {"m^2": 1.0},
    "cd0": {"1": 1.0},
    "cd_rev": {"1": 1.0},
    "cd_cross": {"1": 1.0},
    "cn_alpha": {"1/rad": 1.0},
    "x_cp": {"m": 1.0},
    "rho": {"kg/m^3": 1.0},
    "fov_h": {"rad": 1.0},
    "min_range": {"m": 1.0},
    "max_range": {"m": 1.0},
    "sigma0": {"m": 1.0},
    "sigma1": {"1": 1.0},
    "sigma2": {"1/m": 1.0},
    "resolution": {"m": 1.0},
    "dropout": {"1": 1.0},
    "level": {"m": 1.0},
    "depth": {"m": 1.0},
    "exit_area": {"m^2": 1.0},  # jet vane
    "cn_max": {"1": 1.0},
    "jet_factor": {"1": 1.0},
    "swirl_recovery": {"1": 1.0},  # rotor with a stator (EDF)
    "k_h": {"kg": 1.0},  # rotor H-force coefficient
    "rev_thrust": {"1": 1.0},
    "n_h": {"px": 1.0},
    "n_v": {"px": 1.0},
    "ray_groups": {"bitmask": 1.0},
    # rain (schema Rain; rotor and rangefinder rain_* fields): docs/models.md "Rain"
    "rate": {"mm/h": 1.0},
    "catch_area": {"m^2": 1.0},
    "wet_area": {"m^2": 1.0},
    "film_depth": {"m": 1.0, "mm": 1e-3},
    "film_tau": {"s": 1.0},
    "collect_eff": {"1": 1.0},
    "rain_kt": {"1": 1.0},
    "rain_kq": {"1": 1.0},
    "rain_a": {"1/m": 1.0, "1/km": 1e-3},
    "rain_b": {"1": 1.0},
    "rain_margin_db": {"dB": 1.0},
    "rain_window_db": {"dB": 1.0},
    "rain_beam": {"m": 1.0},
    "rain_zone": {"m": 1.0},
    "rain_det": {"1": 1.0},
    # qdd / foot_contact (SP2; docs/models.md "Qdd", "FootContact")
    "cmd_hz": {"Hz": 1.0},
    "kp_max": {"N*m/rad": 1.0},
    "kd_max": {"N*m*s/rad": 1.0},
    "tau_ff_max": {"N*m": 1.0},
    "timeout_s": {"s": 1.0},
    "kd_timeout": {"N*m*s/rad": 1.0},
    "t_max": {"degC": 1.0},
    "derate_start": {"degC": 1.0},
    "f_max": {"N": 1.0},
    "hysteresis_on": {"N": 1.0},
    "hysteresis_off": {"N": 1.0},
    "debounce_s": {"s": 1.0},
}
INT_FIELDS = {
    "timeout_mode",
    "mode",
    "awg",
    "encoder_cpr",
    "bits",
    "unload_mask",
    "width",
    "height",
    "n_h",
    "n_v",
    "ray_groups",
    "rays_per_frame",  # rosette rangefinder (uint): a float token fails the kernel's JSON load
    "pattern_seed",
}
TYPE_KEY = {
    "battery": "batteries",
    "psu": "psus",
    "wire": "wires",
    "load": "loads",
    "esc": "escs",
    "rotor": "rotors",
    "servo": "servos",
    "imu": "imus",
    "baro": "baros",
    "mag": "mags",
    "gnss": "gnss_receivers",
    "environment": "environments",
    "body_drag": "body_drags",
    "launcher": "launchers",
    "camera": "cameras",
    "solid_motor": "solid_motors",
    "airframe": "airframes",
    "sea": "seas",
    "vessel": "vessels",
    "rangefinder": "rangefinders",
    "jet_vane": "jet_vanes",
    "rain": "rains",
    "qdd": "qdds",
    "foot_contact": "foot_contacts",
}
NODE_TYPES = {"battery", "psu", "load", "esc", "servo", "qdd"}
TEXT_FIELDS = {
    "projection"
}  # binding keys whose string value is emitted verbatim (rangefinder: "pinhole" | "spherical")
RESERVED = {
    "type",
    "name",
    "instance",
    "map",
    "set",
    "port",
    "a",
    "b",
    "joint",
    "backlash_joint",
    "body",
    "ref_body",
    "stator_body",  # rotor: the housing instance that takes the swirl torque back
    "rotor",  # jet_vane: the name of the rotor device whose exhaust it sits in
    "shape",  # foot_contact: index of the instance part's geometry shape that is the foot (default 0)
    "site",
    "site_port",
    "ocv",
    "cells",
    "reynolds",
    "cq_ref",
    "auto",  # written by Design.auto_devices: this binding is regenerated from the parts
}
MAX_NODES = 128  # kMaxNodes of the kernel (devices.h)
VEHICLE_GROUP = 4  # MJCF geom group of every vehicle geom; the floor stays in group 0
RAY_GROUPS = 63 & ~(
    1 << VEHICLE_GROUP
)  # rotor ground-effect rays skip the vehicle's own geoms


def _cat_wire(part) -> bool:
    return part.category == "wire" or part.category.startswith("wire.")


def _port_of(ctx: Ctx, ref: str, where: str):
    iid, _, pn = ref.partition(".")
    inst = ctx.inst(iid, where)
    if not pn:
        raise DesignError(f"{where}: '{ref}' needs '<inst>.<port>'")
    return inst, inst.part.port(pn)


# ---- wires --------------------------------------------------------------------------------------------------------
def resolve_wires(ctx: Ctx) -> list:
    out = []
    for i, w in enumerate(ctx.doc.get("wires", [])):
        where = f"wire {i} ({w['from']} -> {w['to']})"
        ia, pa = _port_of(ctx, w["from"], where)
        ib, pb = _port_of(ctx, w["to"], where)
        part = ctx.lib.get(w["part"])
        if pa.get("kind") not in ("power", "signal") or pa.get("kind") != pb.get(
            "kind"
        ):
            raise DesignError(
                f"{where}: both ends must be power ports or both signal ports"
            )
        cable = _cat_wire(part)
        if not cable and (why := incompatible(pa, pb)):
            raise DesignError(
                f"{where}: {why} (use a wire-category part to bridge different connector families)"
            )
        length, lentry = ctx.num(w["length"], f"{where} length", UNITS["m"], "length")
        cond = w.get("conductors", 1)
        if isinstance(cond, bool) or not isinstance(cond, int) or cond < 1:
            raise DesignError(f"{where}: conductors must be an integer >= 1")
        incl = w.get("included_in")
        if incl:
            ctx.inst(incl, where)
        rec = {
            "index": i,
            "name": w.get("name", f"wire{i}"),
            "from": w["from"],
            "to": w["to"],
            "part": part.id,
            "kind": pa["kind"],
            "length": length,
            "conductors": cond,
            "included_in": incl,
            "mass": None,
            "mass_fidelity": None,
            "mass_tol": 0.0,
            "mass_note": "",
            "connectors": [],
            "_ia": ia,
            "_ib": ib,
            "_pa": pa,
            "_pb": pb,
            "_part": part,
            "_len": lentry,
            "group": ia.group,
            "_pairs": _pins.wire_pairs(
                ctx, w
            ),  # [(ref_a, ref_b)] pin of each conductor; [] = no pin data (port-level wire)
        }
        # mass
        if cable:
            rates = {
                k: e
                for k, e in sorted(part.params.items())
                if isinstance(e, dict) and e.get("unit") == "kg/m"
            }
            if "mass_per_m" in rates:
                rates = {"mass_per_m": rates["mass_per_m"]}
            if rates:
                for k, e in rates.items():
                    ctx.note(where, k, e)
                # a wire part with a `conductors` param describes a whole multi-conductor cable: its rate is per cable
                per_cable = part.value("conductors") is not None
                rec["mass"] = (
                    sum(e["value"] for e in rates.values())
                    * length
                    * (1 if per_cable else cond)
                )
                fids = [e["fidelity"] for e in rates.values()] + [lentry["fidelity"]]
                rec["mass_fidelity"] = (
                    "estimated"
                    if "estimated" in fids
                    else next(iter(rates.values()))["fidelity"]
                )
                rec["mass_note"] = (
                    f"{'+'.join(rates)} x length (per cable)"
                    if per_cable
                    else f"{'+'.join(rates)} x length x conductors"
                )
            else:
                ctx.unresolved.append(
                    {
                        "where": where,
                        "what": f"part {part.id} has no mass per metre (unit kg/m): wire mass not counted",
                    }
                )
        elif part.value("mass") is not None:
            rec["mass"] = part.value("mass")
            rec["mass_fidelity"] = part.params["mass"]["fidelity"]
            rec["mass_note"] = "connector part mass"
            ctx.note(where, "mass", part.params["mass"])
        else:
            ctx.unresolved.append(
                {"where": where, "what": f"part {part.id} has no mass"}
            )
        if rec["mass"] is not None and rec["mass_fidelity"] == "estimated":
            rec["mass_tol"] = 0.5 * rec["mass"]
        # implied connectors (cable ends terminating in a connector family)
        if cable:
            for end, inst, port in (("from", ia, pa), ("to", ib, pb)):
                g = port.get("gender")
                if g not in ("male", "female"):
                    continue
                opp = "female" if g == "male" else "male"
                cands = sorted(
                    (not p.category.startswith("connector"), p.id)
                    for p, pt in ctx.lib.compatible(port["type"], g)
                    if pt.get("gender") == opp and p.category == "connector"
                )
                c = {
                    "wire": i,
                    "end": end,
                    "instance": inst.id,
                    "port": port["name"],
                    "port_type": port["type"],
                    "gender": opp,
                    "part": None,
                    "mass": None,
                    "mass_fidelity": None,
                    "mass_tol": 0.0,
                }
                mpc = part.params.get("mass_per_connector")
                if mpc is not None:  # the wire part states its own end connectors
                    c["part"] = part.id
                    c["mass"], c["mass_fidelity"] = mpc["value"], mpc["fidelity"]
                    ctx.note(where + f" {end} connector", "mass_per_connector", mpc)
                    if c["mass_fidelity"] == "estimated":
                        c["mass_tol"] = 0.5 * c["mass"]
                elif cands:
                    pin = w.get("connectors", {}).get(end)
                    if pin is not None and pin not in {cid for _, cid in cands}:
                        raise DesignError(
                            f"{where}: connectors.{end} = {pin!r} has no {opp} {port['type']} port "
                            f"(candidates: {', '.join(cid for _, cid in cands)})"
                        )
                    if pin is None and len(cands) > 1:
                        # an unpinned choice would change silently as the library grows
                        ctx.warn(
                            "CONNECTOR_AMBIGUOUS",
                            f"{where} {end} end: {len(cands)} library connectors fit, using {cands[0][1]}; "
                            f'pin one with connectors = {{ {end} = "..." }}',
                        )
                    cp = ctx.lib.get(pin or cands[0][1])
                    c["part"] = cp.id
                    if cp.value("mass") is not None:
                        c["mass"], c["mass_fidelity"] = (
                            cp.value("mass"),
                            cp.params["mass"]["fidelity"],
                        )
                        ctx.note(where + f" {end} connector", "mass", cp.params["mass"])
                        if c["mass_fidelity"] == "estimated":
                            c["mass_tol"] = 0.5 * c["mass"]
                    else:
                        ctx.unresolved.append(
                            {"where": where, "what": f"connector {cp.id} has no mass"}
                        )
                else:
                    ctx.unresolved.append(
                        {
                            "where": where,
                            "what": f"no library connector with a {opp} {port['type']} port ({end} end)",
                        }
                    )
                rec["connectors"].append(c)
        out.append(rec)
    return out


def wire_points(rec: dict, T: dict):
    """World positions of the two wire ends (port pos, else instance origin). An offboard end has no pose: it
    collapses onto the onboard end. Returns None when both ends are offboard."""
    pts = []
    for inst, port in ((rec["_ia"], rec["_pa"]), (rec["_ib"], rec["_pb"])):
        if inst.id in T:
            Ti = T[inst.id]
            pts.append(
                Ti[:3, :3] @ np.array(port.get("pos", [0, 0, 0]), dtype=float)
                + Ti[:3, 3]
            )
        else:
            pts.append(None)
    if pts[0] is None and pts[1] is None:
        return None
    return (
        pts[0] if pts[0] is not None else pts[1],
        pts[1] if pts[1] is not None else pts[0],
    )


# ---- devices ------------------------------------------------------------------------------------------------------
def _setpath(d: dict, path: str, v):
    *head, last = path.split(".")
    for p in head:
        d = d.setdefault(p, {})
    d[last] = v


def _fix(key: str, v):
    if key.rsplit(".", 1)[-1] in INT_FIELDS:
        if isinstance(v, float) and v != int(v):
            raise DesignError(f"field '{key}' must be an integer, got {v}")
        return int(v)
    return v


def _spec_source(ctx: Ctx, binding: dict, spec: str, where: str):
    """'<param>' | '<field>:<param>' -> (Inst, param spec)."""
    field, sep, pname = spec.partition(":")
    if sep:
        iid = binding.get(field)
        if not isinstance(iid, str):
            raise DesignError(
                f"{where}: map spec '{spec}': the binding has no instance-valued field '{field}'"
            )
        return ctx.inst(iid, where), pname
    return ctx.inst(binding["instance"], where), spec


def _mapped(ctx: Ctx, binding: dict, key: str, spec: str, where: str, parts: set):
    inst, p = _spec_source(ctx, binding, spec, where)
    w = f"{where} {key} <- {inst.id}.{p}"
    if p.startswith("@"):  # whole table column as a list
        tname, _, col = p[1:].partition(".")
        t = inst.part.raw.get("tables", {}).get(tname)
        if not t or col not in t.get("columns", []):
            raise DesignError(
                f"{w}: part {inst.part.id} has no table column {tname}.{col}"
            )
        ctx.note(
            w,
            p,
            {
                "fidelity": t.get("fidelity"),
                "value": f"table {tname}",
                "unit": "",
                "rationale": t.get("rationale", ""),
            },
        )
        parts.add(inst.part.id)
        i = t["columns"].index(col)
        return [float(r[i]) for r in t["rows"]]
    if not isinstance(inst.entry(p), dict):
        raise DesignError(
            f"{w}: instance '{inst.id}' (part {inst.part.id}) has no param '{p}'"
        )
    last = key.rsplit(".", 1)[-1]
    units = FIELD_UNITS.get(last)
    if units is None:
        ctx.warn("UNIT_UNCHECKED", f"{w}: no expected unit known for field '{last}'")
    v, _ = ctx.num(f"{inst.id}.{p}", w, units, p)
    parts.add(inst.part.id)
    return _fix(key, v)


def _union_find():
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    return find, union


def build_devices(ctx: Ctx, wires: list, joint_of: dict):
    """-> (devices dict or None, report [{name,type,parts}], sites {inst: {site name: (pos, R)}})."""
    doc = ctx.doc
    bindings = doc.get("devices", [])
    power_wires = [w for w in wires if w["kind"] == "power"]
    # -- nodes: union-find over (instance, power port); passthrough ports of one instance are one node
    internal = {}
    for b in bindings:
        if b["type"] == "wire":
            for k in ("a", "b"):
                internal.setdefault(b.get("instance"), set()).add(b.get(k))
    find, union = _union_find()
    for iid, inst in ctx.insts.items():
        pt = [
            p["name"]
            for p in inst.part.ports
            if p.get("kind") == "power"
            and p.get("role") == "passthrough"
            and p["name"] not in internal.get(iid, ())
        ]
        for n in pt[
            1:
        ]:  # ponytail: ports named by an internal wire binding stay separate nodes
            union((iid, pt[0]), (iid, n))
    order = {iid: i for i, iid in enumerate(ctx.insts)}
    used = {}  # (iid, port) -> sort key

    def use(iid, pn, where):
        inst = ctx.inst(iid, where)
        names = [p["name"] for p in inst.part.ports]
        if pn not in names:
            raise DesignError(
                f"{where}: part {inst.part.id} has no port '{pn}' (ports: {names})"
            )
        if inst.part.port(pn).get("kind") != "power":
            raise DesignError(f"{where}: port {iid}.{pn} is not a power port")
        used[(iid, pn)] = (order[iid], names.index(pn))

    for w in power_wires:
        use(w["_ia"].id, w["_pa"]["name"], f"wire {w['index']}")
        use(w["_ib"].id, w["_pb"]["name"], f"wire {w['index']}")
    for b in bindings:
        w = f"device '{b['name']}'"
        if b["type"] in NODE_TYPES:
            use(b.get("instance", ""), b.get("port", ""), w)
        if b["type"] == "wire":
            for k in ("a", "b"):
                if b.get(k) != "ground":
                    use(b.get("instance", ""), b.get(k, ""), w)
    node_no = {}
    for key in sorted(used, key=used.get):
        r = find(key)
        if r not in node_no:
            node_no[r] = len(node_no) + 1
    if len(node_no) > MAX_NODES:
        raise DesignError(
            f"{len(node_no)} electrical nodes; the network solver supports {MAX_NODES}"
        )

    def node(iid, pn):
        return 0 if pn == "ground" else node_no[find((iid, pn))]

    dev: dict[str, list] = {}
    report, sites, names = [], {}, set()

    def add(typ, d, parts):
        if d["name"] in names:
            raise DesignError(f"duplicate device name '{d['name']}'")
        names.add(d["name"])
        d["parts"] = sorted(parts)
        dev.setdefault(TYPE_KEY[typ], []).append(d)
        report.append({"name": d["name"], "type": typ, "parts": d["parts"]})

    for w in power_wires:
        part, where = w["_part"], f"wire {w['index']}"
        d = {
            "name": w["name"],
            "a": node(w["_ia"].id, w["_pa"]["name"]),
            "b": node(w["_ib"].id, w["_pb"]["name"]),
        }
        if part.value("awg") is not None:
            d.update(
                awg=int(part.value("awg")),
                length_m=w["length"],
                conductors=w["conductors"],
            )
        elif _cat_wire(part) and part.value("resistance_per_m") is not None:
            ctx.note(where, "resistance_per_m", part.params["resistance_per_m"])
            d["resistance"] = (
                part.value("resistance_per_m") * w["length"] * w["conductors"]
            )
        else:
            r = (
                part.value("resistance")
                if part.value("resistance") is not None
                else part.value("contact_resistance")
            )
            if r is None:
                raise DesignError(
                    f"{where}: part {part.id} has no awg, resistance_per_m, resistance or contact_resistance"
                )
            d["resistance"] = r
        add("wire", d, {part.id})

    for b in bindings:
        typ, where = b["type"], f"device '{b['name']}'"
        d, parts = {"name": b["name"]}, set()
        inst = ctx.inst(b["instance"], where) if "instance" in b else None
        if inst:
            parts.add(inst.part.id)
        if typ in NODE_TYPES:
            d["node"] = node(inst.id, b["port"]) if inst and "port" in b else None
            if d["node"] is None:
                raise DesignError(f"{where}: needs instance and port")
        if typ == "wire":
            if not inst or "a" not in b or "b" not in b:
                raise DesignError(
                    f"{where}: type wire needs instance, a and b (port names or 'ground')"
                )
            d["a"], d["b"] = node(inst.id, b["a"]), node(inst.id, b["b"])
        for k in ("joint", "backlash_joint"):
            if k in b:
                if b[k] not in joint_of:
                    raise DesignError(
                        f"{where}: {k} '{b[k]}' is not a mate with a joint"
                    )
                d[k] = joint_of[b[k]]
        if typ == "jet_vane":
            d["rotor"] = b["rotor"]
        if typ == "foot_contact":
            d["geom"] = _foot_geom(inst, b.get("shape", 0), where)
        for k in ("body", "ref_body", "stator_body"):
            if k in b:
                bi = ctx.inst(b[k], where)
                if bi.offboard:
                    raise DesignError(f"{where}: {k} '{bi.id}' is offboard (no body)")
                d[k] = bi.id
        if "site" in b:
            si = ctx.inst(b["site"], where)
            if si.offboard:
                raise DesignError(f"{where}: site instance '{si.id}' is offboard")
            sp = b.get("site_port")
            sname = f"site:{si.id}" + (f":{sp}" if sp else "")
            if sp:
                P = port_frame(si.part.port(sp), f"{where} ")
                sites.setdefault(si.id, {})[sname] = (P[:3, 3], P[:3, :3])
            else:
                sites.setdefault(si.id, {})[sname] = (np.zeros(3), np.eye(3))
            d["site"] = sname
        for (
            k,
            v,
        ) in (
            b.items()
        ):  # role fields (instances) are not emitted; plain flags / dir are
            if k in RESERVED:
                continue
            if (
                k in TEXT_FIELDS
            ):  # a string the device takes as is (not an instance reference)
                d[k] = v
            elif isinstance(v, str):
                ctx.inst(v, f"{where} field '{k}'")
            else:
                d[k] = v
        # values from parts
        if "map" in b:
            for key, spec in doc["device_maps"][b["map"]].items():
                _setpath(d, key, _mapped(ctx, b, key, spec, where, parts))
        for key, x in b.get("set", {}).items():
            units = FIELD_UNITS.get(key.rsplit(".", 1)[-1])
            w = f"{where} set.{key}"
            if isinstance(x, dict) and isinstance(
                x.get("value"), list
            ):  # provenance table holding a list
                if units is not None and x.get("unit") not in units:
                    raise DesignError(
                        f"{w}: unit {x.get('unit')!r}, expected one of {sorted(units)}"
                    )
                ctx.note(w, key, x)
                f = units[x["unit"]] if units else 1.0
                _setpath(d, key, [_fix(key, float(v) * f) for v in x["value"]])
                continue
            xs = x if isinstance(x, list) else [x]
            vals = [ctx.num(e, w, units, key)[0] for e in xs]
            _setpath(
                d,
                key,
                [_fix(key, v) for v in vals]
                if isinstance(x, list)
                else _fix(key, vals[0]),
            )
        if typ == "battery":
            _battery_ocv(ctx, b, d, where, parts)
        if typ == "launcher":
            d.setdefault(
                "ray_groups", 63 & ~(1 << 5)
            )  # hits the vehicle geoms (group 4); group 5 = the pellet visuals
        if typ == "vessel" and "hulls" not in d:
            d["hulls"] = _hulls(inst, where)
            if fins := _fins(ctx, inst, where, parts):
                d["fins"] = fins
        if typ == "rangefinder":
            d.setdefault(
                "ray_groups", RAY_GROUPS
            )  # sees the world, not the vehicle's own geoms (group 4)
            if d.get("projection", "pinhole") == "pinhole":
                # Rain defaults of an active-stereo depth camera (kernel defaults = a 905 nm lidar / ToF). ESTIMATES: stereo matching
                # loses its texture and rain streaks add false matches sooner than a ToF pulse loses signal (docs/models.md "Rain").
                d.setdefault("rain_margin_db", 4.0)
                d.setdefault("rain_det", 0.02)
        if typ == "rotor":
            d.setdefault("ray_groups", RAY_GROUPS)
            if b.get("reynolds"):
                _rotor_reynolds(ctx, inst, d, where, parts)
            if "cq_ref" in b:
                _rotor_cq_ref(ctx, b["cq_ref"], d, where)
        add(typ, d, parts)

    if not dev:
        return None, [], sites
    if "seed" in doc["assembly"]:
        dev["seed"] = doc["assembly"]["seed"]
    return dev, report, sites


def _foot_geom(inst, k, where: str) -> str:
    """MJCF geom name of shape `k` of the instance's part (build.py names a single-shape part's geom after the instance, else
    '<inst>:<k>'); the shape must collide, or the sensor would read 0 forever."""
    if inst is None:
        raise DesignError(
            f"{where}: type foot_contact needs an instance (the foot part)"
        )
    g = inst.part.geometry
    shapes = g.get("shapes") or ([g] if "shape" in g else [])
    if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < len(shapes):
        raise DesignError(
            f"{where}: shape {k!r}: part {inst.part.id} has {len(shapes)} geometry shape(s)"
        )
    if inst.offboard:
        raise DesignError(f"{where}: instance '{inst.id}' is offboard (no geom)")
    return inst.id if len(shapes) == 1 else f"{inst.id}:{k}"


def _hulls(inst, where: str) -> list:
    """Buoyancy hulls of a vessel: the instance part's geometry boxes that carry `cells = [nx, ny, nz]` (box half extents -> full)."""
    out = []
    for s in inst.part.geometry.get("shapes", []):
        if "cells" not in s:
            continue
        if s.get("shape") != "box":
            raise DesignError(
                f"{where}: hull shape {s.get('shape')!r} of part {inst.part.id}: only boxes are sampled"
            )
        out.append(
            {
                "shape": "box",
                "pos": [float(v) for v in s.get("pos", [0, 0, 0])],
                "size": [2.0 * float(v) for v in s["size"]],
                "n": [int(v) for v in s["cells"]],
            }
        )
    if not out:
        raise DesignError(
            f"{where}: part {inst.part.id} has no geometry box with `cells = [nx, ny, nz]` (the vessel's buoyancy hulls)"
        )
    return out


def _fins(ctx: Ctx, inst, where: str, parts: set) -> list:
    """Skegs / keels of a vessel: the rows of the instance part's table `fins` (columns x, y, z, area, span, cl_alpha, cd_cross; metres, m^2, 1/rad, 1)."""
    t = inst.part.raw.get("tables", {}).get("fins")
    if not t:
        return []
    need = ["x", "y", "z", "area", "span", "cl_alpha", "cd_cross"]
    if not set(need) <= set(t.get("columns", [])):
        raise DesignError(
            f"{where}: part {inst.part.id} table 'fins' needs the columns {need}"
        )
    idx = [t["columns"].index(c) for c in need]
    ctx.note(
        where,
        "fins",
        {
            "fidelity": t.get("fidelity"),
            "value": "table fins",
            "unit": "",
            "rationale": t.get("rationale", ""),
        },
    )
    parts.add(inst.part.id)
    return [
        {
            "pos": [float(r[i]) for i in idx[:3]],
            "area": float(r[idx[3]]),
            "span": float(r[idx[4]]),
            "cl_alpha": float(r[idx[5]]),
            "cd_cross": float(r[idx[6]]),
        }
        for r in t["rows"]
    ]


def _rotor_reynolds(ctx: Ctx, inst, d: dict, where: str, parts: set):
    """Binding flag `reynolds = true`: emit the pooled UIUC Reynolds factor tables (re_x / re_ft / re_fq) and re-reference
    the mapped ct / cq lists to x_ref. The prop part's ct / cq are means over its own static table, whose rows sit at
    higher x, so with f(x) switched on they must be the values AT x_ref (else the factor is applied twice): the lists are
    scaled by k / ct_static (k = least-squares C_ref of the part's [tables.ct_cq_static_derived] against f(x)), which
    keeps the J shape. Needs that table (columns rpm, C_T, C_Q) and the params ct_static / cq_static / diameter."""
    t = inst.part.raw.get("tables", {}).get("ct_cq_static_derived")
    if not t or not {"rpm", "C_T", "C_Q"} <= set(t.get("columns", [])):
        raise DesignError(
            f"{where}: reynolds needs part {inst.part.id} to have a table ct_cq_static_derived (rpm, C_T, C_Q)"
        )
    ct0, cq0, dia = (inst.value(k) for k in ("ct_static", "cq_static", "diameter"))
    if None in (ct0, cq0, dia) or "ct" not in d:
        raise DesignError(
            f"{where}: reynolds needs ct_static, cq_static, diameter on {inst.part.id} and a ct / cq map"
        )
    sh = reynolds_shape(ctx.lib)
    col = {c: [float(r[i]) for r in t["rows"]] for i, c in enumerate(t["columns"])}
    kt = ref_scale(sh, dia, col["rpm"], col["C_T"], "ft") / ct0
    kq = ref_scale(sh, dia, col["rpm"], col["C_Q"], "fq") / cq0
    d["ct"], d["cq"] = [v * kt for v in d["ct"]], [v * kq for v in d["cq"]]
    d["re_x"], d["re_ft"], d["re_fq"] = sh["x"], sh["ft"], sh["fq"]
    parts.update(sh["props"])
    ctx.note(
        where,
        "reynolds",
        {
            "fidelity": "derived",
            "value": "table",
            "unit": "",
            "rationale": "f_T, f_Q pooled from 7 UIUC APC static tables (dvsim.design.reynolds); ct / cq re-referenced to x = 4.5 m^2/s",
        },
    )


def _rotor_cq_ref(ctx: Ctx, spec, d: dict, where: str):
    """Binding field `cq_ref = {value, unit = "1", fidelity, derivation}`: the static torque coefficient C_Q(J = 0) (at x_ref
    when `reynolds` is on) to use in place of the prop part's own, applied as a uniform scale of the mapped cq list (keeps the
    J shape). For a prop whose part value comes from a vendor-rounded torque column that contradicts the vendor's own electrical
    power (docs/validation/x500-hover.md): the value is then a fitted / derived number carrying its provenance."""
    if "cq" not in d or not d["cq"]:
        raise DesignError(f"{where}: cq_ref needs a mapped cq list")
    v, _ = ctx.num(spec, f"{where} cq_ref", {"1": 1.0}, "cq_ref")
    if not 0 < v < 0.05:
        raise DesignError(f"{where}: cq_ref {v} outside (0, 0.05)")
    k = v / d["cq"][0]
    d["cq"] = [c * k for c in d["cq"]]


def _battery_ocv(ctx: Ctx, b: dict, d: dict, where: str, parts: set):
    if "ocv" not in b:
        return
    part = ctx.lib.get(b["ocv"])
    t = part.raw.get("tables", {}).get("ocv_vs_soc")
    if not t or not {"soc", "ocv_V"} <= set(t.get("columns", [])):
        raise DesignError(
            f"{where}: part {part.id} has no table ocv_vs_soc with columns soc, ocv_V"
        )
    cells, _ = ctx.num(
        f"{b['instance']}.{b.get('cells', 'cells_series')}",
        f"{where} cells",
        {"1": 1.0},
        "cells",
    )
    if cells != int(cells) or cells < 1:
        raise DesignError(
            f"{where}: series cell count must be a positive integer, got {cells}"
        )
    ci, cv = t["columns"].index("soc"), t["columns"].index("ocv_V")
    soc = [float(r[ci]) for r in t["rows"]]
    if any(y <= x for x, y in zip(soc, soc[1:])):
        raise DesignError(f"{where}: ocv table soc is not strictly increasing")
    d["ocv_soc"], d["ocv_v"] = soc, [float(r[cv]) * int(cells) for r in t["rows"]]
    ctx.note(
        where,
        f"ocv:{part.id}",
        {
            "fidelity": t.get("fidelity"),
            "value": "table",
            "unit": "V",
            "rationale": t.get("rationale", ""),
        },
    )
    parts.add(part.id)
