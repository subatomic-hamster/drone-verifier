"""Device bindings derived from parts + topology (`Design.auto_devices`, op `auto_devices`).

After a mechanical / electrical edit the simulation devices should follow: this module reads the instances, mates and wires of a
design document and writes the `[[devices]]` bindings (and the `[device_maps.auto_*]` tables they use) that a person would
otherwise write by hand (assemblies/x500v2.toml, aos5, so101, koch are the reference). Rules, in short (assemblies/README.md has
the full list):

* A value is never typed in here: a bound field is a device-map entry `"<field>" = "<param>"` (or `"motor:<param>"`), so the
  builder resolves it from the part with the part's own provenance. A value that needs arithmetic (an avionics load, the
  GNSS sigma, the battery RC capacitance) is a `set` entry whose `derivation` / `rationale` names the part fields; its
  fidelity is `estimated` as soon as one input is.
* A field the parts do not carry stays unset and is reported (`unset`); nothing is invented. The user supplies it with
  `fill = {"<device name glob>": {"set": {"<field>": <provenance table>}, "<binding field>": ...}}`, which is merged into
  the generated binding (a filled field wins over a mapped one).
* A binding written by this module carries `auto = true`; it is regenerated on every run. A binding without the flag is a
  hand-written one and wins unless `overwrite` (True, or a list of device names).
"""

import copy
import fnmatch
import math
import re
from dataclasses import dataclass, field

import numpy as np

from . import assembly as A
from .assembly import Ctx
from .build import compute
from .devices import FIELD_UNITS
from .drc import _World
from .library import DesignError
from .wiring import _State

GROUND_EFFECT = True  # rotor model switch, on in every reference assembly (not part data; the schema default is off)
DEFAULT_OCV = "lipo-cell-ocv-chen2006"  # the reference assemblies' OCV curve (a modelling choice: no battery part names one)
CEP_TO_SIGMA = math.sqrt(
    2 * math.log(2)
)  # circular normal: CEP (50 % radius) = 1.17741 sigma
BEC_PARAMS = ("bec_voltage", "bec_efficiency")
CONSUMER_CURRENT = ("supply_current_typ", "supply_current_avg", "supply_current")

# field -> part params that can feed it (first one the instance has wins). "@t.c" = a table column.
MOTOR = {
    "motor.kv": ["kv", "kv_equivalent"],
    "motor.r20": ["r_winding", "r_effective"],
    "motor.rotor_inertia": ["j_rotor_bell", "rotor_inertia_motor"],
    "motor.i0_visc": ["i0_visc"],
}
ESC = {
    "i_limit": ["current_continuous", "i_cont_per_ch", "i_cont"],
    "r_on": ["r_on_total"],
    "sw_loss": ["switching_loss_fraction"],
    "i_q": ["quiescent_current"],
}
SERVO = {
    "gear_ratio": ["gear_ratio"],
    "efficiency": ["gear_efficiency"],
    "encoder_cpr": ["encoder_resolution"],
    "firmware_hz": ["firmware_hz"],
    "p": ["device_p_default", "reg_p_coefficient_default"],
    "i": ["device_i_default", "reg_i_coefficient_default"],
    "d": ["device_d_default", "reg_d_coefficient_default"],
    "pid_scale": ["pid_scale"],
    "d_scale": ["d_scale"],
    "deadband": ["dead_zone_default", "dead_zone_cw_default"],
    "max_torque": ["device_max_torque_default", "reg_max_torque_limit_default"],
    "min_voltage": ["reg_min_voltage_limit_default", "reg_min_input_voltage_default"],
    "max_voltage": ["reg_max_voltage_limit_default", "reg_max_input_voltage_default"],
    "max_temp": ["reg_temperature_limit_default", "reg_max_temperature_default"],
    "overload_load": ["reg_overload_torque_default"],
    "overload_time": ["reg_protection_time_default"],
    "unload_mask": ["reg_unloading_condition_default", "reg_shutdown_default"],
}
# Integrated-driver QDD actuator (category actuator.bldc.qdd; schema Qdd). The CubeMars AK-series part carries kv, the
# line-to-line phase resistance, the rotor inertia and the gear ratio; efficiency, the thermal pair, the encoder, the current
# limit and the CAN-driver numbers are the part's own params (a part without them is reported `unset`, nothing is invented).
QDD_MOTOR = {
    "motor.kv": ["kv"],
    "motor.r20": ["r_phase_to_phase", "r_winding"],
    "motor.rotor_inertia": ["rotor_inertia"],
    "motor.i0": ["i0"],
    "motor.c_th": ["c_th_winding", "c_th"],
    "motor.r_th": ["r_th_winding", "r_th"],
}
QDD = {
    "gear_ratio": ["gear_ratio"],
    "efficiency": ["gear_efficiency", "efficiency"],
    "encoder_cpr": ["encoder_cpr"],
    "cmd_hz": ["can_cmd_hz", "cmd_hz"],
    "latency_s": ["can_latency", "latency_s"],
    "i_limit": ["i_limit_phase", "i_limit"],
    "kp_max": ["mit_kp_max", "kp_max"],
    "kd_max": ["mit_kd_max", "kd_max"],
    "tau_ff_max": ["mit_tau_ff_max", "tau_ff_max"],
    "timeout_s": ["can_timeout", "timeout_s"],
    "kd_timeout": ["kd_timeout"],
    "t_max": ["temp_fault", "t_max"],
    "derate_start": ["temp_derate_start", "derate_start"],
}
BATTERY = {
    "capacity_ah": ["capacity_ah"],
    "r0": ["r_internal_pack"],
    "r1": ["thevenin_r1"],
}
PSU = {
    "voltage": ["v_out_nominal"],
    "r_series": ["r_out"],
    "i_limit": ["current_limit"],
}
ROTOR = {
    "diameter": ["diameter"],
    "j": ["@ct_cq_j.j"],
    "ct": ["@ct_cq_j.ct"],
    "cq": ["@ct_cq_j.cq"],
}
IMU = {
    "gyro.noise_density": ["gyro_noise_density"],
    "gyro.full_scale": ["gyro_fs_max"],
    "gyro.bits": ["gyro_adc_bits"],
    "accel.noise_density": ["accel_noise_density_xy", "accel_noise_density"],
    "accel.full_scale": ["accel_fs_max"],
    "accel.bits": ["accel_adc_bits"],
}
MAG = {"noise_std": ["rms_noise"]}
GNSS = {
    "rate_hz": ["max_nav_rate_gps_glonass", "max_nav_rate_gps_only"],
    "vel_std": ["velocity_accuracy"],
}
# device fields the C++ validation needs > 0 / present (devices.cpp); anything else has a schema default
REQUIRED = {
    "esc": [
        "joint",
        "motor.kv",
        "motor.r20",
        "motor.rotor_inertia",
        "motor.c_th",
        "motor.r_th",
    ],
    "servo": [
        "joint",
        "motor.kv",
        "motor.r20",
        "motor.rotor_inertia",
        "motor.c_th",
        "motor.r_th",
        "gear_ratio",
        "efficiency",
        "encoder_cpr",
        "firmware_hz",
        "pid_scale",
        "max_torque",
    ],
    "qdd": [
        "joint",
        "motor.kv",
        "motor.r20",
        "motor.rotor_inertia",
        "motor.c_th",
        "motor.r_th",
        "gear_ratio",
        "efficiency",
        "encoder_cpr",
        "cmd_hz",
        "i_limit",
        "kp_max",
        "kd_max",
        "tau_ff_max",
        "timeout_s",
        "kd_timeout",
        "t_max",
        "derate_start",
    ],
    "rotor": ["body", "joint", "diameter", "j", "ct", "cq"],
    "battery": ["ocv", "capacity_ah", "r0"],
    "psu": ["voltage", "r_series"],
    "wire": ["resistance"],
    "load": ["i0"],
    "imu": ["site", "rate_hz"],
    "baro": ["site", "rate_hz"],
    "mag": ["site", "rate_hz", "earth_field"],
    "gnss": [
        "site",
        "rate_hz",
        "lat0",
        "lon0",
        "alt0",
    ],  # the world origin: where on Earth is the user's call
    "environment": [],
}
HOST = {
    "imu": "pcb.fc",
    "baro": "pcb.fc",
    "mag": "pcb.gps",
    "gnss": "pcb.gps",
}  # board an offboard sensor chip sits on
CHIP = {
    "sensor.imu": "imu",
    "sensor.baro": "baro",
    "sensor.mag": "mag",
    "sensor.gnss": "gnss",
}
# `set` fields this module computes from part values: never carried over (a swapped part that no longer feeds them drops them)
FORMULA = {
    "battery": {"c1"},
    "load": {"i0"},
    "gnss": {"horiz_std", "vert_std"},
    "rotor": {"j", "ct", "cq"},
    "environment": {"origin_alt"},
}
# binding keys the derivation owns; any other key of a generated binding (cq_ref, a filled site) is carried over on regeneration
STRUCTURAL = {
    "type", "name", "map", "set", "auto", "instance", "port", "motor", "joint", "body",
    "a", "b", "ocv", "dir", "ground_effect", "reynolds",
}  # fmt: skip
ROLE_FIELDS = (
    "instance",
    "body",
    "ref_body",
    "site",
    "motor",
)  # binding fields that name an instance


def _split(ref):
    i, _, p = ref.partition(".")
    return i, (p or None)


def _digits(s):
    m = re.search(r"(\d+)$", s)
    return m.group(1) if m else ""


def _num(e):
    v = e.get("value") if isinstance(e, dict) else None
    return v if A._is_num(v) else None


@dataclass
class Plan:
    typ: str
    name: str
    binding: (
        dict  # type / name / instance / port / joint / ... (no map, no set, no auto)
    )
    key: tuple
    anchors: set
    parts: set = field(
        default_factory=set
    )  # before `set` below: that name shadows the builtin in this class body
    map: dict = field(
        default_factory=dict
    )  # device field -> "<param>" | "<role>:<param>"
    set: dict = field(
        default_factory=dict
    )  # device field -> provenance table (derived here, or filled)
    src: dict = field(default_factory=dict)  # device field -> "inst.param" (report)
    notes: list = field(default_factory=list)
    filled: list = field(default_factory=list)

    def has(self, f):
        return f in self.map or f in self.set or f in self.binding

    def missing(self):
        return [f for f in REQUIRED[self.typ] if not self.has(f)]

    def emit(self, mapname):
        b = dict(self.binding)
        if self.map:
            b["map"] = mapname
        b["auto"] = True
        if self.set:
            b["set"] = copy.deepcopy(self.set)
        return b


def _prov(value, unit, inputs, formula):
    """A derived / estimated provenance table; `inputs` = [(label, entry)]. Estimated as soon as one input is."""
    if isinstance(value, float):
        value = float(f"{value:.10g}")  # 412.49999999999994 -> 412.5
    est = [lab for lab, e in inputs if e.get("fidelity") == "estimated"]
    text = f"{formula}; inputs: " + ", ".join(
        f"{lab} = {e['value']} {e.get('unit', '')} ({e.get('fidelity')})"
        for lab, e in inputs
    )
    if est:
        return {
            "value": value,
            "unit": unit,
            "fidelity": "estimated",
            "rationale": text + f"; estimated because {', '.join(est)} is",
        }
    return {"value": value, "unit": unit, "fidelity": "derived", "derivation": text}


class _Topo:
    def __init__(self, d, doc):
        self.d, self.doc = d, doc
        self.ctx = Ctx(A.Assembly(doc), d.lib)
        self.insts = self.ctx.insts
        self.mates = [
            {
                "name": m.get("name", f"{m['a']}--{m['b']}"),
                "a": _split(m["a"]),
                "b": _split(m["b"]),
                "joint": m.get("joint"),
                "closure": bool(m.get("closure")),
            }
            for m in doc.get("mates", [])
        ]
        self.wires = [
            {"from": _split(w["from"]), "to": _split(w["to"]), "w": w}
            for w in doc.get("wires", [])
        ]
        self._world = False
        self.notes = []

    def port(self, ref):
        iid, pn = ref
        return self.insts[iid].part.port(pn) if iid in self.insts and pn else {}

    def world(self):
        if self._world is False:
            try:
                xml, _ = compute(
                    A.Assembly(self.doc),
                    self.d.lib.dir,
                    self.d.lib.root,
                    self.d.lib,
                    route=False,
                )
                self._world = _World(xml, self.d.lib.root)
            except (DesignError, KeyError, ValueError, StopIteration):
                self._world = None
        return self._world


# ---- field -> map entries --------------------------------------------------------------------------------------------
def _fields(plan, inst, table, role=None, bad=None):
    """Map every field of `table` the instance can feed; the others go to plan.notes only when required."""
    for f, aliases in table.items():
        hit = None
        for a in aliases:
            if a.startswith("@"):
                tn, _, col = a[1:].partition(".")
                t = inst.part.raw.get("tables", {}).get(tn)
                if t and col in t.get("columns", []):
                    hit = a
                    break
                continue
            e = inst.entry(a)
            if _num(e) is None:
                continue
            units = FIELD_UNITS.get(f.rsplit(".", 1)[-1])
            if units is not None and e.get("unit") not in units:
                plan.notes.append(
                    f"{inst.id}.{a} has unit {e.get('unit')!r}, device field {f} needs one of {sorted(units)}: not mapped"
                )
                continue
            hit = a
            break
        if hit:
            plan.map[f] = f"{role}:{hit}" if role else hit
            plan.src[f] = f"{inst.id}.{hit}"
            plan.parts.add(inst.part.id)


def _motor_fields(plan, motor, role, tag=None):
    _fields(plan, motor, MOTOR, role)
    i0 = ["i0"] + ([f"no_load_current_{tag}"] if tag else [])
    _fields(plan, motor, {"motor.i0": i0}, role)
    if tag:
        _fields(plan, motor, {"motor.i0_w": [f"omega_noload_motor_{tag}"]}, role)


def _tag(inst, vbus):
    """Servo no-load point: the voltage tag (`no_load_current_<tag>` with `omega_noload_motor_<tag>`, i0 is measured at that
    speed) nearest to the supply voltage; ties go to the lower voltage."""
    tags = []
    for p in inst.part.params:
        m = re.fullmatch(r"no_load_current_(\d+)v(\d*)", p)
        if (
            m
            and _num(inst.entry(f"omega_noload_motor_{m.group(1)}v{m.group(2)}"))
            is not None
        ):
            tags.append(
                (float(f"{m.group(1)}.{m.group(2) or 0}"), f"{m.group(1)}v{m.group(2)}")
            )
    if not tags:
        return None
    ref = vbus if vbus else (inst.value("v_nominal") or 0.0)
    return min(tags, key=lambda t: (abs(t[0] - ref), t[0]))[1]


def _first_port(inst, kind, role=None, not_type=None):
    for p in inst.part.ports:
        if (
            p.get("kind") == kind
            and (role is None or p.get("role") == role)
            and p.get("type") != not_type
        ):
            return p["name"]
    return None


# ---- the rules ---------------------------------------------------------------------------------------------------------
def _dir(topo, mate, prop_id):
    """+1 / -1 so that dir x (world joint axis) points along world +z (thrust up). Falls back to the mate's `reverse` flag
    when the design does not build or the axis is not within 60 degrees of vertical."""
    fallback = -1 if (mate["joint"] or {}).get("reverse") else 1
    w = topo.world()
    if w is None or prop_id not in w.idx:
        return fallback, "from the mate's reverse flag (the design does not build)"
    b = w.bodies[w.idx[prop_id]]
    if not b.get("joint"):
        return fallback, "from the mate's reverse flag"
    az = float((b["T"][:3, :3] @ np.asarray(b["joint"]["axis"], dtype=float))[2])
    if abs(az) < 0.5:
        return fallback, "from the mate's reverse flag (the spin axis is not vertical)"
    return (
        1 if az > 0 else -1
    ), f"thrust up: spin joint axis z = {az:+.2f} in the world"


def _propulsion(topo, plans):
    ctx = topo.ctx
    drives = []  # (esc inst, esc port, motor inst)
    for w in topo.wires:
        for a, b in ((w["from"], w["to"]), (w["to"], w["from"])):
            pa, pb = topo.port(a), topo.port(b)
            if (
                pa.get("type") == "motor-3phase"
                and pa.get("role") == "source"
                and pb.get("role") == "sink"
            ):
                drives.append((ctx.insts[a[0]], a[1], ctx.insts[b[0]]))
    per_esc = {}
    for esc, port, mot in drives:
        per_esc.setdefault(esc.id, []).append(port)
    for esc, port, mot in drives:
        mate = next(
            (
                m
                for m in topo.mates
                if not m["closure"]
                and mot.id in (m["a"][0], m["b"][0])
                and any(
                    ctx.insts[x].part.category.startswith("prop")
                    for x in (m["a"][0], m["b"][0])
                    if x != mot.id and x in ctx.insts
                )
                and (m["joint"] or {}).get("type") == "hinge"
            ),
            None,
        )
        prop = None
        if mate:
            other = mate["b"][0] if mate["a"][0] == mot.id else mate["a"][0]
            prop = ctx.insts[other]
        many = len(per_esc[esc.id]) > 1
        name = (
            esc.id + _digits(port)
            if many and _digits(port)
            else (f"{esc.id}_{mot.id}" if many else esc.id)
        )
        b = {"type": "esc", "name": name, "instance": esc.id, "motor": mot.id}
        pw = _first_port(esc, "power", "sink")
        if pw:
            b["port"] = pw
        if mate:
            b["joint"] = mate["name"]
        p = Plan(
            "esc", name, b, ("esc", mate["name"] if mate else name), {esc.id, mot.id}
        )
        p.parts |= {esc.part.id, mot.part.id}
        _fields(p, esc, ESC)
        _motor_fields(p, mot, "motor")
        if not mate:
            p.notes.append(f"no spin joint found: {mot.id} has no hinge mate to a prop")
        plans.append(p)
        if prop is None:
            continue
        d, why = _dir(topo, mate, prop.id)
        rb = {
            "type": "rotor",
            "name": f"rotor_{prop.id}",
            "instance": prop.id,
            "body": prop.id,
            "joint": mate["name"],
            "dir": d,
        }
        r = Plan("rotor", rb["name"], rb, ("rotor", mate["name"]), {prop.id, mot.id})
        r.parts.add(prop.part.id)
        r.notes.append(
            f"dir {d:+d}: {why}; no part states a prop handedness, so the prop is assumed to match the spin sense"
        )
        _fields(r, prop, ROTOR)
        if "ct" not in r.map:
            _static_rotor(r, prop)
        r.binding["ground_effect"] = GROUND_EFFECT
        t = prop.part.raw.get("tables", {}).get("ct_cq_static_derived")
        if (
            "ct" in r.map
            and t
            and {"rpm", "C_T", "C_Q"} <= set(t.get("columns", []))
            and all(
                prop.value(k) is not None
                for k in ("ct_static", "cq_static", "diameter")
            )
        ):
            r.binding["reynolds"] = True
        plans.append(r)


def _static_rotor(r, prop):
    """A prop with no J sweep, only static coefficients: a one-point table at J = 0 (Rotor tables may have one point)."""
    ct = next(
        (
            n
            for n in ("ct_static", "ct_static_at_max_rpm")
            if _num(prop.entry(n)) is not None
        ),
        None,
    )
    cp = next(
        (
            n
            for n in ("cq_static", "cp_static_at_max_rpm")
            if _num(prop.entry(n)) is not None
        ),
        None,
    )
    if not ct or not cp:
        r.notes.append(
            f"{prop.part.id}: no ct_cq_j table and no static ct / cq (or cp): j / ct / cq unset"
        )
        return
    ect, ecp = prop.entry(ct), prop.entry(cp)
    cq = ecp["value"] if cp == "cq_static" else ecp["value"] / (2 * math.pi)
    how = "" if cp == "cq_static" else " C_Q = C_P / (2 pi)."
    inputs = [(f"{prop.id}.{ct}", ect), (f"{prop.id}.{cp}", ecp)]
    r.set["j"] = {
        "value": [0.0],
        "unit": "1",
        "fidelity": "derived",
        "derivation": "static point J = 0: the part has no J sweep",
    }
    r.set["ct"] = _prov(
        [ect["value"]],
        "1",
        inputs[:1],
        f"C_T at J = 0 = {prop.id}.{ct} (no J sweep in the part)",
    )
    r.set["cq"] = _prov(
        [cq],
        "1",
        inputs[1:],
        f"C_Q at J = 0 from {prop.id}.{cp}.{how} (no J sweep in the part)",
    )
    r.src.update({"ct": f"{prop.id}.{ct}", "cq": f"{prop.id}.{cp}"})
    r.parts.add(prop.part.id)
    r.notes.append(
        "one-point rotor table (J = 0 only): thrust does not fall with advance ratio"
    )


def _servos(topo, plans, st):
    ctx = topo.ctx
    seen = set()
    for m in topo.mates:
        if (
            m["closure"]
            or (m["joint"] or {}).get("type") not in ("hinge", "slide")
            or (m["joint"] or {}).get("spin")
        ):
            continue
        for iid, pn in (m["a"], m["b"]):
            inst = ctx.insts.get(iid)
            if (
                inst is None
                or not inst.part.category.startswith("motor.servo")
                or iid in seen
            ):
                continue
            seen.add(iid)
            pw = _first_port(inst, "power")
            b = {"type": "servo", "name": iid, "instance": iid, "joint": m["name"]}
            if pw:
                b["port"] = pw
            p = Plan("servo", iid, b, ("servo", m["name"]), {iid})
            p.parts.add(inst.part.id)
            vbus = None
            if pw and st is not None:
                v = st.net.voltage((iid, pw))
                vbus = v[1] if v else None
            tag = _tag(inst, vbus)
            if tag:
                p.notes.append(
                    f"no-load point {tag}: nearest to the supply "
                    + (
                        f"{vbus:g} V"
                        if vbus
                        else "(no source reaches the servo: the part's v_nominal)"
                    )
                )
            _motor_fields(p, inst, None, tag)
            _fields(p, inst, SERVO)
            plans.append(p)


def _qdds(topo, plans):
    """One qdd device per actuator.bldc.qdd instance on a hinge mate (its output joint)."""
    seen = set()
    for m in topo.mates:
        if (
            m["closure"]
            or (m["joint"] or {}).get("type") != "hinge"
            or (m["joint"] or {}).get("spin")
        ):
            continue
        for iid, pn in (m["a"], m["b"]):
            inst = topo.ctx.insts.get(iid)
            if (
                inst is None
                or not inst.part.category.startswith("actuator.bldc.qdd")
                or iid in seen
            ):
                continue
            seen.add(iid)
            pw = _first_port(inst, "power")
            b = {"type": "qdd", "name": iid, "instance": iid, "joint": m["name"]}
            if pw:
                b["port"] = pw
            p = Plan("qdd", iid, b, ("qdd", m["name"]), {iid})
            p.parts.add(inst.part.id)
            _fields(p, inst, QDD_MOTOR)
            _fields(p, inst, QDD)
            plans.append(p)


def _batteries(topo, plans, ocv):
    for inst in topo.insts.values():
        if (
            not inst.part.category.startswith("battery")
            or inst.value("capacity_ah") is None
        ):
            continue
        port = _first_port(inst, "power", "source")
        if not port:
            continue
        name = (
            "battery"
            if sum(
                1
                for i in topo.insts.values()
                if i.part.category.startswith("battery") and i.value("capacity_ah")
            )
            == 1
            else f"battery_{inst.id}"
        )
        p = Plan(
            "battery",
            name,
            {"type": "battery", "name": name, "instance": inst.id, "port": port},
            ("battery", inst.id),
            {inst.id},
        )
        p.parts.add(inst.part.id)
        _fields(p, inst, BATTERY)
        if (
            inst.part.category == "battery.lipo"
            and inst.value("cells_series") is not None
        ):
            lib = topo.d.lib
            pick = ocv or DEFAULT_OCV
            if pick in lib.parts and "ocv_vs_soc" in lib.parts[pick].raw.get(
                "tables", {}
            ):
                p.binding["ocv"] = pick
                p.parts.add(pick)
                p.src["ocv"] = f"{pick} (table ocv_vs_soc) x {inst.id}.cells_series"
                alts = [
                    x.id
                    for x in lib.by_category("battery.cell_curve")
                    if x.id != pick and "ocv_vs_soc" in x.raw.get("tables", {})
                ]
                p.notes.append(
                    f"OCV curve {pick} (the reference assemblies' choice; no battery part names one); others: {', '.join(alts) or 'none'}"
                )
            else:
                p.notes.append(
                    f"OCV curve {pick!r} is not in the library with a table ocv_vs_soc"
                )
        else:
            p.notes.append(
                f"{inst.part.category}: no OCV curve for this chemistry in the library, `ocv` unset"
            )
        r1, tau = inst.entry("thevenin_r1"), inst.entry("thevenin_tau")
        if _num(r1) and _num(tau):
            p.set["c1"] = _prov(
                tau["value"] / r1["value"],
                "F",
                [(f"{inst.id}.thevenin_tau", tau), (f"{inst.id}.thevenin_r1", r1)],
                "c1 = thevenin_tau / thevenin_r1",
            )
            p.src["c1"] = f"{inst.id}.thevenin_tau / {inst.id}.thevenin_r1"
        elif "r1" in p.map:
            p.notes.append(
                "thevenin_tau missing: r1 is set but c1 is not (the RC pair needs both)"
            )
            del p.map["r1"]
        plans.append(p)


def _psus(topo, plans):
    for inst in topo.insts.values():
        if inst.part.category.split(".")[0] != "psu":
            continue
        port = _first_port(inst, "power", "source")
        if not port:
            continue
        p = Plan(
            "psu",
            inst.id,
            {"type": "psu", "name": inst.id, "instance": inst.id, "port": port},
            ("psu", inst.id),
            {inst.id},
        )
        p.parts.add(inst.part.id)
        _fields(p, inst, PSU)
        plans.append(p)


def _sources(topo, plans):
    """Instances that are a net source: those of the battery / psu plans, as {inst: dist 0}."""
    return {p.binding["instance"] for p in plans if p.typ in ("battery", "psu")}


def _boards(topo, plans, src):
    """Resistive edges inside boards (`wire` bindings): a part with >= 2 pass-through power ports and r_series / r_path gets an edge
    from its input port (the wired pass-through port whose far end is nearer the source) to every wired output; a pcb with a
    sink and a source port and no pass-through (a servo bus adapter) needs an edge too, but no part states its resistance."""
    ctx = topo.ctx
    adj = {}
    for w in topo.wires:
        pa, pb = topo.port(w["from"]), topo.port(w["to"])
        if pa.get("kind") != "power" or pb.get("kind") != "power":
            continue
        adj.setdefault(w["from"][0], set()).add(w["to"][0])
        adj.setdefault(w["to"][0], set()).add(w["from"][0])
    dist, todo = {i: 0 for i in src}, list(src)
    while todo:
        n = todo.pop(0)
        for m in adj.get(n, ()):
            if m not in dist:
                dist[m] = dist[n] + 1
                todo.append(m)
    for inst in ctx.insts.values():
        pt = [
            p["name"]
            for p in inst.part.ports
            if p.get("kind") == "power" and p.get("role") == "passthrough"
        ]
        res = next(
            (n for n in ("r_series", "r_path") if _num(inst.entry(n)) is not None), None
        )
        if len(pt) >= 2 and inst.id in dist:
            ends = {}
            for w in topo.wires:
                for mine, other in ((w["from"], w["to"]), (w["to"], w["from"])):
                    if mine[0] == inst.id and mine[1] in pt and other[0] in dist:
                        ends.setdefault(mine[1], []).append(dist[other[0]])
            inp = [p for p, ds in ends.items() if min(ds) < dist[inst.id]]
            outs = [p for p in pt if p in ends and p not in inp]
            if len(inp) != 1 or not outs:
                continue
            if res is None:
                topo.notes.append(
                    f"{inst.id} ({inst.part.id}): no r_series / r_path param, its ports stay one zero-resistance node"
                )
                continue
            for o in outs:
                name = (
                    f"{inst.id}_board"
                    if len(outs) == 1
                    else f"{inst.id}_path{_digits(o) or o}"
                )
                b = {
                    "type": "wire",
                    "name": name,
                    "instance": inst.id,
                    "a": inp[0],
                    "b": o,
                }
                p = Plan(
                    "wire",
                    name,
                    b,
                    ("wire", inst.id, frozenset((inp[0], o))),
                    {inst.id},
                )
                p.parts.add(inst.part.id)
                _fields(p, inst, {"resistance": [res]})
                plans.append(p)
        elif not pt and inst.part.category == "pcb.servo_driver":
            sink = _first_port(inst, "power", "sink")
            source = _first_port(inst, "power", "source")
            if sink and source:
                name = f"{inst.id}_board"
                b = {
                    "type": "wire",
                    "name": name,
                    "instance": inst.id,
                    "a": sink,
                    "b": source,
                }
                p = Plan(
                    "wire",
                    name,
                    b,
                    ("wire", inst.id, frozenset((sink, source))),
                    {inst.id},
                )
                p.parts.add(inst.part.id)
                _fields(p, inst, {"resistance": ["r_series", "r_path"]})
                p.notes.append(
                    "separate sink / source nodes: without this edge the downstream net has no source"
                )
                plans.append(p)
    return dist


def _loads(topo, plans, st):
    """Avionics as constant-current sinks on the battery side of the BEC (the schema's Load has no constant-power form):
    I_bat = V_rail x I_part / (eta x V_nominal). A consumer is a part with supply_current_*, power-wired to a BEC / source rail
    or signal-wired to one that is (a GPS on the FC's port is powered through it)."""
    ctx = topo.ctx
    cons = {
        i.id: next((n for n in CONSUMER_CURRENT if _num(i.entry(n)) is not None), None)
        for i in ctx.insts.values()
    }
    cons = {k: v for k, v in cons.items() if v}
    rail = {}  # consumer -> (source instance, source port)
    for w in topo.wires:
        pa, pb = topo.port(w["from"]), topo.port(w["to"])
        if pa.get("kind") != "power":
            continue
        for s, c in ((w["from"], w["to"]), (w["to"], w["from"])):
            if c[0] in cons and topo.port(s).get("role") == "source":
                rail[c[0]] = s
    for _ in range(len(cons)):  # signal wires carry the rail one hop at a time
        for w in topo.wires:
            pa = topo.port(w["from"])
            if pa.get("kind") != "signal":
                continue
            a, b = w["from"][0], w["to"][0]
            for x, y in ((a, b), (b, a)):
                if x in rail and y in cons and y not in rail:
                    rail[y] = rail[x]
    for cid, s in sorted(rail.items()):
        c, bec = ctx.insts[cid], ctx.insts[s[0]]
        iin = c.entry(cons[cid])
        if all(_num(bec.entry(k)) is not None for k in BEC_PARAMS):
            port = _first_port(bec, "power", "passthrough") or _first_port(
                bec, "power", "sink"
            )
            srcs = st.net.sources((bec.id, port)) if port else []
            bat = ctx.insts[srcs[0][0]] if srcs else None
            vn = bat.entry("v_nominal") if bat else None
            if _num(vn) is None:
                topo.notes.append(
                    f"load_{cid}: the net of {bec.id}.{port} has no source with v_nominal: not derived"
                )
                continue
            ev, ee = bec.entry("bec_voltage"), bec.entry("bec_efficiency")
            i0 = -ev["value"] * iin["value"] / (ee["value"] * vn["value"])
            inputs = [
                (f"{bec.id}.bec_voltage", ev),
                (f"{cid}.{cons[cid]}", iin),
                (f"{bec.id}.bec_efficiency", ee),
                (f"{bat.id}.v_nominal", vn),
            ]
            formula = "i0 = -V_rail x I_part / (eta x V_nominal): constant-current sink on the battery side of the BEC"
            where = bec
        else:
            port = _first_port(c, "power", "sink")
            if not port:
                continue
            i0, inputs, formula, where = (
                -iin["value"],
                [(f"{cid}.{cons[cid]}", iin)],
                "i0 = -I_part (fed directly)",
                c,
            )
        name = f"load_{cid}"
        b = {"type": "load", "name": name, "instance": where.id, "port": port}
        p = Plan("load", name, b, ("load", name), {cid, where.id})
        p.parts |= {c.part.id, where.part.id}
        p.set["i0"] = _prov(round(i0, 6), "A", inputs, formula)
        p.src["i0"] = f"{cid}.{cons[cid]}"
        plans.append(p)


def _sensors(topo, plans):
    ctx = topo.ctx
    for inst in ctx.insts.values():
        typ = next((v for k, v in CHIP.items() if inst.part.category == k), None)
        if typ is None:
            continue
        if not inst.offboard:
            site, why = inst.id, "the chip's own body"
        else:
            hosts = [
                i.id
                for i in ctx.insts.values()
                if not i.offboard and i.part.category.startswith(HOST[typ])
            ]
            site, why = (
                (hosts[0], f"the only {HOST[typ]} board")
                if len(hosts) == 1
                else (None, f"{len(hosts)} {HOST[typ]} boards: name one with fill site")
            )
        b = {
            "type": typ,
            "name": typ
            if sum(
                1 for i in ctx.insts.values() if i.part.category == inst.part.category
            )
            == 1
            else f"{typ}_{inst.id}",
            "instance": inst.id,
        }
        if site:
            b["site"] = site
        p = Plan(
            typ, b["name"], b, (typ, inst.id), {inst.id} | ({site} if site else set())
        )
        p.parts.add(inst.part.id)
        p.notes.append(f"site: {why}" if site else f"site unset: {why}")
        if typ == "imu":
            _fields(p, inst, IMU)
        elif typ == "baro":
            noise = [
                (inst.value(n), n)
                for n in inst.part.params
                if n.startswith("pressure_rms_noise") and inst.value(n) is not None
            ]
            if noise:
                _fields(p, inst, {"noise_std": [min(noise)[1]]})
                p.notes.append(
                    f"noise_std = the lowest of {len(noise)} pressure_rms_noise_* settings ({min(noise)[1]})"
                )
        elif typ == "mag":
            _fields(p, inst, MAG)
        else:
            _fields(p, inst, GNSS)
            cep, vcep = (
                inst.entry("horizontal_accuracy_cep_gps"),
                inst.entry("vertical_accuracy_cep"),
            )
            if _num(cep):
                h = cep["value"] / CEP_TO_SIGMA
                p.set["horiz_std"] = _prov(
                    h,
                    "m",
                    [(f"{inst.id}.horizontal_accuracy_cep_gps", cep)],
                    "per-coordinate sigma of a circular normal = CEP / 1.17741",
                )
                p.src["horiz_std"] = f"{inst.id}.horizontal_accuracy_cep_gps"
                if _num(vcep):
                    p.set["vert_std"] = _prov(
                        h * vcep["value"] / cep["value"],
                        "m",
                        [
                            (f"{inst.id}.horizontal_accuracy_cep_gps", cep),
                            (f"{inst.id}.vertical_accuracy_cep", vcep),
                        ],
                        "vert_std = horiz_std x vertical_accuracy_cep / horizontal_accuracy_cep",
                    )
        plans.append(p)


def derive(d, doc, ocv=None) -> tuple:
    """All bindings this design implies, in the reference assemblies' order -> (plans, topo)."""
    topo = _Topo(d, doc)
    plans: list = []
    _batteries(topo, plans, ocv)
    _psus(topo, plans)
    src = _sources(topo, plans)
    # fed / voltage questions are answered by the DRC's own net model on the document with the source bindings in it
    tmp = copy.deepcopy(doc)
    tmp["devices"] = [dict(p.binding) for p in plans]
    st = _State(d, tmp)
    _boards(topo, plans, src)
    _loads(topo, plans, st)
    _propulsion(topo, plans)
    _servos(topo, plans, st)
    _qdds(topo, plans)
    _sensors(topo, plans)
    return plans, topo


def _apply_fill(plans, fill):
    hit = set()
    for pat, patch in (fill or {}).items():
        for p in plans:
            if not fnmatch.fnmatchcase(p.name, pat):
                continue
            hit.add(pat)
            for k, v in patch.items():
                if k == "set":
                    for f, x in v.items():
                        p.set[f] = copy.deepcopy(x)
                        p.map.pop(f, None)
                        p.filled.append(f)
                else:
                    p.binding[k] = copy.deepcopy(v)
                    p.filled.append(k)
    return [x for x in (fill or {}) if x not in hit]


def _environment(topo, plans):
    for p in plans:
        e = p.set.get("alt0") if p.typ == "gnss" else None
        if A._is_num(e.get("value") if isinstance(e, dict) else None) and not any(
            q.typ == "environment" for q in plans
        ):
            env = Plan(
                "environment",
                "environment",
                {
                    "type": "environment",
                    "name": "environment",
                    "instance": p.binding["instance"],
                },
                ("environment",),
                {p.binding["instance"]},
            )
            env.set["origin_alt"] = copy.deepcopy(e)
            env.src["origin_alt"] = (
                f"{p.name}.alt0 (world z = 0 sits at the GNSS origin altitude)"
            )
            env.parts.add(topo.insts[p.binding["instance"]].part.id)
            plans.append(env)


# ---- installing -------------------------------------------------------------------------------------------------------------
def _refs(b):
    out = {b[k] for k in ROLE_FIELDS if isinstance(b.get(k), str)}
    return out


def _key(b):
    t = b["type"]
    if t == "esc":
        return ("esc", b.get("joint") or b["name"])
    if t == "rotor":
        return ("rotor", b.get("joint") or b["name"])
    if t in ("servo", "qdd"):
        return (t, b.get("joint") or b["name"])
    if t == "wire":
        return ("wire", b.get("instance"), frozenset((b.get("a"), b.get("b"))))
    if t == "load":
        return ("load", b["name"])
    if t == "environment":
        return ("environment",)
    return (t, b.get("instance"))


def stale(doc, iid) -> list:
    """Device bindings that name instance `iid` (as instance / motor / body / site): their values came from the part it replaced."""
    return [
        {
            "device": b["name"],
            "type": b["type"],
            "generated": bool(b.get("auto")),
            "why": f"names {iid} ({', '.join(k for k in ROLE_FIELDS if b.get(k) == iid)})",
        }
        for b in doc.get("devices", [])
        if iid in _refs(b)
    ]


def _mapname(doc, plan, used):
    base = "auto_" + re.sub(
        r"[^A-Za-z0-9_-]", "_", plan.typ + "_" + "_".join(sorted(plan.parts))
    )
    n, name = 1, base
    while name in used and used[name] != plan.map:
        n += 1
        name = f"{base}_{n}"
    used[name] = dict(plan.map)
    return name


def _carry(plans, devices) -> set:
    """Values a person supplied (fill) for what no part carries survive the regeneration of a generated binding: its `set`
    entries the parts cannot feed now, and extra binding keys (cq_ref, a filled site). -> {(plan key, field)}."""
    old = {_key(b): b for b in devices if b.get("auto")}
    out = set()
    for p in plans:
        b = old.get(p.key)
        if b is None:
            continue
        for f, v in b.get("set", {}).items():
            if f not in p.map and f not in p.set and f not in FORMULA.get(p.typ, ()):
                p.set[f] = v
                out.add((p.key, f))
        for k, v in b.items():
            if k not in p.binding and k not in STRUCTURAL:
                p.binding[k] = v
                out.add((p.key, k))
    return out


def auto_devices(d, doc, s) -> dict:
    """Op `auto_devices` {scope?, overwrite?, fill?, ocv?}: write the bindings the parts and topology imply into `doc`."""
    scope = s.get("scope")
    scope = {scope} if isinstance(scope, str) else (set(scope) if scope else None)
    ow = s.get("overwrite") or False
    plans, topo = derive(d, doc, s.get("ocv"))
    devices = doc.setdefault("devices", [])
    unmatched = _apply_fill(plans, s.get("fill"))
    carried = _carry(plans, devices)
    _environment(topo, plans)
    unmatched = [
        x for x in unmatched if not any(fnmatch.fnmatchcase(p.name, x) for p in plans)
    ]
    env = [p for p in plans if p.typ == "environment"]
    _apply_fill(env, s.get("fill"))
    carried |= _carry(env, devices)
    existing = {_key(b): i for i, b in enumerate(devices)}
    produced = {p.key for p in plans}
    rec = {
        "op": "auto_devices",
        "created": [],
        "refreshed": [],
        "unchanged": [],
        "kept": [],
        "removed": [],
        "unset": [],
        "notes": list(topo.notes),
    }
    maps = doc.setdefault("device_maps", {})
    used = {k: v for k, v in maps.items()}
    for p in plans:
        if scope is not None and not (p.anchors & scope):
            continue
        i = existing.get(p.key)
        old = devices[i] if i is not None else None
        if (
            old is not None
            and not old.get("auto")
            and not (ow is True or p.name in (ow or ()))
        ):
            rec["kept"].append(
                {
                    "device": old["name"],
                    "why": "hand-written binding wins (overwrite = True replaces it)",
                }
            )
            continue
        taken = {b["name"] for j, b in enumerate(devices) if j != i}
        while p.name in taken:
            p.name += "_auto"
            p.binding["name"] = p.name
        new = p.emit(_mapname(doc, p, used) if p.map else "")
        if p.map:
            maps[new["map"]] = dict(p.map)
        r = {
            "device": p.name,
            "type": p.typ,
            "instance": p.binding.get("instance"),
            "parts": sorted(p.parts),
            "from": dict(sorted(p.src.items())),
            "filled": p.filled,
            "carried": sorted(c for k, c in carried if k == p.key),
            "notes": p.notes,
        }
        if old is not None:
            same = old == new and (not p.map or maps.get(new["map"]) == dict(p.map))
            devices[i] = new
            (rec["unchanged"] if same else rec["refreshed"]).append(
                r if not same else {**r, "was": old.get("map")}
            )
        else:
            devices.append(new)
            rec["created"].append(r)
        if p.missing():
            rec["unset"].append({"device": p.name, "fields": p.missing()})
    for b in list(devices):  # generated bindings that the design no longer implies
        if (
            b.get("auto")
            and _key(b) not in produced
            and (scope is None or _refs(b) & scope)
        ):
            devices.remove(b)
            rec["removed"].append(
                {"device": b["name"], "why": "no longer implied by the design"}
            )
    for n in [
        n
        for n in maps
        if n.startswith("auto_") and not any(b.get("map") == n for b in devices)
    ]:
        del maps[n]
    if not maps:
        del doc["device_maps"]
    if not devices:
        del doc["devices"]
    rec["maps"] = sorted(k for k in doc.get("device_maps", {}) if k.startswith("auto_"))
    if unmatched:
        rec["notes"].append(
            f"fill patterns that matched no device: {', '.join(unmatched)}"
        )
    rec["summary"] = {
        k: len(rec[k])
        for k in ("created", "refreshed", "unchanged", "kept", "removed", "unset")
    }
    return rec
