"""Design-rule checks (DRC / ERC) over an assembly: `check(asm, lib=None) -> [Finding]`.

Every finding has a stable UPPER_SNAKE `code`, a severity (error | warning | info), a message, `where` (instance / port /
wire / mate ids) and a concrete `fix` suggestion (a library part id, a fastener id, ...). The rule table (code, severity,
threshold and where the threshold comes from) is in assemblies/README.md, "Design rules"; RULES below lists the same codes.

Everything is computed from part data (ports, params, thrust / stall tables) and from the built MJCF (geometry, masses). No
kernel is needed. Design margins that are not facts about a part are named constants at the top of this file, each with its
source or the statement that it is a design margin.
"""

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path

import numpy as np

from .assembly import UNITS, Assembly, Ctx, load
from .build import _mate_spec, compute
from .drc_pins import pin_rules
from .library import DesignError, Library, genders_ok, types_ok
from .mass import _STL, mesh_props
from .snap import (
    ENGAGEMENT_FACTOR,
    EPS_LEN,
    incompatible,
    select_fasteners,
    thread_of,
)

G = 9.80665  # m/s^2, standard gravity
# ---- design margins (not facts about a part) -------------------------------------------------------------------------
TW_WARN = 2.0  # thrust-to-weight below this: warning. Multirotor community rule of thumb (2:1 minimum), e.g. the ArduPilot
#                forum thread "Minimum Thrust to Weight Ratio" https://discuss.ardupilot.org/t/minimum-thrust-to-weight-ratio/50980
TW_ERROR = 1.3  # below this: error. DESIGN MARGIN: hover then needs 1 / 1.3 = 77 % of the maximum thrust, 23 % is left for control
USABLE_FRACTION = 0.8  # flight time uses 80 % of the pack capacity. DESIGN MARGIN: the usual "land with 20 % left" LiPo practice
TRIM_WARN = 0.2  # CG offset warning when trimming it takes more than 20 % of the per-motor thrust headroom (DESIGN MARGIN)
INTERFERENCE_TOL = 1e-3  # m: overlaps shallower than 1 mm are contact, not interference (DESIGN MARGIN: STL / print tolerance)
CG_DIFF_WARN = 0.15  # CG offset warning when the hover thrusts must differ from the equal-throttle ones by more than 15 % of hover
#                      thrust at some motor (DESIGN MARGIN; datum: a 50 mm CG shift on an X quad needs a 16-17 % motor speed difference,
#                      E-MASS, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12736611/ -- speed, i.e. ~2x that in thrust)
PROP_GAP_FRAC = 0.05  # prop tip clearance (to another disc / any collision geometry) below 5 % of the prop diameter: warning. DESIGN
#                       MARGIN (no standard: builders quote 3 mm to 35 mm, ardupilot forum "Recommended minimum prop tip clearance")
GROUND_GAP_FRAC = 0.05  # lowest disc point above the lowest collision point of the airframe below 5 % of the prop diameter: warning
#                         (DESIGN MARGIN: a prop strike on a slightly tilted landing). Ground effect info: Cheeseman & Bennett 1955,
#                         thrust ratio 1 / (1 - (R / 4 z)^2), z = height of the disc above the ground
GROUND_EFFECT_INFO = 0.05  # report the Cheeseman-Bennett thrust gain at rest height when it is above 5 % (z < 1.15 R)
COPPER_LOSS_INFO = 0.20  # copper loss above 20 % of the electrical input at hover: THERMAL says so (DESIGN MARGIN: leaves < 80 %
#                          efficiency before iron, friction and ESC losses)
T_AMB = 25.0  # degC ambient for the winding estimate: the convention of assemblies/*.toml (their r_th derivations) and the kernel t_amb
ALPHA_CU = 0.00393  # 1/K, copper resistance temperature coefficient at 20 degC (R(T) = R20 (1 + a (T - 20)); same value as the r_th derivations)
SERVO_MARGIN = 2.0  # stall / worst static holding torque below 2: warning. Servo sizing guidance for vertical (gravity) loads: safety
#                     factor 1.5-2 on the static torque (Kpower servo-selection notes, https://www.kpower.com/insight_servo/7642.html);
#                     DESIGN MARGIN: the top of that range, because a servo holding near stall overheats and the pose grid is a lower bound
MAX_POSES = 20000  # torque pose grid: n samples per joint with n ** joints <= MAX_POSES (n between 2 and 7)
SERVO_ID_MAX = 252  # highest usable servo bus ID: Feetech STS 0-253, Dynamixel 0-252 (254 = broadcast); the stricter of the two
REACH_TOL = 0.005  # m: an ARM_REACH target counts as reached within 5 mm. DESIGN MARGIN: a gripper is not a point, and the STS3215's
#                    own backlash (0.5 deg) is 2-3 mm at the end of a 0.3 m arm
REACH_STARTS, REACH_ITERS = (
    24,
    80,
)  # IK: seeded restarts x damped-least-squares steps (the first start is the range middle)
REACH_DAMPING = 0.05  # m: damping of the IK step (Levenberg-Marquardt lambda, in metres of end-effector error)
REACH_CONVERGED = 1e-4  # m: an IK run stops at 0.1 mm

RULES = {  # code -> (default severity, what it checks); the long form is the README table
    "BUILD_FAILED": (
        "error",
        "the assembly does not build for a reason no other rule explains",
    ),
    "CHECKS_SKIPPED": (
        "info",
        "a rule group was skipped because the model could not be built",
    ),
    "DRC_INTERNAL_ERROR": ("warning", "a rule crashed (bug); the others still ran"),
    "MATE_INVALID": (
        "error",
        "a mate that cannot be formed (unknown or offboard instance / port, type or gender mismatch, ambiguous)",
    ),
    "FLOATING_INSTANCE": (
        "error",
        "an on-board instance not connected to the root by mates",
    ),
    "PORT_UNMATED_REQUIRED": (
        "error",
        "a mech port the part marks `required = true` that no mate uses",
    ),
    "FASTENER_THREAD": ("error", "fastener thread does not match the mated ports"),
    "FASTENER_SHORT": (
        "warning",
        "fastener shorter than stack + engagement (the builder's own code)",
    ),
    "FASTENER_GRIP": (
        "warning",
        "screw longer than stack + tapped depth: it bottoms out before it clamps",
    ),
    "FASTENER_UNAVAILABLE": ("error", "no fastener / nut can be selected for the mate"),
    "INTERFERENCE": (
        "error",
        "geometry of two instances that are not mated to each other overlaps at rest",
    ),
    "PROP_OVERLAP": ("error", "two rotor discs overlap in the plane of rotation"),
    "PROP_FRAME_CLASH": ("error", "a rotor disc sweeps through other geometry"),
    "PROP_CLEARANCE": (
        "warning",
        "prop tip within 5 % of D of another disc / geometry, or the disc too near the ground (info: ground-effect gain)",
    ),
    "CG_OFFSET": (
        "warning",
        "centre of gravity is off the thrust centre (needs trim thrust)",
    ),
    "DEVICE_UNSET": (
        "error",
        "a simulation device binding lacks a field the device validation needs (the run would fail); the fix names the part param or `set` that supplies it",
    ),
    "THERMAL": (
        "info",
        "motor copper loss at hover (and winding temperature when a device gives r_th)",
    ),
    "SERVO_TORQUE_MARGIN": (
        "warning",
        "servo stall torque less than 2x the worst static holding torque",
    ),
    "SERVO_ID_CONFLICT": (
        "error",
        "two servos on one bus declare the same servo_id, or an ID outside 0..252",
    ),
    "SERVO_ID_UNSET": (
        "info",
        "several servos on one bus and some declare no servo_id (all factory-new servos answer to the same ID)",
    ),
    "SERVO_BAUD_MISMATCH": (
        "error",
        "servos on one bus have different baud rates (the `baud` override, else the part's baud_default)",
    ),
    "SERVO_BUS_CURRENT": (
        "warning",
        "all servos of a bus at stall draw more than its driver board's max_continuous_current (error when the holding current does)",
    ),
    "SERVO_CHAIN_LENGTH": (
        "warning",
        "more servos than the board drives (error), or the cable drop to the farthest servo at all-stall leaves it below its minimum voltage",
    ),
    "ARM_REACH": (
        "warning",
        "a `reach` / `workspace` target the arm's numerical IK cannot reach within the joint limits (info when all points are reachable)",
    ),
    "WIRE_INVALID": (
        "error",
        "a wire whose ends / part cannot be resolved or whose port kinds differ",
    ),
    "WIRE_CONNECTOR_MISMATCH": (
        "error",
        "connector type or gender at a wire end does not match the port",
    ),
    "WIRE_RATING_UNKNOWN": (
        "info",
        "a wire part with no current rating: the wire cannot be checked",
    ),
    "VOLTAGE_OVER_MAX": ("error", "net voltage above a port's v_max"),
    "VOLTAGE_BELOW_MIN": ("warning", "net voltage below a port's v_min"),
    "CURRENT_OVER_RATING": (
        "error",
        "sustained current above a port / wire rating, or peak above its burst rating",
    ),
    "CURRENT_PEAK_OVER_RATING": (
        "warning",
        "worst-case (peak) current above a continuous rating",
    ),
    "BATTERY_C_RATING": (
        "warning",
        "battery C rating times capacity below the load current",
    ),
    "PSU_OVERLOAD": ("warning", "supply rated current below the load current"),
    "FLOATING_PORT": ("warning", "a power port with no wire"),
    "NET_NO_SOURCE": ("error", "a wired net with no source"),
    "THRUST_TO_WEIGHT": (
        "info",
        "maximum static thrust over weight (warning < 2.0, error < 1.3)",
    ),
    "HOVER_THROTTLE": (
        "info",
        "hover throttle and current from the motor thrust table",
    ),
    "FLIGHT_TIME": ("info", "pack capacity x usable fraction / hover current"),
    "SERVO_TORQUE": (
        "info",
        "gravity torque at the worst static pose vs the servo stall / rated torque",
    ),
    "PAYLOAD_LIMIT": (
        "info",
        "extra end-effector mass that keeps every servo within stall torque",
    ),
    "PIN_ALREADY_USED": ("error", "one connector pin is on two conductors"),
    "PIN_KIND_MISMATCH": (
        "error",
        "a conductor joins unlike pins (power to ground, signal to power); a no-connect pin is a warning",
    ),
    "PIN_VOLTAGE_MISMATCH": (
        "error",
        "a driving pin's level is outside the other pin's accepted range (a level difference between signals is a warning)",
    ),
    "PIN_TWO_OUTPUTS": ("error", "two signal outputs drive one net"),
    "PIN_GND_MISSING": (
        "warning",
        "a multi-conductor DC wire carries power or signal but no ground return",
    ),
    "PIN_ALT_CONFLICT": (
        "error",
        "one peripheral function claimed by two pins, or a wire's TX / RX on different peripherals",
    ),
    "PIN_FUNCTION_MISSING": (
        "warning",
        "a multiplexed pin is wired to a device pin whose role (TX, SDA, ...) it cannot provide",
    ),
}
_RANK = {"error": 0, "warning": 1, "info": 2}


@dataclass
class Finding:
    code: str
    severity: str  # error | warning | info
    message: str
    where: list = field(default_factory=list)  # instance / port / wire / mate ids
    fix: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _add(F, code, sev, msg, where=(), fix=""):
    F.append(Finding(code, sev, msg, [str(w) for w in where], fix))


def _n(x):
    """Numeric value of a provenance entry (or None)."""
    v = x.get("value") if isinstance(x, dict) else None
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _default_parts() -> Path:
    repo = Path(__file__).resolve().parents[3] / "parts"
    return repo if repo.is_dir() else Path("parts")


def _ids(xs, k=3) -> str:
    xs = list(dict.fromkeys(xs))[:k]
    return ", ".join(xs) if xs else "(none in the library)"


def _is_cable(part) -> bool:
    return part.category == "wire" or part.category.startswith("wire.")


# ---- mechanical, no geometry --------------------------------------------------------------------------------------------
def _alts(lib, inst, port, key, need):
    """Library parts of the same category with a power port of the same type / gender / role rated >= need for `key`."""
    out = []
    for pid, p in sorted(lib.parts.items()):
        if pid == inst.part.id or p.category != inst.part.category:
            continue
        if any(
            q.get("kind") == "power"
            and q.get("type") == port.get("type")
            and q.get("gender") == port.get("gender")
            and q.get("role") == port.get("role")
            and (_n(q.get(key)) or 0.0) >= need
            for q in p.ports
        ):
            out.append(pid)
    return out


def _compatible_ids(lib, port, categories=None):
    return [
        p.id
        for p, _ in lib.compatible(port.get("type", ""), port.get("gender", ""))
        if categories is None or any(p.category.startswith(c) for c in categories)
    ]


def _auto_fastener(lib, name, spec, nut_id):
    try:
        rec = select_fasteners(lib, name, spec.pa, spec.pb, "auto", nut_id)[0]
    except DesignError:
        return ""
    return f'{rec["part"]} x{rec["count"]} (fasteners = "auto" selects it)'


def _mechanical(ctx, F):
    """Mates, floating instances, required ports, fasteners. Returns {mate name: MateSpec} of the valid mates."""
    lib, doc = ctx.lib, ctx.doc
    onboard = [i for i, x in ctx.insts.items() if not x.offboard]
    root = next(i for i, x in ctx.insts.items() if x.root)
    specs, mdict, edges, used = {}, {}, [], set()
    for m in doc.get("mates", []):
        name = m.get("name", f"{m['a']}--{m['b']}")
        edges.append((m["a"].partition(".")[0], m["b"].partition(".")[0]))
        try:
            spec = _mate_spec(ctx, m)
        except DesignError as e:
            fix = "name two compatible mech ports (or fix the part ids)"
            try:
                iid, _, pn = m["a"].partition(".")
                if pn:
                    ids = _compatible_ids(lib, ctx.inst(iid).part.port(pn))
                    fix = f"parts with a port that mates {m['a']}: {_ids(ids)}"
            except DesignError:
                pass
            _add(F, "MATE_INVALID", "error", str(e), [name, m["a"], m["b"]], fix)
            continue
        specs[name], mdict[name] = spec, m
        used |= {(spec.a, spec.pa["name"]), (spec.b, spec.pb["name"])}
    # floating instances: BFS over every mate (also the invalid ones: their own finding already explains them);
    # a free-placed instance (pos / rpy) is welded to the root
    edges += [(root, i) for i, x in ctx.insts.items() if x.placed and not x.root]
    adj = {i: set() for i in ctx.insts}
    for a, b in edges:
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)
    seen, todo = {root}, [root]
    while todo:
        for n in adj[todo.pop()] - seen:
            seen.add(n)
            todo.append(n)
    for iid in onboard:
        if iid in seen:
            continue
        cands = []
        for oid in sorted(seen):
            for pb in ctx.insts[oid].part.ports:
                if pb.get("kind") != "mech" or (oid, pb["name"]) in used:
                    continue
                for pa in ctx.insts[iid].part.ports:
                    if pa.get("kind") == "mech" and incompatible(pa, pb) is None:
                        cands.append(f"{oid}.{pb['name']} <-> {iid}.{pa['name']}")
        _add(
            F,
            "FLOATING_INSTANCE",
            "error",
            f"instance '{iid}' ({ctx.insts[iid].part.id}) is not connected to the root '{root}' by any mate",
            [iid],
            "add a mate, free compatible port pairs: " + _ids(cands),
        )
    for iid in onboard:
        for p in ctx.insts[iid].part.ports:
            if (
                p.get("kind") == "mech"
                and p.get("required")
                and (iid, p["name"]) not in used
            ):
                _add(
                    F,
                    "PORT_UNMATED_REQUIRED",
                    "error",
                    f"required mech port {iid}.{p['name']} (type {p.get('type')}) has no mate",
                    [f"{iid}.{p['name']}"],
                    f"mate it to a compatible port: {_ids(_compatible_ids(lib, p))}",
                )
    # fasteners
    for name, spec in specs.items():
        f = mdict[name].get("fasteners")
        if f is None:
            continue
        nut_id = mdict[name].get("nut") or (
            f.get("nut") if isinstance(f, dict) else None
        )
        auto = _auto_fastener(lib, name, spec, nut_id)
        ta, tb = thread_of(spec.pa), thread_of(spec.pb)
        bad = None
        if ta and tb and ta != tb:
            bad = f"thread mismatch {ta.upper()} vs {tb.upper()} between {spec.a}.{spec.pa['name']} and {spec.b}.{spec.pb['name']}"
        elif isinstance(f, dict):
            try:
                scr, th = lib.get(f["part"]), ta or tb
                if th and not scr.has_port(f"thread.{th}", "male"):
                    bad = f"fastener {scr.id} is not a {th.upper()} screw but the mated ports are {th.upper()}"
            except DesignError as e:
                _add(
                    F,
                    "FASTENER_UNAVAILABLE",
                    "error",
                    f"mate '{name}': {e}",
                    [name],
                    auto,
                )
                continue
        if bad:
            _add(
                F,
                "FASTENER_THREAD",
                "error",
                f"mate '{name}': {bad}",
                [name, f["part"] if isinstance(f, dict) else "auto"],
                auto,
            )
            continue
        try:
            rec, _, _, _, warns = select_fasteners(
                lib, name, spec.pa, spec.pb, f, nut_id
            )
        except DesignError as e:
            _add(
                F,
                "FASTENER_UNAVAILABLE",
                "error",
                str(e),
                [name],
                "pick a fastener part that fits, or add the missing nut / screw to the library",
            )
            continue
        for code, msg in warns:
            _add(
                F,
                code,
                "warning",
                msg,
                [name, rec["part"]],
                f"use a screw of length >= {rec['L_min']:.4g} m: {auto or 'none in the library'}",
            )
        # too long: a blind tapped hole of known depth. The screw must reach L_min (FASTENER_SHORT) and stop before the hole bottom,
        # otherwise it bottoms out and leaves the stack unclamped. Through-bolts: the tip position is not modelled, so no check.
        tapped = [p for p in (spec.pa, spec.pb) if p.get("tapped")]
        depth = tapped[0].get("thread_depth") if tapped else None
        if depth is not None:
            room = rec["stack"] + float(depth)
            if rec["length"] > room + EPS_LEN:
                d = float(rec["thread"][1:]) / 1000.0
                _add(
                    F,
                    "FASTENER_GRIP",
                    "warning",
                    f"mate '{name}': {rec['part']} length {rec['length'] * 1000:.1f} mm > clamped stack {rec['stack'] * 1000:.1f} mm + tapped depth "
                    f"{float(depth) * 1000:.1f} mm = {room * 1000:.1f} mm: it bottoms out {1000 * (rec['length'] - room):.1f} mm before the head clamps "
                    f"(the hole offers {float(depth) / d:.1f} x d of thread; rule of thumb >= 1 x d in steel, 2 x d in aluminium, Nord-Lock "
                    f"'Thread engagement in a tapped hole'; the builder's L_min uses {ENGAGEMENT_FACTOR} x d)",
                    [name, rec["part"]],
                    f"use a screw of length {rec['L_min'] * 1000:.1f} to {room * 1000:.1f} mm, or a deeper tapped hole",
                )
    return specs


# ---- wires and nets -------------------------------------------------------------------------------------------------------
class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _wires(ctx, F):
    """Resolve [[wires]] (own light check, so a bad wire is a finding, not an exception)."""
    out = []
    for i, w in enumerate(ctx.doc.get("wires", [])):
        name = w.get("name", f"wire{i}")
        try:
            ends = []
            for k in ("from", "to"):
                iid, _, pn = w[k].partition(".")
                inst = ctx.inst(iid, f"wire '{name}'")
                ends.append((inst, inst.part.port(pn)))
            (ia, pa), (ib, pb) = ends
            part = ctx.lib.get(w["part"])
            if pa.get("kind") not in ("power", "signal") or pa.get("kind") != pb.get(
                "kind"
            ):
                raise DesignError(
                    f"wire '{name}': both ends must be power ports or both signal ports"
                )
        except DesignError as e:
            _add(
                F, "WIRE_INVALID", "error", str(e), [name], "fix the wire's ends / part"
            )
            continue
        bad = None
        if _is_cable(part):
            leads = part.ports[:2]
            for e, p, side in zip(leads, (pa, pb), ("from", "to")):
                if e.get("type") != "wire-lead" and not (
                    types_ok(e.get("type", ""), p.get("type", ""))
                    and genders_ok(e.get("gender", ""), p.get("gender", ""))
                ):
                    bad = f"cable {part.id} ends in {e.get('type')} {e.get('gender')} but the {side} port is {p.get('type')} {p.get('gender')}"
        else:
            why = incompatible(pa, pb)
            if why:
                bad = f"{why} ({w['from']} <-> {w['to']})"
            else:
                for p in (pa, pb):
                    if not any(
                        types_ok(q.get("type", ""), p.get("type", ""))
                        for q in part.ports
                    ):
                        bad = f"connector part {part.id} has no {p.get('type')} port"
        if bad:
            fix = _ids(
                _compatible_ids(ctx.lib, pa, ("wire", "connector"))
                if _is_cable(part)
                else _compatible_ids(ctx.lib, pa, ("connector",))
            )
            _add(
                F,
                "WIRE_CONNECTOR_MISMATCH",
                "error",
                f"wire '{name}': {bad}",
                [name, w["from"], w["to"]],
                f"use a part that mates both ends, e.g. {fix}",
            )
        out.append(
            {
                "i": i,
                "name": name,
                "ia": ia,
                "pa": pa,
                "ib": ib,
                "pb": pb,
                "part": part,
                "kind": pa["kind"],
                "w": w,
            }
        )
    return out


class _Net:
    """Power ports -> nodes (passthrough ports of one instance and internal `wire` bindings are one node) -> nets (nodes
    joined by explicit wires)."""

    def __init__(self, ctx, wires):
        self.ctx, self.node, self.net = ctx, _UF(), _UF()
        self.ports = {}
        for iid, inst in ctx.insts.items():
            for p in inst.part.ports:
                if p.get("kind") == "power":
                    self.ports[(iid, p["name"])] = (inst, p)
                    self.node.find((iid, p["name"]))
            pt = [
                p["name"]
                for p in inst.part.ports
                if p.get("kind") == "power" and p.get("role") == "passthrough"
            ]
            for n in pt[1:]:
                self.node.union((iid, pt[0]), (iid, n))
        devs = ctx.doc.get("devices", [])
        for b in devs:
            if (
                b["type"] == "wire"
                and b.get("a") != "ground"
                and b.get("b") != "ground"
            ):
                ka, kb = (
                    (b.get("instance"), b.get("a")),
                    (b.get("instance"), b.get("b")),
                )
                if ka in self.ports and kb in self.ports:
                    self.node.union(ka, kb)
        self.wires = [w for w in wires if w["kind"] == "power"]
        for k in self.ports:
            self.net.union(k, self.node.find(k))
        for w in self.wires:
            w["ka"], w["kb"] = (
                (w["ia"].id, w["pa"]["name"]),
                (w["ib"].id, w["pb"]["name"]),
            )
            self.net.union(w["ka"], w["kb"])
        # sources
        self.dev_sources = {}  # port key -> device binding (battery / psu)
        for b in devs:
            k = (b.get("instance"), b.get("port"))
            if b["type"] in ("battery", "psu") and k in self.ports:
                self.dev_sources[k] = b
        self.role_sources = sorted(
            k for k, (_, p) in self.ports.items() if p.get("role") == "source"
        )

    def nroot(self, key):
        return self.net.find(key)

    def members(self, key):
        r = self.nroot(key)
        return [k for k in self.ports if self.net.find(k) == r]

    def sources(self, key):
        """Source port keys of the net holding `key`: battery / psu devices, else role = source ports."""
        m = self.members(key)
        dev = sorted(k for k in m if k in self.dev_sources)
        return dev or [k for k in self.role_sources if k in m]

    def voltage(self, key):
        """(v_max, v_nominal) of the net holding `key`, or None when no source states a voltage."""
        vmax = []
        vnom = []
        for k in self.sources(key):
            inst, p = self.ports[k]
            up_key = (k[0], p["v_from"]) if p.get("v_from") else None
            if up_key in self.ports and up_key != k:
                # a converter-less output (an ESC's motor leads) swings no higher than the net on its input port `v_from`
                up = self.voltage(up_key)
                if up:
                    hi = _n(p.get("v_max"))
                    vmax.append(min(hi, up[0]) if hi else up[0])
                    vnom.append(up[1])
                    continue
            hi = (
                _n(p.get("v_max"))
                or inst.value("v_full")
                or inst.value("v_out_nominal")
            )
            if hi is None:
                continue
            vmax.append(hi)
            vnom.append(inst.value("v_nominal") or inst.value("v_out_nominal") or hi)
        return (max(vmax), max(vnom)) if vmax else None

    def flows(self, loads):
        """loads {port key: (peak A, sustained A or None)} -> ({wire index: (peak, sustained)}, {source key: (peak, sus)}).
        Tree flow from the first source of each net (a net with several sources is split by that one; a loop edge carries 0)."""
        node_load = {}
        for k, (pk, su) in loads.items():
            r = self.node.find(k)
            a, b = node_load.get(r, (0.0, 0.0))
            node_load[r] = (a + pk, None if (b is None or su is None) else b + su)
        adj = {}
        for w in self.wires:
            ra, rb = self.node.find(w["ka"]), self.node.find(w["kb"])
            if ra != rb:
                adj.setdefault(ra, []).append((rb, w))
                adj.setdefault(rb, []).append((ra, w))
        cur, total, done = {}, {}, set()
        for key in [*self.dev_sources, *self.role_sources]:
            r = self.node.find(key)
            if r in done or self.nroot(key) in done:
                continue
            done.add(self.nroot(key))
            order, par, seen = [], {r: None}, {r}
            todo = [r]
            while todo:
                n = todo.pop(0)
                order.append(n)
                for m, w in adj.get(n, ()):
                    if m not in seen:
                        seen.add(m)
                        par[m] = (n, w)
                        todo.append(m)
            acc = {n: node_load.get(n, (0.0, 0.0)) for n in order}
            for n in reversed(order):
                if par[n] is not None:
                    p, w = par[n]
                    cur[w["i"]] = acc[n]
                    a, b = acc[p]
                    acc[p] = (
                        a + acc[n][0],
                        None if (b is None or acc[n][1] is None) else b + acc[n][1],
                    )
            total[key] = acc[r]
        return cur, total


def _voltage_rules(ctx, net, F):
    for k, (inst, p) in sorted(net.ports.items()):
        if p.get("role") == "source":
            continue
        v = net.voltage(k)
        if v is None:
            continue
        vmax, vnom = v
        src = ", ".join(f"{a}.{b}" for a, b in net.sources(k))
        hi, lo = _n(p.get("v_max")), _n(p.get("v_min"))
        if hi is not None and vmax > hi + 1e-9:
            alts = _alts(ctx.lib, inst, p, "v_max", vmax)
            _add(
                F,
                "VOLTAGE_OVER_MAX",
                "error",
                f"{k[0]}.{k[1]} is rated v_max {hi:g} V but its net ({src}) reaches {vmax:g} V",
                [f"{k[0]}.{k[1]}", src],
                f"replace {inst.part.id} by a part with the same port rated >= {vmax:g} V: {_ids(alts)}; or use a lower-voltage source",
            )
        if lo is not None and vnom < lo - 1e-9:
            _add(
                F,
                "VOLTAGE_BELOW_MIN",
                "warning",
                f"{k[0]}.{k[1]} needs v_min {lo:g} V but its net ({src}) is {vnom:g} V nominal",
                [f"{k[0]}.{k[1]}", src],
                "raise the supply voltage or use a part whose v_min is below it",
            )


def _topology_rules(ctx, net, F):
    wired = {net.node.find(w["ka"]) for w in net.wires} | {
        net.node.find(w["kb"]) for w in net.wires
    }
    for iid, inst in ctx.insts.items():
        if inst.offboard:
            continue
        mine = [k for k in net.ports if k[0] == iid]
        loose = [
            k
            for k in mine
            if net.node.find(k) not in wired and net.ports[k][1].get("role") != "source"
        ]
        if not loose:
            continue
        powered = [
            k
            for k in mine
            if net.node.find(k) in wired and net.ports[k][1].get("role") != "source"
        ]
        for k in loose:
            _add(
                F,
                "FLOATING_PORT",
                "warning" if powered else "error",
                f"power port {k[0]}.{k[1]} ({net.ports[k][1].get('type')}, {net.ports[k][1].get('role', 'no role')}) has no wire"
                + ("" if powered else f"; {iid} has no powered port at all"),
                [f"{k[0]}.{k[1]}"],
                f"add a wire to it, e.g. from a source port with type {net.ports[k][1].get('type')}",
            )
    seen = set()
    for w in net.wires:
        r = net.nroot(w["ka"])
        if r in seen:
            continue
        seen.add(r)
        if not net.sources(w["ka"]):
            ports = sorted(f"{a}.{b}" for a, b in net.members(w["ka"]))
            _add(
                F,
                "NET_NO_SOURCE",
                "error",
                f"the net through wire '{w['name']}' has no source (no battery / psu device, no source-role port): {', '.join(ports)}",
                [w["name"], *ports[:6]],
                "wire a battery or supply into the net",
            )


# ---- propulsion data ------------------------------------------------------------------------------------------------------
def _thrust_rows(part):
    t = part.raw.get("tables", {}).get("thrust_test")
    if not t or "current_A" not in t.get("columns", []):
        return None
    c = {n: i for i, n in enumerate(t["columns"])}
    if "thrust_N" in c:
        tn = lambda r: float(r[c["thrust_N"]])
    elif "thrust_g" in c:
        tn = lambda r: float(r[c["thrust_g"]]) * G / 1000.0
    else:
        return None
    rows = [
        {
            "thr": r[c["throttle_pct"]] / 100.0 if "throttle_pct" in c else None,
            "V": float(r[c["voltage_V"]]) if "voltage_V" in c else None,
            "I": float(r[c["current_A"]]),
            "T": tn(r),
        }
        for r in t["rows"]
    ]
    return sorted(rows, key=lambda r: r["T"]) or None


def _units(ctx):
    """One per ESC device with a `motor`: the ESC, motor, thrust table, and the rotor device on the same joint."""
    devs = ctx.doc.get("devices", [])
    rotors = [d for d in devs if d["type"] == "rotor"]
    out = []
    for d in devs:
        if (
            d["type"] != "esc"
            or d.get("motor") not in ctx.insts
            or d.get("instance") not in ctx.insts
        ):
            continue
        mot = ctx.insts[d["motor"]]
        rot = next(
            (r for r in rotors if d.get("joint") and r.get("joint") == d["joint"]), None
        )
        out.append(
            {
                "dev": d,
                "esc": ctx.insts[d["instance"]],
                "motor": mot,
                "rotor": rot,
                "prop": ctx.insts.get(rot["instance"]) if rot else None,
                "rows": _thrust_rows(mot.part),
                "key": (d["instance"], d.get("port", "power_in")),
            }
        )
    return out


def _scaled(u, vnom):
    """Table rows scaled to pack voltage: thrust and current both x (V / V_row)^2 (rpm ~ V, thrust ~ rpm^2, P ~ rpm^3, I = P / V).
    First-order; rows without a voltage column are used as measured."""
    out = []
    for r in u["rows"]:
        s = (vnom / r["V"]) ** 2 if r["V"] and vnom else 1.0
        out.append({**r, "T": r["T"] * s, "I": r["I"] * s})
    return out


def _motor_curve(kt, r, i0, kq, kT):
    """Static-thrust motor model, T = kT w^2, shaft torque Q = kq T (C_Q / C_T x D, same for every speed), constant friction I0,
    ideal ESC. Returns f(T, V) -> (pack current A, duty = motor voltage / V)."""

    def f(T, V):
        im = kq * T / kt + i0  # Q = Kt (Im - I0)
        vm = kt * np.sqrt(T / kT) + im * r  # w = sqrt(T / kT), Vm = Ke w + Im R
        return vm * im / V, vm / V

    return f


@cache
def _fit_motor(kv, r, i0, rows):
    """Fit (kq, kT) of `_motor_curve` to table rows ((T, I, V, throttle) tuples) by least squares on log current plus a 0.25-weight
    log duty term (duty ~ throttle: current alone cannot separate shaft power from friction; the vendor throttle is not exactly
    duty, hence the low weight), log grid refined 5x. Returns (curve, rms log-current error). ponytail: a 2-parameter grid fit, not a solver; fine for <= ~10 rows."""
    kt = 60.0 / (2 * math.pi * kv)
    T, I, V, thr = (np.array(c, dtype=float) for c in zip(*rows))

    def cost(a, b):
        mi, d = _motor_curve(kt, r, i0, math.exp(a), math.exp(b))(T, V)
        return float(
            np.mean(np.log(mi / I) ** 2) + 0.25 * np.mean(np.log(d / thr) ** 2)
        )

    ca, cb, sa, sb = math.log(3e-3), math.log(1e-6), 3.5, 5.0
    for _ in range(5):
        best = min(
            (cost(a, b), a, b)
            for a in np.linspace(ca - sa, ca + sa, 25)
            for b in np.linspace(cb - sb, cb + sb, 25)
        )
        _, ca, cb = best
        sa, sb = sa / 5, sb / 5
    curve = _motor_curve(kt, r, i0, math.exp(ca), math.exp(cb))
    return curve, float(np.sqrt(np.mean(np.log(curve(T, V)[0] / I) ** 2)))


def _below_table(u, rows, t_req, vnom):
    """(throttle or None, current A, how) at a thrust below the lowest table row (rows scaled to vnom). Never a linear
    extrapolation: the physics fitted to the table rows (motor kv / R / I0 + T ~ n^2, Q ~ n^2, anchored to the lowest row), or,
    when the part lacks kv / R / row voltages or the fit is poor (> 15 % rms), the scaling law T ~ n^2, P ~ n^3 from the lowest row."""
    m, r0 = u["motor"], rows[0]
    kv, rw, i0 = m.value("kv"), m.value("r_winding"), m.value("i0") or 0.0
    fit = None
    if kv and rw and vnom and all(x["V"] and x["thr"] for x in u["rows"]):
        fit = _fit_motor(
            kv,
            rw,
            i0,
            tuple((x["T"], x["I"], x["V"], x["thr"]) for x in u["rows"][:3]),
        )  # the 3 lowest rows: the ones nearest the extrapolation (the model is not exact up to full throttle)
    if fit and fit[1] < 0.15:
        curve = fit[0]
        i_m, d = curve(np.array([t_req, r0["T"]]), vnom)
        cur = float(
            r0["I"] * i_m[0] / i_m[1]
        )  # anchored: continuous with the lowest row
        thr = None if r0["thr"] is None else float(r0["thr"] * d[0] / d[1])
        return thr, cur, "motor model fitted to the table (kv, R, I0, T ~ n^2, Q ~ n^2)"
    s = t_req / r0["T"]
    thr = (
        None if r0["thr"] is None else r0["thr"] * math.sqrt(s)
    )  # n ~ sqrt(T), throttle ~ n
    return thr, r0["I"] * s**1.5, "scaling law T ~ n^2, P ~ n^3 from the lowest row"


def _hover(u, t_req, vnom):
    """(throttle or None, current A, extrapolation note or "") at thrust t_req [N]: linear between the scaled table rows,
    the fitted physics below the first row (`_below_table`), None above the last."""
    rows = _scaled(u, vnom)
    if t_req > rows[-1]["T"]:
        return None
    if t_req < rows[0]["T"]:
        return _below_table(u, rows, t_req, vnom)
    ts = [r["T"] for r in rows]
    cur = float(np.interp(t_req, ts, [r["I"] for r in rows]))
    thr = None
    if all(r["thr"] is not None for r in rows):
        thr = float(np.interp(t_req, ts, [r["thr"] for r in rows]))
    return thr, cur, ""


def _performance(ctx, net, units, mass, F):
    """Thrust-to-weight, hover throttle / current, flight time. Fills u["Tmax"], u["Ipeak"], u["Ihover"]."""
    n = 0
    for u in units:
        v = net.voltage(u["key"])
        u["vmax"], u["vnom"] = v if v else (None, None)
        if not u["rows"]:
            _add(
                F,
                "HOVER_THROTTLE",
                "info",
                f"motor {u['motor'].id} ({u['motor'].part.id}) has no thrust_test table: no thrust / current model",
                [u["motor"].id],
            )
            continue
        rows = u["rows"]
        u["Tmax"] = _scaled(u, u["vnom"])[-1]["T"]
        imax = max(rows, key=lambda r: r["I"])
        s = max(1.0, (u["vmax"] / imax["V"]) ** 2) if (u["vmax"] and imax["V"]) else 1.0
        u["Ipeak"] = (
            imax["I"] * s
        )  # worst case: table maximum, scaled up when the pack is fuller than the table voltage
        n += 1
    ok = [u for u in units if "Tmax" in u]
    if not ok or mass is None:
        return
    w_n = mass * G
    tmax = sum(u["Tmax"] for u in ok)
    tw = tmax / w_n
    names = [u["motor"].id for u in ok]
    msg = f"max static thrust {tmax:.1f} N (motor tables scaled to pack nominal voltage) / weight {w_n:.2f} N ({mass:.3f} kg) = {tw:.2f}"
    if tw < TW_WARN:
        per = TW_WARN * w_n / len(ok)
        cands = []
        for p in ctx.lib.by_category("motor.bldc"):
            rows = _thrust_rows(p)
            if rows and rows[-1]["T"] >= per:
                cands.append((rows[-1]["T"], p.id))
        _add(
            F,
            "THRUST_TO_WEIGHT",
            "error" if tw < TW_ERROR else "warning",
            msg
            + f" (< {TW_ERROR if tw < TW_ERROR else TW_WARN}; margins: 2.0 community rule of thumb, 1.3 leaves 23 % thrust for control)",
            names,
            f"cut {max(0.0, mass - tmax / G / TW_WARN):.2f} kg, or fit motors with >= {per:.1f} N max thrust each: {_ids([i for _, i in sorted(cands)])}",
        )
    else:
        _add(F, "THRUST_TO_WEIGHT", "info", msg, names)
    # hover: equal throttle fraction f = W / sum(Tmax); each motor makes f x its max thrust
    f = w_n / tmax
    tot_i, thrs, parts, extrap = 0.0, [], [], set()
    for u in ok:
        h = _hover(u, f * u["Tmax"], u["vnom"])
        if h is None:
            continue
        u["Ihover"], u["Hthr"] = h[1], h[0]
        tot_i += h[1]
        if h[0] is not None:
            thrs.append(h[0])
        parts.append(f"{u['motor'].id} {f * u['Tmax']:.2f} N {h[1]:.2f} A")
        if h[2]:
            extrap.add(h[2])
    note = (
        " [extrapolated: hover thrust is below the motor table's lowest row; "
        + "; ".join(sorted(extrap))
        + "]"
        if extrap
        else ""
    )
    if parts and tw >= 1.0:
        _add(
            F,
            "HOVER_THROTTLE",
            "info",
            "hover: "
            + (
                "throttle "
                + ", ".join(
                    f"{t * 100:.0f} %" for t in sorted(set(round(t, 3) for t in thrs))
                )
                + "; "
                if thrs
                else ""
            )
            + f"total {tot_i:.2f} A ("
            + "; ".join(parts[:4])
            + (" ..." if len(parts) > 4 else "")
            + ")"
            + note,
            names,
        )
        for k in {u["key"] for u in ok if "Ihover" in u}:
            for b in ctx.doc.get("devices", []):
                if b["type"] != "battery":
                    continue
                bk = (b.get("instance"), b.get("port"))
                if bk in net.ports and net.nroot(bk) == net.nroot(k):
                    cap = ctx.insts[b["instance"]].value("capacity_ah")
                    cur = sum(
                        u["Ihover"]
                        for u in ok
                        if "Ihover" in u and net.nroot(u["key"]) == net.nroot(k)
                    )
                    if cap and cur > 0:
                        _add(
                            F,
                            "FLIGHT_TIME",
                            "info",
                            f"{cap:g} Ah x {USABLE_FRACTION:g} usable / {cur:.2f} A hover = {cap * USABLE_FRACTION / cur * 60:.1f} min "
                            "(propulsion current only; avionics and wiring losses not included)"
                            + note,
                            [b["instance"]],
                        )
            break


# ---- geometry: MJCF -> world -------------------------------------------------------------------------------------------------
def _quat_R(q):
    w, x, y, z = np.array(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _vec(s, default=(0.0, 0.0, 0.0)):
    return (
        np.array([float(x) for x in s.split()]) if s else np.array(default, dtype=float)
    )


def _pose(pos, quat):
    T = np.eye(4)
    if quat:
        T[:3, :3] = _quat_R([float(x) for x in quat.split()])
    T[:3, 3] = _vec(pos)
    return T


def _z_to(v):
    v = v / np.linalg.norm(v)
    a = np.cross([0.0, 0.0, 1.0], v)
    s = np.linalg.norm(a)
    if s < 1e-12:
        return np.eye(3) if v[2] > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - v[2]) / s**2)


@cache
def _mesh_info(path):
    """(volume centroid, AABB lo, AABB hi) of a binary STL in its file frame."""
    raw = Path(path).read_bytes()
    n = int.from_bytes(raw[80:84], "little")
    tri = np.frombuffer(raw, dtype=_STL, count=n, offset=84)["v"].astype(np.float64)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    v6 = np.einsum("ij,ij->i", a, np.cross(b, c))
    vol = v6.sum()
    cen = (
        (v6[:, None] * (a + b + c) / 4).sum(0) / vol
        if abs(vol) > 0
        else tri.reshape(-1, 3).mean(0)
    )
    verts = tri.reshape(-1, 3)
    return cen, verts.min(0), verts.max(0)


class _World:
    def __init__(self, xml, mesh_base):
        root = ET.fromstring(xml)
        assets = {m.get("name"): m.get("file") for m in root.iter("mesh")}
        self.bodies, self.geoms, self.idx = [], [], {}

        def walk(el, parent, Tp):
            for b in el.findall("body"):
                Tl = _pose(b.get("pos"), b.get("quat"))
                T = Tp @ Tl
                j = None
                for je in b.findall("joint"):
                    r = je.get("range")
                    j = {
                        "name": je.get("name"),
                        "type": je.get("type", "hinge"),
                        "axis": _vec(je.get("axis"), (0, 0, 1)),
                        "pos": _vec(je.get("pos")),
                        "range": [float(x) for x in r.split()] if r else None,
                    }
                bi = len(self.bodies)
                self.idx[b.get("name")] = bi
                self.bodies.append(
                    {
                        "name": b.get("name"),
                        "parent": parent,
                        "Tl": Tl,
                        "T": T,
                        "joint": j,
                        "geoms": [],
                    }
                )
                for g in b.findall("geom"):
                    kind, size = g.get("type", "sphere"), _vec(g.get("size"))
                    if g.get("fromto"):
                        ft = _vec(g.get("fromto"))
                        a, c = ft[:3], ft[3:]
                        Tg = np.eye(4)
                        Tg[:3, :3], Tg[:3, 3] = (
                            _z_to(c - a)
                            if np.linalg.norm(c - a) > 1e-12
                            else np.eye(3),
                            (a + c) / 2,
                        )
                        size = np.array([size[0], np.linalg.norm(c - a) / 2])
                    else:
                        Tg = _pose(g.get("pos"), g.get("quat"))
                    mesh = None
                    if kind == "mesh":
                        mesh = str((Path(mesh_base) / assets[g.get("mesh")]).resolve())
                    geom = {
                        "name": g.get("name"),
                        "body": bi,
                        "kind": kind,
                        "size": size,
                        "T": Tg,
                        "mass": float(g.get("mass", 0.0)),
                        "collide": g.get("contype") == "1",
                        "mesh": mesh,
                    }
                    self.bodies[bi]["geoms"].append(len(self.geoms))
                    self.geoms.append(geom)
                walk(b, bi, T)

        walk(root.find("worldbody"), None, np.eye(4))
        for b in self.bodies:
            m = sum(self.geoms[g]["mass"] for g in b["geoms"])
            b["mass"] = m
            b["com"] = (
                sum(
                    self.geoms[g]["mass"]
                    * (self.geoms[g]["T"] @ np.append(self._com(self.geoms[g]), 1.0))[
                        :3
                    ]
                    for g in b["geoms"]
                )
                / m
                if m > 0
                else np.zeros(3)
            )

    @staticmethod
    def _com(g):
        return _mesh_info(g["mesh"])[0] if g["kind"] == "mesh" else np.zeros(3)

    def obb(self, gi):
        """World OBB (centre, axes as columns, half extents) of a geom, exact for a box, bounding box otherwise."""
        g = self.geoms[gi]
        T = self.bodies[g["body"]]["T"] @ g["T"]
        k, s = g["kind"], g["size"]
        if k == "box":
            c, h = np.zeros(3), s[:3]
        elif k == "sphere":
            c, h = np.zeros(3), np.full(3, s[0])
        elif k == "cylinder":
            c, h = np.zeros(3), np.array([s[0], s[0], s[1]])
        elif k == "capsule":
            c, h = np.zeros(3), np.array([s[0], s[0], s[1] + s[0]])
        else:
            _, lo, hi = _mesh_info(g["mesh"])
            c, h = (lo + hi) / 2, (hi - lo) / 2
        return T[:3, :3] @ c + T[:3, 3], T[:3, :3], h

    def world_com(self):
        m = sum(b["mass"] for b in self.bodies)
        return sum(
            b["mass"] * (b["T"] @ np.append(b["com"], 1.0))[:3] for b in self.bodies
        ) / m, m


def _obb_depth(a, b):
    """Penetration depth of two OBBs by the separating-axis test (0 when separated)."""
    (c1, R1, h1), (c2, R2, h2) = a, b
    d, best = c2 - c1, math.inf
    axes = [R1[:, i] for i in range(3)] + [R2[:, i] for i in range(3)]
    axes += [np.cross(R1[:, i], R2[:, j]) for i in range(3) for j in range(3)]
    for ax in axes:
        nrm = np.linalg.norm(ax)
        if nrm < 1e-9:
            continue
        ax = ax / nrm
        ov = h1 @ np.abs(R1.T @ ax) + h2 @ np.abs(R2.T @ ax) - abs(d @ ax)
        if ov <= 0:
            return 0.0
        best = min(best, ov)
    return best


def _pair_depth(world, obbs, gi, gj):
    """Overlap depth of two collision geoms: the OBB test first (bounding boxes contain the shapes: no overlap there, none for
    real), then the sampled test for anything that is not box-box. 0 when the OBB depth is within INTERFERENCE_TOL."""
    dep = _obb_depth(obbs[gi], obbs[gj])
    if dep <= INTERFERENCE_TOL:
        return 0.0
    if world.geoms[gi]["kind"] != "box" or world.geoms[gj]["kind"] != "box":
        dep = max(
            _sampled_depth(world, gi, gj, obbs), _sampled_depth(world, gj, gi, obbs)
        )
    return dep


def _obb_gap(a, b):
    """Lower bound (m) of the distance between two OBBs: the best separating axis of the SAT; 0 when none separates."""
    (c1, R1, h1), (c2, R2, h2) = a, b
    d, best = c2 - c1, 0.0
    axes = [R1[:, i] for i in range(3)] + [R2[:, i] for i in range(3)]
    axes += [np.cross(R1[:, i], R2[:, j]) for i in range(3) for j in range(3)]
    for ax in axes:
        nrm = np.linalg.norm(ax)
        if nrm < 1e-9:
            continue
        ax = ax / nrm
        best = max(best, abs(d @ ax) - h1 @ np.abs(R1.T @ ax) - h2 @ np.abs(R2.T @ ax))
    return best


def placement_metrics(world, moved, mated, reach=0.05):
    """Geometry of newly placed instances against everything else, for choosing between placements.
    `moved`: instance ids (MJCF body names) just placed; `mated`: {frozenset((id, id))} pairs allowed to touch.
    -> {"interference": {other id: depth m} (same rule as the INTERFERENCE finding, 1 mm tolerance),
        "gap": smallest bounding-box clearance (m, a lower bound of the true distance) to a non-mated instance within `reach`
        of a moved one, or None when nothing is that close}."""
    obbs = {gi: world.obb(gi) for gi, g in enumerate(world.geoms) if g["collide"]}
    rad = {gi: float(np.linalg.norm(o[2])) for gi, o in obbs.items()}
    name = lambda gi: world.bodies[world.geoms[gi]["body"]]["name"]
    hit, gap = {}, None
    for gi in obbs:
        if name(gi) not in moved:
            continue
        for gj in obbs:
            nj = name(gj)
            if nj in moved or frozenset((name(gi), nj)) in mated:
                continue
            if np.linalg.norm(obbs[gi][0] - obbs[gj][0]) > rad[gi] + rad[gj] + reach:
                continue
            dep = _pair_depth(world, obbs, gi, gj)
            if dep > INTERFERENCE_TOL:
                hit[nj] = max(hit.get(nj, 0.0), dep)
            else:
                g = _obb_gap(obbs[gi], obbs[gj])
                gap = g if gap is None else min(gap, g)
    return {"interference": hit, "gap": gap}


def _interference(ctx, world, specs, F):
    mated = {frozenset((s.a, s.b)) for s in specs.values()}
    mated |= {
        frozenset((m["a"].partition(".")[0], m["b"].partition(".")[0]))
        for m in ctx.doc.get("mates", [])
    }
    obbs = {gi: world.obb(gi) for gi, g in enumerate(world.geoms) if g["collide"]}
    rad = {gi: float(np.linalg.norm(o[2])) for gi, o in obbs.items()}
    best = {}
    gis = sorted(obbs)
    for x, gi in enumerate(gis):
        for gj in gis[x + 1 :]:
            bi, bj = world.geoms[gi]["body"], world.geoms[gj]["body"]
            if bi == bj:
                continue
            ni, nj = world.bodies[bi]["name"], world.bodies[bj]["name"]
            if frozenset((ni, nj)) in mated:
                continue
            if np.linalg.norm(obbs[gi][0] - obbs[gj][0]) > rad[gi] + rad[gj]:
                continue
            dep = _pair_depth(world, obbs, gi, gj)
            if dep > INTERFERENCE_TOL:
                key = tuple(sorted((ni, nj)))
                best[key] = max(best.get(key, 0.0), dep)
    for (a, b), dep in sorted(best.items()):
        _add(
            F,
            "INTERFERENCE",
            "error",
            f"{a} and {b} overlap by {dep * 1000:.1f} mm at rest and are not mated to each other",
            [a, b],
            "move one of them (a different mate or part) so that the shapes only touch",
        )


# ---- point sampling: depth of a sample of solid A inside solid B (meshes by ray parity, primitives analytic) ----
SAMPLE_CAP = 60  # sample points per shape and pair (ponytail: a thin sliver crossing between samples is missed)


@cache
def _mesh_tris(path):
    raw = Path(path).read_bytes()
    n = int.from_bytes(raw[80:84], "little")
    return np.frombuffer(raw, dtype=_STL, count=n, offset=84)["v"].astype(np.float64)


def _world_T(world, gi):
    g = world.geoms[gi]
    return world.bodies[g["body"]]["T"] @ g["T"]


def _sample_points(world, gi, lo, hi):
    """Surface sample points of geom gi (mesh vertices, else corners / edge and face centres / cap rims) inside the box lo..hi."""
    g, T = world.geoms[gi], _world_T(world, gi)
    if g["kind"] == "mesh":
        pts = mesh_props(g["mesh"])[2]
    elif g["kind"] == "box":  # 27 points: corners, edge and face centres, the centre
        pts = np.array(
            [(np.array(s) - 1.0) * g["size"][:3] for s in np.ndindex(3, 3, 3)]
        )
    else:  # cylinder / sphere / capsule: rims at z = -hz, 0, hz and the two cap / pole points
        r = g["size"][0]
        hz = 0.0 if g["kind"] == "sphere" else g["size"][1]
        tip = hz + (r if g["kind"] != "cylinder" else 0.0)
        a = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        ring = np.stack([r * np.cos(a), r * np.sin(a), np.zeros_like(a)], 1)
        pts = np.vstack(
            [ring + [0, 0, z] for z in (-hz, 0.0, hz)]
            + [[[0, 0, -tip], [0, 0, tip], [0, 0, 0]]]
        )
    P = pts @ T[:3, :3].T + T[:3, 3]
    P = P[np.all((P >= lo) & (P <= hi), axis=1)]
    if len(P) > SAMPLE_CAP:
        P = P[np.linspace(0, len(P) - 1, SAMPLE_CAP).astype(int)]
    return P


def _prim_depth(g, T, P):
    """Depth of points P inside primitive g (0 outside)."""
    L = (P - T[:3, 3]) @ T[:3, :3]
    s = g["size"]
    if g["kind"] == "box":
        d = (s[:3] - np.abs(L)).min(axis=1)
    elif g["kind"] == "sphere":
        d = s[0] - np.linalg.norm(L, axis=1)
    elif g["kind"] == "cylinder":
        d = np.minimum(s[0] - np.hypot(L[:, 0], L[:, 1]), s[1] - np.abs(L[:, 2]))
    else:  # capsule: distance to the axis segment
        z = np.clip(L[:, 2], -s[1], s[1])
        d = s[0] - np.hypot(np.hypot(L[:, 0], L[:, 1]), L[:, 2] - z)
    return np.maximum(d, 0.0)


def _ray_first_hit(p, d, tri, tmin, tmax):
    """(number of crossings, nearest hit distance) of the ray p + t d, t > 0, with triangles (k,3,3)."""
    ax = int(np.argmax(np.abs(d)))
    o1, o2 = [i for i in range(3) if i != ax]
    ok = (
        (tmin[:, o1] <= p[o1])
        & (p[o1] <= tmax[:, o1])
        & (tmin[:, o2] <= p[o2])
        & (p[o2] <= tmax[:, o2])
    )
    tr = tri[ok]
    if not len(tr):
        return 0, math.inf
    v0, e1, e2 = tr[:, 0], tr[:, 1] - tr[:, 0], tr[:, 2] - tr[:, 0]
    h = np.cross(d, e2)
    a = np.einsum("ij,ij->i", e1, h)
    good = np.abs(a) > 1e-18
    f = np.where(good, 1.0 / np.where(good, a, 1.0), 0.0)
    s = p - v0
    u = f * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, e1)
    v = f * (q @ d)
    t = f * np.einsum("ij,ij->i", e2, q)
    hit = good & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-12)
    return int(hit.sum()), float(t[hit].min()) if hit.any() else math.inf


_DIRS = [np.array(v, dtype=float) * s for v in np.eye(3) for s in (1.0, -1.0)]


def _depth_in(world, gj, P, cache_):
    """Depth of each point of P inside geom gj."""
    g = world.geoms[gj]
    if g["kind"] != "mesh":
        return _prim_depth(g, _world_T(world, gj), P)
    if gj not in cache_:
        T = _world_T(world, gj)
        tri = _mesh_tris(g["mesh"]) @ T[:3, :3].T + T[:3, 3]
        cache_[gj] = (tri, tri.min(axis=1), tri.max(axis=1))
    tri, tmin, tmax = cache_[gj]
    out = np.zeros(len(P))
    for k, p in enumerate(P):
        n, _ = _ray_first_hit(p, _DIRS[0], tri, tmin, tmax)
        if (
            n % 2 == 1
        ):  # inside: the nearest surface along the six axis rays bounds the penetration depth
            out[k] = min(_ray_first_hit(p, d, tri, tmin, tmax)[1] for d in _DIRS)
    return out


def _sampled_depth(world, gi, gj, obbs):
    """Deepest sampled point of geom gi inside geom gj (m)."""
    c, R, h = obbs[gj]
    ext = np.abs(R) @ h
    P = _sample_points(world, gi, c - ext, c + ext)
    if not len(P):
        return 0.0
    d = _depth_in(world, gj, P, world.__dict__.setdefault("_tri_cache", {}))
    return float(d.max()) if len(d) else 0.0


# ---- rotors: discs, clearance, CG -----------------------------------------------------------------------------------------------
def _rotor_discs(ctx, world, units):
    out = []
    for u in units:
        r, prop = u["rotor"], u["prop"]
        if not r or not prop or prop.id not in world.idx:
            continue
        b = world.bodies[world.idx[prop.id]]
        j = b["joint"]
        if not j or j["name"] != r.get("joint"):
            continue
        R = b["T"][:3, :3]
        axis = R @ j["axis"]
        d = prop.value("diameter")
        if d is None:
            continue
        out.append(
            {
                "u": u,
                "prop": prop.id,
                "c": R @ j["pos"] + b["T"][:3, 3],
                "n": axis / np.linalg.norm(axis),
                "R": d / 2,
                "t": prop.value("geom_thickness") or 0.0,
            }
        )
    return out


def _prop_rules(ctx, world, discs, F):
    for x, a in enumerate(discs):
        for b in discs[x + 1 :]:
            d = b["c"] - a["c"]
            h = abs(d @ a["n"])
            xy = float(np.linalg.norm(d - (d @ a["n"]) * a["n"]))
            gap = xy - (a["R"] + b["R"])
            if gap < 0 and h < a["t"] + b["t"]:
                _add(
                    F,
                    "PROP_OVERLAP",
                    "error",
                    f"rotor discs of {a['prop']} (D {2 * a['R'] * 1000:.0f} mm) and {b['prop']} (D {2 * b['R'] * 1000:.0f} mm) overlap by {-gap * 1000:.0f} mm "
                    f"(centres {xy * 1000:.0f} mm apart in the disc plane)",
                    [a["prop"], b["prop"]],
                    f"use props with D1 + D2 <= {2 * xy * 1000:.0f} mm, or move the motors apart / to different heights",
                )
            elif gap < PROP_GAP_FRAC * 2 * max(a["R"], b["R"]) and h < a["t"] + b["t"]:
                _add(
                    F,
                    "PROP_CLEARANCE",
                    "warning",
                    f"tip-to-tip gap of {a['prop']} and {b['prop']} is {gap * 1000:.1f} mm, under {PROP_GAP_FRAC * 100:.0f} % of the larger diameter "
                    f"({PROP_GAP_FRAC * 2 * max(a['R'], b['R']) * 1000:.1f} mm); the wakes interact and a flexed blade can touch",
                    [a["prop"], b["prop"]],
                    f"use props with D1 + D2 <= {(xy - PROP_GAP_FRAC * 2 * max(a['R'], b['R'])) * 2 * 1000:.0f} mm, or move the motors apart",
                )
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    for a in discs:
        n = a["n"]
        e1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        pts = [
            a["c"] + n * dz + rr * (np.cos(t) * e1 + np.sin(t) * e2)
            for dz in (-a["t"] / 2, 0.0, a["t"] / 2)
            for rr in a["R"] * np.array([0.25, 0.5, 0.75, 1.0])
            for t in ang
        ]
        pts = np.array(pts)
        own = {a["prop"], a["u"]["motor"].id}
        p = world.bodies[world.idx[a["prop"]]]["parent"]
        while p is not None and len(own) < 3:
            own.add(world.bodies[p]["name"])
            p = None
        # tip ring a margin outside the disc (PROP_CLEARANCE): the same sampling at r = R (1 + PROP_GAP_FRAC) x 3 layers x 16 angles
        ring = np.array(
            [
                a["c"]
                + n * dz
                + a["R"] * (1 + PROP_GAP_FRAC) * (np.cos(t) * e1 + np.sin(t) * e2)
                for dz in (-a["t"] / 2, 0.0, a["t"] / 2)
                for t in ang
            ]
        )

        def hits(P):
            out = set()
            for gi, g in enumerate(world.geoms):
                nm = world.bodies[g["body"]]["name"]
                if (
                    not g["collide"]
                    or nm in own
                    or world.bodies[g["body"]]["parent"] == world.idx[a["prop"]]
                ):
                    continue
                c, R, h = world.obb(gi)
                loc = (P - c) @ R
                if (
                    g["kind"] == "cylinder"
                ):  # exact: the bounding box of a yawed prop disc pokes into its neighbour
                    inside = (np.hypot(loc[:, 0], loc[:, 1]) <= h[0]) & (
                        np.abs(loc[:, 2]) <= h[2]
                    )
                else:
                    inside = (np.abs(loc) - h <= 0).all(axis=1)
                if inside.any():
                    out.add(nm)
            return out

        hit = hits(pts)
        for nm in sorted(hits(ring) - hit):
            _add(
                F,
                "PROP_CLEARANCE",
                "warning",
                f"{nm} is within {PROP_GAP_FRAC * 100:.0f} % of the diameter ({PROP_GAP_FRAC * 2 * a['R'] * 1000:.1f} mm) of the tip of {a['prop']} "
                f"(D {2 * a['R'] * 1000:.0f} mm); sampled at 16 angles",
                [a["prop"], nm],
                "move it out of the disc by a margin, or use a smaller prop (a tip near structure loses thrust and is noisy)",
            )
        for nm in sorted(hit):
            _add(
                F,
                "PROP_FRAME_CLASH",
                "error",
                f"the disc of {a['prop']} (D {2 * a['R'] * 1000:.0f} mm) sweeps through {nm}",
                [a["prop"], nm],
                "use a smaller prop, a taller motor mount, or move the part out of the disc",
            )
    # ground: a free-base vehicle is placed with its lowest collision point on z = 0, so that plane is where it rests
    if ctx.doc.get("assembly", {}).get("base", "free") != "free":
        return
    low = [
        (
            a["c"][2]
            - a["R"] * math.sqrt(max(0.0, 1 - a["n"][2] ** 2))
            - a["t"] / 2 * abs(a["n"][2]),
            a,
        )
        for a in discs
    ]
    if not low:
        return
    z, a = min(low, key=lambda x: x[0])
    D = 2 * a["R"]
    if z < GROUND_GAP_FRAC * D:
        _add(
            F,
            "PROP_CLEARANCE",
            "error" if z < 0 else "warning",
            f"the lowest point of the disc of {a['prop']} (D {D * 1000:.0f} mm) is {z * 1000:.0f} mm above the lowest collision point of the airframe, "
            f"under {GROUND_GAP_FRAC * 100:.0f} % of D ({GROUND_GAP_FRAC * D * 1000:.0f} mm): the blades touch the ground on a slightly tilted landing"
            + (" (below it at rest)" if z < 0 else ""),
            [a["prop"]],
            "lengthen the landing gear / skids or fit a smaller prop",
        )
    if z >= GROUND_GAP_FRAC * D:
        zr = z / a["R"]
        gain = (
            1 / (1 - (1 / (4 * zr)) ** 2) - 1 if zr >= 0.6 else None
        )  # Cheeseman-Bennett, only valid for z / R > ~0.6
        if gain is None or gain > GROUND_EFFECT_INFO:
            _add(
                F,
                "PROP_CLEARANCE",
                "info",
                f"at rest the lowest disc ({a['prop']}) is {z * 1000:.0f} mm = {zr:.2f} R above the ground: "
                + (
                    f"ground effect adds ~{gain * 100:.0f} % thrust (Cheeseman & Bennett 1 / (1 - (R / 4 z)^2))"
                    if gain is not None
                    else "below z / R = 0.6, where the Cheeseman-Bennett model stops being valid; ground effect is strong, not estimated"
                )
                + " at take-off / landing: a thrust step as the vehicle leaves the ground",
                [a["prop"]],
            )


def _cg_rule(ctx, world, discs, mass, F):
    ok = [d for d in discs if "Tmax" in d["u"]]
    if len(ok) < 3:
        return
    cg, wm = world.world_com()
    t = np.array([d["u"]["Tmax"] for d in ok])
    p = np.array([d["c"][:2] for d in ok])
    tc = (t[:, None] * p).sum(0) / t.sum()
    off = float(np.linalg.norm(cg[:2] - tc))
    w = (mass or wm) * G
    # least-norm change of the equal-throttle thrusts T0 that puts the net horizontal moment about the CG at zero
    T0 = w * t / t.sum()
    A = np.vstack([np.ones(len(ok)), (p - cg[:2]).T])
    T = T0 + A.T @ np.linalg.solve(A @ A.T, np.array([w, 0.0, 0.0]) - A @ T0)
    f0 = w / t.sum()
    used = (T / t).max()
    headroom = (used - f0) / (1 - f0) if f0 < 1 else math.inf
    diff = float(
        np.max(np.abs(T - T0) / T0)
    )  # extra differential thrust at the worst motor, as a fraction of its hover thrust
    msg = (
        f"CG is {off * 1000:.1f} mm from the thrust centre (horizontal); hover needs up to {diff * 100:.0f} % extra differential thrust "
        f"at one motor, which takes {headroom * 100:.0f} % of the per-motor thrust headroom"
    )
    where = [d["prop"] for d in ok]
    if used > 1 or T.min() < 0:
        _add(
            F,
            "CG_OFFSET",
            "error",
            msg + ": hover is not reachable",
            where,
            "move the battery / payload toward the thrust centre",
        )
    elif headroom > TRIM_WARN or diff > CG_DIFF_WARN:
        _add(
            F,
            "CG_OFFSET",
            "warning",
            msg
            + (
                f" (> {TRIM_WARN * 100:.0f} % headroom)"
                if headroom > TRIM_WARN
                else f" (> {CG_DIFF_WARN * 100:.0f} % of hover thrust)"
            ),
            where,
            "move the battery / payload toward the thrust centre",
        )


# ---- motor losses at hover --------------------------------------------------------------------------------------------------------
def _thermal(ctx, units, F):
    """Copper loss per motor at hover: winding current I_m = I_pack / throttle (ideal ESC: V_pack I_pack = duty V_pack I_m, throttle ~ duty;
    ESC and iron losses are not split out), P_cu = I_m^2 R_winding(T). With a thermal resistance from the device binding
    (`motor.r_th`, K/W, winding to ambient; no part publishes one) the winding temperature solves T = T_AMB + r_th P_cu(T),
    R(T) = R20 (1 + ALPHA_CU (T - 20)). Never invents r_th or a temperature limit: no part states a winding limit, so this is info."""
    rows = []
    for u in units:
        r = u["motor"].value("r_winding")
        if not (r and u.get("Ihover") and u.get("Hthr") and u.get("vnom")):
            continue
        im = u["Ihover"] / u["Hthr"]
        pin = u["Ihover"] * u["vnom"]
        pcu = im * im * r
        rth = _n(u["dev"].get("set", {}).get("motor.r_th"))
        tw = None
        if rth is not None:
            k = rth * im * im * r * ALPHA_CU
            tw = (
                (T_AMB + rth * im * im * r * (1 - 20 * ALPHA_CU)) / (1 - k)
                if k < 1
                else math.inf
            )
        rows.append((pcu / pin, pcu, im, pin, tw, rth, u))
    if not rows:
        return
    frac, pcu, im, pin, tw, rth, u = max(rows, key=lambda x: x[0])
    msg = (
        f"hover copper loss {pcu:.1f} W = {frac * 100:.0f} % of the {pin:.0f} W electrical input at the worst motor {u['motor'].id} "
        f"(winding current {im:.1f} A = pack current / throttle, R_winding {u['motor'].value('r_winding') * 1000:.0f} mOhm)"
    )
    if tw is not None:
        msg += f"; winding ~{tw:.0f} degC at {T_AMB:g} degC ambient with the device's r_th {rth:g} K/W"
    else:
        msg += "; no r_th in the device binding, so no temperature"
    if frac > COPPER_LOSS_INFO:
        msg += f"; above {COPPER_LOSS_INFO * 100:.0f} %: a lighter prop / lower kv / larger motor loses less"
    _add(F, "THERMAL", "info", msg, [u["motor"].id])


def _device_unset(ctx, F):
    """Device fields the simulator needs (auto_devices.REQUIRED: devices.cpp refuses the run without them) that a binding neither
    sets, maps nor carries: the stripped `motor.c_th`, or a field a swap to a part without that param left unfed.
    Error: a run would fail. Never invented here, only named, with where the number can come from."""
    from . import auto_devices as AD

    feeds = {}
    for tab in (
        AD.MOTOR,
        AD.ESC,
        AD.SERVO,
        AD.BATTERY,
        AD.PSU,
        AD.ROTOR,
        AD.IMU,
        AD.MAG,
        AD.GNSS,
    ):
        for fld, params in tab.items():
            feeds.setdefault(fld, params)
    maps = ctx.doc.get("device_maps", {})
    for b in ctx.doc.get("devices", []):
        req = AD.REQUIRED.get(b.get("type"), [])
        have = set(b) | set(b.get("set", {})) | set(maps.get(b.get("map"), {}))
        for f in [f for f in req if f not in have]:
            holder = b.get("motor") if f.startswith("motor.") else b.get("instance")
            pid = ctx.insts[holder].part.id if holder in ctx.insts else "the part"
            fill = f"design.auto_devices(fill={{'{b['name']}': {{'set': {{'{f}': <provenance table>}}}}}})"
            if f in feeds:
                fix = (
                    f"add a '{feeds[f][0].lstrip('@')}' param to {pid} (or `set` it on '{holder}') and map it with auto_devices, "
                    f"or supply the number yourself: {fill}"
                )
            else:
                fix = (
                    f"no part carries it: put `{f}` in the `set` of device '{b['name']}' as a provenance table "
                    f"(estimated values need a rationale), e.g. {fill}"
                )
            _add(
                F,
                "DEVICE_UNSET",
                "error",
                f"device '{b['name']}' ({b['type']}) has no '{f}': the simulation run would fail",
                [b["name"], f],
                fix,
            )


# ---- servo torque ---------------------------------------------------------------------------------------------------------------
def _servo_points(part):
    """[(V, stall torque N*m, stall current A or None, rated torque N*m or None)] from params stall_torque_<V>v, ... (7v4 -> 7.4)."""
    pts = {}
    for k, e in part.params.items():
        m = re.fullmatch(r"(stall_torque|stall_current|rated_torque)_(\d+)v(\d+)?", k)
        if m and _n(e) is not None:
            if (
                m.group(1) in ("stall_torque", "rated_torque")
                and e.get("unit") != "N*m"
            ):
                continue
            pts.setdefault(float(f"{m.group(2)}.{m.group(3) or 0}"), {})[m.group(1)] = (
                _n(e)
            )
    return [
        (v, d["stall_torque"], d.get("stall_current"), d.get("rated_torque"))
        for v, d in sorted(pts.items())
        if "stall_torque" in d
    ]


def _at_voltage(points, v, col):
    """Value at bus voltage v: linear between table voltages, proportional to V below the lowest (stall values scale with V / R),
    clamped at the highest (no extrapolation upward)."""
    pts = [(p[0], p[col]) for p in points if p[col] is not None]
    if not pts:
        return None
    if v <= pts[0][0]:
        return pts[0][1] * v / pts[0][0]
    return float(np.interp(v, [p[0] for p in pts], [p[1] for p in pts]))


def _samples(rng, n):
    return np.linspace(rng[0], rng[1], n) if rng else np.zeros(1)


def _driven(world):
    """{joint name: body index} of the ranged hinge joints: the ones a pose grid / IK moves."""
    return {
        b["joint"]["name"]: i
        for i, b in enumerate(world.bodies)
        if b["joint"] and b["joint"]["type"] == "hinge" and b["joint"]["range"]
    }


def _fk(world, jb, names, Q):
    """Forward kinematics of P poses at once: Q (P, len(names)) angles of the joints `names` (the others stay at 0) ->
    (T, base), per body the world pose (P, 4, 4) and the pose of its joint frame before the joint turns."""
    P, nb = len(Q), len(world.bodies)
    T, base = [None] * nb, [None] * nb
    qi = {jb[j]: k for k, j in enumerate(names)}
    for i, b in enumerate(world.bodies):
        Tp = (
            np.broadcast_to(np.eye(4), (P, 4, 4))
            if b["parent"] is None
            else T[b["parent"]]
        )
        base[i] = Tp @ b["Tl"]
        if i in qi:
            j, q = b["joint"], Q[:, qi[i]]
            a = j["axis"] / np.linalg.norm(j["axis"])
            K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
            R = (
                np.eye(3)
                + np.sin(q)[:, None, None] * K
                + (1 - np.cos(q))[:, None, None] * (K @ K)
            )
            Tj = np.broadcast_to(np.eye(4), (P, 4, 4)).copy()
            Tj[:, :3, :3] = R
            Tj[:, :3, 3] = j["pos"] - R @ j["pos"]
            T[i] = base[i] @ Tj
        else:
            T[i] = base[i]
    return T, base


def _subtree(world):
    kids = {i: [] for i in range(len(world.bodies))}
    for i, b in enumerate(world.bodies):
        if b["parent"] is not None:
            kids[b["parent"]].append(i)

    def subtree(i):
        return [i] + [x for c in kids[i] for x in subtree(c)]

    return subtree


def _com(world, T):
    return {
        i: np.einsum("pij,j->pi", T[i], np.append(b["com"], 1.0))[:, :3]
        for i, b in enumerate(world.bodies)
    }


def _gravity_tau(world, jb, base, com, subtree, joint):
    """Gravity torque (N*m, one per pose) about the axis of `joint` from everything it carries -> (tau, axis, pivot) per pose."""
    i = jb[joint]
    j = world.bodies[i]["joint"]
    Rb, tb = base[i][:, :3, :3], base[i][:, :3, 3]
    axis = Rb @ (j["axis"] / np.linalg.norm(j["axis"]))
    piv = Rb @ j["pos"] + tb
    S = sum(world.bodies[k]["mass"] * (com[k] - piv) for k in subtree(i))
    return G * (-axis[:, 0] * S[:, 1] + axis[:, 1] * S[:, 0]), axis, piv


def _tip(world, jb, first):
    """(leaf body, tip point in the leaf's frame): the leaf = the deepest driven body; the tip = the farthest collision-geometry
    point of it from the pivot of joint `first` at rest (the payload point of PAYLOAD_LIMIT, the end effector of ARM_REACH)."""
    subtree = _subtree(world)
    depth = {i: len(subtree(i)) for i in jb.values()}
    leaf = min(depth, key=lambda i: (depth[i], -i)) if depth else None
    tip_loc = None
    if leaf is not None:
        piv = world.bodies[jb[first]]["T"] @ np.append(
            world.bodies[jb[first]]["joint"]["pos"], 1.0
        )
        best = -1.0
        for gi in world.bodies[leaf]["geoms"]:
            g = world.geoms[gi]
            if not g["collide"]:
                continue
            c, R, h = world.obb(gi)
            Tb = world.bodies[leaf]["T"]
            corners = [
                c + R @ (h * np.array(s))
                for s in np.ndindex(2, 2, 2)
                for s in [np.array(s) * 2 - 1]
            ]
            for pt in corners:
                dd = float(np.linalg.norm(pt - piv[:3]))
                if dd > best:
                    best, tip_loc = dd, np.linalg.solve(Tb[:3, :3], pt - Tb[:3, 3])
    return leaf, tip_loc


def _servo_rules(ctx, net, world, F):
    servos = [
        d
        for d in ctx.doc.get("devices", [])
        if d["type"] == "servo" and d.get("joint") and d.get("instance") in ctx.insts
    ]
    jb = _driven(world)
    servos = [d for d in servos if d["joint"] in jb]
    out = {}
    if not servos:
        return out
    names = [d["joint"] for d in servos]
    J = len(names)
    n = int(max(2, min(7, math.floor(MAX_POSES ** (1.0 / J)))))
    grids = np.meshgrid(
        *[_samples(world.bodies[jb[j]]["joint"]["range"], n) for j in names],
        indexing="ij",
    )
    Q = np.stack([g.ravel() for g in grids], axis=1)  # (P, J)
    P = len(Q)
    nb = len(world.bodies)
    T, base = _fk(world, jb, names, Q)
    subtree = _subtree(world)

    com = _com(world, T)
    leaf, tip_loc = _tip(world, jb, names[0])
    worst_margin, worst, pay_min, pay_who, pay_need = (
        math.inf,
        None,
        math.inf,
        None,
        0.0,
    )
    rows = []
    for d in servos:
        i = jb[d["joint"]]
        tau, axis, piv = _gravity_tau(world, jb, base, com, subtree, d["joint"])
        k = int(np.argmax(np.abs(tau)))
        tmax = float(abs(tau[k]))
        inst = ctx.insts[d["instance"]]
        pts = _servo_points(inst.part)
        v = net.voltage((d["instance"], d.get("port", "port_a")))
        vb = v[1] if v else inst.value("v_nominal")
        stall = _at_voltage(pts, vb, 1) if pts and vb else None
        rated = _at_voltage(pts, vb, 3) if pts and vb else None
        if stall is None:
            continue
        rows.append((d, tmax, stall, rated, vb, Q[k]))
        out[d["name"]] = {
            "tau": tmax,
            "stall": stall,
            "vb": vb,
            "inst": inst,
            "rated": rated,
        }
        if tip_loc is not None:
            e = np.einsum("pij,j->pi", T[leaf], np.append(tip_loc, 1.0))[:, :3] - piv
            s = (
                axis[:, 1] * e[:, 0] - axis[:, 0] * e[:, 1]
            )  # lever of 1 kg (about the horizontal) times g gives torque
            num = stall - np.sign(s) * tau
            with np.errstate(divide="ignore", invalid="ignore"):
                m = np.where(
                    np.abs(s) > 1e-12,
                    num / (G * np.abs(s)),
                    np.where(np.abs(tau) <= stall, np.inf, 0.0),
                )
            mm = max(0.0, float(m.min()))
            if mm < pay_min:
                pay_min, pay_who, pay_need = mm, d["name"], tmax
    for d, tmax, stall, rated, vb, q in rows:
        margin = stall / tmax if tmax > 0 else math.inf
        pose = ", ".join(f"{nm} {np.degrees(x):.0f} deg" for nm, x in zip(names, q))
        msg = f"{d['name']} ({d['joint']}): worst static gravity torque {tmax:.3f} N*m at [{pose}] vs stall {stall:.3f} N*m at {vb:g} V (margin {margin:.2f})"
        if margin < worst_margin:
            worst_margin, worst = margin, (d, msg)
        pts = _servo_points(ctx.insts[d["instance"]].part)
        if margin < 1:
            alts = [
                p.id
                for p in ctx.lib.by_category("motor.servo")
                if (s := _servo_points(p)) and (_at_voltage(s, vb, 1) or 0) >= tmax
            ]
            _add(
                F,
                "SERVO_TORQUE",
                "error",
                msg + ": the servo cannot hold this pose",
                [d["name"], d["joint"]],
                f"use a servo with stall torque >= {tmax:.2f} N*m at {vb:g} V: {_ids(alts)}; or lighten / shorten the arm",
            )
        elif rated is not None and tmax > rated:
            _add(
                F,
                "SERVO_TORQUE",
                "warning",
                msg
                + f"; above the rated (continuous) torque {rated:.3f} N*m, it will overheat when holding",
                [d["name"], d["joint"]],
                "use a stronger servo or reduce the lever arm / mass",
            )
        if 1 <= margin < SERVO_MARGIN:
            _add(
                F,
                "SERVO_TORQUE_MARGIN",
                "warning",
                msg
                + f" < {SERVO_MARGIN:g}: no reserve for acceleration, payload or a voltage sag (the static torque is the arm's own weight; a payload adds to it, see PAYLOAD_LIMIT)",
                [d["name"], d["joint"]],
                f"use a servo with stall torque >= {SERVO_MARGIN * tmax:.2f} N*m at {vb:g} V, or lighten / shorten the arm",
            )
    if worst and worst_margin >= 1 and not any(f.code == "SERVO_TORQUE" for f in F):
        _add(
            F,
            "SERVO_TORQUE",
            "info",
            worst[1] + " (the tightest servo)",
            [worst[0]["name"]],
        )
    if tip_loc is not None and rows:
        if pay_min <= 0:
            _add(
                F,
                "PAYLOAD_LIMIT",
                "warning",
                f"no payload fits: {pay_who} is already at or past its stall torque at the worst pose",
                [pay_who],
                "see SERVO_TORQUE",
            )
        elif math.isfinite(pay_min):
            _add(
                F,
                "PAYLOAD_LIMIT",
                "info",
                f"max extra end-effector mass {pay_min * 1000:.0f} g at {ctx.insts[world.bodies[leaf]['name']].id} (limited by {pay_who}, margin >= 1 at every sampled pose)",
                [pay_who],
            )
    return out


# ---- servo bus --------------------------------------------------------------------------------------------------------------------
def _servo_bus(ctx, net, servo_info, F):
    """One half-duplex bus = the servos whose power ports are in one net (the 3-pin cable carries supply and signal together).
    SERVO_ID_CONFLICT / SERVO_BAUD_MISMATCH (error), SERVO_ID_UNSET (info), SERVO_BUS_CURRENT, SERVO_CHAIN_LENGTH."""
    buses = {}
    for d in ctx.doc.get("devices", []):
        inst = ctx.insts.get(d.get("instance")) if d["type"] == "servo" else None
        if inst is not None:
            key = (inst.id, d.get("port", "port_a"))
            if key in net.ports:
                buses.setdefault(net.nroot(key), []).append((d, inst, key))
    for members in buses.values():
        names = [d["name"] for d, _, _ in members]
        ids = {}
        for d, inst, _ in members:
            sid = inst.value("servo_id")
            if sid is None:
                continue
            ids.setdefault(int(sid), []).append(d["name"])
            if not 0 <= sid <= SERVO_ID_MAX:
                _add(
                    F,
                    "SERVO_ID_CONFLICT",
                    "error",
                    f"{d['name']}: servo_id {sid:g} is outside 0..{SERVO_ID_MAX} (254 is the broadcast ID)",
                    [d["name"]],
                    f"use an ID in 0..{SERVO_ID_MAX}",
                )
        free = [i for i in range(1, SERVO_ID_MAX + 1) if i not in ids]
        for sid, who in sorted(ids.items()):
            if len(who) > 1:
                _add(
                    F,
                    "SERVO_ID_CONFLICT",
                    "error",
                    f"servos {', '.join(who)} share bus ID {sid}: the bus answers with a collision and the host cannot address them separately",
                    who,
                    f"give them different IDs (`set` servo_id on each instance), e.g. {', '.join(str(i) for i in free[: len(who) - 1])} for all but one",
                )
        unset = [d["name"] for d, inst, _ in members if inst.value("servo_id") is None]
        if unset and len(members) > 1:
            _add(
                F,
                "SERVO_ID_UNSET",
                "info",
                f"{len(members)} servos share a bus and {', '.join(unset)} declare no servo_id: new servos all answer to the factory ID "
                "(1 on the STS3215), so each must be given its own ID before the bus works",
                unset,
                "`set` a servo_id on every servo instance (e.g. its position along the chain)",
            )
        bauds = {}
        for d, inst, _ in members:
            b = inst.value("baud") or inst.value("baud_default")
            if b:
                bauds.setdefault(int(b), []).append(d["name"])
        if len(bauds) > 1:
            _add(
                F,
                "SERVO_BAUD_MISMATCH",
                "error",
                "servos on one bus run at different baud rates: "
                + "; ".join(
                    f"{b} bit/s: {', '.join(w)}" for b, w in sorted(bauds.items())
                ),
                names,
                "set one baud on every servo (`set` baud) or move the odd one to its own bus",
            )
        _bus_electrics(ctx, net, servo_info, members, F)


def _bus_electrics(ctx, net, servo_info, members, F):
    """SERVO_BUS_CURRENT: all servos stalled vs the bus driver board's max_continuous_current (the supply: PSU_OVERLOAD).
    SERVO_CHAIN_LENGTH: servo count vs the board's max_servos, and the cable drop to the farthest servo with all of them
    stalled vs the servo's minimum operating voltage."""
    loads, peak, sus = {}, 0.0, 0.0
    for d, inst, key in members:
        v = net.voltage(key)
        pts = _servo_points(inst.part)
        pk = _at_voltage(pts, v[1], 2) if (pts and v) else None
        if pk is None:
            return
        si = servo_info.get(d["name"])
        kt, idle = inst.value("kt_output_datasheet"), inst.value("idle_current")
        su = idle + si["tau"] / kt if (si and kt and idle is not None) else None
        loads[key] = (pk, su)
        peak += pk
        sus = None if (sus is None or su is None) else sus + su
    root = net.nroot(members[0][2])
    boards = [
        (i, i.value("max_continuous_current"))
        for i in {net.ports[k][0] for k in net.members(members[0][2])}
        if i.part.category == "pcb.servo_driver"
    ]
    for board, cont in boards:
        sev = _sev(peak, sus, cont, None) if cont else None
        if sev:
            _add(
                F,
                "SERVO_BUS_CURRENT",
                "error" if sev == "error" else "warning",
                f"{board.id}: {len(members)} servos at stall draw {peak:.1f} A"
                + (f", {sus:.1f} A holding" if sus is not None else "")
                + f" vs the board's max_continuous_current {cont:g} A",
                [board.id, *[d["name"] for d, _, _ in members]],
                f"use a driver board rated >= {peak:.0f} A, split the servos over two boards, or cap the servos' max_torque so they cannot all stall",
            )
        if (n := board.value("max_servos")) and len(members) > n:
            _add(
                F,
                "SERVO_CHAIN_LENGTH",
                "error",
                f"{len(members)} servos on a bus whose board ({board.id}) drives at most {n:g}",
                [board.id],
                "split the servos over a second bus",
            )
    # cable drop along the chain: sum over the wires on the path from the source of (current through it x its loop resistance)
    src = net.sources(members[0][2])
    if not src:
        return
    cur, _ = net.flows(loads)
    adj = {}
    for w in net.wires:
        if net.nroot(w["ka"]) == root and w["i"] in cur:
            r = _wire_r(ctx, w)
            if r is None:
                continue
            a, b = net.node.find(w["ka"]), net.node.find(w["kb"])
            adj.setdefault(a, []).append((b, w, r))
            adj.setdefault(b, []).append((a, w, r))
    drop, hops, todo = (
        {net.node.find(src[0]): 0.0},
        {net.node.find(src[0]): 0},
        [net.node.find(src[0])],
    )
    while todo:
        n = todo.pop()
        for m, w, r in adj.get(n, ()):
            if m not in drop:
                drop[m], hops[m] = drop[n] + cur[w["i"]][0] * r, hops[n] + 1
                todo.append(m)
    vnom = net.voltage(members[0][2])[1]
    worst = None
    for d, inst, key in members:
        node = net.node.find(key)
        vmin = next(
            (
                inst.value(p)
                for p in (
                    "v_operating_min",
                    "reg_min_input_voltage_default",
                    "reg_min_voltage_limit_default",
                )
                if inst.value(p)
            ),
            None,
        )
        if vmin and node in drop and (worst is None or vnom - drop[node] < worst[0]):
            worst = (vnom - drop[node], drop[node], hops[node], d["name"], vmin)
    if worst and worst[0] < worst[4]:
        v, dr, h, who, vmin = worst
        _add(
            F,
            "SERVO_CHAIN_LENGTH",
            "warning",
            f"{who} is {h} cable hops from the bus source: with all {len(members)} servos stalled the cables drop {dr:.2f} V, "
            f"leaving {v:.2f} V of {vnom:g} V, below the servo's minimum {vmin:g} V (wire runs from the source only: the supply's output resistance and the board's are not counted)",
            [who],
            "shorten the chain (feed the bus in the middle or from a second port), use a heavier cable (lower ohm/m), or limit the servos' max_torque",
        )


def _wire_r(ctx, w):
    """Loop resistance (ohm) of a wire entry: resistance_per_m x length x conductors, or None when the part states no ohm/m."""
    rpm = w["part"].value("resistance_per_m")
    if rpm is None:
        return None
    try:
        length, _ = ctx.num(w["w"]["length"], f"wire {w['name']}", UNITS["m"], "length")
    except DesignError:
        return None
    return rpm * length * w["w"].get("conductors", 1)


# ---- arm reach ---------------------------------------------------------------------------------------------------------------------
def _reach_points(ctx, t):
    """[[x, y, z]] m of a [[targets]] entry that declares `reach` (a list of provenance tables, value = [x, y, z]) or `workspace`
    (one provenance table, value = [x0, y0, z0, x1, y1, z1], the box is sampled on a 3 x 3 x 3 grid)."""
    pts = []
    for k, e in enumerate(t.get("reach", [])):
        v = ctx.entry(e, f"target {t['name']} reach[{k}]")["value"]
        if not (isinstance(v, list) and len(v) == 3):
            raise DesignError(f"target {t['name']} reach[{k}]: value must be [x, y, z]")
        pts.append([float(x) for x in v])
    if "workspace" in t:
        v = ctx.entry(t["workspace"], f"target {t['name']} workspace")["value"]
        if not (isinstance(v, list) and len(v) == 6):
            raise DesignError(
                f"target {t['name']} workspace: value must be [x0, y0, z0, x1, y1, z1]"
            )
        g = [np.linspace(v[i], v[i + 3], 3) for i in range(3)]
        pts += [[x, y, z] for x in g[0] for y in g[1] for z in g[2]]
    return pts


def _ik(world, jb, names, leaf, tip, target, starts=REACH_STARTS):
    """Damped least squares from `starts` joint-limit-respecting starts (the range middle, then seeded random poses) ->
    (closest approach in m, joint angles there). The Jacobian is by forward difference of `_fk` (all columns in one batch)."""
    lo = np.array([world.bodies[jb[j]]["joint"]["range"][0] for j in names])
    hi = np.array([world.bodies[jb[j]]["joint"]["range"][1] for j in names])
    rng = np.random.default_rng(0)
    th, eps = np.append(tip, 1.0), 1e-6

    def pos(Q):
        T, _ = _fk(world, jb, names, Q)
        return np.einsum("pij,j->pi", T[leaf], th)[:, :3]

    best = (math.inf, None, None)
    for s in range(starts):
        q = (lo + hi) / 2 if s == 0 else rng.uniform(lo, hi)
        for _ in range(REACH_ITERS):
            p = pos(np.vstack([q, q + eps * np.eye(len(q))]))
            e = target - p[0]
            err = float(np.linalg.norm(e))
            if err < best[0]:
                best = (err, q.copy(), p[0].copy())
            if err < REACH_CONVERGED:
                break
            Jm = (p[1:] - p[0]).T / eps
            dq = Jm.T @ np.linalg.solve(Jm @ Jm.T + REACH_DAMPING**2 * np.eye(3), e)
            q = np.clip(q + dq, lo, hi)
        if best[0] < REACH_CONVERGED:
            break
    return best


def _reach_rules(ctx, world, F):
    tg = [t for t in ctx.doc.get("targets", []) if "reach" in t or "workspace" in t]
    jb = _driven(world)
    if not tg or not jb:
        return
    first = next(iter(jb))
    leaf, tip = _tip(world, jb, first)
    pivot = (
        world.bodies[jb[first]]["T"]
        @ np.append(world.bodies[jb[first]]["joint"]["pos"], 1.0)
    )[:3]
    names = [
        j for j, i in jb.items() if i != leaf
    ]  # the end effector's own joint (the gripper) does not move its tip point
    for t in tg:
        res = [
            (_ik(world, jb, names, leaf, tip, np.array(p)), p)
            for p in _reach_points(ctx, t)
        ]
        bad = [(r, p) for r, p in res if r[0] > REACH_TOL]
        if not bad:
            worst = max((r[0] for r, _ in res), default=0.0)
            _add(
                F,
                "ARM_REACH",
                "info",
                f"target '{t['name']}': all {len(res)} point(s) reachable within the joint limits (worst residual {worst * 1000:.1f} mm, tolerance {REACH_TOL * 1000:g} mm)",
                [t["name"]],
            )
            continue
        (gap, q, near), p = max(bad, key=lambda x: x[0][0])
        pose = ", ".join(f"{n} {np.degrees(x):.0f} deg" for n, x in zip(names, q))
        # too far (the closest tip is nearer the shoulder than the target) or too close (the arm cannot fold that tight)?
        far = np.linalg.norm(np.array(p) - pivot) > np.linalg.norm(near - pivot)
        mm = f"{gap * 1000:.0f} mm"
        fix = (
            f"lengthen a link by about {mm} or mount the base about {mm} closer to the target (a different mount); or move the target inside the workspace"
            if far
            else f"the arm cannot fold that close within its joint limits: mount the base about {mm} farther from the target, or move the target outward"
        )
        _add(
            F,
            "ARM_REACH",
            "warning",
            f"target '{t['name']}': {len(bad)} of {len(res)} point(s) out of reach; worst [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}] m, "
            f"closest approach {gap * 1000:.0f} mm at [{pose}] (end effector = the farthest point of {world.bodies[leaf]['name']}; numerical IK, joint limits on)",
            [t["name"]],
            fix,
        )


# ---- servo choice for a joint (Design.suggest) --------------------------------------------------------------------------------------
def joint_holding(asm, lib, joint):
    """Servos ranked by holding-torque margin at `joint`: stall torque (at the bus voltage) over the worst static gravity torque
    the joint carries over the pose grid of SERVO_TORQUE_MARGIN. -> {"joint", "tau", "pose", "vb", "rows"}; best margin first.
    The servo now on the joint (if any) sets the bus voltage (candidates not operable at it are left out: v_operating_min /
    max) and the mate-fit check (mates_fit); the rest of the arm keeps its mass, so a heavier servo adds to the torque."""
    if not isinstance(asm, Assembly):
        asm = load(asm)
    ctx = Ctx(asm, lib)
    xml, _ = compute(asm, lib.dir, lib.root, lib, route=False)
    world = _World(xml, lib.root)
    jb = _driven(world)
    if joint not in jb:
        raise DesignError(
            f"'{joint}' is not a ranged hinge joint of this design (joints: {', '.join(jb)})"
        )
    names = list(jb)
    n = int(max(2, min(7, math.floor(MAX_POSES ** (1.0 / len(names))))))
    grids = np.meshgrid(
        *[_samples(world.bodies[jb[j]]["joint"]["range"], n) for j in names],
        indexing="ij",
    )
    Q = np.stack([g.ravel() for g in grids], axis=1)
    T, base = _fk(world, jb, names, Q)
    tau, _, _ = _gravity_tau(world, jb, base, _com(world, T), _subtree(world), joint)
    k = int(np.argmax(np.abs(tau)))
    tmax = float(abs(tau[k]))
    now = next(
        (
            d
            for d in ctx.doc.get("devices", [])
            if d["type"] == "servo"
            and d.get("joint") == joint
            and d.get("instance") in ctx.insts
        ),
        None,
    )
    vb, mine = None, None
    if now:
        mine = ctx.insts[now["instance"]]
        v = _Net(ctx, _wires(ctx, [])).voltage((mine.id, now.get("port", "port_a")))
        vb = v[1] if v else None
    have = (
        {p.get("type") for p in mine.part.ports if p.get("kind") == "mech"}
        if mine
        else None
    )
    rows = []
    for p in lib.by_category("motor.servo"):
        pts = _servo_points(p)
        v = vb or p.value("v_nominal")
        stall = _at_voltage(pts, v, 1) if pts and v else None
        lo, hi = p.value("v_operating_min"), p.value("v_operating_max")
        if stall is None or (vb and lo and hi and not lo <= vb <= hi):
            continue  # no torque table, or not operable at the bus voltage
        rated = _at_voltage(pts, v, 3)
        margin = stall / tmax if tmax > 1e-9 else 1e9
        fits = (
            None
            if have is None
            else have <= {q.get("type") for q in p.ports if q.get("kind") == "mech"}
        )
        verdict = (
            "cannot hold the worst pose"
            if margin < 1
            else f"holds, margin below {SERVO_MARGIN:g} (SERVO_TORQUE_MARGIN)"
            if margin < SERVO_MARGIN
            else "holds with margin"
        )
        if rated is not None and tmax > rated and margin >= 1:
            verdict += "; above the rated (continuous) torque: it will overheat holding"
        if mine is not None and p.id == mine.part.id:
            verdict += "; the servo on this joint now"
        rows.append(
            {
                "part": p.id,
                "category": p.category,
                "name": p.raw.get("name", ""),
                "score": round(margin, 2),
                "margin": margin,
                "stall_Nm": stall,
                "rated_Nm": rated,
                "voltage_V": v,
                "tau_worst_Nm": tmax,
                "mass": p.value("mass"),
                "mates_fit": fits,
                "why": [verdict],
            }
        )
    rows.sort(key=lambda r: (-r["margin"], r["part"]))
    pose = {nm: float(x) for nm, x in zip(names, Q[k])}
    return {"joint": joint, "tau": tmax, "pose": pose, "vb": vb, "rows": rows}


# ---- current rules ---------------------------------------------------------------------------------------------------------------
def _sev(peak, sus, cont, burst):
    if burst is not None and peak > burst + 1e-9:
        return "error"
    if sus is not None and sus > cont + 1e-9:
        return "error"
    return "warning" if peak > cont + 1e-9 else None


def _emit_current(F, sev, what, peak, sus, cont, where, fix):
    if sev is None:
        return
    s = f"{peak:.1f} A peak" + (f", {sus:.1f} A sustained" if sus is not None else "")
    if sev == "error":
        _add(
            F,
            "CURRENT_OVER_RATING",
            "error",
            f"{what}: {s} vs rating {cont:g} A",
            where,
            fix,
        )
    else:
        _add(
            F,
            "CURRENT_PEAK_OVER_RATING",
            "warning",
            f"{what}: {s} (worst case) above the continuous rating {cont:g} A",
            where,
            fix,
        )


def _current_rules(ctx, net, units, servo_info, F, diag=None):
    loads, labels = {}, {}
    for u in units:
        pk = u.get("Ipeak")
        if pk is None:
            pk = u["motor"].value("i_peak")
        if pk is None:
            continue
        sus = u.get("Ihover")
        if (
            u["key"] in loads
        ):  # several motors on one ESC input (4-in-1): the loads add, they do not overwrite
            opk, osus = loads[u["key"]]
            loads[u["key"]] = (
                opk + pk,
                None if sus is None or osus is None else osus + sus,
            )
        else:
            loads[u["key"]] = (pk, sus)
        for p in u["motor"].part.ports:
            if p.get("kind") == "power" and p.get("role") == "sink":
                loads[(u["motor"].id, p["name"])] = (pk, sus)
    for d in ctx.doc.get("devices", []):
        if d["type"] != "servo":
            continue
        inst = ctx.insts.get(d.get("instance"))
        if not inst:
            continue
        key = (inst.id, d.get("port", "port_a"))
        v = net.voltage(key)
        pts = _servo_points(inst.part)
        pk = _at_voltage(pts, v[1], 2) if (pts and v) else None
        if pk is None:
            continue
        si = servo_info.get(d["name"])
        kt, idle = inst.value("kt_output_datasheet"), inst.value("idle_current")
        sus = idle + si["tau"] / kt if (si and kt and idle is not None) else None
        loads[key] = (pk, sus)
    cur, total = net.flows(loads)
    seen_unknown = set()
    for w in net.wires:
        if w["i"] not in cur:
            continue
        pk, sus = cur[w["i"]]
        if diag is not None:
            diag.setdefault("currents", {})[w["name"]] = (pk, sus)
        if pk <= 0:
            continue
        items = [
            (f"{w['ka'][0]}.{w['ka'][1]}", w["pa"], w["ia"]),
            (f"{w['kb'][0]}.{w['kb'][1]}", w["pb"], w["ib"]),
        ]
        lim = []
        for label, p, inst in items:
            if "i_peak" not in p and _n(p.get("i_max")) is not None:
                lim.append((_n(p["i_max"]), _n(p.get("i_burst")), label, inst, p))
        rated = [
            (_n(q.get("i_max")), _n(q.get("i_burst")))
            for q in w["part"].ports
            if _n(q.get("i_max")) is not None
        ]
        if rated:
            lim.append(
                (
                    min(r[0] for r in rated),
                    None,
                    f"wire part {w['part'].id}",
                    None,
                    None,
                )
            )
        elif _is_cable(w["part"]) and w["part"].id not in seen_unknown:
            seen_unknown.add(w["part"].id)
            _add(
                F,
                "WIRE_RATING_UNKNOWN",
                "info",
                f"wire part {w['part'].id} publishes no current rating; wire '{w['name']}' carries {pk:.1f} A peak",
                [w["name"], w["part"].id],
                "add an ampacity (i_max on the wire-lead ports) to the part file",
            )
        if not lim:
            continue
        cont, burst, label, inst, port = min(lim, key=lambda x: x[0])
        sev = _sev(pk, sus, cont, burst)
        if inst is not None:
            alts = _alts(ctx.lib, inst, port, "i_max", pk)
            fix = f"replace {inst.part.id} by a part with the same port rated >= {pk:.0f} A: {_ids(alts)}"
        else:
            alts = [
                p.id
                for p in ctx.lib.by_category("wire")
                if any((_n(q.get("i_max")) or 0) >= pk for q in p.ports)
            ]
            fix = f"use a heavier wire (rated >= {pk:.0f} A): {_ids(alts)}; or shorten the load"
        _emit_current(
            F,
            sev,
            f"wire '{w['name']}' limited by {label}",
            pk,
            sus,
            cont,
            [w["name"], label],
            fix,
        )
    for key, (pk, sus) in total.items():
        inst, p = net.ports[key]
        if pk <= 0:
            continue
        if inst.value("c_rating_cont") and inst.value("capacity_ah"):
            ah = inst.value("capacity_ah")
            cont, burst = (
                inst.value("c_rating_cont") * ah,
                (inst.value("c_rating_burst") or 0) * ah or None,
            )
            sev = _sev(pk, sus, cont, burst)
            if sev:
                alts = [
                    q.id
                    for q in ctx.lib.by_category(inst.part.category)
                    if q.value("c_rating_cont")
                    and q.value("capacity_ah")
                    and q.value("c_rating_cont") * q.value("capacity_ah") >= pk
                    and q.id != inst.part.id
                ]
                _add(
                    F,
                    "BATTERY_C_RATING",
                    sev,
                    f"{inst.id}: load {pk:.1f} A peak"
                    + (f", {sus:.1f} A sustained" if sus is not None else "")
                    + f" vs {inst.value('c_rating_cont'):g} C x {ah:g} Ah = {cont:.0f} A continuous"
                    + (f" ({burst:.0f} A burst)" if burst else ""),
                    [inst.id],
                    f"fit a pack with C x Ah >= {pk:.0f} A: {_ids(alts)}",
                )
        elif inst.value("i_out_rated") is not None:
            cont = inst.value("i_out_rated")
            sev = _sev(pk, sus, cont, None)
            if sev:
                alts = [
                    q.id
                    for q in ctx.lib.by_category("psu")
                    if (q.value("i_out_rated") or 0) >= pk and q.id != inst.part.id
                ]
                _add(
                    F,
                    "PSU_OVERLOAD",
                    sev,
                    f"{inst.id}: load {pk:.1f} A at worst case (all servos stalled)"
                    + (f", {sus:.1f} A holding" if sus is not None else "")
                    + f" vs rated {cont:g} A",
                    [inst.id],
                    f"use a supply rated >= {pk:.0f} A: {_ids(alts)}",
                )


def _mirror_rules(ctx, F):
    """MIRROR_UNSUPPORTED: a part inside a mirrored sub-assembly that declares neither [mirror] symmetry nor opposite."""
    from .subasm import PLANES

    for iid, inst in ctx.insts.items():
        if inst.mirror and inst.orig_part.mirror_plane() not in PLANES:
            _add(
                F,
                "MIRROR_UNSUPPORTED",
                "error",
                f"instance '{iid}' (part {inst.orig_part.id}) is in mirrored sub-assembly '{inst.mirror['sub']}' but the part "
                "declares no [mirror] symmetry / opposite",
                [iid],
                f'add a [mirror] table to {inst.orig_part.id} (symmetry = "xz" | "yz" | "xy", or opposite = "<part-id>"), '
                "or do not mirror this sub-assembly",
            )


# ---- driver ----------------------------------------------------------------------------------------------------------------------
def _guard(F, name, fn, *a):
    try:
        return fn(*a)
    except Exception as e:  # a rule bug must not hide the other rules or fail a build
        _add(
            F,
            "DRC_INTERNAL_ERROR",
            "warning",
            f"rule group {name} crashed: {type(e).__name__}: {e}",
            [name],
        )


def check(asm, lib=None, *, model=None, mesh_base=None, diag=None) -> list:
    """All design-rule findings for an assembly (Assembly or path), most severe first.
    `lib`: a Library (default <repo>/parts). `model`: (mjcf text, report) of an already computed build (build() passes it);
    `mesh_base`: the output directory that mesh paths in that MJCF are relative to (default: the repo root, as compute uses).
    `diag`: a dict that receives {"currents": {wire name: (worst-case peak A, sustained A or None)}} from the electrical model."""
    if not isinstance(asm, Assembly):
        asm = load(asm)
    lib = lib or Library(_default_parts())
    F = []
    try:
        ctx = Ctx(asm, lib)
    except DesignError as e:
        return [
            Finding("BUILD_FAILED", "error", str(e), [], "fix the part ids / library")
        ]
    specs = _guard(F, "mechanical", _mechanical, ctx, F) or {}
    wires = _guard(F, "wires", _wires, ctx, F) or []
    net = _guard(F, "nets", _Net, ctx, wires)
    _guard(F, "pins", pin_rules, ctx, F, _add)
    _guard(F, "mirror", _mirror_rules, ctx, F)
    if net:
        _guard(F, "voltage", _voltage_rules, ctx, net, F)
        _guard(F, "topology", _topology_rules, ctx, net, F)
    base = Path(mesh_base) if mesh_base is not None else lib.root
    report = None
    if model is None:
        try:
            model = compute(
                asm, lib.dir, base, lib, route=False
            )  # the rules never read the wire tubes: skip planning them
        except DesignError as e:
            if not any(f.severity == "error" for f in F):
                _add(
                    F,
                    "BUILD_FAILED",
                    "error",
                    str(e),
                    [],
                    "fix the assembly so that `dvsim design build` succeeds",
                )
            else:
                _add(
                    F,
                    "CHECKS_SKIPPED",
                    "info",
                    f"geometry and mass rules skipped, the assembly does not build ({e})",
                    [],
                )
    world = None
    if model:
        xml, report = model
        world = _guard(F, "geometry", _World, xml, base)
    mass = report["mass"]["total"] if report else None
    units = _units(ctx)
    if net:
        _guard(F, "performance", _performance, ctx, net, units, mass, F)
    servo_info = {}
    if world:
        _guard(F, "interference", _interference, ctx, world, specs, F)
        discs = _guard(F, "rotors", _rotor_discs, ctx, world, units) or []
        _guard(F, "props", _prop_rules, ctx, world, discs, F)
        _guard(F, "cg", _cg_rule, ctx, world, discs, mass, F)
        if net:
            servo_info = _guard(F, "servo", _servo_rules, ctx, net, world, F) or {}
        _guard(F, "reach", _reach_rules, ctx, world, F)
    if net:
        _guard(F, "servo bus", _servo_bus, ctx, net, servo_info, F)
    _guard(F, "thermal", _thermal, ctx, units, F)
    _guard(F, "devices", _device_unset, ctx, F)
    if net:
        _guard(F, "currents", _current_rules, ctx, net, units, servo_info, F, diag)
    F.sort(key=lambda f: (_RANK[f.severity], f.code, f.where))
    return F
