"""`dvsim program <scenario.toml>`: run an ordinary program against the simulated robot over the real protocol, with evidence.

USER CODE: the program (and a scenario's `model.script`) is user-authored code run on the user's machine. This is a CLI feature
only; there is deliberately no MCP tool for it (v1 rule: agents author declarative data - designs, TestSpecs - never code).
Limits are resource limits (setrlimit) and a process-group kill, NOT a security sandbox: a program that calls setsid() leaves
the group and is not killed, and it runs with the invoking user's permissions.

Scenario (TOML; relative paths are relative to the scenario file):

    name = "so101-pick"
    [model]                      # exactly one of (optional for px4: the bridge config names the model)
    assembly = "../../assemblies/so101.toml"   # + optional parts = "dir": built with dvsim.design.build
    mjcf = "robot.xml"                         # MJCF / URDF as is
    script = "scene.py"          # python file defining build(outdir: Path) -> Path of the MJCF (for composed scenes)
    [interfaces.feetech]         # a Feetech STS pty on a Host sim running at `limits.rate`; env DVSIM_SERIAL=<pty path>
    servos = { 1 = "/devices/servo1", 2 = "/devices/servo2" }
    baud = 1000000
    [interfaces.dynamixel]       # instead of feetech: a Dynamixel Protocol 2.0 pty, same Host sim / env; models = { id = "xl430-w250-t" |
    servos = { 1 = "/devices/servo1" }         #   "xl330-m288-t" | "xl330-m077-t" } per bus id (default xl430-w250-t), baud as above
    [interfaces.roboclaw]        # instead: a Basicmicro RoboClaw Packet Serial pty (dvsim.roboclaw): motors = { 1 = "/devices/wheel_l", 2 = ... },
    address = 128                #   address 0x80..0x87 (default 128), baud (default 38400)
    # controllers = [{ address = 129, motors = { 1 = "/devices/wheel_rl", 2 = "/devices/wheel_rr" } }]   # roboclaw: further RoboClaws, each its own
    #   pty (env DVSIM_SERIAL_2, ...), same baud; added for assemblies/car4.toml (4 motors on 2 RoboClaw 2x7A)
    [interfaces.native]          # the PROGRAM owns the World (a composed multi-vehicle scene from [model] script = ..., stepped by the program, e.g.
    #   dvsim.cuas.World): env DVSIM_MODEL = the MJCF, DVSIM_RUN_DIR = the run dir. The program calls World.finish(), which writes trace.csv
    #   (t + one column per monitor `column`), replay.dvsimog and native.json {tick, dt, state_hash}; the runner judges the monitors on
    #   trace.csv (as px4: `column = ...`) and re-runs the replay log through dvsim-replay to compare the hash. examples/cuas/cuas.scenario.toml.
    [interfaces.px4]             # PX4 SITL + dvsim-px4-bridge (own World); env DVSIM_MAVLINK=udp://:14540. Exactly one interface per scenario.
    bridge = "../../px4/bridge_x500.toml"
    # optional: params = { EKF2_EV_CTRL = 15, ... } (PX4 parameters on top of px4/params/dvsim_sitl.params), ctl = true (the bridge's --ctl socket
    # for the program: env DVSIM_CTL, DVSIM_MODEL, DVSIM_RUN_DIR; a companion computer reads its sensors there, examples/skydio/px4_fly.py),
    # monitor_trace = "companion.csv" (judge monitors on that CSV of the run dir, t in sim s, instead of the bridge's trace.csv)
    # px4 from a DESIGN instead of a hand-written bridge file: [design] path = "my-quad.toml" (a saved dvsim.design design / assembly; optional
    # parts = "dir") with a bare [interfaces.px4] (no `bridge`, no [model]). The world is built from the design and the bridge config
    # (motor index <-> ESC <-> PX4 output from the fc.pwmN -> escK.signal wires, sensors, home) is generated into <run dir>/bridge.toml.
    # A PWM wiring / prop direction that disagrees with the PX4 quad-X airframe (dvsim.design.px4cfg) is refused before PX4 starts.
    # arm from a DESIGN: [design] path = "my-arm.toml" with a feetech / dynamixel interface (no `servos` / `baud` needed: they come from the
    # design's servo_id / baud instance params); [model] script = "scene.py" then gets build(outdir, design=<path>, parts=<dir>).
    # px4 + feetech TOGETHER (the one allowed combination): one World, the bridge's, stepped in lockstep with PX4; the Feetech pty is
    # served from that World over the bridge's --ctl socket (dvsim.ctl). The programs run in order, each with DVSIM_SERIAL and
    # DVSIM_MAVLINK set: [[programs]] name = "arm"  cmd = [...]   then   name = "fly"  cmd = [...]   (optional wall_s / cpu_s each).
    # Monitors there: `column` (trace) or path + attr [+ frame = a body prim: the value in that body's frame] sampled at 10 Hz;
    # `phase = "<program name>"` makes from / to relative to that program's start (to defaults to its end), sim seconds.
    [[setup]]                    # optional SetAttr commands before the run (logged, replayed)
    path = "/devices/servo1"; attr = "torque_enable"; values = [0]
    [program]
    cmd = ["python", "{dir}/pick.py"]          # "python" = this interpreter, {dir} = scenario dir; cwd = a temp run dir
    # env = { KEY = "value" }                  # optional extra environment variables of the program
    [limits]                     # all optional
    wall_s = 120  sim_s = 60  rate = 1.0  cpu_s = 300  address_space_mb = 4096  open_files = 256
    [[monitors]]                 # TestSpec-style: op "always" (every sample in [lo, hi]) | "eventually" | "final" (last sample)
    name = "cube_at_target"      # sim kinds: signal = path/paths + attr + (index | indices [+ ref] -> norm of the difference)
    path = "/world/cube"  attr = "pos"  indices = [0, 1]  ref = [0.2, 0.04]  op = "final"  hi = 0.01
    # px4: column = a trace.csv column or the derived `alt` (z - z at start) / `dist_home` (horizontal m from the start) / `tilt` (deg)
    # optional from / to = seconds of sim time since the start (sim kinds: ticks x `dt`, default 0.001).
    # phase = "<program name>" | "fault:<fault name>" (all kinds; a single [program] is "program"): from / to are then relative to
    # that phase's start, `to` defaulting to its end (a fault: its duration, else the run's end) - "the quad recovers within 5 s
    # of the outage": phase = "fault:gps_out", from = 5, op = "final", ...
    [[faults]]                   # fault injection: scripted failures through the bus (SetAttr, logged, replayed; hash check holds)
    name = "gps_out"             # unique; default "<kind><index>"
    at = 40.0                    # sim s since the run's start (px4: since the bridge started, PX4 boot included) ...
    # after = "program+12.5"     # ... or sim s after a program (a [[programs]] name, or "program") started
    # log_marker = "HOLDING"     # optional with after: counts from when this text first shows in that program's log (it prints it)
    kind = "gnss_outage"         # see below
    target = "/devices/gnss"     # the device prim (not for comms_drop: the interface's serial bus)
    params = { }                 # per kind, below; unknown keys are an error
    duration = 10.0              # optional: the fault is undone after this many sim s (else it lasts to the end)
    # Kinds and their physics (the device attr behind each is in src/mujoco/devices/devices.cpp):
    #   motor_out          esc: output stage / winding open circuit (attr fault_open): no drive, the rotor coasts down on the prop's torque
    #   motor_degraded     rotor: damaged prop (attr damage = thrust_loss, default 0.3): thrust and torque x (1 - thrust_loss)
    #   battery_cell_fault battery: one weak / dead cell: r_extra_ohm (series resistance, attr r_fault), ocv_drop_v (EMF lost, attr
    #                      ocv_offset); the pack voltage sag follows from the Thevenin model: V = OCV - drop - I (r0 + r_extra) - v_rc
    #   gnss_outage        gnss: no fix (attr fix_lost): the output holds, the PX4 bridge sends HIL_GPS fix_type 0 / 0 satellites
    #   gnss_jump          gnss: position step params east / north / up (m, default 0), added to the samples from then on
    #   baro_drift         baro: pressure bias ramps at params rate_pa_s (default 1.0 Pa/s ~ 0.085 m/s of altitude); the bias stays
    #   imu_bias_step      imu: params gyro = [x, y, z] rad/s and / or accel = [x, y, z] m/s^2 ADDED to the current bias (undone: restored)
    #   servo_overload     servo: external torque params torque_nm (default 0.5) on the output joint (attr ext_torque)
    #   servo_stall        servo: the output joint is seized (attr jam): stiff spring + damper at the angle it had
    #   comms_drop         feetech / dynamixel: the pty discards every byte the host sends for `duration` (default 0.5 s): the real SDK
    #                      sees timeouts. Bus-side only, so nothing about it is in the replay log (the commands that did arrive are).
    # result.json["faults"]: per fault {name, kind, start_t, end_t, error}, times in sim s of the World; a fault that never fired
    # (program ended first) or failed to apply is a failed monitor `fault_<name>_applied`.

Env of the program: DVSIM_SERIAL / DVSIM_MAVLINK, DVSIM_SCENARIO_DIR, plus the caller's environment. Run dir `<runs-dir>/<name>-<time>/`:
result.json, program.log (stdout+stderr), replay.dvsimog (replay log), trace.csv (sampled monitor signals / bridge trace), and for
px4 bridge.log + px4.log. result.json["pass"] = program exit 0 AND no timeout AND every monitor passed. The final state hash of
the run is replayed with dvsim-replay when that exe exists (result.json["replay"]).
Exit code (see cli): 0 pass, 1 fail, 2 error.
"""

import csv
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
REPLAY = Path(os.environ.get("DVSIM_REPLAY_EXE", REPO / "build" / "dvsim-replay"))
LIMITS = dict(wall_s=120.0, cpu_s=300, address_space_mb=4096, open_files=256, rate=1.0)
# Runs in the child before exec: apply the rlimits, then become the program (no preexec_fn: this process has threads).
LAUNCH = (
    "import os,sys,json,resource;"
    "[resource.setrlimit(getattr(resource,k),tuple(v)) for k,v in json.loads(sys.argv[1]).items()];"
    "os.execvp(sys.argv[2],sys.argv[2:])"
)


class ScenarioError(Exception):
    pass


def load_scenario(path) -> dict:
    path = Path(path).resolve()
    try:
        sc = tomllib.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise ScenarioError(f"cannot read scenario '{path}': {e}") from None
    sc["dir"] = path.parent
    sc.setdefault("name", path.stem.removesuffix(".scenario"))
    sc["limits"] = {**LIMITS, **sc.get("limits", {})}
    ifs = sc.get("interfaces", {})
    if not (set(ifs) <= {"feetech", "dynamixel", "roboclaw", "px4", "native"} and ifs):
        raise ScenarioError(
            "[interfaces] needs one of feetech / dynamixel / roboclaw / px4 / native"
        )
    if (
        "native" in ifs
    ):  # the program owns the World: no bus, no faults, the model comes from a [model] script
        if len(ifs) > 1 or sc.get("faults") or "script" not in sc.get("model", {}):
            raise ScenarioError(
                "[interfaces.native] stands alone, takes [model] script = ... and no [[faults]]"
            )
        if not sc.get("program", {}).get("cmd"):
            raise ScenarioError("[program] cmd is required")
        return sc
    _check_faults(sc, ifs)
    if "design" in sc and "px4" not in ifs:
        _arm_design(sc, ifs)
    elif (
        "design" in sc
    ):  # a saved design: the world AND the bridge config come from it (_bridge_cfg)
        if "bridge" in ifs["px4"] or "model" in sc:
            raise ScenarioError(
                "[design] goes with [interfaces.px4] only, without `bridge` and without [model]"
            )
        sc["model"] = {k: v for k, v in sc["design"].items() if k in ("parts",)} | {
            "assembly": sc["design"].get("path", "")
        }
        if not sc["design"].get("path"):
            raise ScenarioError(
                "[design] path = <saved design / assembly toml> is required"
            )
    elif "px4" in ifs and "bridge" not in ifs["px4"]:
        raise ScenarioError(
            "[interfaces.px4] needs `bridge` (or a [design] to generate it from)"
        )
    if set(ifs) == {
        "feetech",
        "px4",
    }:  # one World in the bridge, the pty served over its ctl socket (_combo)
        if not sc.get("programs") or any(
            not p.get("cmd") or not p.get("name") for p in sc["programs"]
        ):
            raise ScenarioError(
                "px4 + feetech needs [[programs]], each with name and cmd"
            )
        return sc
    if (
        len(ifs) > 1
    ):  # px4 has its own World in the bridge; the two serial buses would share one DVSIM_SERIAL
        raise ScenarioError(
            f"[interfaces] takes exactly one interface, got {sorted(ifs)}"
        )
    if not sc.get("program", {}).get("cmd"):
        raise ScenarioError("[program] cmd is required")
    return sc


def _arm_design(sc, ifs):
    """[design] with a feetech / dynamixel interface: the saved design (path, optional parts) is the arm. Without [model] the
    design is built as it is; with [model] script = "..." the script composes the scene around it and gets
    build(outdir, design=<path>, parts=<dir>). An interface without `servos` takes {servo_id: "/devices/<servo>"} from the design's
    servo devices (the instance's `servo_id` param) and, without `baud`, the bus baud (the instances' `baud` / baud_default)."""
    from .design import DesignError, Library, load
    from .design.assembly import Ctx

    d = sc["design"]
    if not d.get("path"):
        raise ScenarioError(
            "[design] path = <saved design / assembly toml> is required"
        )
    path, parts = sc["dir"] / d["path"], sc["dir"] / d.get("parts", REPO / "parts")
    m = sc.setdefault("model", {})
    if m and set(m) != {"script"}:
        raise ScenarioError(
            "[design] with feetech / dynamixel: [model] may only be a `script`"
        )
    if not m:
        m.update(assembly=d["path"], **({"parts": d["parts"]} if "parts" in d else {}))
    if all("servos" in ifs[k] for k in ifs):
        return
    try:
        asm = load(path)
        ctx = Ctx(asm, Library(parts))
        devs = [b for b in asm.doc.get("devices", []) if b["type"] == "servo"]
        ids = {b["name"]: ctx.insts[b["instance"]].value("servo_id") for b in devs}
        bauds = {
            ctx.insts[b["instance"]].value("baud")
            or ctx.insts[b["instance"]].value("baud_default")
            for b in devs
        }
    except (DesignError, KeyError, OSError) as e:
        raise ScenarioError(f"[design] {path}: {e}") from None
    if not devs or None in ids.values():
        raise ScenarioError(
            f"[design] {path}: every servo device needs a servo_id param to derive `servos` (missing: {[n for n, i in ids.items() if i is None]})"
        )
    for k in ifs:
        ifs[k].setdefault("servos", {int(i): f"/devices/{n}" for n, i in ids.items()})
        if len(bauds) == 1 and None not in bauds:
            ifs[k].setdefault("baud", int(bauds.pop()))


# ---- fault injection ([[faults]]) ----
def _vec(p, key):
    v = [float(x) for x in p[key]]
    if len(v) != 3:
        raise ScenarioError(f"fault param {key} needs 3 values, got {len(v)}")
    return v


def _imu_bias(t, p, get):
    ap, rs = [], []
    for key, attr in (("gyro", "gyro_bias"), ("accel", "accel_bias")):
        if key in p:
            cur = get(t, attr)
            ap.append((t, attr, list(cur + _vec(p, key))))
            rs.append((t, attr, list(cur)))
    return ap, rs


# kind -> (allowed params with defaults, ops(target, params, get) -> (apply, restore) lists of (path, attr, values)).
# get(path, attr) -> ndarray (the current value, for steps that are relative). comms_drop has no ops: it flips the port's `drop`.
FAULTS = {
    "motor_out": (
        {},
        lambda t, p, g: ([(t, "fault_open", [1])], [(t, "fault_open", [0])]),
    ),
    "motor_degraded": (
        {"thrust_loss": 0.3},
        lambda t, p, g: ([(t, "damage", [p["thrust_loss"]])], [(t, "damage", [0])]),
    ),
    "battery_cell_fault": (
        {"r_extra_ohm": 0.0, "ocv_drop_v": 0.0},
        lambda t, p, g: (
            [(t, "r_fault", [p["r_extra_ohm"]]), (t, "ocv_offset", [-p["ocv_drop_v"]])],
            [(t, "r_fault", [0]), (t, "ocv_offset", [0])],
        ),
    ),
    "gnss_outage": (
        {},
        lambda t, p, g: ([(t, "fix_lost", [1])], [(t, "fix_lost", [0])]),
    ),
    "gnss_jump": (
        {"east": 0.0, "north": 0.0, "up": 0.0},
        lambda t, p, g: (
            [(t, "jump", [p["east"], p["north"], p["up"]])],
            [(t, "jump", [0, 0, 0])],
        ),
    ),
    "baro_drift": (
        {"rate_pa_s": 1.0},
        lambda t, p, g: (
            [(t, "drift_rate", [p["rate_pa_s"]])],
            [(t, "drift_rate", [0])],
        ),
    ),
    "imu_bias_step": ({"gyro": None, "accel": None}, _imu_bias),
    "servo_overload": (
        {"torque_nm": 0.5},
        lambda t, p, g: (
            [(t, "ext_torque", [p["torque_nm"]])],
            [(t, "ext_torque", [0])],
        ),
    ),
    "servo_stall": ({}, lambda t, p, g: ([(t, "jam", [1])], [(t, "jam", [0])])),
    "comms_drop": ({}, None),
}


def _check_faults(sc, ifs):
    seen = set()
    for i, f in enumerate(sc.get("faults", [])):
        k = f.get("kind")
        if k not in FAULTS:
            raise ScenarioError(f"[[faults]] kind {k!r}: one of {sorted(FAULTS)}")
        f.setdefault("name", f"{k}{i}")
        if f["name"] in seen:
            raise ScenarioError(f"[[faults]] duplicate name '{f['name']}'")
        seen.add(f["name"])
        if ("at" in f) == ("after" in f):
            raise ScenarioError(f"fault '{f['name']}': exactly one of at / after")
        if "after" in f and not re.fullmatch(r"[^+\s]+\+[0-9.]+", f["after"]):
            raise ScenarioError(
                f"fault '{f['name']}': after = \"<program>+<seconds>\", got {f['after']!r}"
            )
        if "log_marker" in f and "after" not in f:
            raise ScenarioError(f"fault '{f['name']}': log_marker needs after")
        if k == "comms_drop":
            if not set(ifs) & {"feetech", "dynamixel"}:
                raise ScenarioError(
                    f"fault '{f['name']}': comms_drop needs a feetech / dynamixel interface"
                )
            if "duration" not in f:
                f["duration"] = 0.5
        elif not f.get("target"):
            raise ScenarioError(
                f"fault '{f['name']}': target (a /devices/... prim) is required"
            )
        allowed = FAULTS[k][0]
        if bad := set(f.get("params", {})) - set(allowed):
            raise ScenarioError(
                f"fault '{f['name']}': unknown params {sorted(bad)} (kind {k}: {sorted(allowed)})"
            )
        f["params"] = {
            **{a: v for a, v in allowed.items() if v is not None},
            **f.get("params", {}),
        }
        if k == "imu_bias_step" and not ({"gyro", "accel"} & set(f["params"])):
            raise ScenarioError(
                f"fault '{f['name']}': imu_bias_step needs params gyro and / or accel"
            )
        if f.get("duration", 1) <= 0:
            raise ScenarioError(f"fault '{f['name']}': duration must be > 0")


class _Faults:
    """Applies [[faults]] from a thread by polling the World's tick (~100 Hz), through sim.set (SetAttr: logged, replayed).
    sim: Sim / RemoteSim (tick, set, query); origin: sim s that `at` counts from; port: Feetech / Dynamixel port (comms_drop);
    logs: {program name: log path} (for log_marker); starts: {program name: sim s it started}, filled in by the runner."""

    def __init__(self, faults, sim, dt, origin=0.0, port=None, logs=None):
        self.f, self.sim, self.dt, self.origin, self.port, self.logs = (
            faults,
            sim,
            dt,
            origin,
            port,
            logs or {},
        )
        self.starts, self.undo, self.marker_t = {}, {}, {}
        self.log = [dict(name=f["name"], kind=f["kind"], target=f.get("target"), params=f["params"], start_t=None,
                         end_t=None, error=None) for f in faults]  # fmt: skip
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        if self.f:
            self._th.start()

    def close(self, end_t):
        self._stop.set()
        if self.f:
            self._th.join()
        for r in self.log:
            if r["start_t"] is not None and r["end_t"] is None:
                r["end_t"] = end_t  # still active when the run ended

    def _due(self, f, t):
        if "at" in f:
            return self.origin + float(f["at"])
        prog, _, s = f["after"].rpartition("+")
        if prog not in self.starts:
            return None
        base = self.starts[prog]
        if "log_marker" in f:
            if prog not in self.marker_t:
                try:
                    if f["log_marker"] in self.logs[prog].read_text(errors="replace"):
                        self.marker_t[prog] = t
                except (OSError, KeyError):
                    pass
            if prog not in self.marker_t:
                return None
            base = self.marker_t[prog]
        return base + float(s)

    def _apply(self, f, r):
        try:
            if f["kind"] == "comms_drop":
                self.port.drop = True
                self.undo[f["name"]] = lambda: setattr(self.port, "drop", False)
            else:
                ap, rs = FAULTS[f["kind"]][1](
                    f["target"], f["params"], lambda p, a: self.sim.query(p, a)[0]
                )
                for op in ap:
                    self.sim.set(*op)
                self.undo[f["name"]] = lambda: [self.sim.set(*op) for op in rs]
            r["start_t"] = self.sim.tick * self.dt
        except (ValueError, KeyError, ScenarioError) as e:
            r["error"] = str(e)

    def _loop(self):
        while not self._stop.wait(0.01):
            try:
                t = self.sim.tick * self.dt
                for f, r in zip(self.f, self.log):
                    if r["start_t"] is None and r["error"] is None:
                        due = self._due(f, t)
                        if due is not None and t >= due:
                            self._apply(f, r)
                    elif r["start_t"] is not None and r["end_t"] is None and "duration" in f \
                            and t >= r["start_t"] + float(f["duration"]):  # fmt: skip
                        self.undo[f["name"]]()
                        r["end_t"] = self.sim.tick * self.dt
            except (OSError, ValueError):  # the bridge / sim went away
                return

    def verdicts(self):
        """Failed monitors for faults that never fired or could not be applied."""
        return [{"name": f"fault_{r['name']}_applied", "pass": False,
                 "error": r["error"] or "never triggered (the program ended first, or its phase / log_marker never came)"}
                for r in self.log if r["start_t"] is None]  # fmt: skip

    def spans(self):
        return {
            f"fault:{r['name']}": (r["start_t"], r["end_t"])
            for r in self.log
            if r["start_t"] is not None
        }


def _phase(m, spans, base=0.0):
    """Monitor with phase = "<program>" | "fault:<name>" -> (monitor with absolute from / to in the series' time base, None), or
    (None, failed result). spans: {phase: (start, end)} in World sim s; base: the series' time origin in World sim s."""
    if "phase" not in m:
        return m, None
    sp = spans.get(m["phase"])
    if sp is None:
        return None, {
            "name": m["name"],
            "pass": False,
            "error": f"phase '{m['phase']}' did not run",
        }
    s, e = sp[0] - base, (sp[1] if sp[1] is not None else spans["end"]) - base
    return {
        **m,
        "from": s + m.get("from", 0),
        "to": s + m["to"] if "to" in m else e,
    }, None


def _derive(tr):
    """Derived px4 trace columns: alt (z - z at start), dist_home (horizontal m from the start), tilt (deg between the body z axis and
    world z: 0 upright, 180 upside down)."""
    tr["alt"] = tr["z"] - tr["z"][0]
    tr["dist_home"] = np.hypot(tr["x"] - tr["x"][0], tr["y"] - tr["y"][0])
    tr["tilt"] = np.degrees(
        np.arccos(np.clip(1 - 2 * (tr["qx"] ** 2 + tr["qy"] ** 2), -1, 1))
    )


def _judge_all(mons, spans, base, series_of):
    out = []
    for m in mons:
        w, err = _phase(m, spans, base)
        out.append(err or judge(w, series_of(m)))
    return out


def _val(m, v):
    if "index" in m:
        return float(v[m["index"]])
    x = v if "indices" not in m else v[m["indices"]]
    return (
        float(np.linalg.norm(x - np.asarray(m["ref"], float)))
        if "ref" in m
        else float(x[0] if len(x) == 1 else np.linalg.norm(x))
    )


def judge(m, series) -> dict:
    """series [(t, value)] -> result dict. Same semantics as dvsim.TestSpec monitors, plus op "final"."""
    lo, hi = float(m.get("lo", -math.inf)), float(m.get("hi", math.inf))
    win = [(t, v) for t, v in series if m.get("from", 0) <= t <= m.get("to", math.inf)]
    r = dict(
        name=m["name"],
        op=m.get("op", "always"),
        lo=lo,
        hi=hi,
        samples=len(win),
    )
    if not win:
        return {**r, "pass": False, "error": "no samples in the window"}
    win = [(float(t), float(v)) for t, v in win]
    vals = [v for _, v in win]
    ok = [lo <= v <= hi for v in vals]
    hit = {"always": lambda: next((i for i, k in enumerate(ok) if not k), None),
           "eventually": lambda: next((i for i, k in enumerate(ok) if k), None),
           "final": lambda: len(ok) - 1}[r["op"]]()  # fmt: skip
    p = {"always": hit is None, "eventually": hit is not None, "final": ok[-1]}[r["op"]]
    r.update(min=min(vals), max=max(vals), final=vals[-1], **{"pass": p})
    if hit is not None:
        r.update(hit_t=win[hit][0], hit_value=win[hit][1])
    return r


def _expand(monitors, kind):
    out = []
    for m in monitors:
        if "name" not in m:
            raise ScenarioError(f"monitor without a name: {m}")
        if kind in ("px4", "native") or (kind == "combo" and "column" in m):
            if "column" not in m:
                raise ScenarioError(f"px4 monitor '{m['name']}' needs `column`")
            out.append(m)
            continue
        paths = m.get("paths") or [m.get("path")]
        if None in paths or "attr" not in m:
            raise ScenarioError(f"monitor '{m['name']}' needs path|paths and attr")
        out += [
            {
                **m,
                "path": p,
                "name": m["name"] if len(paths) == 1 else f"{m['name']}:{p}",
            }
            for p in paths
        ]
    return out


def replay_hash(log, root=None):
    """dvsim-replay on a replay log -> (tick, hash hex) or None if the exe is missing."""
    if not REPLAY.exists():
        return None
    argv = [str(REPLAY), str(log)] + (["--root", str(root)] if root else [])
    r = subprocess.run(argv, capture_output=True, text=True, timeout=600)
    m = re.search(r"tick=(\d+) state_hash=([0-9a-f]+)", r.stdout)
    if r.returncode or not m:
        raise RuntimeError(f"dvsim-replay failed: {r.stderr or r.stdout}")
    return int(m[1]), m[2]


def _replay_root(sc, model):
    """dvsim-replay's --root: the run's model folder, but a live run's log covers the viewer's whole session (its design loads too),
    so those replay under the Sim's own root (live["root"])."""
    return (sc.get("_live") or {}).get("root") or Path(model).parent


def _group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def kill_group(proc, grace=2.0):
    """SIGTERM then SIGKILL proc's whole process group; returns once no member is left (or after a few seconds)."""
    pgid = proc.pid
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        end = time.monotonic() + wait
        while time.monotonic() < end:
            proc.poll()  # reap the leader: a zombie keeps its group "alive"
            if not _group_alive(pgid):
                return
            time.sleep(0.02)


def run_child(argv, env, cwd, log, limits, stop_if=None):
    """Start the program in its own session / process group with rlimits; wait up to wall_s (or until stop_if() is a reason
    string). -> dict(returncode, signal, wall_s, timeout: None|"wall"|<stop_if reason>). The group is always killed at the end."""
    lim = {"RLIMIT_CPU": [int(limits["cpu_s"]), int(limits["cpu_s"]) + 2],
           "RLIMIT_AS": [int(limits["address_space_mb"]) << 20] * 2,
           "RLIMIT_NOFILE": [int(limits["open_files"])] * 2}  # fmt: skip
    t0 = time.monotonic()
    with open(log, "wb") as lf:
        p = subprocess.Popen([sys.executable, "-c", LAUNCH, json.dumps(lim), *argv], cwd=cwd, env=env,
                             stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)  # fmt: skip
        timeout = None
        while p.poll() is None:
            if time.monotonic() - t0 > limits["wall_s"]:
                timeout = "wall"
            elif stop_if and (timeout := stop_if()):
                pass
            if timeout:
                break
            time.sleep(0.05)
        kill_group(p)  # also reaps background children a finished program left behind
        rc = p.wait()
    return dict(returncode=rc, signal=signal.Signals(-rc).name if rc < 0 else None,
                wall_s=time.monotonic() - t0, timeout=timeout, pgid=p.pid)  # fmt: skip


def _build_model(sc, out: Path):
    m = sc.get("model", {})
    d = sc["dir"]
    n = sum(k in m for k in ("assembly", "mjcf", "script"))
    if n != 1:
        raise ScenarioError("[model] needs exactly one of assembly / mjcf / script")
    out.mkdir(parents=True, exist_ok=True)
    if "mjcf" in m:
        return d / m["mjcf"]
    if "assembly" in m:
        from .design import DesignError, build

        try:
            return Path(
                build(d / m["assembly"], d / m.get("parts", REPO / "parts"), out)["xml"]
            )
        except DesignError as e:
            raise ScenarioError(str(e)) from None
    spec = importlib.util.spec_from_file_location(
        "dvsim_scenario_script", d / m["script"]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str((d / m["script"]).parent))  # the script's own siblings
    spec.loader.exec_module(mod)
    kw = (
        dict(  # [design] + script: the script composes its scene around the saved design
            design=d / sc["design"]["path"],
            parts=d / sc["design"].get("parts", REPO / "parts"),
        )
        if "design" in sc
        else {}
    )
    return Path(mod.build(out, **kw))


def _stop_if(sc, other=None):
    """run_child's stop_if: a live run's stop event ("stopped"), else `other`."""
    ev = (sc.get("_live") or {}).get("stop")
    return lambda: "stopped" if ev and ev.is_set() else other and other()


def _live_sim(sc, model):
    """-> (the Sim to run in, the pacing rate) for a live run (the viewer's World, scenario model loaded into it), else (None, None)."""
    live = sc.get("_live")
    if not live:
        return None, None
    sim = live["sim"]
    sim.call({"cmd_type": "Pause", "cmd": {}})
    r = sim.call(
        {"cmd_type": "LoadModel", "cmd": {"path": str(model)}}
    )  # confined to the Sim's root: the caller builds under it
    if not r["ok"]:
        raise ScenarioError(
            f"live run: cannot load the scenario model into the viewer's World: {r['message']}"
        )
    return sim, float(
        sc["limits"]["rate"] if live.get("rate") is None else live["rate"]
    )


def _model_dt(model):
    """Seconds per tick of a built scene: its scene.json `dt`, else the MJCF option timestep (kernel default 0.002)."""
    sj = Path(model).parent / "scene.json"
    if sj.exists():
        return float(json.loads(sj.read_text())["dt"])
    o = ET.parse(model).getroot().find("option")
    return float(o.get("timestep")) if o is not None and o.get("timestep") else 0.002


def run(scenario, runs_dir="runs", live=None) -> tuple:
    """-> (result dict, run dir). Raises ScenarioError for a bad scenario / missing prerequisites.
    `live` = dict(sim=<host=True Sim, e.g. the viewer's>, rate=<sim s per wall s, None = limits.rate, 0 = unpaced>, stop=<Event>,
    on_dir=<callable(run dir)>): the scenario's model is loaded INTO that Sim and the program drives it (native: over dvsim.attach;
    feetech / dynamixel / roboclaw: the ports use it directly), so whoever shows the Sim sees the run as it happens. The Sim's replay
    log then covers its whole session; the replay check replays all of it. px4 / combo cannot run live."""
    sc = load_scenario(scenario)
    ifs = sc["interfaces"]
    lim, kind = sc["limits"], ("combo" if len(ifs) > 1 else next(iter(ifs)))
    if live:
        if kind in ("px4", "combo"):
            raise ScenarioError(
                f"live run: the {kind} interface owns its own sim processes and cannot run in the viewer's World"
            )
        sc["_live"] = live
    run_dir = (
        Path(runs_dir).resolve() / f"{sc['name']}-{time.strftime('%Y%m%d-%H%M%S')}"
    )  # absolute: programs run in run_dir/cwd and get paths under it (DVSIM_MODEL, DVSIM_RUN_DIR)
    run_dir.mkdir(parents=True)
    cwd = run_dir / "cwd"
    cwd.mkdir()
    sub = lambda s: s.replace("{dir}", str(sc["dir"]))
    argvs = lambda cmd: [
        sys.executable if a == "python" and i == 0 else sub(a)
        for i, a in enumerate(cmd)
    ]
    argv = argvs(sc["program"]["cmd"]) if kind != "combo" else None
    env = {
        **os.environ,
        "DVSIM_SCENARIO_DIR": str(sc["dir"]),
        **sc.get("program", {}).get("env", {}),
    }
    mons = _expand(sc.get("monitors", []), kind)
    res = dict(
        name=sc["name"],
        kind=kind,
        scenario=str(Path(scenario).resolve()),
        argv=argv,
        limits=lim,
    )
    (run_dir / "program.log").touch()
    log = run_dir / "program.log"
    if live and live.get("on_dir"):
        live["on_dir"](run_dir)
    if kind == "combo":
        res["argv"] = argv = [argvs(p["cmd"]) for p in sc["programs"]]
    {"px4": _px4, "combo": _combo, "native": _native}.get(kind, _sim)(
        sc, mons, run_dir, cwd, argv, env, log, res
    )
    p = res["program"]
    why = []
    if p["returncode"]:
        why.append(
            f"program exit {p['returncode']}"
            + (f" ({p['signal']})" if p["signal"] else "")
        )
    if p["timeout"]:
        why.append(f"timeout: {p['timeout']}")
    why += [f"monitor {m['name']}" for m in res["monitors"] if not m["pass"]]
    if not res["replay"].get("match", True):
        why.append("replay hash mismatch")
    res.update({"pass": not why, "fail_reasons": why})
    (run_dir / "result.json").write_text(json.dumps(res, indent=2))
    return res, run_dir


def _sim(sc, mons, run_dir, cwd, argv, env, log, res):
    from . import Sim
    from .dynamixel import DynamixelPort
    from .feetech import FeetechPort
    from .roboclaw import RoboclawPort

    fc, lim = sc["interfaces"][res["kind"]], sc["limits"]
    dxl, rc = res["kind"] == "dynamixel", res["kind"] == "roboclaw"
    Port = RoboclawPort if rc else DynamixelPort if dxl else FeetechPort
    extra = (
        dict(models={int(i): m for i, m in fc.get("models", {}).items()})
        if dxl
        else dict(address=fc.get("address", 0x80))
        if rc
        else {}
    )
    model = _build_model(sc, run_dir / "model")
    sim, rate = _live_sim(sc, model)
    sim = sim or Sim.load(model, host=True)
    try:
        for s in sc.get("setup", []):
            for p in s.get("paths") or [s["path"]]:
                sim.set(p, s["attr"], s["values"])
        port = Port(
            sim,
            {int(i): p for i, p in fc["motors" if rc else "servos"].items()},
            baud=fc.get("baud", 38400 if rc else 1_000_000),
            realtime=True,
            **extra,
        )
        # roboclaw only: `controllers = [{address, motors}, ...]` = further RoboClaws, each on its own pty (separate USB cables),
        # env DVSIM_SERIAL_2, DVSIM_SERIAL_3, ... (DVSIM_SERIAL = the first)
        more = [
            Port(
                sim,
                {int(i): p for i, p in c["motors"].items()},
                baud=fc.get("baud", 38400),
                realtime=True,
                address=c.get("address", 0x80),
            )
            for c in (fc.get("controllers", []) if rc else [])
        ]
    except (ValueError, KeyError) as e:
        raise ScenarioError(str(e)) from None
    dt, series, rows, stop = (
        sc.get("dt", 0.001),
        {m["name"]: [] for m in mons},
        [],
        threading.Event(),
    )
    tick0 = sim.tick
    wall0 = time.monotonic()
    F = _Faults(sc.get("faults", []), sim, dt, tick0 * dt, port, {"program": log})

    def sample():
        t, row = (
            (sim.tick - tick0) * dt,
            [time.monotonic() - wall0, sim.tick - tick0, (sim.tick - tick0) * dt],
        )
        for m in mons:
            v = _val(m, sim.query(m["path"], m["attr"], m.get("frame", ""))[0])
            series[m["name"]].append((t, v))
            row.append(v)
        rows.append(row)

    def sampler():
        while not stop.wait(0.02):
            sample()

    env["DVSIM_SERIAL"] = port.device_path
    for n, p in enumerate(more, 2):
        env[f"DVSIM_SERIAL_{n}"] = p.device_path
    th = threading.Thread(target=sampler, daemon=True)
    try:
        r = sim.call(
            {
                "cmd_type": "Run",
                "cmd": {"rate": float(lim["rate"] if rate is None else rate)},
            }
        )
        if not r["ok"]:
            raise ScenarioError(r["message"])
        th.start()
        sim_s = lim.get("sim_s")
        F.starts["program"] = sim.tick * dt
        F.start()
        res["program"] = run_child(argv, env, cwd, log, lim, _stop_if(sc,
                                   (lambda: "sim_time" if (sim.tick - tick0) * dt > sim_s else None) if sim_s else None))  # fmt: skip
    finally:
        stop.set()
        th.join()
        F.close(sim.tick * dt)
        for p in (port, *more):
            p.close()
        sim.call({"cmd_type": "Pause", "cmd": {}})
    sample()
    h = f"{sim.state_hash():016x}"
    (run_dir / "replay.dvsimog").write_bytes(sim.log())
    res["sim"] = dict(tick=sim.tick - tick0, sim_s=(sim.tick - tick0) * dt, state_hash=h, diverged=sim.diverged,
                      port_errors=[e for p in (port, *more) for e in p.errors])  # fmt: skip
    end = sim.tick * dt
    spans = {"program": (F.starts["program"], end), **F.spans(), "end": end}
    res["monitors"] = (
        _judge_all(mons, spans, tick0 * dt, lambda m: series[m["name"]]) + F.verdicts()
    )
    res["faults"] = F.log
    with open(run_dir / "trace.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wall_s", "tick", "t", *[m["name"] for m in mons]])
        w.writerows(rows)
    rp = replay_hash(run_dir / "replay.dvsimog", _replay_root(sc, model))
    res["replay"] = (
        {"available": False}
        if rp is None
        else dict(
            available=True, tick=rp[0], state_hash=rp[1], match=rp == (sim.tick, h)
        )
    )
    if res["sim"]["diverged"]:
        res["monitors"].append({"name": "solver_diverged", "pass": False})


def _native(sc, mons, run_dir, cwd, argv, env, log, res):
    """[interfaces.native]: the program owns the World (a multi-vehicle scene it steps itself, e.g. dvsim.cuas.World). The runner builds
    the model from [model] script, starts the program with env DVSIM_MODEL (the MJCF) and DVSIM_RUN_DIR, then judges what the program
    left in the run dir: native.json {tick, dt, state_hash, diverged}, replay.dvsimog and trace.csv (t + one column per monitor `column`),
    and re-runs the replay log through dvsim-replay to compare the hash. World.finish() writes all three."""
    model = _build_model(sc, run_dir / "model")
    env.update(DVSIM_MODEL=str(model), DVSIM_RUN_DIR=str(run_dir))
    sim, rate = _live_sim(sc, model)
    srv = tmp = None
    if sim:  # live: the program's Sim.load(DVSIM_MODEL) becomes a proxy for the viewer's World (dvsim.attach)
        from .attach import Server

        tmp = tempfile.mkdtemp(
            prefix="dvsim-att-"
        )  # unix socket paths are short; the dir is 0700
        srv = Server(sim, Path(tmp) / "s", _model_dt(model), rate)
        env.update(DVSIM_ATTACH=str(srv.path), DVSIM_ATTACH_MODEL=os.path.realpath(model))
    try:
        res["program"] = run_child(argv, env, cwd, log, sc["limits"], _stop_if(sc))
    finally:
        if srv:
            srv.close()
            shutil.rmtree(tmp, ignore_errors=True)
    done = run_dir / "native.json"
    if not done.exists():  # crashed / killed before World.finish()
        res["sim"], res["replay"] = {}, {"available": False}
        res["monitors"] = [
            {
                "name": "world_finished",
                "pass": False,
                "error": "no native.json: see program.log",
            }
        ]
        return  # fmt: skip
    fin = json.loads(done.read_text())
    res["sim"] = fin
    with open(run_dir / "trace.csv") as f:
        rows = list(csv.DictReader(f))
    tr = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    bad = [m["name"] for m in mons if m["column"] not in tr]
    if bad:
        raise ScenarioError(f"monitors {bad}: no such trace column (have {sorted(tr)})")
    spans = {"program": (0.0, float(tr["t"][-1])), "end": float(tr["t"][-1])}
    res["monitors"] = _judge_all(
        mons, spans, 0.0, lambda m: list(zip(tr["t"], tr[m["column"]]))
    )
    if fin.get("diverged"):
        res["monitors"].append({"name": "solver_diverged", "pass": False})
    rp = replay_hash(run_dir / "replay.dvsimog", _replay_root(sc, model))
    res["replay"] = {"available": False} if rp is None else dict(
        available=True, tick=rp[0], state_hash=rp[1], match=rp == (fin["tick"], fin["state_hash"]))  # fmt: skip


def _bridge_cfg(sc, run_dir, model):
    """The bridge config: the scenario's `bridge` file, or (with [design]) generated from the design into run_dir/bridge.toml, which
    refuses a motor order / prop direction that does not match the PX4 airframe BEFORE PX4 is started (dvsim.design.px4cfg)."""
    if "design" not in sc:
        return sc["dir"] / sc["interfaces"]["px4"]["bridge"]
    from .design import DesignError
    from .design.px4cfg import bridge_config

    d = sc["design"]
    try:
        return bridge_config(
            sc["dir"] / d["path"],
            model,
            run_dir / "bridge.toml",
            sc["dir"] / d.get("parts", REPO / "parts"),
        )
    except DesignError as e:
        raise ScenarioError(str(e)) from None


def _px4(sc, mons, run_dir, cwd, argv, env, log, res):
    from .px4 import Stack

    model = _build_model(sc, run_dir / "model") if sc.get("model") else None
    cfg = _bridge_cfg(sc, run_dir, model)
    dt, faults = sc.get("dt", 0.001), sc.get("faults", [])
    ifc = sc["interfaces"]["px4"]
    st = Stack(
        cfg,
        run_dir,
        model=model,
        ctl=bool(faults) or bool(ifc.get("ctl")),
        params=ifc.get("params"),
    )  # faults reach the bridge's World over its ctl socket; so does a companion program (`ctl = true`: env DVSIM_CTL)
    env["DVSIM_MAVLINK"] = st.mavlink
    env["DVSIM_RUN_DIR"] = str(run_dir)
    if model:
        env["DVSIM_MODEL"] = str(model)
    if st.ctl:
        env["DVSIM_CTL"] = str(st.ctl)
    try:
        st.start()
    except RuntimeError as e:
        raise ScenarioError(f"PX4 stack: {e}") from None
    rs = F = None
    try:
        if faults:
            from .ctl import RemoteSim

            rs = RemoteSim(st.ctl)
        F = _Faults(faults, rs, dt, 0.0, None, {"program": log})
        if faults:
            F.starts["program"] = rs.tick * dt
            F.start()
        res["program"] = run_child(argv, env, cwd, log, sc["limits"])
    finally:
        if rs:
            F.close(rs.tick * dt)
            rs.close()
        final = st.stop()
    res["sim"] = dict(final or {}, bridge_log="bridge.log", px4_log="px4.log")
    with open(
        run_dir / ifc["monitor_trace"] if "monitor_trace" in ifc else st.trace
    ) as f:
        rows = list(
            csv.DictReader(f)
        )  # monitor_trace: the companion program's own trace (t = sim s, x y z qw qx qy qz + its columns)
    tr = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    _derive(tr)
    bad = [m["name"] for m in mons if m["column"] not in tr]
    if bad:
        raise ScenarioError(f"monitors {bad}: no such trace column (have {sorted(tr)})")
    t_end = float(tr["t"][-1])
    prog_start = F.starts.get(
        "program", float(tr["t"][0])
    )  # no faults: the program ran from ~ the first trace row on
    res["program"]["start_t"] = (
        prog_start  # World sim s: program.log's stamps + this = trace.csv's t (flight.py logs wall ~ sim s)
    )
    spans = {"program": (prog_start, t_end), **F.spans(), "end": t_end}
    res["monitors"] = (
        _judge_all(
            mons,
            spans,
            float(tr["t"][0]),
            lambda m: list(zip(tr["t"] - tr["t"][0], tr[m["column"]])),
        )
        + F.verdicts()
    )
    res["faults"] = F.log
    rp = replay_hash(st.log)
    res["replay"] = {"available": False} if rp is None else dict(
        available=True, tick=rp[0], state_hash=rp[1], match=bool(final) and rp == (final["tick"], final["state_hash"]))  # fmt: skip


def _combo(sc, mons, run_dir, cwd, argvs, env, log, res):
    """px4 + feetech. ONE World: the bridge's, stepped in lockstep with PX4 (wall = sim time). The arm's Feetech pty is served by
    FeetechPort in this process over the bridge's --ctl socket (RemoteSim), so its SetAttr writes run in that same World and land in
    the bridge's replay log next to the ESC throttles. The programs run in order (the arm one first, PX4 idling disarmed in the same
    lockstep, then the flight), each with DVSIM_SERIAL + DVSIM_MAVLINK; a failing program stops the rest."""
    from .ctl import RemoteSim
    from .feetech import FeetechPort
    from .px4 import Stack

    lim, dt = sc["limits"], sc.get("dt", 0.001)
    model = _build_model(sc, run_dir / "model")
    st = Stack(_bridge_cfg(sc, run_dir, model), run_dir, model=model, ctl=True)
    env["DVSIM_MAVLINK"] = st.mavlink
    try:
        st.start()
    except RuntimeError as e:
        raise ScenarioError(f"PX4 stack: {e}") from None
    fc = sc["interfaces"]["feetech"]
    sig = [
        m for m in mons if "column" not in m
    ]  # world signals, sampled over ctl at 10 Hz
    series, rows, stop, stages = {m["name"]: [] for m in sig}, [], threading.Event(), []
    rs = port = th = F = None

    def sample():
        row = []
        for m in sig:
            tick, v = rs.get(m["path"], m["attr"], m.get("frame", ""))
            series[m["name"]].append((tick * dt, _val(m, v)))
            row.append(series[m["name"]][-1][1])
        rows.append([tick * dt, *row] if sig else [])

    def sampler():
        while not stop.wait(0.1):
            try:
                sample()
            except (OSError, ValueError):  # the bridge went away
                return

    try:
        rs = RemoteSim(st.ctl)
        for s in sc.get(
            "setup", []
        ):  # SetAttr before the programs (logged, replayed): e.g. torque on, the arm holds its pose
            for p in s.get("paths") or [s["path"]]:
                rs.set(p, s["attr"], s["values"])
        port = FeetechPort(
            rs,
            {int(i): p for i, p in fc["servos"].items()},
            baud=fc.get("baud", 1_000_000),
            realtime=True,
        )
        env["DVSIM_SERIAL"] = port.device_path
        F = _Faults(sc.get("faults", []), rs, dt, 0.0, port,
                    {p["name"]: run_dir / f"program.{p['name']}.log" for p in sc["programs"]})  # fmt: skip
        F.start()
        if sig:
            th = threading.Thread(target=sampler, daemon=True)
            th.start()
        for p, a in zip(sc["programs"], argvs):
            t0 = F.starts[p["name"]] = rs.tick * dt
            r = run_child(
                a,
                env,
                cwd,
                run_dir / f"program.{p['name']}.log",
                {**lim, **{k: p[k] for k in ("wall_s", "cpu_s") if k in p}},
            )
            stages.append(dict(name=p["name"], start_t=t0, end_t=rs.tick * dt, **r))
            if r["returncode"] or r["timeout"]:
                break
        if sig:
            sample()
    except (ValueError, KeyError) as e:
        raise ScenarioError(str(e)) from None
    finally:
        stop.set()
        if th:
            th.join()
        if F:
            F.close(rs.tick * dt)
        if port:
            port.close()
        if rs:
            rs.close()
        final = st.stop()
    log.write_text(
        "".join(
            f"== {s['name']} ==\n" + (run_dir / f"program.{s['name']}.log").read_text()
            for s in stages
        )
    )
    res["programs"] = stages
    res["program"] = next(
        (s for s in stages if s["returncode"] or s["timeout"]), stages[-1]
    )
    res["sim"] = dict(
        final or {},
        bridge_log="bridge.log",
        px4_log="px4.log",
        port_errors=list(port.errors),
    )
    if sig:
        with open(run_dir / "signals.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", *[m["name"] for m in sig]])
            w.writerows(rows)
    with open(st.trace) as f:
        trows = list(csv.DictReader(f))
    tr = {k: np.array([float(r[k]) for r in trows]) for k in trows[0]}
    _derive(tr)
    bad = [m["name"] for m in mons if "column" in m and m["column"] not in tr]
    if bad:
        raise ScenarioError(f"monitors {bad}: no such trace column (have {sorted(tr)})")
    # phase: from / to relative to that program's (or fault's) start, `to` defaulting to its end
    spans = (
        {s["name"]: (s["start_t"], s["end_t"]) for s in stages}
        | F.spans()
        | {"end": float(tr["t"][-1])}
    )
    res["monitors"] = (
        _judge_all(
            mons,
            spans,
            0.0,
            lambda m: (
                list(zip(tr["t"], tr[m["column"]]))
                if "column" in m
                else series[m["name"]]
            ),
        )
        + F.verdicts()
    )
    res["faults"] = F.log
    rp = replay_hash(st.log, Path(model).parent)
    res["replay"] = {"available": False} if rp is None else dict(
        available=True, tick=rp[0], state_hash=rp[1], match=bool(final) and rp == (final["tick"], final["state_hash"]))  # fmt: skip
