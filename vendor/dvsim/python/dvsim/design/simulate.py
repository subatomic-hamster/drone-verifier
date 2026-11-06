"""Built-in simulation checks for `Design.simulate` (need the C++ kernel; run on the built MJCF of the design).

hover: the controller is part of dvsim, not agent code. It sees ground truth (body pose, velocity) and drives only the ESC
  throttle inputs through the normal command bus (Sim.set /devices/<esc> throttle). No state estimator, no yaw-rate
  command, no position hold: altitude (PID on height above the take-off point, ramped to the target) + attitude (geometric
  PD + small integral, level, initial heading), allocated to rotors with a mixer built from the model's own rotor
  positions / axes / spin directions. Rotor thrust-vs-throttle and torque-to-thrust ratios come from a short calibration on
  a fork of the sim (every throttle level held 0.15 s), so any motor / prop / ESC combination works. Inertia is
  approximated by the instance masses as points at their body origins (a controller scaling only: the loop is
  stable for +-50 % errors; the plant is MuJoCo).
hold: servos torque-on at a pose, reports sag and current.
"""

import json
import math
import time
import xml.etree.ElementTree as ET

import numpy as np

from .drc import USABLE_FRACTION
from .library import DesignError

G = 9.80665
CAL_TICKS = 150  # calibration: ticks each throttle level is held
WN_Z, ZETA_Z = 3.0, 1.0  # altitude loop (rad/s): slow compared with the attitude loop
WN_R, ZETA_R, KI_R = 20.0, 0.9, 30.0  # attitude loop
WN_YAW = 6.0  # yaw loop (rad/s)
WN_XY, ZETA_XY = (
    1.5,
    1.0,
)  # horizontal position loop (hold_xy), slower than the attitude loop
TILT_ABORT = 1.2  # rad: the vehicle is lost


def _kernel():
    import dvsim

    if getattr(dvsim, "_dvsim", None) is None:
        raise DesignError(
            "simulate needs the C++ kernel (dvsim._dvsim is not built in this environment)"
        )
    return dvsim


def _model(xml):
    """(device config dict, {body: prim path}, {joint: (prim path, axis string)}, timestep, root body name)."""
    root = ET.parse(xml).getroot()
    dev = json.loads(root.find("custom/text").get("data"))
    paths, joints = {}, {}

    def walk(el, prefix):
        for b in el.findall("body"):
            p = f"{prefix}/{b.get('name')}"
            paths[b.get("name")] = p
            for j in [*b.findall("joint"), *b.findall("freejoint")]:
                joints[j.get("name")] = (f"{p}/{j.get('name')}", j.get("axis"))
            walk(b, p)

    walk(root.find("worldbody"), "/world")
    opt = root.find("option")
    dt = (
        float(opt.get("timestep")) if opt is not None and opt.get("timestep") else 0.002
    )
    return dev, paths, joints, dt, next(iter(paths))


def _R(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _vee(M):
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def _Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def hover(
    xml,
    report,
    altitude=2.0,
    climb_s=4.0,
    hold_s=10.0,
    tol=0.3,
    wall_s=120.0,
    control_ticks=4,
    wind=None,
    hold_xy=False,
    max_tilt=0.6,
):
    """Climb to `altitude` m above the take-off point in `climb_s`, hold `hold_s` s. Pass = height within +-`tol` m for the
    whole hold window and the sim did not diverge. Returns the numbers (see keys).
    `wind`: steady world wind (3 m/s, set on the environment device after the calibration; the design needs one).
    `hold_xy`: also hold the take-off x / y (PD + integral on position -> desired thrust vector, geometric attitude
    target, thrust vector tilted at most `max_tilt` rad from vertical: past that the position error grows = the vehicle
    is blown away). Without it the heading is held and level attitude is commanded as before (x / y drift freely).
    Always reported: horizontal drift `xy_error_max_m` from the take-off point in the hold window, `saturation_pct` (share
    of hold control ticks with at least one motor commanded at full thrust), `motor_current_max_A`."""
    dvsim = _kernel()
    dev, paths, joints, dt, root = _model(xml)
    rotors, escs = dev.get("rotors", []), {e["joint"]: e for e in dev.get("escs", [])}
    if not rotors or any(r["joint"] not in escs for r in rotors):
        raise DesignError(
            "hover: needs rotor devices, each driven by an esc on the same joint (a multirotor)"
        )
    bats = dev.get("batteries", [])
    sim = dvsim.Sim.load(xml)
    q = lambda p, a, fr="": sim.query(p, a, fr)[0]
    rp, n = paths[root], len(rotors)
    esc_p = [f"/devices/{escs[r['joint']]['name']}" for r in rotors]
    rot_p = [f"/devices/{r['name']}" for r in rotors]
    dirs = np.array([float(r.get("dir", 1)) for r in rotors])
    # geometry in the root body frame
    pos = np.array([q(paths[r["body"]], "pos", rp) for r in rotors])
    ax = np.array(
        [
            _R(q(paths[r["body"]], "quat", rp))
            @ np.array(joints[r["joint"]][1].split(), float)
            for r in rotors
        ]
    )
    t = dirs[:, None] * ax  # thrust directions
    mass = float(report["mass"]["total"])
    items = [
        (i["mass"], np.array(q(paths[i["body"]], "pos", rp)))
        for i in report["instances"]
        if i.get("body") and i["mass"] > 0
    ]
    msum = sum(m for m, _ in items)
    com = sum(m * p for m, p in items) / msum
    d2 = np.array(
        [
            sum(m * np.sum(np.delete(p - com, k) ** 2) for m, p in items)
            for k in range(3)
        ]
    )
    inertia = np.maximum(
        d2 * mass / msum, mass * 0.02**2
    )  # ponytail: diagonal point-mass inertia, controller scaling only
    # calibration on a fork: thrust and torque per rotor vs throttle
    f = sim.fork()
    levels = np.concatenate([[0.0], np.linspace(0.1, 1.0, 10)])
    T = np.zeros((len(levels), n))
    Q = np.zeros_like(T)
    for li, u in enumerate(levels):
        for p in esc_p:
            f.set(p, "throttle", float(u))
        f.step(CAL_TICKS)
        T[li] = [
            q_ * d for q_, d in zip((f.query(p, "thrust")[0][0] for p in rot_p), dirs)
        ]
        Q[li] = [abs(f.query(p, "torque")[0][0]) for p in rot_p]
    T = np.maximum.accumulate(np.maximum(T, 0.0), axis=0)
    w_n = mass * G
    out = {
        "kind": "hover",
        "mass_kg": mass,
        "weight_N": w_n,
        "max_thrust_N": float(T[-1].sum()),
        "thrust_to_weight": float(T[-1].sum() / w_n),
        "target_altitude_m": altitude,
        "tolerance_m": tol,
        "hold_s": hold_s,
        "controller": "built-in altitude PID + attitude PD(+I), ground-truth state, ESC throttles via the command bus",
    }
    if T[-1].sum() < 1.05 * w_n:
        return {
            **out,
            "pass": False,
            "reason": "max thrust (calibration) is below 1.05 x weight: cannot hover",
        }
    ok = T[-1] > 1e-6
    kq = float(np.mean(Q[-4:, ok].sum(axis=1) / T[-4:, ok].sum(axis=1)))
    arm = pos - com
    B = np.zeros((4, n))
    for i in range(n):
        B[0, i] = t[i][2]
        B[1:, i] = (
            np.cross(arm[i], t[i]) - kq * ax[i]
        )  # rotor reaction torque = -Q a, Q = kq T
    Bp = np.linalg.pinv(B)
    free = joints[f"{root}:free"][0]
    st = lambda: (q(rp, "pos"), _R(q(rp, "quat")), q(free, "qvel"))
    p0, R0, _ = st()
    z0, yaw0 = float(p0[2]), math.atan2(R0[1, 0], R0[0, 0])
    Rd = _Rz(yaw0)
    kz, kvz = WN_Z**2, 2 * ZETA_Z * WN_Z
    kr, kw = WN_R**2, 2 * ZETA_R * WN_R
    # yaw: slower loop (the point-mass inertia over-estimates Izz and the yaw torque path is the slow rotor spin-up, so at
    # WN_R it limit-cycles every control tick); roll / pitch unchanged
    kr_v = np.array([kr, kr, WN_YAW**2])
    kw_v = np.array([kw, kw, 2 * ZETA_R * WN_YAW])
    ki_v = np.array([KI_R, KI_R, KI_R * (WN_YAW / WN_R) ** 2])
    dtc = control_ticks * dt
    nsteps = int(round((climb_s + hold_s) / dtc))
    iz, ir, ixy = 0.0, np.zeros(3), np.zeros(2)
    kxy, kvxy = WN_XY**2, 2 * ZETA_XY * WN_XY
    xy0 = np.array(p0[:2])
    if wind is not None:
        envs = dev.get("environments", [])
        if not envs:
            raise DesignError("hover: wind needs an environment device in the design")
        sim.set(f"/devices/{envs[0]['name']}", "wind", [float(x) for x in wind])
    rec = {k: [] for k in ("t", "h", "u", "tilt", "ib", "soc", "im", "xy", "sat")}
    wall0, reason = time.monotonic(), None
    for k in range(nsteps):
        ts = k * dtc
        p, R, v6 = st()
        h = float(p[2]) - z0
        s = min(ts / climb_s, 1.0) if climb_s > 0 else 1.0
        zref, vref = (
            altitude * s * s * (3 - 2 * s),
            (altitude * 6 * s * (1 - s) / climb_s if s < 1.0 else 0.0),
        )
        iz = float(np.clip(iz + (zref - h) * dtc * 2.0, -3.0, 3.0))
        az = kz * (zref - h) + kvz * (vref - v6[2]) + iz
        exy = p[:2] - xy0
        if hold_xy:  # position held from lift-off: switching it on after the climb steps the attitude target
            ixy = np.clip(ixy - exy * dtc * 0.5, -8.0, 8.0)
            fv = mass * np.array(
                [*(-kxy * exy - kvxy * v6[:2] + ixy), G + az]
            )  # desired thrust vector (N)
            fh = np.linalg.norm(fv[:2])
            if fh > fv[2] * math.tan(max_tilt):
                fv[:2] *= fv[2] * math.tan(max_tilt) / fh
            b3 = fv / np.linalg.norm(fv)
            b2 = np.cross(b3, Rd[:, 0])
            b2 /= np.linalg.norm(b2)
            Rt = np.column_stack([np.cross(b2, b3), b2, b3])
            fz = float(fv @ R[:, 2])
        else:
            Rt, fz = Rd, mass * (G + az) / max(R[2, 2], 0.5)
        eR = 0.5 * _vee(Rt.T @ R - R.T @ Rt)
        ir = np.clip(ir + eR * dtc, -0.2, 0.2)
        tau = inertia * (-kr_v * eR - kw_v * v6[3:6] - ki_v * ir)
        Tr = Bp @ np.array([fz, *tau])
        Ti = np.clip(Tr, 0.0, T[-1])
        rec["sat"].append(bool(np.any(Tr >= T[-1] - 1e-9)))
        rec["xy"].append(float(np.linalg.norm(exy)))
        u = np.array([np.interp(Ti[i], T[:, i], levels) for i in range(n)])
        for pth, ui in zip(esc_p, u):
            sim.set(pth, "throttle", float(ui))
        sim.step(control_ticks)
        tilt = math.acos(max(-1.0, min(1.0, R[2, 2])))
        (
            rec["t"].append(ts),
            rec["h"].append(h),
            rec["u"].append(u),
            rec["tilt"].append(tilt),
        )
        if bats:
            rec["ib"].append(float(q(f"/devices/{bats[0]['name']}", "current")[0]))
            rec["soc"].append(float(q(f"/devices/{bats[0]['name']}", "soc")[0]))
        if True:  # every control tick: the window mean / max must not alias a per-tick oscillation
            rec["im"].append([float(q(pth, "current")[0]) for pth in esc_p])
        if sim.diverged or tilt > TILT_ABORT or h < -0.5:
            reason = (
                "solver diverged"
                if sim.diverged
                else f"vehicle lost at t = {ts:.2f} s (tilt {math.degrees(tilt):.0f} deg, height {h:.2f} m)"
            )
            break
        if time.monotonic() - wall0 > wall_s:
            reason = f"wall-clock budget {wall_s} s exceeded at sim t = {ts:.1f} s"
            break
    tt, h = np.array(rec["t"]), np.array(rec["h"])
    win = tt >= climb_s
    if reason or win.sum() < 2:
        return {
            **out,
            "pass": False,
            "reason": reason or "no hold window",
            "sim_s": float(tt[-1]) if len(tt) else 0.0,
        }
    err = h[win] - altitude
    u = np.array(rec["u"])[win]
    res = {
        **out,
        "pass": bool(np.abs(err).max() <= tol),
        "altitude_error_max_m": float(np.abs(err).max()),
        "altitude_error_rms_m": float(np.sqrt(np.mean(err**2))),
        "altitude_mean_m": float(h[win].mean()),
        "hover_throttle": float(u.mean()),
        "hover_throttle_per_motor": [float(x) for x in u.mean(axis=0)],
        "tilt_max_deg": float(math.degrees(max(rec["tilt"]))),
        "xy_error_max_m": float(np.max(np.array(rec["xy"])[win])),
        "xy_error_rms_m": float(np.sqrt(np.mean(np.array(rec["xy"])[win] ** 2))),
        "saturation_pct": float(100.0 * np.mean(np.array(rec["sat"])[win])),
        "wind_m_s": [float(x) for x in wind] if wind is not None else [0.0, 0.0, 0.0],
        "hold_xy": bool(hold_xy),
        "sim_s": float(tt[-1] + dtc),
        "calibration": {
            "kq_torque_per_thrust_m": kq,
            "levels": [float(x) for x in levels],
            "thrust_total_N": [float(x) for x in T.sum(axis=1)],
        },
    }
    if rec["im"]:
        im = np.array(rec["im"])[win]  # hold window, every control tick
        res["motor_current_A"] = [float(x) for x in im.mean(axis=0)]
        res["motor_current_max_A"] = float(im.max())
    if bats:
        ib, soc = np.array(rec["ib"])[win], np.array(rec["soc"])[win]
        res["hover_current_A"] = float(ib.mean())
        res["battery_current_max_A"] = float(ib.max())
        slope = -float(np.polyfit(tt[win], soc, 1)[0])  # SoC per s
        res.update(
            soc_start=float(soc[0]), soc_end=float(soc[-1]), soc_slope_per_s=slope
        )
        if slope > 0:
            res["endurance_s"] = max(
                0.0, float((soc[-1] - (1 - USABLE_FRACTION)) / slope)
            )
            res["endurance_note"] = (
                f"time from the end of the run until SoC = {1 - USABLE_FRACTION:.2f} (the DRC's {USABLE_FRACTION:.0%} usable rule), at the hold-window SoC slope"
            )
    return res


def hold(xml, report, pose=None, settle_s=1.0, window_s=0.5, sag_tol=0.15):
    """Servos torque-on at `pose` ({servo name or joint name: rad}, default 0 = the zero pose). After `settle_s` the next
    `window_s` is measured: sag = |joint angle - goal|, motor / bus current, torque. Pass = no divergence and sag <= `sag_tol`
    rad (0.15 rad = 8.6 deg is a choice, not a spec; the stock SO-101 sags 0.09 rad at the zero pose)."""
    dvsim = _kernel()
    dev, paths, joints, dt, _ = _model(xml)
    servos = dev.get("servos", [])
    if not servos:
        raise DesignError("hold: the design has no servo devices (an arm)")
    pose = pose or {}
    sim = dvsim.Sim.load(xml)
    goals = {
        s["name"]: float(pose.get(s["name"], pose.get(s["joint"], 0.0))) for s in servos
    }
    for s in servos:
        sim.set(f"/devices/{s['name']}", "goal_position", goals[s["name"]])
        sim.set(f"/devices/{s['name']}", "torque_enable", 1)
    sim.step(int(round(settle_s / dt)))
    rows = {
        s["name"]: {"sag": [], "i": [], "ib": [], "tau": [], "T": []} for s in servos
    }
    psu = [f"/devices/{p['name']}" for p in dev.get("psus", [])]
    ipsu = []
    for _ in range(max(1, int(round(window_s / dt / 10)))):
        sim.step(10)
        for s in servos:
            r, d = rows[s["name"]], f"/devices/{s['name']}"
            r["sag"].append(
                abs(
                    float(sim.query(joints[s["joint"]][0], "qpos")[0][0])
                    - goals[s["name"]]
                )
            )
            r["i"].append(abs(float(sim.query(d, "current")[0][0])))
            r["ib"].append(abs(float(sim.query(d, "bus_current")[0][0])))
            r["tau"].append(abs(float(sim.query(d, "torque")[0][0])))
            r["T"].append(float(sim.query(d, "temperature")[0][0]))
        if psu:
            ipsu.append(float(sim.query(psu[0], "current")[0][0]))
    res = {
        "kind": "hold",
        "mass_kg": float(report["mass"]["total"]),
        "pose_rad": goals,
        "servos": {
            s["name"]: {
                "joint": s["joint"],
                "goal_rad": goals[s["name"]],
                "sag_max_rad": max(rows[s["name"]]["sag"]),
                "current_max_A": max(rows[s["name"]]["i"]),
                "bus_current_max_A": max(rows[s["name"]]["ib"]),
                "torque_max_Nm": max(rows[s["name"]]["tau"]),
                "temperature_C": rows[s["name"]]["T"][-1],
            }
            for s in servos
        },
    }
    res["sag_max_rad"] = max(v["sag_max_rad"] for v in res["servos"].values())
    res["current_max_A"] = max(v["current_max_A"] for v in res["servos"].values())
    res["bus_current_total_A"] = float(
        sum(v["bus_current_max_A"] for v in res["servos"].values())
    )
    if ipsu:
        res["psu_current_A"] = max(ipsu)
    res["sag_tol_rad"] = sag_tol
    res["pass"] = bool(not sim.diverged and res["sag_max_rad"] <= sag_tol)
    return res


KINDS = {"hover": hover, "hold": hold}
