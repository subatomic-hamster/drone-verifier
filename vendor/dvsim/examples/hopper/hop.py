"""EDF hopper flight harness: the program of the hopper scenarios. Owns the Sim, plays the world around the flight computer (sensor sampling, servo set-up and gear backlash, the
seeded physical scatter, faults), steps the flight software of flight.py at 250 Hz and writes trace.csv / replay.dvsimog / native.json (dvsim.marine.write_native) and telemetry.jsonl
(the HUD stream, dvsim.params contract).

Everything random is drawn from numpy's PCG64 seeded [seed, 7777] in a fixed order, or lives inside the sim (sensor noise, Dryden turbulence: seeded by the model build,
scene.build_seeded), so a (seed, parameters) pair replays to the same state hash. The build-time part of the scatter (fan thrust, mass, battery position, sensor noise, initial
state of charge) is applied to the MJCF by scene.scatter_xml; the run-time part (wind, biases, servo gains and backlash, faults) by this file.

Env: DVSIM_MODEL, DVSIM_RUN_DIR, DVSIM_SCENARIO_DIR (set by `dvsim program`), DVSIM_PARAMS (JSON overrides of the scenario's [[parameters]]), HOPPER_SCENARIO (file name, default
hop10.scenario.toml)."""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import dvsim
import vehicle as veh
from flight import DEFAULTS, Flight
from flight import DT as CTRL_DT
from dvsim import params as nparams
from dvsim.marine import write_native

SIM_DT = 0.001
STRIDE = round(CTRL_DT / SIM_DT)  # sim ticks per control tick
ROW_EVERY = 16  # trace row every 16 ms
TLM_EVERY = 48  # telemetry record every 48 ms (about 21 Hz)
STATE_CODE = dict(
    PAD=0,
    READY=0,
    SPOOL=1,
    CLIMB=2,
    TRANSIT=3,
    HOVER=4,
    DESCENT=5,
    FINAL=6,
    TOUCHDOWN=7,
    LANDED=8,
    ABORT_LAND=9,
    FAULT=10,
)
COLS = [
    "t", "x_m", "y_m", "z_m", "vz_ms", "tilt_deg", "state", "thrust_N", "est_x_m", "est_y_m", "est_alt_m", "pos_err_m", "dist_pad_m", "hover_err_m",
    "vbat_v", "ibat_a", "soc", "rpm", "lidar_m", "vane_max_deg", "sat_frac", "abort", "td_speed_ms", "td_tilt_deg", "td_dist_m", "landed", "range_m",
]  # fmt: skip
TOUCH_Z = 0.01  # m above the standing height: feet down
LIM = math.radians(20.0)  # mechanical range of the vane hinges


def draw_hidden(seed, p):
    """The physical scatter of one flight (the flight computer knows none of it). Fixed draw order. `disperse` = 0 gives the nominal vehicle."""
    r = np.random.default_rng([int(seed), 7777])
    u = lambda a: float(r.uniform(-a, a))
    n = lambda s: float(np.clip(r.standard_normal(), -3, 3) * s)
    d = float(p["disperse"])
    h = {}
    sc = dict(
        ct_scale=1 + d * u(p["thrust_tol"]),
        mass_scale=1 + d * u(p["mass_tol"]),
        batt_dx=d * u(p["cg_tol_m"]) / 0.25,
        batt_dy=d * u(p["cg_tol_m"]) / 0.25,
        soc0=1.0 - d * float(r.uniform(0, p["soc_drop"])),
        imu_noise=1 + d * float(r.uniform(0, p["noise_extra"])),
        baro_noise=1 + d * float(r.uniform(0, p["noise_extra"])),
        lidar_noise=1 + d * float(r.uniform(0, p["noise_extra"])),
    )  # fmt: skip
    if p["gnss_rtk"] < 0.5:
        from scene import M8N

        sc["gnss"] = M8N
    h["scatter"] = sc
    h["backlash"] = math.radians(
        float(r.uniform(0.0, 1.0)) * p["backlash_deg"] * d
        + (1 - d) * p["backlash_deg"] * 0.5
    )
    h["servo_p"] = 1.0 - d * float(r.uniform(0, p["servo_p_drop"]))
    h["gyro_bias"] = [math.radians(n(p["gyro_bias_dps"]) * d) for _ in range(3)]
    h["accel_bias"] = [n(p["accel_bias_ms2"]) * d for _ in range(3)]
    wd = p["wind_dir_deg"] if p["wind_dir_deg"] >= 0 else float(r.uniform(0, 360))
    w = math.radians(wd)
    ws = p["wind_ms"] * (1.0 if p["wind_fixed"] > 0.5 else float(r.uniform(0.4, 1.0)))
    h["wind"] = [ws * math.cos(w), ws * math.sin(w), 0.0]
    h["fault_servo"] = int(r.integers(1, 5))
    return h


def _play(y, x, b):
    """Backlash (play) operator, half width b: the output y does not move until the input x has taken up the play, then follows it at a distance b."""
    if x - y > b:
        return x - b
    if y - x > b:
        return x + b
    return y


def q(sim, path, attr, i=0):
    return float(sim.query(path, attr)[0][i])


def tilt_deg(sim):
    qw, qx, qy, qz = sim.query("/world/frame", "quat")[0]
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (qx * qx + qy * qy)))))


def prepare(seed, p, out):
    """Build the model of one flight (seed, scatter) in `out`. -> (model path, hidden dict)"""
    from scene import build_seeded

    h = draw_hidden(seed, p)
    w20 = p["w20_ms"] if p["w20_ms"] >= 0 else p["wind_ms"]
    return build_seeded(out, seed, w20, h["scatter"]), h


def fly(seed, p, model, h, run_dir=None, log=True, telemetry=True, max_s=None):
    """One flight. -> dict(res, rows, sim, fl). `p` = resolved parameters (dvsim.params.resolve), `h` = draw_hidden(seed, p)."""
    sim = dvsim.Sim.load(str(model))
    if log:
        sim.start_log()  # before the first command
    tw = (
        nparams.TelemetryWriter(Path(run_dir) / "telemetry.jsonl")
        if telemetry and run_dir
        else None
    )
    for s in range(1, 5):
        sim.set(f"/devices/servo{s}", "torque_enable", [1])
        sim.set(f"/devices/servo{s}", "p_gain", [veh.SERVO_P * h["servo_p"]])
        sim.set(f"/devices/servo{s}", "d_gain", [veh.SERVO_D])
        sim.set(f"/devices/servo{s}", "goal_position", [0.0])
    sim.set("/devices/imu", "gyro_bias", h["gyro_bias"])
    sim.set("/devices/imu", "accel_bias", h["accel_bias"])
    sim.set("/devices/air", "wind", h["wind"])
    fl = Flight(**{k: p[k] for k in DEFAULTS if k in p})
    fx0, fy0, fz0 = (q(sim, "/world/frame", "pos", i) for i in range(3))
    tx, ty = fx0 + p["dx_m"], fy0 + p["dy_m"]  # the landing pad
    play = np.zeros(4)  # output shaft angle of each servo inside its play
    last_baro, last_gnss = None, None
    rows, res = [], dict(td_speed=99.0, td_tilt=99.0, td_dist=99.0, landed=0)
    z_prev, vz, sat_n, sat_yaw_n, pow_n, touch_tick, z_max = (
        0.0,
        0.0,
        0,
        0,
        0,
        None,
        0.0,
    )
    fault_done = dict(servo=False, batt=False)
    v_min, soc_min, abort_seen = 99.0, 1.0, 0
    t_end_state = None
    sim.step(1)  # the devices' outputs (bus voltage, ...) exist after the first tick
    for k in range(int((max_s or p["max_s"]) / SIM_DT)):
        if k % STRIDE == 0:
            t = sim.tick * SIM_DT
            gyro = sim.query("/devices/imu", "gyro")[0]
            accel = sim.query("/devices/imu", "accel")[0]
            pr = q(sim, "/devices/baro", "pressure")
            baro = pr if pr != last_baro and pr > 0 else None  # 0 = no sample yet
            last_baro = pr
            gn = tuple(float(x) for x in np.concatenate([sim.query("/devices/gnss", "lat")[0], sim.query("/devices/gnss", "lon")[0], sim.query("/devices/gnss", "alt")[0], sim.query("/devices/gnss", "vel_ned")[0]]))  # fmt: skip
            fix = gn if gn != last_gnss and any(gn) else None
            last_gnss = gn
            rng = q(sim, "/devices/lidar", "range")
            vb = q(sim, "/devices/battery", "voltage")
            c = fl.step(t, gyro, accel, baro, fix, rng, vb)
            for i in range(4):
                play[i] = _play(
                    play[i], float(np.clip(c.goals[i], -LIM, LIM)), h["backlash"]
                )
                sim.set(f"/devices/servo{i + 1}", "goal_position", [play[i]])
            sim.set("/devices/esc", "throttle", [float(np.clip(c.duty, 0.0, 1.0))])
            # faults of the scenario
            if p["fault_servo"] > 0.5 and not fault_done["servo"] and t >= p["t_fault"]:
                s = h["fault_servo"]
                sim.set(
                    f"/devices/servo{s}", "torque_limit", [p["fault_torque"] * 1000]
                )
                sim.set(
                    f"/devices/servo{s}",
                    "p_gain",
                    [veh.SERVO_P * h["servo_p"] * p["fault_p"]],
                )
                fault_done["servo"] = True
            if (
                p["fault_batt_v"] != 0.0
                and not fault_done["batt"]
                and t >= p["t_fault"] + 3.0
            ):
                sim.set("/devices/battery", "ocv_offset", [-abs(p["fault_batt_v"])])
                fault_done["batt"] = True
            # truth
            fx, fy, fz = (q(sim, "/world/frame", "pos", i) for i in range(3))
            zt = fz - fz0
            vz = (zt - z_prev) / CTRL_DT
            z_prev = zt
            thrust = q(sim, "/devices/rotor", "thrust")
            tl = tilt_deg(sim)
            soc = q(sim, "/devices/battery", "soc")
            if thrust > 1.0:
                pow_n += 1
                sat_n += bool(fl.sat)
                sat_yaw_n += bool(fl.sat_yaw)
            if fl.state in (
                "CLIMB",
                "TRANSIT",
                "HOVER",
                "DESCENT",
                "FINAL",
                "ABORT_LAND",
            ):
                v_min, soc_min = min(v_min, vb), min(soc_min, soc)
            abort_seen = max(abort_seen, int(bool(fl.abort_reason)))
            z_max = max(z_max, zt)
            if (
                z_max > 0.5
                and zt <= TOUCH_Z
                and vz < 0
                and (touch_tick is None or -vz > res["td_speed"])
            ):  # the HARDEST ground contact counts
                touch_tick = k if touch_tick is None else touch_tick
                res.update(
                    td_speed=-vz,
                    td_tilt=tl,
                    td_dist=math.hypot(fx - tx, fy - ty),
                    landed=1,
                )
            if k % ROW_EVERY == 0:
                est = fl.telemetry()
                ex, ey, ea = (
                    est.get("x_m", 0.0),
                    est.get("y_m", 0.0),
                    est.get("alt_m", 0.0),
                )
                he = (
                    abs(zt - p["alt_m"])
                    if (fl.state == "HOVER" and t - fl.t_state > 2.0)
                    else 0.0
                )
                rows.append([
                    t, fx - fx0, fy - fy0, zt, vz, tl, STATE_CODE[fl.state], thrust, ex, ey, ea, math.hypot(ex - (fx - fx0), ey - (fy - fy0)),
                    math.hypot(fx - tx, fy - ty), he, vb, q(sim, "/devices/battery", "current"), soc, q(sim, "/devices/esc", "omega") * 30.0 / math.pi, rng,
                    math.degrees(float(np.max(np.abs(c.goals)))), sat_n / max(pow_n, 1), abort_seen, res["td_speed"], res["td_tilt"], res["td_dist"], res["landed"], math.hypot(fx - fx0, fy - fy0),
                ])  # fmt: skip
            if tw and k % TLM_EVERY == 0:
                tw.emit(
                    _record(
                        fl,
                        sim,
                        t,
                        zt,
                        vz,
                        tl,
                        thrust,
                        fx - fx0,
                        fy - fy0,
                        vb,
                        soc,
                        rng,
                        h,
                        c,
                    )
                )
            if fl.state in ("LANDED", "FAULT") and t_end_state is None:
                t_end_state = t
            if t_end_state is not None and t - t_end_state > 1.0:
                break
            if touch_tick is not None and k - touch_tick > 6000:
                break
            if sim.diverged:
                break
        sim.step(1)
    if tw:
        tw.close()
    res.update(
        seed=seed, state=fl.state, abort=fl.abort_reason, sat_frac=sat_n / max(pow_n, 1), sat_yaw_frac=sat_yaw_n / max(pow_n, 1), t_end=sim.tick * SIM_DT, diverged=bool(sim.diverged),
        v_min=v_min, soc_min=soc_min, events=fl.events, hidden=h, final_tilt=tilt_deg(sim),
    )  # fmt: skip
    return dict(res=res, rows=rows, sim=sim, fl=fl)


def _record(fl, sim, t, zt, vz, tl, thrust, x, y, vb, soc, rng, h, c):
    e = fl.telemetry()
    est = {
        k: e.get(k)
        for k in (
            "alt_m",
            "vz_ms",
            "x_m",
            "y_m",
            "vx_ms",
            "vy_ms",
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
            "tilt_deg",
        )
    }
    return dict(
        t=round(t, 3),
        state=e["state"],
        est={k: (None if v is None else round(v, 4)) for k, v in est.items()},
        setpoint=dict(x_m=e.get("ref_x_m"), y_m=e.get("ref_y_m"), alt_m=e.get("ref_alt_m"), vz_ms=e.get("ref_vz_ms")),
        throttle=round(float(c.duty), 4),
        vanes={
            **{f"a{i}_deg": round(math.degrees(q(sim, f"/devices/servo{i}", "position")), 3) for i in range(1, 5)},
            **{f"g{i}_deg": round(g, 3) for i, g in enumerate(e.get("goals_deg", [0, 0, 0, 0]), 1)},
            **{f"f{i}_N": round(float(np.linalg.norm(sim.query(f"/devices/vane{i}", "force")[0])), 3) for i in range(1, 5)},
        },
        thrust_N=round(thrust, 3),
        thrust_cmd_N=None if e.get("thrust_cmd_N") is None else round(e["thrust_cmd_N"], 3),
        thrust_est_N=None if e.get("thrust_est_N") is None else round(e["thrust_est_N"], 3),
        battery=dict(v=round(vb, 3), i_a=round(q(sim, "/devices/battery", "current"), 2), soc=round(soc, 4)),
        motor=dict(rpm=round(q(sim, "/devices/esc", "omega") * 30.0 / math.pi, 1)),
        lidar_m=round(rng, 3),
        torque_cmd_Nm=dict(zip("xyz", (round(x, 4) for x in e.get("tau", [0.0, 0.0, 0.0])))),
        flags=dict(abort=e.get("abort", ""), vane_sat=bool(e.get("vane_sat")), yaw_sat=bool(e.get("yaw_sat")), lidar_ok=bool(e.get("lidar_ok"))),
        truth=dict(
            alt_m=round(zt, 3), vz_ms=round(vz, 3), x_m=round(x, 3), y_m=round(y, 3), tilt_deg=round(tl, 2),
        ),
        wind_ms=[round(w, 2) for w in h["wind"]],
    )  # fmt: skip


def main():
    name = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.environ.get("HOPPER_SCENARIO", "hop10.scenario.toml")
    )
    sc = Path(os.environ.get("DVSIM_SCENARIO_DIR", Path(__file__).parent)) / name
    p = nparams.values(sc)
    run_dir = Path(os.environ["DVSIM_RUN_DIR"])
    model, h = prepare(p["seed"], p, run_dir / "model")
    r = fly(p["seed"], p, model, h, run_dir)
    write_native(r["sim"], SIM_DT, COLS, r["rows"])
    brief = {k: v for k, v in r["res"].items() if k not in ("hidden", "events")}
    (run_dir / "result.json").write_text(json.dumps(brief, default=str, indent=1))
    sys.stdout.write(json.dumps(brief, default=str) + "\n")
    for ev in r["res"]["events"]:
        sys.stdout.write(f"  {ev[0]:7.3f}s {ev[1]}\n")


if __name__ == "__main__":
    main()
