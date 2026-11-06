"""One flight in its own process: `python -m dv.flight SPEC.json`, stdout = JSON lines (flight_tick events, then {"result": ...}). The scene is the agent's built MJCF, seeded and scattered; the
harness around the controller follows examples/hopper/hop.py (sensors, servo set-up, gear backlash, truth) with the design's own numbers in place of hop.py's constants. The controller sees
estimates only, from examples/hopper/estimator.py, never truth.

spec: {xml, props, gnss:{horiz_std,vert_std}, range:{min_range,max_range}, servo:{p,d}, scenario, seed, scatter:{...} | None, code, cfg, mission, sandbox, log, rain_mmh, ...}"""

import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

DVSIM = Path(os.environ["DVSIM_ROOT"])
sys.path.insert(0, str(DVSIM / "examples" / "hopper"))

import estimator as est_mod
import vehicle as veh

SIM_DT = 0.001
STRIDE = 4  # sim ticks per control tick (250 Hz)
CTRL_DT = SIM_DT * STRIDE
LIM = math.radians(20.0)  # mechanical range of the vane hinges
TOUCH_Z = 0.02  # m above the standing height: feet down
TICK_EVERY = 25  # control ticks between flight_tick events (10 Hz)
TELEM_EVERY = 125  # control ticks between telemetry rows returned to the agent (2 Hz)
RECOVER_DEG = 10.0  # recovery time = longest stretch in the air above this tilt


def _play(y, x, b):
    """Backlash operator (hop._play of examples/hopper)."""
    if x - y > b:
        return x - b
    if y - x > b:
        return x + b
    return y


class _VertFilter(est_mod.AxisFilter):
    """The vertical axis filter with the receiver's real vertical noise for GNSS fixes (the only fixes that come with a lag); the original hard-codes the RTK figure."""

    vert_std = 0.10

    def update(self, idx, z, sigma, lag=0):
        if idx == 0 and lag:
            sigma = max(sigma, self.vert_std)
        return super().update(idx, z, sigma, lag)


def draw(seed, scatter, sc):
    """Everything random in a flight, from the seed in a fixed order. scatter None = the nominal vehicle."""
    r = np.random.default_rng([int(seed), 7777])
    u = lambda a: float(r.uniform(-a, a))
    n = lambda s: float(np.clip(r.standard_normal(), -3, 3) * s)
    w = float(r.uniform(0, 2 * math.pi))
    ws = float(sc.get("wind_ms", 0.0))
    h = dict(
        wind=[ws * math.cos(w), ws * math.sin(w), 0.0],
        backlash=math.radians(0.25),
        gyro_bias=[0.0] * 3,
        accel_bias=[0.0] * 3,
        xml={},
    )
    s = scatter
    a = dict(ct=u(s["thrust_tol"]), m=u(s["mass_tol"]), dx=u(s["cg_tol_m"]), dy=u(s["cg_tol_m"]), n1=float(r.uniform(0, s["noise_extra"])), n2=float(r.uniform(0, s["noise_extra"])), n3=float(r.uniform(0, s["noise_extra"])), bl=float(r.uniform(0, s["backlash_deg"]))) if s else None  # fmt: skip
    if s:
        h["backlash"] = math.radians(a["bl"])
        h["gyro_bias"] = [math.radians(n(s["gyro_bias_dps"])) for _ in range(3)]
        h["accel_bias"] = [n(s["accel_bias_ms2"]) for _ in range(3)]
        h["xml"] = dict(
            ct_scale=1 + a["ct"],
            mass_scale=1 + a["m"],
            cg_dx=a["dx"],
            cg_dy=a["dy"],
            imu_noise=1 + a["n1"],
            baro_noise=1 + a["n2"],
            lidar_noise=1 + a["n3"],
        )
    return h


def scene_xml(xml, seed, sc, x):
    """The built MJCF with this flight's seeds, Dryden intensity, rain device and physical scatter. Small port of examples/hopper/scene.scatter_xml (which assumes the stock battery share)
    and examples/hopper-land/landscene (the rain device, its estimated catch areas)."""
    from scene import scatter_xml

    m = re.search(r"(<text name=\"dvsim.devices\" data=')(.*?)('\s*/>)", xml, re.DOTALL)
    d = json.loads(m.group(2))

    def seeded(o, salt):
        if isinstance(o, dict):
            if "seed" in o:
                o["seed"] = int(seed) * 1000 + salt
            for i, v in enumerate(o.values()):
                seeded(v, salt + i + 1)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                seeded(v, salt + 37 * (i + 1))

    seeded(d, 1)
    d["seed"] = int(seed)
    for e in d["environments"]:
        e["w20"] = float(sc.get("w20_ms", 0.0))
    if float(sc.get("rain_mmh", 0.0)) > 0:
        d["rains"] = [
            dict(name="rain", rate=0.0, body="frame", catch_area=0.045, wet_area=0.12)
        ]
    xml = (
        xml[: m.start(2)]
        + json.dumps(d, sort_keys=True, separators=(",", ":"))
        + xml[m.end(2) :]
    )
    if x:
        # the CG shift moves the battery by shift / battery share, as scene.scatter_xml does with the stock 0.25
        share = x["batt_share"]
        xml = scatter_xml(
            xml,
            dict(
                ct_scale=x["ct_scale"],
                mass_scale=x["mass_scale"],
                batt_dx=x["cg_dx"] / share,
                batt_dy=x["cg_dy"] / share,
                imu_noise=x["imu_noise"],
                baro_noise=x["baro_noise"],
                lidar_noise=x["lidar_noise"],
            ),
        )
    return xml


def tilt_deg(q):
    _, qx, qy, _ = q
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (qx * qx + qy * qy)))))


class Judge:
    """The mission on truth: first violation and metrics."""

    def __init__(self, m, soc_min):
        self.m, self.soc_min = m, soc_min
        self.first = None
        self.tilt_max = 0.0
        self.z_max = 0.0
        self.t_lift = None
        self.in_band_since = None
        self.hold_best = 0.0
        self.held = False
        self.td_speed = 0.0
        self.td_dist = None
        self.t_touch = None
        self.rest_since = None
        self.alt_err_hold = 0.0

    def violate(self, t, why):
        if self.first is None:
            self.first = dict(t=round(t, 2), reason=why)

    def tick(self, t, x, y, z, vz, tilt):
        m = self.m
        self.z_max = max(self.z_max, z)
        self.tilt_max = max(self.tilt_max, tilt)
        if self.t_lift is None and z > 0.05:
            self.t_lift = t
        if tilt > m["tilt_max_deg"]:
            self.violate(
                t,
                f"tilt {tilt:.1f} deg over the {m['tilt_max_deg']:g} deg limit at height {z:.2f} m, t={t:.1f} s",
            )
        band = abs(z - m["alt_m"]) <= m["alt_tol_m"]
        if band and not self.held:
            if self.in_band_since is None:
                self.in_band_since = t
            self.hold_best = max(self.hold_best, t - self.in_band_since)
            if t - self.in_band_since >= m["hold_s"]:
                self.held = True
        elif not band:
            self.in_band_since = None
        if self.z_max > 0.3 and z <= TOUCH_Z and vz < 0:  # the hardest contact counts
            if self.t_touch is None:
                self.t_touch = t
            if -vz > self.td_speed:
                self.td_speed = -vz
                self.td_dist = math.hypot(x, y)
        if self.t_touch is not None and abs(vz) < 0.05 and z <= TOUCH_Z + 0.03:
            self.rest_since = self.rest_since if self.rest_since is not None else t
        else:
            self.rest_since = None


def run_flight(spec, emit=None, load_sim=None, on_tick=None):
    """The flight loop, in this process. emit(dict) gets flight_tick events. load_sim(path) -> Sim (default dvsim.Sim.load: the GUI passes one bound to its window); on_tick(sim, t) runs every control
    tick (the GUI's pacing). -> result dict (never raises for agent mistakes)."""
    import dvsim

    from dv.sandbox import Controller

    t_wall = time.perf_counter()
    props, sc, m = spec["props"], spec.get("scenario_params", {}), spec["mission"]
    # the design's numbers in place of vehicle.py's stock constants (this process flies one vehicle)
    veh.IMU_REL = np.array(props["imu_rel"])
    veh.LIDAR_REL = np.array(props["lidar_rel"])
    veh.H_STAND = props["h_stand"]
    veh.LIDAR_MIN, veh.LIDAR_MAX = (
        spec["range"]["min_range"],
        spec["range"]["max_range"],
    )
    h = draw(spec["seed"], spec.get("scatter"), sc)
    if h["xml"]:
        h["xml"]["batt_share"] = spec["batt_mass"] / props["mass"]
    wd = Path(tempfile.mkdtemp(prefix="dv-flight-", dir=spec.get("tmp")))
    model = wd / "scene.xml"
    text = scene_xml(Path(spec["xml"]).read_text(), spec["seed"], sc, h["xml"])
    sim_dt = spec.get("sim_dt") or SIM_DT  # the crosscheck flies at half the timestep
    stride = round(CTRL_DT / sim_dt)
    if sim_dt != SIM_DT:
        text, hits = re.subn(r'timestep="[^"]+"', f'timestep="{sim_dt}"', text)
        assert hits, "no timestep attribute in the MJCF"
    model.write_text(text)
    sim = (load_sim or dvsim.Sim.load)(str(model))
    if spec.get("log"):
        sim.start_log()
    for s in range(1, 5):
        sim.set(f"/devices/servo{s}", "torque_enable", [1])
        sim.set(f"/devices/servo{s}", "p_gain", [spec["servo"]["p"]])
        sim.set(f"/devices/servo{s}", "d_gain", [spec["servo"]["d"]])
        sim.set(f"/devices/servo{s}", "goal_position", [0.0])
    sim.set("/devices/imu", "gyro_bias", h["gyro_bias"])
    sim.set("/devices/imu", "accel_bias", h["accel_bias"])
    sim.set("/devices/air", "wind", h["wind"])
    sb = spec["sandbox"]
    ctl = Controller(
        spec["code"],
        spec["cfg"],
        deadline_s=sb["step_deadline_ms"] / 1000.0,
        hang_s=sb["hang_s"],
    )
    q = lambda path, attr, i=0: float(sim.query(path, attr)[0][i])
    fx0, fy0, fz0 = (q("/world/frame", "pos", i) for i in range(3))
    est = est_mod.Estimator(CTRL_DT)
    est.sig_gnss = max(0.05, spec["gnss"]["horiz_std"])
    est.sig_vel = 0.06
    calibrated = False
    t_cal = m["t_cal_s"]
    play = np.zeros(4)
    last_baro, last_gnss = None, None
    judge = Judge(m, spec["soc_min_pct"])
    z_prev = 0.0
    soc = i_peak = 1.0
    soc_min_seen, v_min, i_peak, thrust_peak = 1.0, 99.0, 0.0, 0.0
    duty, vanes = 0.0, [0.0] * 4
    telem, stop_reason = [], None
    rain = float(sc.get("rain_mmh", 0.0))
    rain_last = -1.0
    sim.step(1)
    k = 0
    max_ticks = int((m["mission_s"] + 2.0) / sim_dt)
    soc0, e_wh, ah = None, 0.0, 0.0
    sat_ticks, air_ticks, tilt_since, recovery = 0, 0, None, 0.0
    hover_n, hover_duty, hover_i = 0, 0.0, 0.0
    while k < max_ticks:
        if k % stride == 0:
            t = sim.tick * sim_dt
            n = k // stride
            gyro = sim.query("/devices/imu", "gyro")[0]
            accel = sim.query("/devices/imu", "accel")[0]
            pr = q("/devices/baro", "pressure")
            baro = pr if pr != last_baro and pr > 0 else None
            last_baro = pr
            gn = tuple(float(x) for x in np.concatenate([sim.query("/devices/gnss", "lat")[0], sim.query("/devices/gnss", "lon")[0], sim.query("/devices/gnss", "alt")[0], sim.query("/devices/gnss", "vel_ned")[0]]))  # fmt: skip
            fix = gn if gn != last_gnss and any(gn) else None
            last_gnss = gn
            rng = q("/devices/lidar", "range")
            vb = q("/devices/battery", "voltage")
            # ---- estimator: what the flight computer has
            armed = False
            if not calibrated:
                est.calibrate(gyro, accel, baro, fix, rng)
                if t >= t_cal and est.ready():
                    est.finish()
                    est.kf[2].__class__ = _VertFilter
                    est.kf[2].vert_std = spec["gnss"]["vert_std"]
                    calibrated = True
            else:
                est.update(gyro, accel, baro, fix, rng)
                armed = True
            # ---- controller
            if armed:
                s = dict(
                    pos=[float(v) for v in est.pos],
                    vel=[float(v) for v in est.vel],
                    att=[float(v) for v in est.att.q],
                    rate=[float(v) for v in est.omega],
                    height=float(est.lidar_z + props["h_stand"])
                    if (est.lidar_ok and est.lidar_z is not None)
                    else None,
                    vbat=vb,
                    armed=True,
                    imu=dict(
                        gyro=[float(v) for v in gyro], accel=[float(v) for v in accel]
                    ),
                    baro=None if baro is None else float(baro),
                    gnss=None if fix is None else [float(v) for v in fix],
                )
            else:
                s = dict(pos=[0.0, 0.0, 0.0], vel=[0.0, 0.0, 0.0], att=[1.0, 0.0, 0.0, 0.0], rate=[0.0, 0.0, 0.0], height=None, vbat=vb, armed=False, imu=dict(gyro=[float(v) for v in gyro], accel=[float(v) for v in accel]), baro=None if baro is None else float(baro), gnss=None)  # fmt: skip
            duty, vanes = ctl.step(t, s)
            if not armed:
                duty, vanes = 0.0, [0.0] * 4  # disarmed on the pad
            for i in range(4):
                play[i] = _play(
                    play[i], float(np.clip(vanes[i], -LIM, LIM)), h["backlash"]
                )
                sim.set(f"/devices/servo{i + 1}", "goal_position", [float(play[i])])
            sim.set("/devices/esc", "throttle", [float(np.clip(duty, 0.0, 1.0))])
            if rain > 0:
                r = rain * min(
                    1.0,
                    max(
                        0.0,
                        (t - sc.get("rain_t_on_s", 3.0))
                        / max(sc.get("rain_ramp_s", 2.0), 1e-3),
                    ),
                )
                if abs(r - rain_last) > 0.05 * rain or (
                    r != rain_last and r in (0.0, rain)
                ):
                    sim.set("/devices/rain", "rate", [r])
                    rain_last = r
            # ---- truth, for the judge and the log only
            fx, fy, fz = (q("/world/frame", "pos", i) for i in range(3))
            x, y, z = fx - fx0, fy - fy0, fz - fz0
            vz = (z - z_prev) / CTRL_DT
            z_prev = z
            tl = tilt_deg(sim.query("/world/frame", "quat")[0])
            judge.tick(t, x, y, z, vz, tl)
            soc = q("/devices/battery", "soc")
            ib = q("/devices/battery", "current")
            th = q("/devices/rotor", "thrust")
            soc_min_seen, v_min, i_peak, thrust_peak = (
                min(soc_min_seen, soc),
                min(v_min, vb),
                max(i_peak, abs(ib)),
                max(thrust_peak, th),
            )
            soc0 = soc if soc0 is None else soc0
            e_wh += vb * abs(ib) * CTRL_DT / 3600.0
            ah += abs(ib) * CTRL_DT / 3600.0
            if judge.t_lift is not None and judge.t_touch is None:
                air_ticks += 1
                sat_ticks += (
                    duty >= 0.995 or max(abs(v) for v in vanes) >= veh.VANE_LIMIT
                )
                if tl > RECOVER_DEG:
                    tilt_since = t if tilt_since is None else tilt_since
                    recovery = max(recovery, t - tilt_since)
                else:
                    tilt_since = None
                if judge.in_band_since is not None and t - judge.in_band_since > 0.5:
                    hover_n, hover_duty, hover_i = hover_n + 1, hover_duty + duty, hover_i + abs(ib)  # fmt: skip
            if n % TICK_EVERY == 0 and emit:
                emit(
                    dict(
                        kind="flight_tick",
                        scenario=spec["scenario"],
                        t=round(t, 2),
                        z=round(z, 3),
                        x=round(x, 3),
                        y=round(y, 3),
                        tilt_deg=round(tl, 2),
                        duty=round(float(duty), 3),
                        vbat=round(vb, 2),
                    )
                )
            if n % TELEM_EVERY == 0:
                ex = est.pos if calibrated else [0.0, 0.0, 0.0]
                telem.append(dict(t=round(t, 1), z=round(z, 2), x=round(x, 2), y=round(y, 2), est_z=round(float(ex[2]), 2), est_x=round(float(ex[0]), 2), est_y=round(float(ex[1]), 2), vz=round(vz, 2), tilt_deg=round(tl, 1), duty=round(float(duty), 3), vanes_deg=[round(math.degrees(float(v)), 1) for v in vanes], thrust_n=round(th, 1), vbat=round(vb, 1), soc=round(soc, 3)))  # fmt: skip
            if on_tick:
                on_tick(sim, t)
            # ---- stop conditions
            if ctl.dead and stop_reason is None:
                stop_reason = ctl.error
                judge.violate(t, ctl.error)
            if judge.rest_since is not None and t - judge.rest_since > 1.0:
                break
            if judge.t_lift is None and t > m["liftoff_by_s"]:
                judge.violate(
                    t,
                    f"did not lift off by t={m['liftoff_by_s']:g} s: thrust peaked at {thrust_peak:.1f} N, weight {props['mass'] * veh.G:.1f} N, last duty {duty:.2f}",
                )
                break
            if tl > 100.0 and z < 0.5:  # on its side on the ground
                break
            if ctl.dead and t > judge.first["t"] + 3.0:
                break
            if sim.diverged:
                judge.violate(t, "the simulation diverged")
                break
        sim.step(1)
        k += 1
    t_end = sim.tick * sim_dt
    ctl.close()
    q_end = sim.query("/world/frame", "quat")[0]
    final_tilt = tilt_deg(q_end)
    fx, fy, fz = (q("/world/frame", "pos", i) for i in range(3))
    x_end, y_end, z_end = fx - fx0, fy - fy0, fz - fz0
    dist = math.hypot(x_end, y_end)
    # ---- end-of-flight checks, in mission order
    j = judge
    if not j.held and j.first is None:
        if j.z_max < m["alt_m"] - m["alt_tol_m"]:
            j.violate(
                t_end,
                f"never reached the altitude band: peak {j.z_max:.2f} m against {m['alt_m']:g} +- {m['alt_tol_m']:g} m",
            )
        else:
            j.violate(
                t_end,
                f"did not hold the altitude band for {m['hold_s']:g} s: longest {j.hold_best:.1f} s (peak {j.z_max:.2f} m)",
            )
    if j.first is None and j.t_touch is None:
        j.violate(
            t_end,
            f"did not land inside {m['mission_s']:g} s: ended at height {z_end:.2f} m",
        )
    if j.first is None and j.td_speed > m["touchdown_m_s"]:
        j.violate(
            j.t_touch,
            f"touchdown at {j.td_speed:.2f} m/s, limit {m['touchdown_m_s']:g} m/s",
        )
    if j.first is None and dist > m["land_radius_m"]:
        j.violate(
            t_end,
            f"landed {dist:.2f} m from the takeoff point, limit {m['land_radius_m']:g} m",
        )
    if j.first is None and final_tilt > 30.0:
        j.violate(t_end, f"not upright at rest: tilt {final_tilt:.0f} deg")
    if j.first is None and j.rest_since is None:
        j.violate(
            t_end, f"did not come to rest on the ground inside {m['mission_s']:g} s"
        )
    if j.first is None and t_end > m["mission_s"]:
        j.violate(t_end, f"mission took {t_end:.1f} s, limit {m['mission_s']:g} s")
    if j.first is None and soc * 100.0 < spec["soc_min_pct"]:
        j.violate(
            t_end,
            f"battery at {soc * 100:.0f} % at landing, limit {spec['soc_min_pct']:g} %",
        )
    late_frac = ctl.late / max(ctl.steps, 1)
    if j.first is None and late_frac > sb["late_frac_max"]:
        j.violate(
            t_end,
            f"controller too slow: {ctl.late} of {ctl.steps} steps missed the {sb['step_deadline_ms']} ms deadline",
        )
    # the anti-exploit invariant: what the pack gave cannot exceed what it holds, and the integrated power must match the charge the pack's state of charge says it gave
    pack = spec["pack"]
    v_mean = e_wh / ah if ah > 0 else 0.0
    soc_wh = ((soc0 or soc) - soc) * pack["capacity_ah"] * v_mean
    available = (soc0 or soc) * pack["capacity_ah"] * pack["v_full"]
    energy = dict(
        used_wh=round(e_wh, 4),
        from_soc_wh=round(soc_wh, 4),
        available_wh=round(available, 3),
        ok=bool(e_wh <= available and abs(e_wh - soc_wh) <= max(0.03 * e_wh, 0.02)),
    )
    if e_wh > available:
        j.violate(t_end, f"energy invariant broken: {e_wh:.1f} Wh used, {available:.1f} Wh available")  # fmt: skip
    res = dict(
        scenario=spec["scenario"],
        seed=spec["seed"],
        **{"pass": j.first is None},
        reason=j.first["reason"] if j.first else "mission complete",
        first_violation=j.first,
        metrics=dict(
            t_end_s=round(t_end, 2),
            t_liftoff_s=j.t_lift and round(j.t_lift, 2),
            z_max_m=round(j.z_max, 3),
            hold_s=round(j.hold_best, 2),
            touchdown_m_s=round(j.td_speed, 3),
            land_dist_m=round(dist, 3),
            miss_distance_m=round(dist, 3),
            saturation_pct=round(100.0 * sat_ticks / max(air_ticks, 1), 1),
            recovery_s=round(recovery, 2),
            hover_duty_pct=round(100.0 * hover_duty / hover_n, 1) if hover_n else None,
            hover_i_a=round(hover_i / hover_n, 2) if hover_n else None,
            energy=energy,
            tilt_max_deg=round(j.tilt_max, 2),
            final_tilt_deg=round(final_tilt, 2),
            soc_end_pct=round(soc * 100, 1),
            v_min=round(v_min, 2),
            i_peak_a=round(i_peak, 1),
            thrust_peak_n=round(thrust_peak, 2),
            late_steps=ctl.late,
            steps=ctl.steps,
            wall_s=round(time.perf_counter() - t_wall, 2),
        ),  # fmt: skip
        telemetry=telem,
    )
    if spec.get("log"):
        Path(spec["log"]).write_bytes(sim.log())
        res["state_hash"] = f"{sim.state_hash():016x}"
        # what the GUI needs to replay this flight (and any score to be re-checked): the dvsim native.json, the flight's own MJCF
        (Path(spec["log"]).parent / "native.json").write_text(
            json.dumps(
                dict(
                    tick=sim.tick,
                    dt=sim_dt,
                    state_hash=res["state_hash"],
                    diverged=bool(sim.diverged),
                )
            )
        )
        (Path(spec["log"]).parent / "scene.xml").write_text(model.read_text())
    model.unlink(missing_ok=True)
    try:
        wd.rmdir()
    except OSError:
        pass
    return res


def static_thrust(xml, secs=2.0):
    """Static thrust run at full charge: the vehicle's free joint removed, full duty. -> {thrust_n, i_batt_a, i_motor_a, vbat}. Peak values after the spin-up."""
    import dvsim

    wd = Path(tempfile.mkdtemp(prefix="dv-static-"))
    p = wd / "scene.xml"
    p.write_text(re.sub(r'<freejoint name="frame:free"\s*/>', "", xml, count=1))
    try:
        sim = dvsim.Sim.load(str(p))
        sim.step(1)
        sim.set("/devices/esc", "throttle", [1.0])
        thrust, ib, v = 0.0, 0.0, 99.0
        steps = int(secs / SIM_DT)
        for i in range(steps // 4):
            if (
                i == steps // 8
            ):  # spin-up current is a transient: the numbers are those of the second half
                thrust, ib, v = 0.0, 0.0, 99.0
            sim.step(4)
            thrust = max(thrust, float(sim.query("/devices/rotor", "thrust")[0][0]))
            ib = max(ib, abs(float(sim.query("/devices/battery", "current")[0][0])))
            v = min(v, float(sim.query("/devices/battery", "voltage")[0][0]))
        return dict(thrust_n=thrust, i_batt_a=ib, vbat=v)
    finally:
        p.unlink(missing_ok=True)
        wd.rmdir()


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())

    def emit(ev):
        sys.stdout.write(json.dumps(ev) + "\n")
        sys.stdout.flush()

    res = run_flight(spec, emit)
    sys.stdout.write(json.dumps(dict(result=res)) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
