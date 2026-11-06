"""Flight software of the EDF hopper: sensors in (IMU, barometer, GNSS, lidar, battery monitor), four vane servo goals and the ESC throttle out, 250 Hz. No simulator imports; hop.py
owns the Sim.

States   PAD (gyro / pad calibration) -> SPOOL (thrust up to 1.12 W until the legs leave the ground) -> CLIMB -> TRANSIT -> HOVER -> DESCENT -> FINAL (lidar-governed sink rate, position
         hold) -> TOUCHDOWN (thrust cut on contact) -> LANDED.  ABORT_LAND (any time in the air): stop where it is, sink on the same final law; FAULT: thrust cut (tilt beyond
         60 deg: it can no longer be recovered).
Aborts   tilt > 40 deg, battery < 18 V for 0.5 s, GNSS silent > 2 s, estimated position > 12 m from the line of the plan, no lift-off 3 s after the spool-up, mission time out.
The mission is a list of waypoints set by the parameters (alt_m, hold_s, dx_m, dy_m): climb to alt_m above the pad, transit (dx, dy), hold, descend to the pad at (dx, dy)."""

import math
from dataclasses import dataclass, field

import control as ctl
import numpy as np
import trajectory as trj
import vehicle as veh
from estimator import Estimator

DT = 0.004  # s, control period (250 Hz)
G = veh.G
W = veh.MASS * G
DEFAULTS = dict(
    t_cal=2.0, t_launch=3.0, alt_m=10.0, hold_s=5.0, dx_m=0.0, dy_m=0.0,
    vmax_up=3.5, vmax_xy=2.5, vmax_down=3.0, amax_z=2.0, amax_xy=1.2,
    z_final=2.0, v_sink_max=1.2, v_sink_min=0.25, k_sink=0.55,
    wn_att=9.0, zeta_att=0.8, kp_xy=2.5, kd_xy=3.3, kp_z=2.5, kd_z=3.0, ki_xy=1.0, ki_z=0.4, tilt_max_deg=20.0, dT_max=15.0,
    dT_spool=10.0, t_lead=0.065,
    abort_tilt_deg=40.0, abort_v=18.0, abort_gnss_s=2.0, abort_off_m=12.0, t_max_s=120.0,
)  # fmt: skip


@dataclass
class Cmd:
    goals: np.ndarray = field(
        default_factory=lambda: np.zeros(4)
    )  # servo goal angles (rad)
    duty: float = 0.0  # ESC throttle 0..1


class Flight:
    def __init__(self, **kw):
        p = dict(DEFAULTS)
        p.update({k: v for k, v in kw.items() if k in DEFAULTS})
        self.p = p
        self.est = Estimator(DT)
        self.att = ctl.AttitudeCtl(
            wn=p["wn_att"], zeta=p["zeta_att"], t_lead=p["t_lead"]
        )
        self.pos = ctl.PositionCtl(
            p["kp_xy"],
            p["kd_xy"],
            p["kp_z"],
            p["kd_z"],
            p["ki_xy"],
            p["ki_z"],
            a_xy_max=p["amax_xy"] * 2.5,
            tilt_max_deg=p["tilt_max_deg"],
        )
        self.thr = ctl.ThrustCtl(dT_max=p["dT_max"])
        self.ref = trj.Reference()
        self.state = "PAD"
        self.events = []
        self.t_state = 0.0
        self.t_last = 0.0
        self.abort_reason = ""
        self.vbat = 0.0
        self.v_low_t = 0.0
        self.sat = False  # roll / pitch demand beyond the vane range
        self.sat_yaw = False  # yaw demand cut short by the vane range
        self.F_des = np.array([0.0, 0.0, W])
        self.tau = np.zeros(3)
        self.goals = np.zeros(4)
        self.duty = 0.0
        self.p_ref = np.zeros(3)
        self.v_ref = np.zeros(3)
        self.target = np.array([p["dx_m"], p["dy_m"]])
        self.hold_until = 0.0
        self.sink = 0.0
        self.t_touch = 0.0
        self.leg = 0
        self.lidar_last = 0.0

    # ------------------------------------------------------------------------------------------------------------- helpers
    def _go(self, name, t):
        self.events.append((round(t, 3), name))
        self.state = name
        self.t_state = t

    def _plan(self, t, p1, vmax_xy, vmax_z):
        """Chain a min-jerk segment to p1 (z above the standing height of the centre of mass)."""
        p, _, _ = self.ref.state(t)
        d = np.asarray(p1, float) - p
        T = max(
            trj.duration(math.hypot(d[0], d[1]), vmax_xy, self.p["amax_xy"]),
            trj.duration(abs(d[2]), vmax_z, self.p["amax_z"]),
        )
        self.ref.go(t, p1, T)

    def _abort(self, t, why):
        if self.state in ("ABORT_LAND", "TOUCHDOWN", "LANDED", "FAULT"):
            return
        self.abort_reason = why
        self.events.append((round(t, 3), f"abort: {why}"))
        self.target = self.est.pos[:2].copy()
        self._go("ABORT_LAND", t)

    # ------------------------------------------------------------------------------------------------------------- one control tick
    def step(self, t, gyro, accel, baro, gnss, lidar, vbat):
        p = self.p
        e = self.est
        self.t_last = t
        self.vbat += 0.05 * (vbat - self.vbat) if self.vbat > 1 else (vbat - self.vbat)
        if self.state == "PAD":
            e.calibrate(gyro, accel, baro, gnss, lidar)
            if t >= p["t_cal"] and e.ready():
                e.finish()
                self.events.append((round(t, 3), "calibrated"))
                self._go("READY", t)
            return Cmd()
        e.update(gyro, accel, baro, gnss, lidar)
        pos, vel, R = e.pos, e.vel, e.att.R
        tilt = e.att.tilt
        if self.state == "READY":
            if t >= p["t_launch"]:
                self.thr.reset()
                self._go("SPOOL", t)
            return Cmd()
        if self.state in ("LANDED", "FAULT"):
            return Cmd()
        # ---- aborts
        airborne = self.state not in ("SPOOL", "TOUCHDOWN")
        if airborne and self.state != "ABORT_LAND":
            if tilt > math.radians(p["abort_tilt_deg"]):
                self._abort(t, "tilt")
            self.v_low_t = self.v_low_t + DT if self.vbat < p["abort_v"] else 0.0
            if self.v_low_t > 0.5:
                self._abort(t, "battery")
            if e.gnss_age > p["abort_gnss_s"]:
                self._abort(t, "gnss")
            if (
                t - self.t_state > p["t_max_s"]
                or np.hypot(*(pos[:2] - self._plan_line_pt())) > p["abort_off_m"]
            ):
                self._abort(t, "off-plan")
        if tilt > math.radians(60.0) and self.state != "SPOOL":
            self.events.append((round(t, 3), "fault: tilt > 60 deg, thrust cut"))
            self._go("FAULT", t)
            return Cmd()
        # ---- state machine -> reference
        hold_int = False
        v_ref, a_ref = np.zeros(3), np.zeros(3)
        if self.state == "SPOOL":
            self.p_ref = np.array([0.0, 0.0, 0.0])
            if pos[2] > 0.04:
                self.ref.start(t, pos, vel)
                self._plan(
                    t, np.array([0.0, 0.0, p["alt_m"]]), p["vmax_xy"], p["vmax_up"]
                )
                self._go("CLIMB", t)
            elif t - self.t_state > 6.0:
                self.events.append((round(t, 3), "fault: no lift-off"))
                self._go("FAULT", t)
                return Cmd()
        elif (
            self.state == "CLIMB"
            and self.ref.done(t)
            and abs(pos[2] - p["alt_m"]) < 0.5
        ):
            if np.hypot(*self.target) > 0.05:
                self._plan(
                    t, np.array([*self.target, p["alt_m"]]), p["vmax_xy"], p["vmax_up"]
                )
                self._go("TRANSIT", t)
            else:
                self.hold_until = t + p["hold_s"]
                self._go("HOVER", t)
        elif self.state == "TRANSIT" and self.ref.done(t):
            self.hold_until = t + p["hold_s"]
            self._go("HOVER", t)
        elif self.state == "HOVER" and t >= self.hold_until:
            self._plan(
                t, np.array([*self.target, p["z_final"]]), p["vmax_xy"], p["vmax_down"]
            )
            self._go("DESCENT", t)
        elif (
            self.state == "DESCENT"
            and self.ref.done(t)
            and abs(pos[2] - p["z_final"]) < 0.5
        ):
            self.ref.start(t, np.array([*self.target, pos[2]]))
            self._go("FINAL", t)
        elif self.state == "ABORT_LAND" and self.state != "FINAL":
            if t - self.t_state > 0.5:
                self.ref.start(t, np.array([*self.target, pos[2]]))
        p_ref, v_ref, a_ref = self.ref.update(t)
        kp_scale = 1.0
        if self.state in ("FINAL", "ABORT_LAND"):
            z = pos[2]
            sink = min(p["v_sink_max"], max(p["v_sink_min"], p["k_sink"] * z))
            self.sink += min(
                max(sink - self.sink, -3.0 * DT), 3.0 * DT
            )  # sink rate changes gently
            p_ref = np.array([*self.target, pos[2]])
            v_ref = np.array([0.0, 0.0, -self.sink])
            a_ref = np.zeros(3)
            hold_int = z < 0.15
            if z < 0.02:
                self.t_touch = t
                self._go("TOUCHDOWN", t)
        if self.state == "TOUCHDOWN":
            self.thr.T_cmd = max(self.thr.T_cmd - 80.0 * DT, 0.0)
            if self.thr.T_cmd <= 0.0 and t - self.t_state > 0.8:
                self._go("LANDED", t)
            duty = veh.duty_for_thrust(self.thr.T_cmd, self.vbat)
            self.duty = duty
            self.goals = np.zeros(4)
            return Cmd(self.goals.copy(), duty)
        # ---- controllers
        ground = self.state == "SPOOL"
        if ground:
            # 1.12 W, and after 2 s still on the ground (weak fan, heavy vehicle) 6 % of W more per second up to 1.6 W
            F = np.array(
                [0.0, 0.0, W * min(1.6, 1.12 + 0.06 * max(0.0, t - self.t_state - 2.0))]
            )
        else:
            F = self.pos.force(
                p_ref, v_ref, a_ref, pos, vel, DT, hold_int=hold_int, kp_scale=kp_scale
            )
            if self.state == "ABORT_LAND":
                F[:2] *= 0.7
        self.F_des = F
        n_des = F / np.linalg.norm(F)
        zb = R[:, 2]
        T_want = float(max(F @ zb, 0.0)) if not ground else float(F[2])
        duty, T_cmd = self.thr.throttle(
            T_want,
            e.thrust_accel,
            self.vbat,
            DT,
            learn=not ground,
            rate=p["dT_spool"] if ground else None,
        )
        self.tau = self.att.torque(
            R,
            e.omega,
            n_des,
            0.0,
            T_cmd,
            DT,
            freeze=ground or hold_int,
            dT=self.thr.dT,
            alpha=e.alpha,
        )
        d, self.sat, self.sat_yaw = ctl.allocate(self.tau, max(T_cmd, 5.0))
        self.goals = d
        self.duty = duty
        self.p_ref, self.v_ref = p_ref, v_ref
        return Cmd(d.copy(), duty)

    def _plan_line_pt(self):
        """Nearest point of the planned ground track to the estimate: the segment from the pad to the target."""
        a, b = np.zeros(2), self.target
        pe = self.est.pos[:2]
        ab = b - a
        n = float(ab @ ab)
        s = 0.0 if n < 1e-9 else min(max(float((pe - a) @ ab) / n, 0.0), 1.0)
        return a + s * ab

    # ------------------------------------------------------------------------------------------------------------- telemetry
    def telemetry(self):
        e = self.est
        if e.kf is None:
            return dict(state=self.state)
        pos, vel = e.pos, e.vel
        roll, pitch, yaw = e.att.euler()
        return dict(
            state=self.state,
            x_m=pos[0], y_m=pos[1], alt_m=pos[2], vx_ms=vel[0], vy_ms=vel[1], vz_ms=vel[2],
            roll_deg=math.degrees(roll), pitch_deg=math.degrees(pitch), yaw_deg=math.degrees(yaw), tilt_deg=math.degrees(e.att.tilt),
            ref_x_m=float(self.p_ref[0]), ref_y_m=float(self.p_ref[1]), ref_alt_m=float(self.p_ref[2]),
            ref_vz_ms=float(self.v_ref[2]), tau=[float(x) for x in self.tau], goals_deg=[math.degrees(float(g)) for g in self.goals],
            thrust_cmd_N=float(self.thr.T_cmd), thrust_est_N=float(self.thr.T_meas), thrust_trim_N=float(self.thr.trim), duty=float(self.duty),
            vane_sat=bool(self.sat), yaw_sat=bool(self.sat_yaw),lidar_ok=bool(e.lidar_ok), gnss_age_s=float(e.gnss_age), abort=self.abort_reason,
        )  # fmt: skip
