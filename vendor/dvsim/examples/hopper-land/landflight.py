"""Flight software of the hopper-land flight computer: examples/hopper/flight.py with a landing pad at another height, air at 2500-4000 m, gusts and a moving deck. Same states and interfaces
(sensors in, four servo goals and the ESC duty out, 250 Hz, no simulator imports); what changed:

  mission     `Mission(alt_m, dx_m, dy_m, dz_m, pad_r)`: the landing pad is dz_m above the take-off pad (a surveyed point, from the RTK base survey / site map); alt_m is the cruise height above the
              take-off pad (the planner puts it above the terrain between the pads). The final approach, the touchdown test and the sink law use the height above the LANDING pad, `pos z - dz`,
              and the estimator trusts the lidar only over a surveyed pad (landest). Deck landing: `step(..., deck=DeckState)`, see landdeck.py.
  air         the throttle map, the thrust limit and the gyroscopic / spin-up feed-forward use the density the estimator derives from the barometer (the original's RHO0 = 1.225). Before the fan is
              started the firmware checks full-throttle thrust / weight at that density (landveh.thrust_margin) and refuses with "fault: thrust margin" below `margin_min`.
  gusts       the wind's push on the body, measured by the accelerometer (landest.a_dist), is subtracted from the position loop's force with gain `k_dob` (0 = the original).
"""

import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "examples" / "hopper"))
sys.path.insert(0, str(HERE))

import control as ctl  # examples/hopper (unchanged)
import landveh
import trajectory as trj
import vehicle as veh
from flight import DEFAULTS as BASE_DEFAULTS
from flight import DT, Cmd
from landest import LandEstimator

G = veh.G
W = veh.MASS * G
DEFAULTS = dict(
    BASE_DEFAULTS, k_dob=0.7, margin_min=landveh.MARGIN_MIN, v_sink_min=0.25, z_final=2.0, t_max_s=160.0, abort_off_m=12.0,
    use_window=1.0, window_wait_s=20.0, deck_hold_h=0.45, deck_vz_hold=0.15, hold_horizon_s=1.8, hold_lead_s=0.9, hold_max_s=12.0,
)  # fmt: skip


@dataclass
class Mission:
    alt_m: float = 10.0  # cruise height above the take-off pad
    dx_m: float = 0.0
    dy_m: float = 0.0
    dz_m: float = 0.0  # landing pad height above the take-off pad
    pad_r: float = 3.0  # flat radius of the pads (the lidar is trusted inside it)
    hold_s: float = 3.0
    deck_half: float = 1.5  # half width of a moving deck's landing surface
    fixed_pads: bool = True  # the take-off pad stands still in the flight frame; False on a ship (the deck is the only pad and it moves, the climb follows it)


class ThrustCtlRho(ctl.ThrustCtl):
    """ctl.ThrustCtl with the throttle map at the air density `rho` (set every tick from the barometer)."""

    rho = veh.RHO0

    def throttle(self, T_want, f_z, v_bus, dt, learn=True, rate=None):
        T_want = min(max(T_want, self.t_min), self.t_max)
        r = (self.dT_max if rate is None else rate) * dt
        old = self.T_cmd
        self.T_cmd += min(max(T_want - self.T_cmd, -r), r)
        self.dT = (self.T_cmd - old) / dt
        self.T_meas += (veh.MASS * f_z - self.T_meas) * min(1.0, dt / self.f_tau)
        if learn:
            self.trim += self.ki * (self.T_cmd - self.T_meas) * dt
            self.trim = min(max(self.trim, -self.trim_max), self.trim_max)
        return landveh.duty_for_thrust(
            max(self.T_cmd + self.trim, 0.0), v_bus, self.rho
        ), self.T_cmd


class AttitudeCtlRho(ctl.AttitudeCtl):
    """ctl.AttitudeCtl with the fan speed behind the gyroscopic moment and the spin-up reaction taken at the density `rho`: n = sqrt(T / (C_T rho D^4)) rises as the air thins (x 1.17 at 3800 m). The
    parent computes both terms at RHO0; the difference is added."""

    rho = veh.RHO0

    def torque(
        self,
        R,
        omega,
        n_des,
        yaw_ref,
        thrust,
        dt,
        freeze=False,
        dT=0.0,
        alpha=np.zeros(3),
    ):
        tau = super().torque(R, omega, n_des, yaw_ref, thrust, dt, freeze, dT, alpha)
        s = math.sqrt(veh.RHO0 / self.rho) - 1.0
        n0 = math.sqrt(max(thrust, 0.0) / (veh.CT0 * veh.RHO0 * veh.D_FAN**4))
        H0 = np.array([0.0, 0.0, veh.J_FAN * 2 * math.pi * n0])
        tau = tau + s * np.cross(omega + self.t_lead * alpha, H0)
        if n0 > 1.0 and thrust > 1.0:
            tau[2] += s * veh.J_FAN * math.pi * n0 / thrust * dT
        return tau


class Flight:
    def __init__(self, mission=None, **kw):
        p = dict(DEFAULTS)
        p.update({k: v for k, v in kw.items() if k in DEFAULTS})
        self.p = p
        self.m = mission or Mission()
        self.est = LandEstimator(DT)
        self.est.pads = (
            [
                (0.0, 0.0, 0.0, self.m.pad_r),
                (self.m.dx_m, self.m.dy_m, self.m.dz_m, self.m.pad_r),
            ]
            if self.m.fixed_pads
            else []
        )
        self.t_gate = None  # when the descent reached the hold point over the deck (quiet-window gate)
        self.est.deck_half = self.m.deck_half
        self.att = AttitudeCtlRho(
            wn=p["wn_att"], zeta=p["zeta_att"], t_lead=p["t_lead"]
        )
        self.pos = ctl.PositionCtl(
            p["kp_xy"], p["kd_xy"], p["kp_z"], p["kd_z"], p["ki_xy"], p["ki_z"], a_xy_max=p["amax_xy"] * 2.5, tilt_max_deg=p["tilt_max_deg"],
        )  # fmt: skip
        self.thr = ThrustCtlRho(dT_max=p["dT_max"])
        self.ref = trj.Reference()
        self.state = "PAD"
        self.events = []
        self.t_state = 0.0
        self.abort_reason = ""
        self.vbat = 0.0
        self.v_low_t = 0.0
        self.sat = False
        self.sat_yaw = False
        self.F_des = np.array([0.0, 0.0, W])
        self.tau = np.zeros(3)
        self.goals = np.zeros(4)
        self.duty = 0.0
        self.p_ref = np.zeros(3)
        self.v_ref = np.zeros(3)
        self.target = np.array([self.m.dx_m, self.m.dy_m])
        self.hold_until = 0.0
        self.sink = 0.0
        self.t_touch = 0.0
        self.h_hist = deque(
            maxlen=25
        )  # the last 0.1 s of height above the surface (touchdown test)
        self.margin = float("nan")
        self.deck = None  # landdeck.DeckState (latest)
        self.t_last = 0.0

    # ------------------------------------------------------------------------------------------------------------- helpers
    def _go(self, name, t):
        self.events.append((round(t, 3), name))
        self.state = name
        self.t_state = t

    def _plan(self, t, p1, vmax_xy, vmax_z, follow=False):
        """Chain a min-jerk segment to p1. follow: p1's xy is the landing point and the deck is moving: aim at where the deck will be when the segment ends and arrive with its velocity."""
        p, _, _ = self.ref.state(t)
        p1 = np.asarray(p1, float)
        if not self.m.fixed_pads:  # over a moving deck only the height is planned; the horizontal reference IS the deck (step)
            p1, follow = np.array([p[0], p[1], p1[2]]), False
        d = p1 - p
        T = max(
            trj.duration(math.hypot(d[0], d[1]), vmax_xy, self.p["amax_xy"]),
            trj.duration(abs(d[2]), vmax_z, self.p["amax_z"]),
        )
        v1 = (0.0, 0.0, 0.0)
        if follow and self.deck is not None:
            for _ in range(
                2
            ):  # the deck moves while we fly: re-time the segment to the intercept point
                txy, vtxy = self._target_xy(t, T)
                d = np.array([txy[0] - p[0], txy[1] - p[1], p1[2] - p[2]])
                T = max(
                    trj.duration(math.hypot(d[0], d[1]), vmax_xy, self.p["amax_xy"]),
                    trj.duration(abs(d[2]), vmax_z, self.p["amax_z"]),
                )
            txy, vtxy = self._target_xy(t, T)
            p1 = np.array([txy[0], txy[1], p1[2]])
            v1 = (vtxy[0], vtxy[1], 0.0)
        self.ref.go(t, p1, T, v1)

    def _abort(self, t, why):
        if self.state in ("ABORT_LAND", "TOUCHDOWN", "LANDED", "FAULT"):
            return
        self.abort_reason = why
        self.events.append((round(t, 3), f"abort: {why}"))
        self.target = self.est.pos[:2].copy()
        self.deck = None  # an abort lands where it is, not on the deck
        self._go("ABORT_LAND", t)

    def ground(self, t):
        """(height of the landing surface above the take-off pad, its vertical velocity) at time t: the surveyed pad, or the deck's predicted state."""
        if self.deck is not None:
            pz, vz = self.deck.predict_z(t)
            return pz, vz
        return self.m.dz_m, 0.0

    def _window_open(self, t):
        """May the final sink start? Always, unless a deck motion predictor is running (DeckState.ready) and `use_window`: then when it names a quiet window starting now, or after `window_wait_s`
        at the hold point (a sea that never goes quiet is landed on anyway: the touchdown monitors report what that costs)."""
        d = self.deck
        if d is None or not self.p["use_window"] or d.ready is None:
            return True
        if self.t_gate is None:
            self.t_gate = t
        return (
            d.window is not None and d.window - t < 0.3
        ) or t - self.t_gate > self.p["window_wait_s"]

    def _target_xy(self, t, lead=0.0):
        """(horizontal position, velocity) of the landing point at time t + lead: the surveyed pad, or the deck as the datalink last reported it, extrapolated."""
        if self.deck is not None:
            p, v = self.deck.predict(t + lead)
            return p[:2], v[:2]
        return self.target, np.zeros(2)

    # ------------------------------------------------------------------------------------------------------------- one control tick
    def step(self, t, gyro, accel, baro, gnss, lidar, vbat, deck=None):
        p, m = self.p, self.m
        e = self.est
        self.t_last = t
        if deck is not None and self.state != "ABORT_LAND":
            self.deck = deck
        self.vbat += 0.05 * (vbat - self.vbat) if self.vbat > 1 else (vbat - self.vbat)
        if self.state == "PAD":
            if deck is not None and deck.omega_b is not None:
                # on a moving deck the gyro bias is the gyro less the deck's own rotation, and the accelerometers' mean points along the vertical in the deck frame, both from the datalink
                gyro = np.asarray(gyro, float) - deck.omega_b
                accel = float(np.linalg.norm(accel)) * deck.up_b
            e.calibrate(gyro, accel, baro, gnss, lidar)
            if t >= p["t_cal"] and e.ready():
                e.finish()
                self.events.append((round(t, 3), "calibrated"))
                self._go("READY", t)
            return Cmd()
        e.update(gyro, accel, baro, gnss, lidar)
        pos, vel, R = e.pos, e.vel, e.att.R
        tilt = e.att.tilt
        self.thr.rho = self.att.rho = e.rho
        e.deck, e.t_now = self.deck, t
        self.thr.t_max = min(26.0, 0.92 * landveh.t_full(e.rho))
        if self.state == "READY":
            if t >= p["t_launch"]:
                self.margin = landveh.thrust_margin(e.rho)
                self.events.append(
                    (
                        round(t, 3),
                        f"air: rho {e.rho:.3f} kg/m3, full-throttle thrust / weight {self.margin:.2f}",
                    )
                )
                if self.margin < p["margin_min"]:
                    self.events.append(
                        (
                            round(t, 3),
                            f"fault: thrust margin {self.margin:.2f} < {p['margin_min']:.2f}: not launching",
                        )
                    )
                    self.state = "FAULT"
                    self.t_state = t
                    return Cmd()
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
            if t - self.t_state > p["t_max_s"] or (
                self.deck is None
                and np.hypot(*(pos[:2] - self._plan_line_pt())) > p["abort_off_m"]
            ):
                self._abort(t, "off-plan")
        if tilt > math.radians(60.0) and self.state != "SPOOL":
            self.events.append((round(t, 3), "fault: tilt > 60 deg, thrust cut"))
            self._go("FAULT", t)
            return Cmd()
        # ---- state machine -> reference
        hold_int = False
        v_ref, a_ref = np.zeros(3), np.zeros(3)
        zg, vzg = self.ground(t)
        txy, vtxy = self._target_xy(t)
        z_fin = zg + p["z_final"]
        if self.state == "SPOOL":
            self.p_ref = np.array([0.0, 0.0, 0.0])
            if pos[2] > 0.04:
                self.ref.start(t, pos, vel)
                if (
                    not m.fixed_pads and self.deck is not None
                ):  # lifting off a moving deck: climb over it
                    self._plan(
                        t,
                        np.array([*txy, m.alt_m]),
                        p["vmax_xy"],
                        p["vmax_up"],
                        follow=True,
                    )
                else:
                    self._plan(
                        t, np.array([0.0, 0.0, m.alt_m]), p["vmax_xy"], p["vmax_up"]
                    )
                self._go("CLIMB", t)
            elif t - self.t_state > 6.0:
                self.events.append((round(t, 3), "fault: no lift-off"))
                self._go("FAULT", t)
                return Cmd()
        elif self.state == "CLIMB" and self.ref.done(t) and abs(pos[2] - m.alt_m) < 0.5:
            if np.hypot(*self.target) > 0.05:
                self._plan(
                    t,
                    np.array([*txy, m.alt_m]),
                    p["vmax_xy"],
                    p["vmax_up"],
                    follow=True,
                )
                self._go("TRANSIT", t)
            else:
                self.hold_until = t + m.hold_s
                self._go("HOVER", t)
        elif self.state == "TRANSIT" and self.ref.done(t):
            self.hold_until = t + m.hold_s
            self._go("HOVER", t)
        elif self.state == "HOVER" and t >= self.hold_until:
            self._plan(
                t, np.array([*txy, z_fin]), p["vmax_xy"], p["vmax_down"], follow=True
            )
            self._go("DESCENT", t)
        elif (
            self.state == "DESCENT"
            and self.ref.done(t)
            and abs(pos[2] - z_fin) < 0.5
            and self._window_open(t)
        ):
            self.ref.start(t, np.array([*txy, pos[2]]))
            self._go("FINAL", t)
        elif self.state == "ABORT_LAND":
            if t - self.t_state > 0.5:
                self.ref.start(t, np.array([*self.target, pos[2]]))
        p_ref, v_ref, a_ref = self.ref.update(t)
        ship = not m.fixed_pads
        if self.deck is not None and self.state in (
            "HOVER",
            "DESCENT",
            "TRANSIT",
            *(("CLIMB",) if ship else ()),
        ):  # follow the deck's drift on top of the planned path
            p_ref = np.array([p_ref[0], p_ref[1], p_ref[2]])
            if self.state == "HOVER" or ship:
                p_ref[:2], v_ref[:2] = txy, vtxy
            if self.state == "DESCENT" and self.ref.done(
                t
            ):  # waiting over the deck for a quiet window: hold station on it
                p_ref, v_ref = np.array([*txy, z_fin]), np.array([*vtxy, vzg])
                a_ref = np.zeros(3)
        if self.state in ("FINAL", "ABORT_LAND"):
            h = pos[2] - zg if self.state == "FINAL" else e.height_above_ground()
            hold = False
            if self.deck is not None and self.state == "FINAL":
                if (
                    e.pad_lidar and e.agl_age < 0.1
                ):  # over the deck: the height above it is what the lidar reads, not the difference of two filtered heights
                    h = max(e.agl - veh.H_STAND, 0.0)
                # a deck rising fast would hit the fan-off hopper and throw it: hold `deck_hold_h` above it until the forecast says it will not rise faster than `deck_vz_hold` for the next `hold_horizon_s`
                # (the time the last sink takes), or `hold_max_s` after FINAL started (a sea that never goes quiet is landed on anyway)
                hold = (
                    h < p["deck_hold_h"] + 0.3
                    and self.deck.vz_max_ahead(t, p["hold_horizon_s"], p["hold_lead_s"])
                    > p["deck_vz_hold"]
                    and t - self.t_state < p["hold_max_s"]
                )
            sink = min(p["v_sink_max"], max(p["v_sink_min"], p["k_sink"] * h))
            if hold:
                sink = min(0.6, max(-0.6, h - p["deck_hold_h"]))
            self.sink += min(max(sink - self.sink, -3.0 * DT), 3.0 * DT)
            xy = txy if self.state == "FINAL" else self.target
            p_ref = np.array([*xy, pos[2]])
            vz_ref = -self.sink
            if self.deck is not None:
                # the deck's heave rate is a feed-forward only near the deck (full below 0.5 m, none above 1.5 m: far above it the datalink's rate would only add to the sink), and the descent
                # in the air stays below `v_sink_max`: faster, the throttle drops under 0.7 W and the jet vanes lose the yaw / roll authority that scales with the exhaust (a spin-up in the seed 21002 flight)
                w_ff = min(1.0, max(0.0, (1.5 - h) / 1.0))
                vz_ref = max(w_ff * vzg - self.sink, -p["v_sink_max"])
            v_ref = np.array([*(vtxy if self.state == "FINAL" else (0.0, 0.0)), vz_ref])
            a_ref = np.zeros(3)
            hold_int = h < 0.15
            # touchdown = the height averaged over 0.1 s is at the feet: one lidar sample (sigma ~3 cm) below 2 cm cut the fan 9 cm above the deck and let the hopper fall on it at 0.55 m/s
            # (seed 21001 of the scripted deck; the original's single-sample test did the same on terrain, where the slower sink hid it)
            self.h_hist.append(h)
            if (
                len(self.h_hist) == self.h_hist.maxlen
                and sum(self.h_hist) / len(self.h_hist) < 0.03
                and not hold
            ):
                self.t_touch = t
                self._go("TOUCHDOWN", t)
        if self.state == "TOUCHDOWN":
            self.thr.T_cmd = max(self.thr.T_cmd - 80.0 * DT, 0.0)
            if self.thr.T_cmd <= 0.0 and t - self.t_state > 0.8:
                self._go("LANDED", t)
            duty = landveh.duty_for_thrust(self.thr.T_cmd, self.vbat, e.rho)
            self.duty = duty
            self.goals = np.zeros(4)
            return Cmd(self.goals.copy(), duty)
        # ---- controllers
        ground = self.state == "SPOOL"
        if ground:
            F = np.array(
                [0.0, 0.0, W * min(1.6, 1.12 + 0.06 * max(0.0, t - self.t_state - 2.0))]
            )
        else:
            F = self.pos.force(p_ref, v_ref, a_ref, pos, vel, DT, hold_int=hold_int)
            F[:2] -= (
                veh.MASS * p["k_dob"] * e.a_dist
            )  # the wind's push, measured: lean into it now
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
        kq = veh.thrust_vane_gain(self.thr.T_meas)
        e.vane_force_b = (kq * d[:, None] * veh.VANE_DIR[:, :2]).sum(axis=0)
        return Cmd(d.copy(), duty)

    def _plan_line_pt(self):
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
            vane_sat=bool(self.sat), yaw_sat=bool(self.sat_yaw), lidar_ok=bool(e.lidar_ok), gnss_age_s=float(e.gnss_age), abort=self.abort_reason,
            rho=float(e.rho), agl_m=None if e.agl is None or e.agl_age > 0.5 else float(e.agl), wind_push_x=float(e.a_dist[0]), wind_push_y=float(e.a_dist[1]),
            margin=None if math.isnan(self.margin) else float(self.margin),
        )  # fmt: skip
