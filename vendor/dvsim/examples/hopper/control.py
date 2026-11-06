"""Cascaded control of the hopper: position / velocity loop (world frame, commands a thrust vector), attitude loop (points the body axis along it, holds the heading) and the
thrust loop (throttle from the thrust command, trimmed by the accelerometer), plus the allocation of the three body torques to the four jet-vane angles.

  position    a_cmd = a_ref + kd (v_ref - v) + kp (p_ref - p) + integral (wind), F_des = M (a_cmd + g e_z), tilt of F_des limited
  attitude    e = z_body x n_des (rad, in the body frame), tau = I (wn^2 e - 2 zeta wn w) + integral trim (centre-of-mass offset, vane zero) + gyro-moment feed-forward w x H_fan
  allocation  deltas = pseudo-inverse of the vane torque matrix B (rows: roll, pitch, yaw per unit force, from the vane sites and force directions of vehicle.py), roll / pitch first,
              yaw in the null space of the roll / pitch rows within the vane limit
  thrust      T_cmd = F_des . z_body, throttle = duty(T_cmd + trim, V) with the trim integrating (T_cmd - M f_z) on the accelerometer, rate-limited (the fan's own spin-up torque
              disturbs yaw: dT/dt <= dT_max)

Sign conventions (tests/python/test_hopper.py checks them against the built model): vane k pushes along VANE_DIR[k] for a positive servo angle; body torque = r x F."""

import math

import numpy as np
import vehicle as veh

G = veh.G
CN_MAX = 1.16


# ---------------------------------------------------------------------------------------------------------------- allocation
def torque_matrix():
    """B (3 x 4): body torque (N m) per newton of force of each vane (lever arm x direction)."""
    return np.array([np.cross(r, d) for r, d in zip(veh.VANE_SITE_REL, veh.VANE_DIR)]).T


_B = torque_matrix()
_BXY = _B[:2]
_PINV_XY = np.linalg.pinv(_BXY)
_BZ = _B[2]
# the null space of _BXY is 2-D: equal forces on vanes 1 and 3 (F1 = F3) and on 2 and 4 make no roll / pitch torque: that is the yaw
assert abs(_BXY @ np.array([1.0, 0, 1, 0])).max() < 1e-3 * abs(_BXY).max()
assert abs(_BXY @ np.array([0, 1.0, 0, 1])).max() < 1e-3 * abs(_BXY).max()


def force_to_angle(force, thrust):
    """Servo angle (rad) of a vane for the normal force `force` (N) in a jet driven by `thrust` (N): the flat-plate law of the jet_vane device inverted."""
    qa = 0.5 * veh.JET_FACTOR**2 * max(thrust, 1.0) / veh.A_EXIT * veh.VANE_AREA
    c = float(np.clip(force / qa, -0.95 * CN_MAX, 0.95 * CN_MAX))
    s = CN_MAX / veh.CN_ALPHA * math.atanh(c / CN_MAX)
    return math.asin(max(-1.0, min(1.0, s)))


def allocate(tau, thrust, limit=veh.VANE_LIMIT):
    """Body torque tau = (roll, pitch, yaw) N m -> four servo angles (rad), saturation flag. Roll / pitch have priority; yaw uses what is left of the vane range."""
    T = max(thrust, 4.0)
    forces = _PINV_XY @ np.asarray(tau[:2], float)  # N per vane: roll / pitch only
    d_rp = np.array([force_to_angle(f, T) for f in forces])
    sat_rp = False
    m = float(np.max(np.abs(d_rp)))
    if m > limit:
        d_rp *= limit / m
        sat_rp = True
    # yaw: the same angle a1 on vanes 1 and 3 (no roll torque), a2 on vanes 2 and 4 (no pitch torque), each as far as the range left by the roll / pitch part allows; linearised at zero
    kq = (
        0.5 * veh.JET_FACTOR**2 * T / veh.A_EXIT * veh.VANE_AREA * veh.CN_ALPHA
    )  # N per rad
    z1, z2 = (
        kq * (_BZ[0] + _BZ[2]),
        kq * (_BZ[1] + _BZ[3]),
    )  # N m of yaw per rad of the pair
    tz_left = tau[2] - float(_BZ @ (kq * d_rp))
    lo1, hi1 = -limit - min(d_rp[0], d_rp[2]), limit - max(d_rp[0], d_rp[2])
    lo2, hi2 = -limit - min(d_rp[1], d_rp[3]), limit - max(d_rp[1], d_rp[3])
    a1 = min(max(tz_left / (z1 + z2), lo1), hi1)
    a2 = min(max((tz_left - z1 * a1) / z2, lo2), hi2)
    a1 = min(max((tz_left - z2 * a2) / z1, lo1), hi1)
    sat_yaw = abs(z1 * a1 + z2 * a2 - tz_left) > 0.05 * abs(tz_left) + 1e-4
    d = d_rp + np.array([a1, a2, a1, a2])
    return np.clip(d, -limit, limit), sat_rp, sat_yaw


def vane_torque(deltas, thrust):
    """Body torque (N m) of the four angles in the linear regime, for checks."""
    kq = (
        0.5
        * veh.JET_FACTOR**2
        * max(thrust, 0.0)
        / veh.A_EXIT
        * veh.VANE_AREA
        * veh.CN_ALPHA
    )
    return _B @ (kq * np.asarray(deltas, float))


# ---------------------------------------------------------------------------------------------------------------- attitude
def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class AttitudeCtl:
    def __init__(
        self, wn=9.0, zeta=0.8, wn_yaw=4.0, t_int=2.0, tau_int_max=0.08, t_lead=0.035
    ):
        self.t_lead = t_lead
        self.wn, self.zeta, self.wn_yaw, self.ki = wn, zeta, wn_yaw, 1.0 / t_int
        self.integ = np.zeros(3)  # N m trim
        self.tau_int_max = tau_int_max

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
        """Body torque command (N m). R body -> world, omega body rates (bias-corrected gyro), n_des desired thrust axis (world unit vector)."""
        zb = R[:, 2]
        e_w = np.cross(zb, n_des)  # rotation needed (world frame), |e| = sin(angle)
        e_b = R.T @ e_w
        yaw = math.atan2(R[1, 0], R[0, 0])
        e_b[2] = wrap(yaw_ref - yaw)
        wn = np.array([self.wn, self.wn, self.wn_yaw])
        tau = veh.I_BODY * (wn**2 * e_b - 2 * self.zeta * wn * omega)
        if not freeze:
            self.integ += self.ki * veh.I_BODY * wn**2 * e_b * dt
            self.integ = np.clip(self.integ, -self.tau_int_max, self.tau_int_max)
        tau += self.integ
        # gyroscopic moment of the spinning fan: the body must supply w x H to turn the spin axis
        n_fan = math.sqrt(max(thrust, 0.0) / (veh.CT0 * veh.RHO0 * veh.D_FAN**4))
        H = np.array(
            [0.0, 0.0, veh.J_FAN * 2 * math.pi * n_fan]
        )  # the fan turns about +z (the MJCF spin joint axis)
        tau += np.cross(
            omega + self.t_lead * alpha, H
        )  # + lead: the vane servos lag the command by ~40 ms; the plant sees the moment late
        # the share of the fan's air torque that the stator does not return rides on the airframe (-z): pre-load the integrator's job with the design value
        tau[2] += veh.Q_UNRECOVERED * veh.CQ0 / veh.CT0 * veh.D_FAN * thrust
        # spin-up reaction: accelerating the fan by dOmega/dt pushes -J dOmega/dt on the airframe (Omega ~ sqrt(T)): cancel it with yaw torque
        if n_fan > 1.0 and thrust > 1.0:
            tau[2] += veh.J_FAN * math.pi * n_fan / thrust * dT
        return tau


# ---------------------------------------------------------------------------------------------------------------- position
class PositionCtl:
    def __init__(
        self,
        kp_xy=1.2,
        kd_xy=2.2,
        kp_z=2.5,
        kd_z=3.0,
        ki_xy=0.25,
        ki_z=0.4,
        a_xy_max=3.0,
        tilt_max_deg=22.0,
        int_max=1.5,
    ):
        self.kp_xy, self.kd_xy, self.kp_z, self.kd_z = kp_xy, kd_xy, kp_z, kd_z
        self.ki_xy, self.ki_z = ki_xy, ki_z
        self.a_xy_max = a_xy_max
        self.tilt_max = math.radians(tilt_max_deg)
        self.int_max = int_max
        self.integ = np.zeros(3)  # m/s2

    def force(self, p_ref, v_ref, a_ref, p, v, dt, hold_int=False, kp_scale=1.0):
        """Desired thrust vector (N, world) of the position loop."""
        e = p_ref - p
        ev = v_ref - v
        a = np.array(a_ref, float)
        a[:2] += self.kp_xy * kp_scale * e[:2] + self.kd_xy * ev[:2]
        a[2] += self.kp_z * e[2] + self.kd_z * ev[2]
        if not hold_int:
            self.integ[:2] += self.ki_xy * e[:2] * dt
            self.integ[2] += self.ki_z * e[2] * dt
            self.integ = np.clip(self.integ, -self.int_max, self.int_max)
        a += self.integ
        axy = float(np.hypot(a[0], a[1]))
        if axy > self.a_xy_max:
            a[:2] *= self.a_xy_max / axy
        F = veh.MASS * (a + np.array([0.0, 0.0, G]))
        F[2] = max(F[2], 0.15 * veh.MASS * G)  # never ask for (almost) negative lift
        # tilt limit of the thrust axis
        t = math.atan2(math.hypot(F[0], F[1]), F[2])
        if t > self.tilt_max:
            k = math.tan(self.tilt_max) / math.tan(t)
            F[:2] *= k
        return F


# ---------------------------------------------------------------------------------------------------------------- thrust
class ThrustCtl:
    def __init__(
        self, ki=6.0, trim_max=4.0, dT_max=15.0, t_min=0.0, t_max=26.0, f_tau=0.03
    ):
        self.ki, self.trim_max, self.dT_max = ki, trim_max, dT_max
        self.t_min, self.t_max = t_min, t_max
        self.trim = 0.0
        self.T_cmd = 0.0
        self.T_meas = 0.0
        self.dT = 0.0  # N/s, rate of the thrust command
        self.f_tau = f_tau

    def reset(self):
        self.trim = 0.0
        self.T_cmd = 0.0
        self.T_meas = 0.0

    def throttle(self, T_want, f_z, v_bus, dt, learn=True, rate=None):
        """-> (duty command, thrust command after the rate limit). f_z = specific force along the body axis (m/s2), v_bus battery voltage (V)."""
        T_want = min(max(T_want, self.t_min), self.t_max)
        r = (self.dT_max if rate is None else rate) * dt
        old = self.T_cmd
        self.T_cmd += min(max(T_want - self.T_cmd, -r), r)
        self.dT = (self.T_cmd - old) / dt
        self.T_meas += (veh.MASS * f_z - self.T_meas) * min(1.0, dt / self.f_tau)
        if learn:
            self.trim += self.ki * (self.T_cmd - self.T_meas) * dt
            self.trim = min(max(self.trim, -self.trim_max), self.trim_max)
        return veh.duty_for_thrust(max(self.T_cmd + self.trim, 0.0), v_bus), self.T_cmd
