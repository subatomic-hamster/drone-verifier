import math

import numpy as np


class Controller:
    """Positive control: climb, hold, descend, cut. Cascaded position -> tilt -> vane torque, thrust by feed-forward plus altitude PID."""

    def __init__(self, cfg):
        self.c = cfg
        self.m, self.g = cfg.mass, cfg.g
        self.I = cfg.inertia
        B = np.cross(
            cfg.vanes.site_rel, cfg.vanes.force_dir
        ).T  # body torque per newton of each vane
        self.pinv_xy = np.linalg.pinv(B[:2])
        self.yaw_per_n = float(B[2] @ np.ones(4))
        tab = np.array(cfg.fan.thrust_motor_volts)
        self.tv_t, self.tv_v = (
            np.concatenate([[0.0], tab[:, 0]]),
            np.concatenate([[0.0], tab[:, 1]]),
        )
        self.alt, self.hold = cfg.mission.alt_m, cfg.mission.hold_s
        self.phase, self.t0, self.t_ref = "wait", cfg.t_cal + 0.5, 0.0
        self.zref, self.iz = 0.0, 0.0
        self.T = 0.0
        self.ixy = np.zeros(2)
        self.itau = np.zeros(2)

    def duty(self, T, vbat):
        return float(
            np.clip(np.interp(T, self.tv_t, self.tv_v) / max(vbat, 1.0), 0.0, 1.0)
        )

    def step(self, t, s):
        off = {"duty": 0.0, "vanes": [0.0, 0.0, 0.0, 0.0]}
        if not s.armed or t < self.t0:
            return off
        dt = self.c.dt
        z, vz = float(s.pos[2]), float(s.vel[2])
        vref = 0.0
        if self.phase == "wait":
            self.phase = "spool"
        if self.phase == "spool":
            self.T = min(self.T + 8.0 * dt, 1.3 * self.m * self.g)
            if z > 0.06 and self.T > 0.95 * self.m * self.g:
                self.phase, self.zref, self.iz = "climb", z, 0.0
        elif self.phase == "climb":
            vref = 1.0
            self.zref += vref * dt
            if self.zref >= self.alt:
                self.zref, vref, self.phase, self.t_ref = self.alt, 0.0, "hold", t
        elif self.phase == "hold":
            if t - self.t_ref > self.hold + 2.0:
                self.phase = "down"
        elif self.phase == "down":
            vref = -0.8 if z > 1.0 else -0.45
            self.zref += vref * dt
            if z < 0.04 or self.zref < -0.3:
                self.phase = "cut"
        if self.phase == "cut":
            return off

        w, x, y, zq = s.att
        R = np.array(
            [
                [1 - 2 * (y * y + zq * zq), 2 * (x * y - w * zq), 2 * (x * zq + w * y)],
                [2 * (x * y + w * zq), 1 - 2 * (x * x + zq * zq), 2 * (y * zq - w * x)],
                [2 * (x * zq - w * y), 2 * (y * zq + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        # position -> desired acceleration
        exy = -s.pos[:2]
        if self.phase != "spool":
            self.ixy = np.clip(self.ixy + 0.3 * exy * dt, -1.0, 1.0)
            self.iz = float(np.clip(self.iz + 0.8 * (self.zref - z) * dt, -3.0, 3.0))
        axy = np.clip(1.2 * exy - 2.0 * s.vel[:2] + self.ixy, -2.5, 2.5)
        if self.phase == "spool":
            axy = np.zeros(
                2
            )  # on the legs: level, whatever the young GNSS estimate says
        az = 2.5 * (self.zref - z) + 3.0 * (vref - vz) + self.iz
        n = np.array([axy[0], axy[1], self.g])
        n = n / np.linalg.norm(n)
        if self.phase != "spool":
            self.T = float(
                np.clip(self.m * (self.g + az) / max(R[2, 2], 0.7), 2.0, 34.0)
            )
        # attitude: rotate body z to n; hold heading
        zb = R[:, 2]
        e = R.T @ np.cross(zb, n)
        e[2] = -math.atan2(R[1, 0], R[0, 0])
        wn = np.array([8.0, 8.0, 3.0])
        rate = np.array(s.rate)
        tau = self.I * (wn * wn * e - 2 * 0.85 * wn * rate)
        T = max(self.T, 3.0)
        if self.phase != "spool":
            self.itau = np.clip(
                self.itau + self.I[:2] * wn[:2] ** 2 * e[:2] * 0.5 * dt, -0.08, 0.08
            )
        tau[:2] += self.itau
        H = (
            self.c.fan.spin_inertia
            * 2
            * math.pi
            * math.sqrt(T / (self.c.fan.ct * self.c.fan.rho * self.c.fan.diameter**4))
        )
        tau += np.cross(rate, np.array([0.0, 0.0, H]))
        tau[2] += self.c.fan.yaw_torque_per_newton * T
        # allocation
        k = self.c.vanes.force_per_rad_per_newton * T
        lim = self.c.vanes.limit_rad
        d = np.clip(self.pinv_xy @ tau[:2] / k, -lim, lim)  # roll and pitch first
        yaw = float(tau[2]) / self.yaw_per_n / k  # yaw gets what range is left
        d = d + float(np.clip(yaw, -lim - d.min(), lim - d.max()))
        return {"duty": self.duty(self.T, s.vbat), "vanes": [float(v) for v in d]}
