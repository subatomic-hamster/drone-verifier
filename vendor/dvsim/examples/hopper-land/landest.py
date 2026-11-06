"""State estimation of the hopper-land flight computer: examples/hopper/estimator.py (imported, unchanged) with the three things the flat-ground, sea-level original assumes removed.

terrain-aware lidar   The original fuses every lidar return as the height above the take-off pad (`z = h - H_STAND`: the ground under the beam is the plane z = 0). Over a mesa edge, a ridge or a
                      ledge that is wrong by the terrain relief. Here the lidar only corrects the vertical filter over a SURVEYED PAD (known elevation `dz` above the take-off pad, flat out to `pad_r`
                      m, the estimated position inside it): `z = h - H_STAND + dz`. Anywhere else the return is kept as `agl` (height of the centre of mass above whatever is below, for the
                      terrain-clearance guard) and the vertical filter runs on the barometer, the RTK height and the accelerometer. A return that disagrees with the filter by more than
                      1 m + 4 sigma is rejected (rain gives false near returns), as in the original.
air density           rho from the barometer alone, standard day (landveh.isa_rho_from_pressure), low-passed over ~0.5 s. The original's feed-forward used 1.225 kg/m3.
gust disturbance      The accelerometer sees the wind's drag on the body directly: the specific force in the body x / y axes is drag + the jet vanes' own force (the fan's thrust is along z). With the vane
                      force removed (known from the commanded deflection), what is left is the wind's push, low-passed (tau_d). The position loop subtracts it (landflight, gain `k_dob`): the thrust
                      vector leans into a gust at the acceleration level instead of waiting for the velocity error to grow.
"""

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "examples" / "hopper"))
sys.path.insert(0, str(HERE))

import estimator as base  # examples/hopper/estimator.py
import landveh
import vehicle as veh

WGS_A, WGS_E2 = (
    6378137.0,
    6.69437999014e-3,
)  # WGS84 semi-major axis, first eccentricity squared
Attitude, AxisFilter, baro_height, lidar_height = (
    base.Attitude,
    base.AxisFilter,
    base.baro_height,
    base.lidar_height,
)


class LandEstimator(base.Estimator):
    def __init__(self, dt, gnss_lag_s=0.1, tau_rho=0.5, tau_d=0.15):
        super().__init__(dt, gnss_lag_s)
        self.pads = [
            (0.0, 0.0, 0.0, 3.0)
        ]  # surveyed flat surfaces: (x, y, dz above the take-off pad, flat radius), local frame (x east, y north of the take-off pad)
        self.deck, self.deck_half, self.t_now = (
            None,
            1.5,
            0.0,
        )  # landdeck.DeckState (latest), half width of the deck (m), clock (s): set by the flight software
        self.rho = veh.RHO0
        self._rho_ok = False
        self.agl = None  # m, centre of mass above the ground under the beam (last valid return), None without
        self.agl_age = 99.0
        self.g_est, self.g_t = (
            None,
            99.0,
        )  # ground level under the last off-pad lidar return (m above the take-off pad datum) and its age
        self.pad_lidar = False  # the last return was fused as a pad return
        self.a_dist = np.zeros(
            2
        )  # m/s2, wind's push on the body, world x / y (low-passed)
        self.vane_force_b = np.zeros(
            2
        )  # N, jet vane force in the body x / y axes (set by the flight software from its commands)
        self.k_rho, self.k_d = min(1.0, dt / tau_rho), min(1.0, dt / tau_d)
        self.sigma_extra = 0.0  # m, extra lidar sigma (rain)
        self.rejected = 0

    def _local(self, fix):
        """GNSS (lat, lon, alt) -> (east, north, up) from the pad with the WGS84 radii of the pad's latitude (the original uses a 6371 km sphere: 0.28 % too short east-west at 47 N, 0.13 m over a
        48 m hop -- invisible in the original's 10 m hops, a bias here)."""
        lat, lon, alt = fix[:3]
        s2 = math.sin(math.radians(self.gnss_ref[0])) ** 2
        w = 1.0 - WGS_E2 * s2
        r_mer, r_pv = WGS_A * (1.0 - WGS_E2) / (w * math.sqrt(w)), WGS_A / math.sqrt(w)
        y = math.radians(lat - self.gnss_ref[0]) * r_mer
        x = (
            math.radians(lon - self.gnss_ref[1])
            * r_pv
            * math.cos(math.radians(self.gnss_ref[0]))
        )
        return x, y, alt - self.gnss_ref[2]

    def height_above_ground(self, max_age=0.3):
        """Height of the feet above the ground below (m), for an abort away from any surveyed pad: the last lidar return if it is fresh, else the filter height over the ground level the
        lidar last saw, else 99 (the ground has not been seen: sink at the fast rate until it is)."""
        if self.agl is not None and self.agl_age < max_age:
            return self.agl - veh.H_STAND
        if self.g_est is not None and self.g_t < 3.0:
            return self.kf[2].pos - self.g_est
        return 99.0

    def pad_at(self, x, y, margin=0.6):
        """-> (dz, index) of the surveyed pad whose flat disc (less `margin`) contains (x, y), else None."""
        if (
            self.deck is not None
        ):  # a moving deck: its surface point now, from the datalink's last state
            p, _ = self.deck.predict(self.t_now)
            if math.hypot(x - p[0], y - p[1]) < self.deck_half - margin:
                return float(p[2]), -1
        for i, (px, py, dz, r) in enumerate(self.pads):
            if math.hypot(x - px, y - py) < r - margin:
                return dz, i
        return None

    def update(self, gyro, accel, baro=None, gnss=None, lidar=None):
        dt = self.dt
        gyro, accel = np.asarray(gyro, float), np.asarray(accel, float)
        w_prev = self.omega
        self.att.update(gyro, dt)
        w = gyro - self.att.bias
        self.omega = w
        a_new = (w - w_prev) / dt
        self.alpha += 0.12 * (a_new - self.alpha)
        r = veh.IMU_REL
        self.f_cg = accel - np.cross(self.alpha, r) - np.cross(w, np.cross(w, r))
        R = self.att.R
        aw = R @ self.f_cg
        for i, a in enumerate((aw[0], aw[1], aw[2] - veh.G)):
            self.kf[i].predict(a)
        # disturbance: body x / y specific force less the vane forces, to the world
        ab = self.f_cg[:2] - self.vane_force_b / veh.MASS
        a_w = (R @ np.array([ab[0], ab[1], 0.0]))[:2]
        self.a_dist += self.k_d * (a_w - self.a_dist)
        # barometer: height above the take-off pad (ISA pressure altitude difference) and the density
        if baro is not None:
            hb = baro_height(baro, self.p_ref) - self.baro_bias
            self.innov["baro"] = self.kf[2].update(0, hb, 0.20)
            rho = landveh.isa_rho_from_pressure(baro)
            self.rho = (
                rho
                if not self._rho_ok
                else self.rho + 5 * self.k_rho * (rho - self.rho)
            )  # a baro sample every 5th tick (50 Hz): time constant tau_rho
            self._rho_ok = True
        # lidar
        self.lidar_ok = False
        self.pad_lidar = False
        self.agl_age += dt
        self.g_t += dt
        if lidar and veh.LIDAR_MIN < lidar < veh.LIDAR_MAX and R[2, 2] > 0.7:
            h = lidar_height(lidar, R)
            sig = 0.04 + 0.012 * lidar + self.sigma_extra
            pad = self.pad_at(self.kf[0].pos, self.kf[1].pos)
            if pad is not None:
                z_l = h - veh.H_STAND + pad[0]
                if abs(z_l - self.kf[2].pos) < 1.0 + 4 * sig:
                    self.innov["lidar"] = self.kf[2].update(0, z_l, sig)
                    self.lidar_ok = self.pad_lidar = True
                    self.lidar_z = z_l
                    self.agl, self.agl_age = h, 0.0
                    if (
                        lidar < 6.0 and baro is not None
                    ):  # the barometer's offset against the lidar, slowly
                        hb = baro_height(baro, self.p_ref)
                        self.baro_bias += 0.01 * ((hb - z_l) - self.baro_bias)
                else:
                    self.rejected += 1
            else:
                self.agl, self.agl_age = h, 0.0
                self.g_est, self.g_t = self.kf[2].pos - (h - veh.H_STAND), 0.0
        # GNSS (delayed)
        if gnss is not None:
            x, y, z = self._local(gnss)
            vn, ve, vd = gnss[3:6]
            self.kf[0].update(0, x, self.sig_gnss, self.lag)
            self.kf[1].update(0, y, self.sig_gnss, self.lag)
            self.innov["gnss_z"] = self.kf[2].update(0, z, 0.10, self.lag)
            self.kf[0].update(1, ve, self.sig_vel, self.lag)
            self.kf[1].update(1, vn, self.sig_vel, self.lag)
            self.kf[2].update(1, -vd, self.sig_vel, self.lag)
            self.gnss_age = 0.0
        else:
            self.gnss_age += dt
