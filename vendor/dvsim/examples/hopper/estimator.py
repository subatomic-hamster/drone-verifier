"""State estimation of the hopper's flight computer from its four sensors: IMU (gyro + accelerometer), barometer, GNSS (RTK rover: position + velocity + height) and the downward
lidar. Pure numpy, no simulator imports: `Estimator.update` takes numbers and returns a state.

  attitude   gyro-integrated quaternion, calibrated on the pad (gyro bias = mean rate, up vector = mean specific force, heading zero). The accelerometer never corrects it in
             flight: with the fan running the specific force is the thrust along the body axis, not gravity (it reads 1 g in every hover, tilted or not).
  position   one 3-state Kalman filter per world axis (position, velocity, accelerometer bias), propagated with the measured acceleration (lever arm of the IMU removed), corrected
             by GNSS position / velocity (delayed fixes) in x, y, z, by the barometer (z, with a slowly estimated bias against the lidar) and by the tilt-compensated lidar (z).
The attitude code is the lander's (examples/lander/estimator.py); the lidar, lever-arm and bias terms are new."""

import math

import numpy as np
import vehicle as veh

G = veh.G
EARTH_R = 6371000.0


# ---------------------------------------------------------------------------------------------------------------- attitude
def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_exp(rv):
    """Unit quaternion of the rotation vector rv (rad)."""
    a = float(np.linalg.norm(rv))
    if a < 1e-12:
        return np.array([1.0, *(0.5 * np.asarray(rv))])
    s = math.sin(0.5 * a) / a
    return np.array([math.cos(0.5 * a), *(s * np.asarray(rv))])


def quat_to_mat(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_from_up(u):
    """Attitude (body -> world) with the body-frame up vector u (the third row of R) and zero heading: the minimal rotation taking u to world z."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    axis = np.cross(u, [0.0, 0.0, 1.0])
    s = float(np.linalg.norm(axis))
    if s < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return quat_exp(axis / s * math.atan2(s, float(u[2])))


class Attitude:
    """Quaternion (body -> world) integrated from the gyro; calibrated on the pad. No accelerometer aiding in flight (see the module docstring)."""

    def __init__(self):
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.bias = np.zeros(3)
        self._g, self._a, self._n = np.zeros(3), np.zeros(3), 0

    def calibrate(self, gyro, accel):
        self._g += gyro
        self._a += accel
        self._n += 1

    def finish_calibration(self):
        self.bias = self._g / self._n
        self.q = quat_from_up(self._a / self._n)

    def update(self, gyro, dt):
        w = np.asarray(gyro, float) - self.bias
        self.q = quat_mul(self.q, quat_exp(w * dt))
        self.q /= np.linalg.norm(self.q)

    @property
    def R(self):
        return quat_to_mat(self.q)

    @property
    def tilt(self):
        """Angle between the body axis and the vertical (rad)."""
        return math.acos(max(-1.0, min(1.0, self.R[2, 2])))

    def euler(self):
        """(roll about x, pitch about y, yaw about z) in rad, ZYX; for the telemetry."""
        R = self.R
        return (
            math.atan2(R[2, 1], R[2, 2]),
            -math.asin(max(-1.0, min(1.0, R[2, 0]))),
            math.atan2(R[1, 0], R[0, 0]),
        )


# ------------------------------------------------------------------------------------------------------------ axis filter
class AxisFilter:
    """x = [position, velocity, accelerometer bias] of one world axis, propagated with the measured world acceleration, corrected by (delayed) position / velocity fixes. A
    measurement `lag` ticks old is compared with the state stored `lag` ticks ago and the Kalman gain is applied to the current state (a cheap delayed-measurement form that
    ignores the correlation between the stored and the current state)."""

    def __init__(
        self, dt, sigma_acc=0.3, sigma_bias_rw=0.02, p0=(1.0, 0.5, 0.3), history=80
    ):
        self.dt = dt
        self.x = np.zeros(3)
        self.P = np.diag(np.square(p0)).astype(float)
        self.sa2, self.sb2 = sigma_acc**2, sigma_bias_rw**2
        self.hist = []
        self.nh = history

    def predict(self, a):
        dt = self.dt
        F = np.array([[1.0, dt, -0.5 * dt * dt], [0.0, 1.0, -dt], [0.0, 0.0, 1.0]])
        g = np.array([0.5 * dt * dt, dt, 0.0])
        p, v, b = self.x
        ac = a - b
        self.x = np.array([p + v * dt + 0.5 * ac * dt * dt, v + ac * dt, b])
        Q = np.outer(g, g) * self.sa2 + np.diag([0.0, 0.0, self.sb2 * dt])
        self.P = F @ self.P @ F.T + Q
        self.hist.append(self.x.copy())
        if len(self.hist) > self.nh:
            self.hist.pop(0)

    def update(self, idx, z, sigma, lag=0):
        """Scalar measurement of state `idx` (0 position, 1 velocity) with std `sigma`, `lag` ticks old. -> the innovation."""
        ref = self.hist[-1 - lag][idx] if lag < len(self.hist) else self.x[idx]
        H = np.zeros(3)
        H[idx] = 1.0
        S = float(H @ self.P @ H) + sigma**2
        K = self.P @ H / S
        self.x = self.x + K * (z - ref)
        self.P = (np.eye(3) - np.outer(K, H)) @ self.P
        return z - ref

    @property
    def pos(self):
        return float(self.x[0])

    @property
    def vel(self):
        return float(self.x[1])


def baro_height(p_pa, p_ref_pa):
    """Height above the pad (m) from static pressure: ISA troposphere inverted about the pad pressure."""
    k = 1.0 / 5.255876
    return 288.15 / 0.0065 * ((p_ref_pa / 101325.0) ** k - (p_pa / 101325.0) ** k)


def lidar_height(rng, R):
    """Height of the centre of mass above flat ground (m) from a downward lidar range `rng` (m) and the attitude R (body -> world): the beam leaves the sensor at LIDAR_REL
    (body) along -z (body); the ground point is on the world plane z = 0 below the centre of mass at z_cg."""
    return float(-(R[2, :] @ veh.LIDAR_REL) + rng * R[2, 2])


# -------------------------------------------------------------------------------------------------------------------- the estimator
class Estimator:
    """Pad calibration (`calibrate` for t_cal s, then `finish`), then `update` every control tick. The local frame: x east, y north, z up, origin on the pad (GNSS average and the
    standing height of the centre of mass), heading zero = the vehicle's x axis on the pad."""

    def __init__(self, dt, gnss_lag_s=0.1):
        self.dt = dt
        self.att = Attitude()
        self.kf = None
        self.lag = round(gnss_lag_s / dt)
        self.cal = dict(p=[], gnss=[], lidar=[], f=[])
        self.p_ref = None
        self.gnss_ref = None  # (lat, lon, alt) of the pad
        self.baro_bias = (
            0.0  # m, baro height minus the lidar's, low-passed below a few metres
        )
        self.omega = np.zeros(3)
        self.alpha = np.zeros(3)
        self.f_cg = np.array([0.0, 0.0, G])
        self.lidar_ok = False
        self.lidar_z = None
        self.gnss_age = 0.0
        self.innov = dict(lidar=0.0, baro=0.0, gnss_z=0.0)
        self.sig_gnss, self.sig_vel = 0.05, 0.06

    # ---- pad calibration
    def calibrate(self, gyro, accel, baro=None, gnss=None, lidar=None):
        self.att.calibrate(np.asarray(gyro, float), np.asarray(accel, float))
        self.cal["f"].append(float(np.linalg.norm(accel)))
        if baro is not None:
            self.cal["p"].append(baro)
        if gnss is not None:
            self.cal["gnss"].append(gnss[:3])
        if lidar:
            self.cal["lidar"].append(lidar)

    def ready(self):
        return len(self.cal["p"]) >= 5 and len(self.cal["gnss"]) >= 3

    def finish(self):
        c = self.cal
        self.att.finish_calibration()
        self.p_ref = float(np.mean(c["p"]))
        self.gnss_ref = np.mean(np.array(c["gnss"]), axis=0)
        self.kf = [
            AxisFilter(self.dt),
            AxisFilter(self.dt),
            AxisFilter(self.dt, p0=(0.3, 0.2, 0.3)),
        ]
        self.kf[2].x[2] = (
            float(np.mean(c["f"])) - G
        )  # accelerometer z bias from |f| at rest
        self.omega = np.zeros(3)

    # ---- flight
    def _local(self, fix):
        lat, lon, alt = fix[:3]
        y = math.radians(lat - self.gnss_ref[0]) * EARTH_R
        x = (
            math.radians(lon - self.gnss_ref[1])
            * EARTH_R
            * math.cos(math.radians(self.gnss_ref[0]))
        )
        return x, y, alt - self.gnss_ref[2]

    def update(self, gyro, accel, baro=None, gnss=None, lidar=None):
        """One control tick. gyro rad/s, accel m/s2 (body), baro Pa or None, gnss (lat, lon, alt, vn, ve, vd) or None, lidar range m (0 / None = no return)."""
        dt = self.dt
        gyro, accel = np.asarray(gyro, float), np.asarray(accel, float)
        w_prev = self.omega
        self.att.update(gyro, dt)
        w = gyro - self.att.bias
        self.omega = w
        a_new = (w - w_prev) / dt
        self.alpha += 0.12 * (
            a_new - self.alpha
        )  # about 30 ms of smoothing: the derivative of a noisy gyro
        r = veh.IMU_REL
        self.f_cg = accel - np.cross(self.alpha, r) - np.cross(w, np.cross(w, r))
        R = self.att.R
        aw = R @ self.f_cg
        for i, a in enumerate((aw[0], aw[1], aw[2] - G)):
            self.kf[i].predict(a)
        # barometer
        if baro is not None:
            hb = baro_height(baro, self.p_ref) - self.baro_bias
            self.innov["baro"] = self.kf[2].update(0, hb, 0.20)
        # lidar
        self.lidar_ok = False
        if lidar and veh.LIDAR_MIN < lidar < veh.LIDAR_MAX and R[2, 2] > 0.7:
            h = lidar_height(lidar, R)
            z_l = h - veh.H_STAND
            sig = 0.04 + 0.012 * lidar
            if abs(z_l - self.kf[2].pos) < 1.0 + 4 * sig:
                self.innov["lidar"] = self.kf[2].update(0, z_l, sig)
                self.lidar_ok = True
                self.lidar_z = z_l
                if (
                    lidar < 6.0 and baro is not None
                ):  # the barometer's offset against the lidar, slowly (tau ~ 2 s at the baro rate)
                    hb = baro_height(baro, self.p_ref)
                    self.baro_bias += 0.01 * ((hb - z_l) - self.baro_bias)
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

    # ---- state
    @property
    def pos(self):
        return np.array([self.kf[0].pos, self.kf[1].pos, self.kf[2].pos])

    @property
    def vel(self):
        return np.array([self.kf[0].vel, self.kf[1].vel, self.kf[2].vel])

    @property
    def thrust_accel(self):
        """Specific force along the body axis (m/s2): thrust / mass."""
        return float(self.f_cg[2])
