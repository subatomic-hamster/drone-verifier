"""Relative navigation for landing on a moving vessel: moving-base RTK GNSS, the deck motion datalink, and deck-motion prediction.

    rg = RelGnss(seed=1)                              # u-blox ZED-F9P moving-base pair: base on the vessel, rover on the lander
    fix = rg.step(t, base_pos, rover_pos)             # every control step; -> RelFix (delayed by the receiver latency) or None
    link = DeckLink(seed=2)                           # the vessel's motion sensor (MRU) broadcast over the deck radio
    link.step(t, deck_state(sim, "osv_hull"))         # every control step; msg = link.latest(t) -> DeckMsg (aged) or None
    pred = DeckPredictor(); pred.add(msg); pred.predict(2.0); pred.next_window(...)   # short-horizon heave / tilt prediction

These are the python halves of the vessel's navigation aids (a datalink is a python channel in this repo, like examples/asv/ais.py): every random draw is seeded, so
a run replays; nothing reads the simulator beyond the truth positions the caller passes in. Frames: world ENU (x East, y North, z Up), SI units, quaternions (w, x, y, z).

RelGnss models what a ZED-F9P rover/base pair in MOVING-BASE mode gives (u-blox ZED-F9P-02B data sheet UBX-21023276, Tables 2, 3, 5, parts/electronics/sparkfun-gps-rtk-sma-16481.toml):
  - an RTK relative position with 0.01 m + 1 ppm CEP (static, 1 km baseline) at up to 8 Hz (GPS+GLO+GAL+BDS), convergence < 10 s; the datasheet gives no dynamic figure,
    so the FIX sigma below is 2x the static CEP-derived value (ESTIMATE) with a short-correlation error;
  - three states: FIX (cm), FLOAT (decimetres, ESTIMATED 0.2 m), SINGLE (no corrections: the two standalone receivers' difference, ESTIMATED ~1 m);
  - the corrections (RTCM from the base) arrive over a radio link that can drop (Gilbert-Elliott bursts, ESTIMATED rates): corrections older than `float_age` s drop the rover to FLOAT,
    older than `single_age` s to SINGLE; when they return it re-converges through FLOAT for U(0.3, 1) x `convergence_s` before FIX is reported again;
  - receiver latency: the relative position is delivered `latency_s` after the epoch it describes (ESTIMATE, one sample at 8 Hz = 0.125 s).
"""

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

FIX, FLOAT, SINGLE = "fix", "float", "single"


@dataclass
class RelFix:
    t_meas: float  # s, epoch the measurement describes
    t_avail: float  # s, when it reached the rover (t_meas + latency)
    rel: (
        np.ndarray
    )  # rover antenna - base antenna, world ENU (m), with the state's error
    vel: (
        np.ndarray
    )  # relative velocity (m/s), 0.05 m/s per axis (data sheet velocity accuracy)
    state: str  # FIX | FLOAT | SINGLE
    sigma_h: float  # m, 1-sigma per horizontal axis of the state
    sigma_v: float  # m, vertical
    age: float = 0.0  # s, set by `step`: now - t_meas


class RelGnss:
    """Moving-base RTK relative position. `step(t, base_pos, rover_pos)` is called every control step with the TRUE antenna phase-centre positions (world)."""

    def __init__(
        self,
        seed=1,
        rate_hz=8.0,
        latency_s=0.125,
        sigma_fix=(
            0.02,
            0.03,
        ),  # m (horizontal per axis, vertical): 2 x the data sheet's static 0.01 m CEP (sigma 0.0085 m) + a margin for motion; ESTIMATE
        sigma_float=(
            0.20,
            0.35,
        ),  # m: float ambiguity solutions are decimetre level; ESTIMATE
        sigma_single=(
            1.0,
            1.6,
        ),  # m: difference of two stand-alone fixes (common errors cancel partly); ESTIMATE
        vel_sigma=0.05,  # m/s per axis: ZED-F9P velocity accuracy (data sheet Table 1)
        corr_time_s=3.0,  # s: correlation of the position error (multipath on a moving base decorrelates quickly); ESTIMATE
        corr_frac=0.5,  # share of the variance that is the correlated part
        convergence_s=10.0,  # s: data sheet "Convergence time RTK < 10 s"
        float_age=2.0,  # s without corrections before FIX degrades to FLOAT (RTCM age limit; ESTIMATE)
        single_age=10.0,  # s without corrections before FLOAT degrades to SINGLE (ESTIMATE)
        rtcm_hz=2.0,  # corrections per second over the link; ESTIMATE
        p_loss_good=0.02,  # Gilbert-Elliott link: loss probability per message in the good state
        p_loss_bad=0.9,  # ... and in the bad state
        p_g2b=0.002,  # per message: good -> bad (fades, e.g. the lander behind the superstructure)
        p_b2g=0.05,  # per message: bad -> good (a fade lasts ~20 messages = 10 s)
        fixed_at_start=True,
        ppm=1.0,  # data sheet: 1 ppm of the baseline length
    ):
        self.rng = np.random.default_rng([int(seed), 7717])
        self.period = 1.0 / rate_hz
        self.latency = latency_s
        self.sig = {FIX: sigma_fix, FLOAT: sigma_float, SINGLE: sigma_single}
        self.vel_sigma = vel_sigma
        self.tau, self.frac = corr_time_s, corr_frac
        self.conv = convergence_s
        self.float_age, self.single_age = float_age, single_age
        self.rtcm_period = 1.0 / rtcm_hz
        self.pl = (p_loss_good, p_loss_bad)
        self.p_gb = (p_g2b, p_b2g)
        self.ppm = ppm * 1e-6
        self.bad = False  # link state
        self.last_rtcm = (
            0.0 if fixed_at_start else -1e9
        )  # time of the last correction that got through
        self.t_next_rtcm = 0.0
        self.state = FIX if fixed_at_start else SINGLE
        self.t_fix_ready = (
            0.0 if fixed_at_start else None
        )  # when the ambiguity is resolved after an outage
        self.t_next = 0.0
        self.c = np.zeros(3)  # correlated error state
        self.queue = deque()  # fixes in flight (latency)
        self.prev = None
        self.out = None
        self.n = {FIX: 0, FLOAT: 0, SINGLE: 0}

    def _link(self, t):
        while self.t_next_rtcm <= t:
            self.bad = (
                (self.rng.random() < self.p_gb[0])
                if not self.bad
                else not (self.rng.random() < self.p_gb[1])
            )
            if self.rng.random() >= self.pl[1 if self.bad else 0]:
                tr = self.t_next_rtcm
                if t - self.last_rtcm > self.float_age and self.t_fix_ready is None:
                    # corrections are back after an outage: re-converge through FLOAT
                    self.t_fix_ready = tr + self.conv * self.rng.uniform(0.3, 1.0)
                self.last_rtcm = tr
            self.t_next_rtcm += self.rtcm_period

    def _state(self, t):
        age = t - self.last_rtcm
        if age > self.single_age:
            self.t_fix_ready = None
            return SINGLE
        if age > self.float_age:
            self.t_fix_ready = None
            return FLOAT
        if (
            self.t_fix_ready is None
        ):  # corrections fresh but no ambiguity solution yet (cold start)
            self.t_fix_ready = t + self.conv * self.rng.uniform(0.3, 1.0)
        return FIX if t >= self.t_fix_ready else FLOAT

    def step(self, t, base_pos, rover_pos):
        """Advance to time t with the TRUE antenna positions; returns the newest RelFix that has arrived (None before the first), or the previous one if nothing new."""
        base_pos, rover_pos = np.asarray(base_pos, float), np.asarray(rover_pos, float)
        rel = rover_pos - base_pos
        self._link(t)
        while self.t_next <= t + 1e-12:
            st = self._state(self.t_next)
            sh, sv = self.sig[st]
            sh_ = math.hypot(sh, self.ppm * float(np.linalg.norm(rel)))
            s = np.array([sh_, sh_, sv])
            phi = math.exp(-self.period / self.tau)
            self.c = phi * self.c + math.sqrt(1 - phi * phi) * math.sqrt(
                self.frac
            ) * s * self.rng.standard_normal(3)
            err = self.c + math.sqrt(1 - self.frac) * s * self.rng.standard_normal(3)
            vel = (
                (rel - self.prev[1]) / (t - self.prev[0])
                if self.prev and t > self.prev[0]
                else np.zeros(3)
            )
            fix = RelFix(
                self.t_next,
                self.t_next + self.latency,
                rel + err,
                vel + self.vel_sigma * self.rng.standard_normal(3),
                st,
                sh_,
                sv,
            )
            self.n[st] += 1
            self.queue.append(fix)
            self.t_next += self.period
        self.prev = (t, rel)
        while self.queue and self.queue[0].t_avail <= t + 1e-12:
            self.out = self.queue.popleft()
        if self.out is not None:
            self.out.age = t - self.out.t_meas
        return self.out


# ---- deck datalink ---------------------------------------------------------------------------------------------------------------------------------


def _qmul(a, b):
    return np.array(
        [
            a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
        ]
    )


def quat_to_mat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def euler_zyx(q):
    """(roll, pitch, yaw) rad of a (w, x, y, z) body quaternion, aerospace ZYX (x forward, y left, z up)."""
    R = quat_to_mat(q)
    return (
        math.atan2(R[2, 1], R[2, 2]),
        -math.asin(max(-1.0, min(1.0, R[2, 0]))),
        math.atan2(R[1, 0], R[0, 0]),
    )


def mat_to_quat(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-6:
        return np.array(
            [
                w,
                (R[2, 1] - R[1, 2]) / (4 * w),
                (R[0, 2] - R[2, 0]) / (4 * w),
                (R[1, 0] - R[0, 1]) / (4 * w),
            ]
        )
    x = math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    return np.array(
        [0.0, x, (R[0, 1] + R[1, 0]) / (4 * x), (R[0, 2] + R[2, 0]) / (4 * x)]
    )


def deck_truth(t, pos, quat, prev=None):
    """dict(t, pos, quat, vel, omega) from a position / quaternion pair (`sim.query("/world/<body>", "pos" | "quat")`); the kernel exposes pos / quat of a body only, so the
    velocities are backward differences against the previous dict over the caller's step (a control step of 10-40 ms), zero without one."""
    pos, quat = np.asarray(pos, float), np.asarray(quat, float)
    vel, om = np.zeros(3), np.zeros(3)
    if prev is not None and t > prev["t"]:
        h = t - prev["t"]
        vel = (pos - prev["pos"]) / h
        dq = _qmul(
            np.array([prev["quat"][0], *-prev["quat"][1:]]), quat
        )  # prev^-1 * q = rotation in the BODY frame
        if dq[0] < 0:
            dq = -dq
        om = 2 * dq[1:] / h / max(dq[0], 1e-9) if dq[0] > 1e-9 else np.zeros(3)
    return dict(t=t, pos=pos, quat=quat / np.linalg.norm(quat), vel=vel, omega=om)


@dataclass
class DeckMsg:
    t_meas: float  # s, epoch of the measurement
    t_avail: float  # s, when it reached the receiver
    pos: (
        np.ndarray
    )  # deck reference point (the MRU / base antenna), world ENU (m): heave = pos[2]
    vel: np.ndarray  # m/s
    quat: np.ndarray  # body attitude (w, x, y, z) with the sensor's attitude error
    omega: np.ndarray  # rad/s, body frame
    roll: float = 0.0  # rad, pitch rad, yaw rad (of `quat`)
    pitch: float = 0.0
    yaw: float = 0.0
    seq: int = 0
    age: float = 0.0  # s, now - t_meas, set by `latest`


class DeckLink:
    """The vessel's motion reference unit broadcast over the deck radio: a message every 1/`rate_hz` s carrying the deck attitude, rates, heave and position, delayed by
    `latency_s` and lost with the burst model of a SiK-class link (ESTIMATES, no link budget). Sensor errors: attitude 0.5 deg rms static, 1.0 deg rms when the deck rotates faster than
    0.1 rad/s (VectorNav VN-100 data sheet, parts/electronics/vectornav-vn-100-rugged.toml; heave from the RTK base antenna: 0.03 m rms); rate noise 0.005 rad/s; velocity 0.05 m/s."""

    def __init__(
        self,
        seed=1,
        rate_hz=25.0,
        latency_s=0.08,  # s: serial + radio airtime at 57.6 kbaud for a ~50-byte frame (~10 ms) + a UART buffer; ESTIMATE
        loss=0.03,  # message loss probability in the good state (ESTIMATE; SiK radios report 0.1-5 % packet loss at short range)
        p_g2b=0.003,
        p_b2g=0.1,  # burst model: a fade lasts ~10 messages (0.4 s at 25 Hz)
        p_loss_bad=0.95,
        att_sigma_deg=(0.5, 1.0),
        heave_sigma=0.03,
        rate_sigma=0.005,
        vel_sigma=0.05,
    ):
        self.rng = np.random.default_rng([int(seed), 5151])
        self.period, self.latency = 1.0 / rate_hz, latency_s
        self.loss, self.pl_bad, self.p_gb = loss, p_loss_bad, (p_g2b, p_b2g)
        self.att = [math.radians(a) for a in att_sigma_deg]
        self.sh, self.sr, self.sv = heave_sigma, rate_sigma, vel_sigma
        self.t_next = 0.0
        self.bad = False
        self.seq = 0
        self.queue = deque()
        self.out = None
        self.sent = self.lost = 0

    def step(self, t, truth):
        """Feed the TRUE deck state (`deck_truth`) every control step; samples are taken at the link rate. Returns the newest delivered DeckMsg (aged) or None."""
        while self.t_next <= t + 1e-12:
            self.sent += 1
            self.bad = (
                (self.rng.random() < self.p_gb[0])
                if not self.bad
                else not (self.rng.random() < self.p_gb[1])
            )
            if self.rng.random() < (self.pl_bad if self.bad else self.loss):
                self.lost += 1
            else:
                sig = (
                    self.att[1]
                    if float(np.linalg.norm(truth["omega"])) > 0.1
                    else self.att[0]
                )
                dq = np.array(
                    [1.0, *(0.5 * sig * self.rng.standard_normal(3))]
                )  # small-angle rotation error, rms `sig` per axis
                q = _qmul(truth["quat"], dq / np.linalg.norm(dq))
                r, p, y = euler_zyx(q)
                pos = truth["pos"] + self.sh * self.rng.standard_normal(
                    3
                )  # RTK base antenna position: cm level in all axes
                self.seq += 1
                self.queue.append(
                    DeckMsg(
                        self.t_next,
                        self.t_next + self.latency,
                        pos,
                        truth["vel"] + self.sv * self.rng.standard_normal(3),
                        q,
                        truth["omega"] + self.sr * self.rng.standard_normal(3),
                        r,
                        p,
                        y,
                        self.seq,
                    )
                )
            self.t_next += self.period
        while self.queue and self.queue[0].t_avail <= t + 1e-12:
            self.out = self.queue.popleft()
        if self.out is not None:
            self.out.age = t - self.out.t_meas
        return self.out


def deck_point(msg, offset):
    """Position and velocity (world) of a point fixed in the vessel at body-frame `offset` from the reference point, from a DeckMsg: p = pos + R offset, v = vel + R (omega x offset)."""
    R = quat_to_mat(msg.quat)
    off = np.asarray(offset, float)
    return msg.pos + R @ off, msg.vel + R @ np.cross(msg.omega, off)


def deck_point_at(msg, offset, t):
    """`deck_point` extrapolated from the message epoch to time t with the message's own rates (constant velocity and angular rate; a 0.1-0.3 s hold bridges the link delay)."""
    h = max(0.0, t - msg.t_meas)
    w = np.asarray(msg.omega, float)
    n = float(np.linalg.norm(w))
    q = msg.quat
    if n > 1e-9 and h > 0:
        a = n * h / 2
        q = _qmul(q, np.array([math.cos(a), *(math.sin(a) * w / n)]))
    R = quat_to_mat(q)
    off = np.asarray(offset, float)
    return msg.pos + msg.vel * h + R @ off, msg.vel + R @ np.cross(w, off)


# ---- deck motion prediction --------------------------------------------------------------------------------------------------------------------------


class DeckPredictor:
    """Short-horizon prediction of the deck at a pad point and quiescent-window detection, from a stream of DeckMsg. A library for any lander (nothing here knows a vehicle).

    The pad point's height h(t), the deck's tilt (roll, pitch) and the pad's vertical speed are resampled to `dt` and modelled as autoregressive series (Yule-Walker, order `order`,
    fitted on the last `window` s): the standard way to predict a ship's heave a few seconds ahead (the motion is a narrow-band process, JONSWAP seas at Tp 4-10 s, so a 2-3 s
    prediction keeps most of the variance). `predict(h)` extends the series h seconds; `next_window(...)` returns the first time in the horizon where the predicted pad height rate
    stays below `vz_max` and the predicted tilt below `tilt_max` for `duration` s (a quiescent period: the landing window), or None.

    Add messages with `add(msg, offset)`; `offset` = the pad point in the body frame (m)."""

    def __init__(self, dt=0.1, window=60.0, order=24, ridge=1e-3, min_history=20.0):
        self.dt, self.window, self.order, self.ridge, self.min_history = (
            dt,
            window,
            order,
            ridge,
            min_history,
        )
        self.t = []
        self.series = {"h": [], "roll": [], "pitch": []}
        self.last_seq = -1

    def add(self, msg, offset=(0.0, 0.0, 0.0)):
        if msg is None or msg.seq == self.last_seq:
            return
        self.last_seq = msg.seq
        p, _ = deck_point(msg, offset)
        self.t.append(msg.t_meas)
        self.series["h"].append(p[2])
        self.series["roll"].append(msg.roll)
        self.series["pitch"].append(msg.pitch)
        while self.t and msg.t_meas - self.t[0] > self.window * 1.2:
            self.t.pop(0)
            for v in self.series.values():
                v.pop(0)

    def ready(self):
        return len(self.t) > 4 and self.t[-1] - self.t[0] >= self.min_history

    def _uniform(self, key):
        t = np.asarray(self.t)
        tt = np.arange(t[-1] - self.window, t[-1] + 1e-9, self.dt)
        tt = tt[tt >= t[0]]
        return tt, np.interp(tt, t, np.asarray(self.series[key]))

    def _ar_predict(self, x, n_ahead):
        mu = x.mean()
        y = x - mu
        p = min(self.order, len(y) // 3)
        r = np.array([np.dot(y[: len(y) - k], y[k:]) / len(y) for k in range(p + 1)])
        r[0] *= 1 + self.ridge
        T = np.array([[r[abs(i - j)] for j in range(p)] for i in range(p)])
        a = np.linalg.solve(T, r[1 : p + 1])
        hist = list(y[-p:])
        out = []
        for _ in range(n_ahead):
            v = float(np.dot(a, hist[::-1][:p]))
            out.append(v)
            hist.append(v)
        return mu + np.array(out)

    def predict(self, horizon, keys=("h", "roll", "pitch")):
        """-> (t_future, {key: predicted series}) for the next `horizon` s at `dt` spacing, or None before `min_history` s of data."""
        if not self.ready():
            return None
        n = int(round(horizon / self.dt))
        tt, _ = self._uniform("h")
        t_f = tt[-1] + self.dt * np.arange(1, n + 1)
        return t_f, {k: self._ar_predict(self._uniform(k)[1], n) for k in keys}

    def next_window(
        self,
        horizon=8.0,
        duration=2.0,
        vz_max=0.15,
        tilt_max=math.radians(3.0),
        t_now=None,
    ):
        """First start time within `horizon` s from now at which the predicted pad vertical speed is below vz_max (m/s) and roll / pitch magnitudes below tilt_max (rad)
        for `duration` s; None when there is none (or no prediction yet). A window starting now returns the current time."""
        pr = self.predict(horizon + duration)
        if pr is None:
            return None
        t_f, p = pr
        t0 = self.t[-1] if t_now is None else t_now
        vz = np.gradient(p["h"], self.dt)
        ok = (
            (np.abs(vz) < vz_max)
            & (np.abs(p["roll"]) < tilt_max)
            & (np.abs(p["pitch"]) < tilt_max)
        )
        m = max(1, int(round(duration / self.dt)))
        run = np.convolve(ok.astype(int), np.ones(m, int), "valid")
        for i in np.nonzero(run == m)[0]:
            if t_f[i] >= t0 - 1e-9 and t_f[i] - t0 <= horizon:
                return float(t_f[i])
        return None

    def rms(self, key="h", seconds=20.0):
        """RMS of the (detrended) recent series, the sea-state gauge a lander can use to decide whether to land at all."""
        if not self.ready():
            return None
        tt, x = self._uniform(key)
        k = max(2, int(seconds / self.dt))
        x = x[-k:]
        return float(np.std(x - x.mean()))
