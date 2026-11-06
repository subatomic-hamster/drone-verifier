"""Deck-relative landing interface of the hopper-land flight computer.

The flight computer lands on a surface that may move. Whatever tells it where that surface is -- RTK moving-baseline GNSS between a base on the deck and the rover on the vehicle, a deck datalink with the
vessel's attitude and motion, a downward lidar -- reaches it as one small record, `DeckState`, a few times per second. `Flight.step(..., deck=DeckState)` (landflight) then
  * follows the deck's horizontal position and velocity (HOVER, DESCENT, FINAL),
  * governs the final sink by the height above the deck surface, with the deck's vertical velocity as feed-forward (relative sink rate `v_sink_min` at touchdown),
  * declares touchdown when the estimated height above the (predicted) surface is below 2 cm.

DeckState is the plug point for the vessel work (the SHIP stage's `dvsim.cuas.relnav` output): anything that produces `DeckState` objects replaces `DeckLink` below. This module also holds a SCRIPTED
deck used to test the interface now: landscene.add_deck builds a platform on three slides and two hinges, `ScriptedDeck` drives it with heave, roll, pitch and a steady drift (sinusoids with seeded
phases, ESTIMATES of a 12 m vessel in sea state 3-4: heave +-0.3 m at 8 s, roll +-2 deg at 9 s, pitch +-1 deg at 7 s), `DeckLink` plays the datalink with rate, latency and noise.
"""

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DeckState:
    """The deck's landing point as the flight computer knows it, in its own local frame (x east, y north, z up; z = the height at which the vehicle's estimate reads when its feet stand on the surface, i.e.
    the surface elevation over the take-off pad)."""

    t: float  # time the state is valid for (flight computer clock, s)
    pos: np.ndarray  # surface point under the landing mark (m)
    vel: np.ndarray  # m/s
    up: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0])
    )  # surface normal (unit)
    sigma_pos: float = 0.03  # m, 1-sigma of each position component (for the log)
    sigma_vel: float = 0.05  # m/s
    ready: bool | None = (
        None  # None: no motion predictor; False: still collecting history; True: it names quiet windows (landship)
    )
    window: float | None = (
        None  # start (flight clock, s) of the next predicted quiet window, if any
    )
    omega_b: np.ndarray | None = (
        None  # the deck's angular rate in its body frame (rad/s), for the moving-base alignment on the pad
    )
    up_b: np.ndarray | None = (
        None  # the vertical in the deck's body frame (unit): where the accelerometers' mean points on a still deck
    )

    zs: tuple | None = (
        None  # (times, heights, rates) arrays of the surface point in this frame: a motion predictor's forecast (landship: relnav.DeckPredictor); None: constant velocity
    )

    def predict(self, t):
        """(position, velocity) at time t: constant-velocity extrapolation from the sample (the link's latency is short against the heave); the vertical part from the forecast `zs` where it covers t."""
        p, v = self.pos + self.vel * (t - self.t), self.vel
        if self.zs is not None and self.zs[0][0] <= t <= self.zs[0][-1]:
            p, v = p.copy(), v.copy()
            p[2], v[2] = (float(np.interp(t, self.zs[0], q)) for q in self.zs[1:])
        return p, v

    def predict_z(self, t):
        p, v = self.predict(t)
        return float(p[2]), float(v[2])

    def vz_max_ahead(self, t, horizon, lead=0.0):
        """The largest upward speed of the surface over [t + lead, t + horizon] (m/s) from the forecast; the current rate when there is none."""
        if self.zs is None or self.zs[0][-1] < t:
            return self.predict_z(t)[1]
        m = (self.zs[0] >= t + lead) & (self.zs[0] <= t + horizon)
        return float(np.max(self.zs[2][m])) if m.any() else self.predict_z(t)[1]


class ScriptedDeck:
    """Drives the deck platform of landscene.add_deck: sinusoidal heave / roll / pitch and a steady drift, phases seeded from the flight's seed. `drive(t)` writes the actuator targets; `surface`,
    `center` read the platform back from the sim (truth, never given to the flight computer)."""

    def __init__(self, sim, info, p, h):
        self.sim, self.info, self.p = sim, info, p
        r = np.random.default_rng([int(p["seed"]), 4242])
        self.ph = r.uniform(0, 2 * math.pi, 3)
        self.x0, self.y0, self.z0 = info.deck["x"], info.deck["y"], info.deck["z"]
        self.vx, self.vy = p["deck_vx_ms"], p["deck_vy_ms"]
        self.A = (
            p["deck_heave_m"],
            math.radians(p["deck_roll_deg"]),
            math.radians(p["deck_pitch_deg"]),
        )
        self.T = (p["deck_heave_s"], p["deck_roll_s"], p["deck_pitch_s"])

    def joints(self, t):
        """(commanded joint positions x y z roll pitch, their velocities) at time t."""
        w = [2 * math.pi / T for T in self.T]
        q = [self.vx * t, self.vy * t] + [
            A * math.sin(wk * t + ph) for A, wk, ph in zip(self.A, w, self.ph)
        ]
        v = [self.vx, self.vy] + [
            A * wk * math.cos(wk * t + ph) for A, wk, ph in zip(self.A, w, self.ph)
        ]
        # ramped in over the first 5 s (s = t / 5): the platform starts at rest, no step in the position actuators
        s, ds = min(t / 5.0, 1.0), (0.2 if t < 5.0 else 0.0)
        return [qi * s for qi in q], [vi * s + qi * ds for qi, vi in zip(q, v)]

    def drive(self, t):
        q, _ = self.joints(t)
        for nm, v in zip("xyzrp", q):
            self.sim.set(f"/actuators/deck_{nm}", "ctrl", [v])

    def _pose(self):
        px, py, pz = self.sim.query("/world/deck", "pos")[0]
        w, x, y, z = self.sim.query("/world/deck", "quat")[0]
        n = np.array(
            [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]
        )
        return np.array([px, py, pz]), n

    def surface(self, x, y, t):
        """(z of the deck surface under world point (x, y), its vertical velocity)."""
        p, n = self._pose()
        z = p[2] - (n[0] * (x - p[0]) + n[1] * (y - p[1])) / n[2]
        _, v = self.joints(t)
        return float(z), float(v[2])

    def center(self, t):
        p, _ = self._pose()
        return float(p[0]), float(p[1])

    def state(self, t):
        """Truth at time t: (position of the landing mark (3), velocity (3), up (3))."""
        p, n = self._pose()
        _, v = self.joints(t)
        return p, np.array(v[:3]), n


class DeckLink:
    """The deck datalink as the flight computer receives it: samples of the deck's true state at `deck_rate_hz`, with noise, delivered `deck_latency_s` late, in the flight computer's local frame
    (relative to the take-off pad: the RTK base / rover difference is what a moving-baseline receiver gives; here the take-off pad's world position stands for the survey).
    ESTIMATES: RTK moving-baseline accuracy 1 cm + 1 ppm static (u-blox ZED-F9P data sheet), degraded by antenna motion to `deck_sigma_pos_m`; velocity from Doppler 0.05 m/s."""

    def __init__(self, deck, p, rng):
        self.deck, self.p, self.rng = deck, p, rng
        self.period = 1.0 / p["deck_rate_hz"]
        self.next_t = 0.0
        self.pending = []  # (arrival time, DeckState)
        ax, ay, az = deck.info.takeoff
        self.origin = np.array([ax, ay, az])

    def sample(self, t):
        """-> the newest DeckState that has arrived by time t and was not handed out before, else None."""
        if t >= self.next_t:
            self.next_t += self.period
            pos, vel, up = self.deck.state(t)
            sp, sv = self.p["deck_sigma_pos_m"], 0.05
            st = DeckState(
                t,
                pos - self.origin + self.rng.normal(0, sp, 3),
                vel + self.rng.normal(0, sv, 3),
                up / np.linalg.norm(up),
                sp,
                sv,
            )
            self.pending.append((t + self.p["deck_latency_s"], st))
        if self.pending and self.pending[0][0] <= t:
            out = None
            while self.pending and self.pending[0][0] <= t:
                out = self.pending.pop(0)[1]
            return out
        return None
