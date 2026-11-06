"""The hopper on the support vessel's helipad: examples/hopper-land wired to dvsim.cuas.ship (the OSV scene) and dvsim.cuas.relnav (moving-baseline RTK `RelGnss`, deck datalink `DeckLink`,
quiescent-window `DeckPredictor`).

    PrefixSim(sim)     the hopper's code asks for /devices/imu and /world/frame; in a ship scene they are /devices/hopper_imu and /world/hopper_frame (compose prefixes the vehicle)
    ShipDeck           TRUTH of the helipad (hull pose -> pad centre / normal / vertical speed), the same interface as landdeck.ScriptedDeck; also runs the vessel's autopilot
    ShipLink           what the flight computer receives: `sample(t, fl)` -> landdeck.DeckState (or None) built from RelGnss + DeckLink + DeckPredictor, never from truth

DeckState construction (the flight frame is the hopper's own GNSS-inertial frame, z = 0 when standing on the take-off pad):
    pad - rover (world, m) = (pad - base)(attitude and rates from the datalink message, at the RelFix epoch) - RelFix.rel      [rel = rover antenna - base antenna]
    DeckState.pos          = estimated position of the hopper at that epoch + (pad - rover) (+ the standing height for z, which is feet level)
so the hopper's own absolute error drops out of the landing point: it lands where the deck is RELATIVE to it, at the accuracy of the RTK fix (cm), the datalink's attitude (0.5 deg x 3 m lever)
and the extrapolation over the message latencies. ESTIMATES: the rover antenna sits at the hopper's frame origin (a few cm from the true GNSS antenna), the vessel's RTK base mast is the
`rtk_base` point of dvsim.cuas.ship.POINTS.
"""

import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)  # examples/ : the package `ship` (ShipPilot) and `asv`
import vehicle as veh
from landdeck import DeckState
from dvsim.cuas import relnav
from dvsim.cuas.ship import PAD_CENTRE, POINTS, body_to_world, pad_pose

PREFIX = "hopper"
SHARED = ("air", "wind", "rain")  # devices of the scene, not of the vehicle


class PrefixSim:
    """Proxy of a dvsim.Sim for code written for the stand-alone hopper scene."""

    def __init__(self, sim, ns=PREFIX):
        self._sim, self._ns = sim, ns

    def _path(self, path):
        if path == "/world/frame":
            return f"/world/{self._ns}_frame"
        if path.startswith("/devices/"):
            n = path[len("/devices/") :]
            return path if n in SHARED else f"/devices/{self._ns}_{n}"
        return path

    def query(self, path, attr):
        return self._sim.query(self._path(path), attr)

    def set(self, path, attr, value):
        return self._sim.set(self._path(path), attr, value)

    def __getattr__(self, name):
        return getattr(self._sim, name)


class _NoStep:
    """The vessel autopilot steps the sim itself (ShipPilot.advance); the harness owns the stepping, so the pilot gets a sim whose step() does nothing."""

    def __init__(self, sim):
        self._sim = sim

    def step(self, n=1):
        pass

    def __getattr__(self, name):
        return getattr(self._sim, name)


class ShipDeck:
    """Truth of the helipad. Interface of landdeck.ScriptedDeck: drive / surface / center / state / _pose."""

    def __init__(self, sim, info, p, h):
        from ship.pilot import ShipPilot

        self.sim, self.info, self.p = sim, info, p
        self.pilot = ShipPilot(sim)
        self.pilot.set_heading(0.0, 0.0)
        self._started = False
        self._t_pilot = -1.0
        self._z = None  # (t, z, vz) of the pad centre for the finite-difference vertical speed
        self.cs = {}

    def drive(self, t):
        """Called every control tick: the vessel's autopilot every 40 ms; it gets underway at `ship_t_start_s`."""
        if t >= self.p["ship_t_start_s"] and not self._started:
            self.pilot.set_heading(0.0, float(self.p["ship_speed_ms"]))
            self._started = True
        if t - self._t_pilot >= 0.04 - 1e-9:  # the pilot's control period
            self._t_pilot = t
            self.pilot.advance(_NoStep(self.sim))

    def hull(self):
        return (
            np.asarray(self.sim.query("/world/osv_hull", "pos")[0], float),
            np.asarray(self.sim.query("/world/osv_hull", "quat")[0], float),
        )

    def _pose(self):
        hp, hq = self.hull()
        return pad_pose(hp, hq)

    def surface(self, x, y, t):
        c, n = self._pose()
        z = c[2] - (n[0] * (x - c[0]) + n[1] * (y - c[1])) / n[2]
        if self._z is None or t - self._z[0] > 0.02:
            vz = 0.0 if self._z is None else (c[2] - self._z[1]) / (t - self._z[0])
            self._z = (t, float(c[2]), vz)
        return float(z), self._z[2]

    def center(self, t):
        c, _ = self._pose()
        return float(c[0]), float(c[1])

    def state(self, t):
        c, n = self._pose()
        return c, np.zeros(3), n


class ShipLink:
    """The flight computer's view of the deck (see the module doc). `sample(t, fl)` every control tick."""

    def __init__(self, deck, p, seed, takeoff):
        self.deck, self.p = deck, p
        self.rg = relnav.RelGnss(seed=int(seed))
        self.dl = relnav.DeckLink(
            seed=int(seed), latency_s=float(p.get("ship_link_latency_s", 0.08))
        )
        self.pred = relnav.DeckPredictor()
        self.prev = None
        self.hist = deque()  # (t, estimated position of the hopper in the flight frame)
        self.last_meas = -1.0
        self.msg = None
        self.rover_pos = lambda: np.zeros(
            3
        )  # set by the harness: true world position of the hopper frame
        self.n_states = 0
        self.off = None  # low-passed (hopper's estimated deck height - the datalink's pad height), m

    def _est_at(self, t):
        h = self.hist
        if not h:
            return None
        if t <= h[0][0]:
            return h[0][1]
        for (t0, p0), (t1, p1) in zip(h, list(h)[1:]):
            if t0 <= t <= t1:
                return p0 + (p1 - p0) * (t - t0) / max(t1 - t0, 1e-9)
        return h[-1][1]

    def sample(self, t, fl):
        hp, hq = self.deck.hull()
        truth = relnav.deck_truth(t, hp, hq, self.prev)
        self.prev = truth
        msg = self.dl.step(t, truth)
        if msg is not None:
            self.pred.add(msg, PAD_CENTRE)
            self.msg = msg
        fix = self.rg.step(
            t, body_to_world(hp, hq, POINTS["rtk_base"]), self.rover_pos()
        )
        if (
            fl.state in ("PAD",)
        ):  # moving-base alignment: the deck's attitude and rates are the reference of the hopper's gyro / accelerometer calibration
            if self.msg is None:
                return None
            R = relnav.quat_to_mat(self.msg.quat)
            return DeckState(
                t,
                np.zeros(3),
                np.zeros(3),
                omega_b=np.asarray(self.msg.omega, float),
                up_b=R[2, :] / np.linalg.norm(R[2, :]),
            )
        self.hist.append((t, fl.est.pos.copy()))
        while self.hist and t - self.hist[0][0] > 3.0:
            self.hist.popleft()
        if fix is None or self.msg is None or fix.t_meas <= self.last_meas:
            return None
        self.last_meas = fix.t_meas
        te = fix.t_meas
        pad, vel = relnav.deck_point_at(self.msg, PAD_CENTRE, te)
        base, _ = relnav.deck_point_at(self.msg, POINTS["rtk_base"], te)
        rel_pad = (pad - base) - fix.rel  # pad - rover, world
        e = self._est_at(te)
        pos = e + rel_pad + np.array([0.0, 0.0, veh.H_STAND])
        up = relnav.quat_to_mat(self.msg.quat)[:, 2]
        window = None
        ready = self.pred.ready()
        if ready:
            window = self.pred.next_window(
                horizon=8.0,
                duration=float(self.p["window_s"]),
                vz_max=float(self.p["window_vz_ms"]),
                tilt_max=math.radians(float(self.p["window_tilt_deg"])),
                t_now=t,
            )
        zs = None
        if ready:  # the pad height's forecast (AR model): the flight computer reads the deck's height and rate now, and ahead, from it instead of the rate of the older datalink sample held constant
            pr = self.pred.predict(3.0, ("h",))
            if pr is not None:
                # the series is the datalink's pad height (history + AR forecast) in the hopper's frame: the offset between the two (the hopper's height error, a few cm of RTK / estimator noise) is
                # low-passed (tau ~1.5 s), and the rate is a 0.5 s moving average of the series' own slope -- splicing the raw RTK height onto the forecast made the rate jump by 0.3 m/s per fix
                raw = pos[2] - pad[2]
                self.off = (
                    raw if self.off is None else self.off + 0.08 * (raw - self.off)
                )
                t_h, h_h = self.pred._uniform(
                    "h"
                )  # ponytail: the predictor's own resampled history (private), a public accessor if relnav grows one
                t_f = np.concatenate([t_h[-8:], pr[0]])
                z_f = np.concatenate([h_h[-8:], pr[1]["h"]]) + self.off
                v_f = np.convolve(
                    np.pad(np.gradient(z_f, t_f), 2, mode="edge"),
                    np.ones(5) / 5,
                    "valid",
                )
                zs = (t_f, z_f, v_f)
        self.n_states += 1
        return DeckState(
            te,
            pos,
            np.asarray(vel, float),
            up / np.linalg.norm(up),
            fix.sigma_h,
            0.05,
            ready,
            window,
            zs=zs,
        )
