"""Minimum-jerk trajectories for the hopper: a quintic per axis between two (position, velocity, acceleration) states, and a `Reference` that chains them so that a new segment
always starts from the state the previous one has reached (no step in the setpoint when the state machine changes its mind). Pure numpy."""

import math

import numpy as np


class MinJerk:
    """p(t) = p0 + v0 t + a0 t^2 / 2 + c3 t^3 + c4 t^4 + c5 t^5 (vectors), reaching (p1, v1, a1) at t = T with the least integrated squared jerk."""

    def __init__(self, p0, v0, a0, p1, v1, a1, T):
        p0, v0, a0, p1, v1, a1 = (
            np.asarray(x, float) for x in (p0, v0, a0, p1, v1, a1)
        )
        self.T = max(float(T), 1e-3)
        self.p0, self.v0, self.a0 = p0, v0, a0
        T = self.T
        A = np.array(
            [
                [T**3, T**4, T**5],
                [3 * T**2, 4 * T**3, 5 * T**4],
                [6 * T, 12 * T**2, 20 * T**3],
            ]
        )
        rhs = np.stack(
            [p1 - p0 - v0 * T - 0.5 * a0 * T**2, v1 - v0 - a0 * T, a1 - a0]
        )  # 3 x n
        self.c = np.linalg.solve(A, rhs)  # 3 x n

    def at(self, t):
        """(position, velocity, acceleration, jerk) at time t after the start; held at the end state beyond T."""
        t = min(max(t, 0.0), self.T)
        c3, c4, c5 = self.c
        p = (
            self.p0
            + self.v0 * t
            + 0.5 * self.a0 * t**2
            + c3 * t**3
            + c4 * t**4
            + c5 * t**5
        )
        v = self.v0 + self.a0 * t + 3 * c3 * t**2 + 4 * c4 * t**3 + 5 * c5 * t**4
        a = self.a0 + 6 * c3 * t + 12 * c4 * t**2 + 20 * c5 * t**3
        j = 6 * c3 + 24 * c4 * t + 60 * c5 * t**2
        return p, v, a, j


def duration(distance, vmax, amax):
    """Shortest rest-to-rest minimum-jerk time over `distance` (m) within speed `vmax` and acceleration `amax`: the profile peaks at 1.875 d / T (speed) and 5.7735 d / T^2."""
    d = abs(float(distance))
    if d < 1e-6:
        return 0.5
    return max(1.875 * d / vmax, math.sqrt(5.7735 * d / amax), 0.5)


class Reference:
    """The setpoint source of the controller: `start(t, p, v, a)` at the beginning, then `go(t, p1, T, v1)` chains a new segment from wherever the reference is at time t."""

    def __init__(self):
        self.seg = None
        self.t0 = 0.0
        self.p = self.v = self.a = np.zeros(3)

    def start(self, t, p, v=(0, 0, 0), a=(0, 0, 0)):
        self.seg = None
        self.p, self.v, self.a = (
            np.asarray(p, float),
            np.asarray(v, float),
            np.asarray(a, float),
        )
        self.t0 = t

    def state(self, t):
        """(p, v, a) of the reference at time t."""
        if self.seg is None:
            return self.p, self.v, self.a
        p, v, a, _ = self.seg.at(t - self.t0)
        return p, v, a

    def go(self, t, p1, T, v1=(0, 0, 0), a1=(0, 0, 0)):
        p, v, a = self.state(t)
        self.seg = MinJerk(p, v, a, p1, v1, a1, T)
        self.t0 = t

    def done(self, t):
        return self.seg is None or (t - self.t0) >= self.seg.T

    def remaining(self, t):
        return 0.0 if self.seg is None else max(self.seg.T - (t - self.t0), 0.0)

    def update(self, t):
        self.p, self.v, self.a = self.state(t)
        return self.p, self.v, self.a
