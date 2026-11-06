"""Numpy views of the sea and range-sensor devices (docs/models.md: sea, rangefinder)."""

import csv
import json
import os
from pathlib import Path

import numpy as np


def write_native(sim, dt: float, cols, rows, run_dir=None) -> None:
    """End of a `[interfaces.native]` program: trace.csv (cols, rows), replay.dvsimog and native.json into the run dir.
    The caller called sim.start_log() before the first command."""
    r = Path(run_dir or os.environ["DVSIM_RUN_DIR"])
    with open(r / "trace.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    (r / "replay.dvsimog").write_bytes(sim.log())
    (r / "native.json").write_text(
        json.dumps(
            dict(
                tick=sim.tick,
                dt=dt,
                state_hash=f"{sim.state_hash():016x}",
                diverged=bool(sim.diverged),
            )
        )
    )


def surface_elevation(sim, sea: str, x, y, t=None, level: float = 0.0) -> np.ndarray:
    """Water surface height (world z) at world points (x, y) at the sea's own clock `t` (default: now).

    Same sum as the device: level + wave_scale * sum a cos(k (x cos th + y sin th) - w t + phi). `level` is the sea's
    still-water level (a bind-time constant, not a device attribute)."""
    w = sim.query(sea, "waves")[0].reshape(-1, 5)  # a, k, w, theta, phi per component
    t = float(sim.query(sea, "time")[0][0]) if t is None else t
    scale = float(sim.query(sea, "wave_scale")[0][0])
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    a, k, om, th, ph = (w[:, i].reshape((-1,) + (1,) * x.ndim) for i in range(5))
    phase = k * (x * np.cos(th) + y * np.sin(th)) - om * t + ph
    return level + scale * (a * np.cos(phase)).sum(axis=0)


def depth_image(sim, sensor: str, n_h: int, n_v: int) -> np.ndarray:
    """Latest frame of a rangefinder as float32 (n_v, n_h) metres, NaN where there is no return (the device writes 0)."""
    r = sim.query(sensor, "range")[0].reshape(n_v, n_h).astype(np.float32)
    r[r == 0] = np.nan
    return r
