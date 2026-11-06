"""Render frames of a finished hopper run (GPU, offscreen Vulkan): python examples/hopper/render_frames.py <run dir> --out <dir> [options]

<run dir> is a `dvsim program` run dir (result.json, replay.dvsimog, trace.csv, model/ present). The run's replay log is re-applied to a fresh World (tools/screenshots.py `Replay`), so the
frames show exactly the simulated flight; the sim is advanced to each requested time and `Sim.render` draws it. The vehicle's pose is read back from the replayed sim.

    --times 1,12,30,46      sim seconds of the stills (default: 5 evenly spaced over the flight)
    --fps 30 --from 0 --to 20   instead: every 1/fps s over [from, to] (a frame sequence for a video; f00000.png ...)
    --cam close|side|wide|top|follow
                       close = 2 m abeam of the vehicle, side = 6 m abeam, follow = 4 m abeam and 1 m above, always on the vehicle,
                       wide = 24 m from the middle of the line between the two pads, 5 m up (the whole hop in view), top = 14 m above the vehicle
    --size 960x540
Writes <out>/f<NNNNN>.png and <out>/frames.json (sim time and tick of every frame, the replay's hash check)."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools"))
sys.path.insert(0, str(HERE.parents[1] / "python"))
import dvsim
from dvsim import _dvsim
from screenshots import Replay


def camera(kind, p, p_start, p_goal):
    if kind == "wide":
        mid = 0.5 * (p_start + p_goal)
        d = p_goal - p_start
        n = np.array([-d[1], d[0], 0.0])
        n = (
            n / np.linalg.norm(n)
            if np.linalg.norm(n) > 1e-6
            else np.array([0.0, -1.0, 0.0])
        )
        far = max(24.0, 1.6 * float(np.linalg.norm(d)))
        return dict(
            pos=mid + far * n + np.array([0, 0, 5.0]),
            lookat=mid + np.array([0, 0, 0.5 * max(p[2], 6.0)]),
            fovy=66.0,
        )
    if kind == "top":
        return dict(pos=p + np.array([0.0, 0.01, 14.0]), lookat=p, fovy=45.0)
    dist, up, fov = dict(
        close=(2.0, 0.2, 45.0), side=(6.0, 0.5, 40.0), follow=(4.0, 1.0, 45.0)
    )[kind]
    return dict(pos=p + np.array([0.0, -dist, up]), lookat=p, fovy=fov)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--times", default="")
    ap.add_argument("--fps", type=float, default=0)
    ap.add_argument("--from", dest="t0", type=float, default=0.0)
    ap.add_argument("--to", dest="t1", type=float, default=0.0)
    ap.add_argument(
        "--cam", default="follow", choices=["close", "side", "wide", "top", "follow"]
    )
    ap.add_argument("--size", default="960x540")
    a = ap.parse_args()
    w, h = (int(v) for v in a.size.split("x"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dvsim.render_style("outdoor,shadows")
    r = Replay(a.run)
    r.dt = (
        0.001  # the kernel step; the trace's last row is not necessarily the last tick
    )
    t_end = r.res["sim"]["tick"] * r.dt
    if a.fps:
        times = list(np.arange(a.t0, a.t1 or t_end, 1.0 / a.fps))
    elif a.times:
        times = [float(v) for v in a.times.split(",")]
    else:
        times = list(np.linspace(0.1 * t_end, 0.95 * t_end, 5))
    p_start = np.array(r.sim.query("/world/frame", "pos")[0])
    p_goal = p_start.copy()
    # the goal pad: the touchdown position of the run (trace x_m / y_m at the end)
    p_goal[:2] = p_start[:2] + np.array([r.col("x_m")[-1], r.col("y_m")[-1]])
    meta = []
    for k, t in enumerate(times):
        tick = r.to(t)
        p = np.array(r.sim.query("/world/frame", "pos")[0])
        img, _ = r.sim.render(width=w, height=h, **camera(a.cam, p, p_start, p_goal))
        (out / f"f{k:05d}.png").write_bytes(_dvsim.encode_png(np.ascontiguousarray(img)))
        meta.append(
            dict(frame=k, t=round(float(t), 3), tick=int(tick), pos=p.round(3).tolist())
        )
    hsh, ok = r.finish()
    (out / "frames.json").write_text(
        json.dumps(
            dict(
                cam=a.cam,
                size=[w, h],
                replay_hash=hsh,
                replay_matches_run=ok,
                frames=meta,
            ),
            indent=1,
        )
    )
    sys.stdout.write(
        f"{len(meta)} frames -> {out} (replay hash {hsh}, {'MATCH' if ok else 'DIFFERENT'})\n"
    )


if __name__ == "__main__":
    main()
