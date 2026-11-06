"""Stills of a finished hopper-land run (GPU, offscreen Vulkan): python examples/hopper-land/render_land.py <run dir> --out <dir> [--times 2,14,30,44] [--cam follow|close|wide|top|side] [--size 960x540]

<run dir> is a `dvsim program` run dir (result.json, replay.dvsimog, model/). The replay log is re-applied to a fresh World (tools/screenshots.py `Replay`), the sim is advanced to each time and
`Sim.render` draws it; the scene's weather look (overcast, rain) follows the scene's own devices. The hopper is `/world/frame`, or `/world/hopper_frame` in a ship scene.
    wide = the whole hop in view: 26 m (more for a longer hop) from the middle of the line from the start to the touchdown point, 6 m up. Never committed; look at them."""

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
        far = max(26.0, 1.6 * float(np.linalg.norm(d)))
        return dict(
            pos=mid + far * n + np.array([0, 0, 6.0]),
            lookat=mid + np.array([0, 0, 0.5 * max(p[2] - mid[2], 6.0)]),
            fovy=66.0,
        )
    if kind == "top":
        return dict(pos=p + np.array([0.0, 0.01, 14.0]), lookat=p, fovy=45.0)
    dist, up, fov = dict(
        close=(2.2, 0.4, 45.0), side=(7.0, 1.0, 40.0), follow=(4.5, 1.5, 45.0)
    )[kind]
    return dict(pos=p + np.array([0.0, -dist, up]), lookat=p, fovy=fov)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--times", default="")
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
    r.dt = 0.001
    t_end = r.res["sim"]["tick"] * r.dt
    times = (
        [float(v) for v in a.times.split(",")]
        if a.times
        else list(np.linspace(0.1 * t_end, 0.95 * t_end, 5))
    )
    body = (
        "/world/hopper_frame"
        if "hopper_frame" in " ".join(r.sim.body_positions())
        else "/world/frame"
    )
    p_start = np.array(r.sim.query(body, "pos")[0])
    # the touchdown point: replay to the end once and read the pose back, then rewind by replaying again
    p_goal = p_start.copy()
    p_goal[:2] = p_start[:2] + np.array([r.col("x_m")[-1], r.col("y_m")[-1]])
    meta = []
    for k, t in enumerate(times):
        tick = r.to(t)
        p = np.array(r.sim.query(body, "pos")[0])
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
