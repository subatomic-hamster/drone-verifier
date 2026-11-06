"""`dvsim render`: showcase PNGs (named camera views) and short animated PNGs (APNG) of a model or assembly.
Stdlib + numpy only: APNG is written by hand (zlib), supersampling is a box filter."""

import struct
import zlib
from pathlib import Path

import numpy as np

from . import Sim, _dvsim


def _chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data))
    )


def apng(frames, fps: int = 20) -> bytes:
    """Animated PNG (loops forever) from equal-size (h, w, 4) uint8 frames."""
    h, w = frames[0].shape[:2]
    out = [
        b"\x89PNG\r\n\x1a\n",
        _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)),
        _chunk(b"acTL", struct.pack(">II", len(frames), 0)),
    ]
    seq = 0
    for i, fr in enumerate(frames):
        raw = b"".join(b"\x00" + fr[y].tobytes() for y in range(h))  # filter 0 per row
        z = zlib.compress(raw, 6)
        out.append(
            _chunk(b"fcTL", struct.pack(">IIIIIHHBB", seq, w, h, 0, 0, 1, fps, 0, 0))
        )
        seq += 1
        if i == 0:
            out.append(_chunk(b"IDAT", z))
        else:
            out.append(_chunk(b"fdAT", struct.pack(">I", seq) + z))
            seq += 1
    out.append(_chunk(b"IEND", b""))
    return b"".join(out)


def shot(sim: Sim, width: int, height: int, aa: int, **cam) -> np.ndarray:
    """One frame, rendered at aa x resolution and box-filtered down."""
    rgba, _ = sim.render(width=width * aa, height=height * aa, **cam)
    if aa > 1:
        rgba = (
            rgba.reshape(height, aa, width, aa, 4)
            .astype(np.float32)
            .mean((1, 3))
            .round()
            .astype(np.uint8)
        )
    return rgba


def _joint(sim: Sim, name: str) -> str:
    prims = sim.call({"cmd_type": "Query", "cmd": {"path": ""}}).get("prims", [])
    hits = [p for p in prims if p.rsplit("/", 1)[-1] == name]
    if len(hits) != 1:
        raise ValueError(f"joint '{name}': {len(hits)} prims end in it")
    return hits[0]


def run(
    model,
    out,
    views=("iso", "front", "top"),
    size=(640, 480),
    aa=2,
    frames=0,
    sweeps=(),
    spins=(),
    orbit=0.0,
    fps=20,
    target=None,
):
    """Writes <out>/<stem>-<view>.png for each view and, if frames > 0, <out>/<stem>-anim.apng. -> list of paths.
    sweeps: ["JOINT:LO:HI", ...] radians, eased there-and-back over the loop; spins: ["JOINT:TURNS", ...] linear."""
    model, out = Path(model), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sim = Sim.load(model)
    w, h = size
    paths = []
    for v in views:
        p = out / f"{model.stem}-{v}.png"
        p.write_bytes(_dvsim.encode_png(shot(sim, w, h, aa, camera=v, target=target)))
        paths.append(p)
    if frames > 0:
        drive = [
            (_joint(sim, n), float(a), float(b))
            for n, a, b in (s.split(":") for s in sweeps)
        ]
        drive_spin = [
            (_joint(sim, n), float(t)) for n, t in (s.split(":") for s in spins)
        ]
        pos, tgt = sim.orbit_camera("iso", target=target)
        d = pos - tgt
        az0 = np.degrees(np.arctan2(d[1], d[0]))
        el = np.degrees(np.arcsin(d[2] / np.linalg.norm(d)))
        dist = float(np.linalg.norm(d))
        imgs = []
        for k in range(frames):
            u = k / frames
            for j, lo, hi in drive:
                sim.set(
                    j, "qpos", [lo + (hi - lo) * (0.5 - 0.5 * np.cos(2 * np.pi * u))]
                )
            for j, turns in drive_spin:
                sim.set(j, "qpos", [2 * np.pi * turns * u])
            imgs.append(
                shot(
                    sim,
                    w,
                    h,
                    aa,
                    camera="orbit",
                    az=az0 + orbit * u,
                    el=el,
                    dist=dist,
                    target=tgt,
                )
            )
        p = out / f"{model.stem}-anim.apng"
        p.write_bytes(apng(imgs, fps))
        paths.append(p)
    return paths
