"""Mass properties: binary-STL volume/area/vertices, printed-part mass, shape volumes."""

import math
from functools import cache
from pathlib import Path

import numpy as np

from .library import DesignError

_STL = np.dtype(
    [("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")]
)  # 50 bytes per triangle


@cache
def mesh_props(path: str):
    """(volume m^3, surface area m^2, vertices (n,3)) of a closed binary STL in metres. Volume = |sum of signed tetra volumes|."""
    raw = Path(path).read_bytes()
    n = int.from_bytes(raw[80:84], "little") if len(raw) >= 84 else -1
    if n < 1 or len(raw) != 84 + 50 * n:
        raise DesignError(f"mesh '{Path(path).name}': not a binary STL")
    tri = np.frombuffer(raw, dtype=_STL, count=n, offset=84)["v"].astype(np.float64)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    vol = abs(float(np.einsum("ij,ij->", a, np.cross(b, c)))) / 6.0
    area = 0.5 * float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())
    verts = np.unique(tri.reshape(-1, 3), axis=0)
    verts.setflags(write=False)
    return vol, area, verts


def printed_mass(
    rho: float, vol: float, area: float, wall_count: float, nozzle: float, infill: float
) -> float:
    """rho x V x f, f = (V_shell + infill (V - V_shell)) / V, V_shell = min(V, A t), t = wall_count x nozzle diameter."""
    v_shell = min(vol, area * wall_count * nozzle)
    return rho * (v_shell + infill * (vol - v_shell))


def shape_volume(kind: str, size: list, mesh_path: str | None = None) -> float:
    if kind == "box":
        return 8 * size[0] * size[1] * size[2]
    if kind == "sphere":
        return 4 / 3 * math.pi * size[0] ** 3
    if kind == "cylinder":
        return math.pi * size[0] ** 2 * 2 * size[1]
    if kind == "capsule":
        return math.pi * size[0] ** 2 * 2 * size[1] + 4 / 3 * math.pi * size[0] ** 3
    if kind == "mesh":
        return mesh_props(mesh_path)[0]
    raise DesignError(f"unknown shape '{kind}'")
