"""Tiny numpy mesh kit for the cosmetic visual layer. Every builder returns a closed, outward-wound triangle `Mesh`
(MuJoCo computes mesh volume / inertia at compile time and wants that), deterministic for the same arguments.
Frames: SI metres, z up. Rings / profiles are counter-clockwise seen from outside."""

import math
import os
import struct

import numpy as np


class Mesh:
    def __init__(self, v=None, f=None):
        self.v = np.zeros((0, 3)) if v is None else np.asarray(v, float).reshape(-1, 3)
        self.f = (
            np.zeros((0, 3), np.int64)
            if f is None
            else np.asarray(f, np.int64).reshape(-1, 3)
        )

    def __add__(self, o: "Mesh") -> "Mesh":
        return Mesh(np.vstack([self.v, o.v]), np.vstack([self.f, o.f + len(self.v)]))

    def volume(self) -> float:
        a, b, c = (self.v[self.f[:, i]] for i in range(3))
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)

    def at(self, pos=(0, 0, 0), R=None) -> "Mesh":
        """Rotated by R (3x3) then translated by pos."""
        v = self.v if R is None else self.v @ np.asarray(R, float).T
        return Mesh(v + np.asarray(pos, float), self.f)

    def bounds(self):
        return self.v.min(0), self.v.max(0)

    def tri_count(self) -> int:
        return len(self.f)


def merge(meshes) -> Mesh:
    out = Mesh()
    for m in meshes:
        out = out + m
    return out


def rot(axis, angle) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def rot_to(axis) -> np.ndarray:
    """Rotation taking +z to `axis`."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    c = a[2]
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        return np.diag([1.0, -1.0, -1.0])
    return rot(np.cross([0, 0, 1.0], a), math.acos(c))


def _outward(m: Mesh) -> Mesh:
    return m if m.volume() >= 0 else Mesh(m.v, m.f[:, ::-1])


def revolve(profile, n=48) -> Mesh:
    """Solid of revolution about z. `profile` = closed (r, z) polygon, counter-clockwise with r to the right
    (region on the left of travel), r >= 0; vertices with r == 0 sit on the axis (collapsed ring)."""
    prof = np.asarray(profile, float)
    ph = np.arange(n) * (2 * math.pi / n)
    v, ring = [], []
    for r, z in prof:
        if abs(r) < 1e-12:
            ring.append(np.full(n, len(v)))
            v.append((0.0, 0.0, z))
        else:
            ring.append(len(v) + np.arange(n))
            v += [(r * math.cos(p), r * math.sin(p), z) for p in ph]
    f = []
    for i in range(len(prof)):
        a, b = ring[i], ring[(i + 1) % len(prof)]
        a1, b1 = np.roll(a, -1), np.roll(b, -1)
        f += list(zip(a, a1, b1)) + list(zip(a, b1, b))
    f = np.array([t for t in f if len(set(t)) == 3])
    return _outward(Mesh(v, f))


def cylinder(r, h, n=48, z0=0.0) -> Mesh:
    return revolve([(0, z0), (r, z0), (r, z0 + h), (0, z0 + h)], n)


def tube(ro, ri, h, n=48, z0=0.0) -> Mesh:
    return revolve([(ri, z0), (ro, z0), (ro, z0 + h), (ri, z0 + h)], n)


def cone(r0, r1, h, n=48, z0=0.0) -> Mesh:
    return revolve([(0, z0), (r0, z0), (r1, z0 + h), (0, z0 + h)], n)


def disc_profile(r, h, ch=0.0, z0=0.0):
    """(r, z) polygon of a solid disc with chamfer `ch` on both rim edges."""
    return [
        (0, z0),
        (r - ch, z0),
        (r, z0 + ch),
        (r, z0 + h - ch),
        (r - ch, z0 + h),
        (0, z0 + h),
    ]


def sphere(r, n=24, centre=(0, 0, 0)) -> Mesh:
    t = np.linspace(-math.pi / 2, math.pi / 2, n // 2 + 1)
    prof = [(r * math.cos(a), r * math.sin(a)) for a in t[::1]]
    prof[0], prof[-1] = (0, -r), (0, r)
    return revolve(prof, n).at(centre)


def torus(R, r, n=48, m=16) -> Mesh:
    t = np.linspace(0, 2 * math.pi, m, endpoint=False)
    return revolve([(R + r * math.cos(a), r * math.sin(a)) for a in t], n)


def rbox(size, rad=0.0, seg=4) -> Mesh:
    """Axis-aligned box centred on the origin, edges and corners rounded with radius `rad` (0 = sharp). size = full extents."""
    h = np.asarray(size, float) / 2
    rad = min(float(rad), float(h.min()) * 0.999)
    if rad <= 0:
        st = [np.array([-x, x]) for x in h]
    else:
        e = rad * np.tan(np.linspace(0, math.pi / 4, seg + 1))
        st = [np.concatenate([-(x - rad) - e[::-1], (x - rad) + e]) for x in h]
        st = [np.unique(np.round(s, 12)) for s in st]
    pts, idx = {}, []

    def vid(p):
        k = tuple(np.round(p, 12))
        if k not in pts:
            pts[k] = len(pts)
        return pts[k]

    f = []
    for ax in range(3):
        u, w = (ax + 1) % 3, (ax + 2) % 3
        for sgn in (-1, 1):
            g = [[None] * len(st[w]) for _ in st[u]]
            for i, a in enumerate(st[u]):
                for j, b in enumerate(st[w]):
                    p = np.zeros(3)
                    p[ax], p[u], p[w] = sgn * h[ax], a, b
                    g[i][j] = vid(p)
            for i in range(len(st[u]) - 1):
                for j in range(len(st[w]) - 1):
                    q = [g[i][j], g[i + 1][j], g[i + 1][j + 1], g[i][j + 1]]
                    q = q if sgn > 0 else q[::-1]
                    f += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
    v = np.array(sorted(pts, key=pts.get))
    if rad > 0:
        c = np.clip(v, -(h - rad), h - rad)
        d = v - c
        n_ = np.linalg.norm(d, axis=1, keepdims=True)
        v = c + np.where(n_ > 1e-15, d / np.maximum(n_, 1e-15) * rad, d)
    return _outward(Mesh(v, f))


def box(size) -> Mesh:
    return rbox(size, 0)


def _area2(p):
    x, y = p[:, 0], p[:, 1]
    return float((x * np.roll(y, -1) - np.roll(x, -1) * y).sum())


def cap(outer, holes=()):
    """Triangulate a 2D polygon with holes by vertical slabs. Every loop edge is split at the x of every vertex, each slab
    interval is a trapezoid zipped into triangles along its side chains (so cap edges match the loop edges exactly: the
    prism built on it is watertight). -> (points (N,2), triangles (M,3), loops: index lists, outer ccw, holes cw)."""
    loops = [np.asarray(outer, float)] + [np.asarray(h, float) for h in holes]
    loops = [L[::-1] if (_area2(L) < 0) == (i == 0) else L for i, L in enumerate(loops)]
    xs = np.unique(np.round(np.concatenate([L[:, 0] for L in loops]), 12))
    pts, key = [], {}

    def vid(x, y):
        k = (round(float(x), 12), round(float(y), 12))
        if k not in key:
            key[k] = len(pts)
            pts.append(k)
        return key[k]

    rings = []
    for L in loops:
        ring = []
        for i in range(len(L)):
            a, b = L[i], L[(i + 1) % len(L)]
            ring.append(vid(*a))
            if abs(a[0] - b[0]) > 1e-12:
                cuts = xs[
                    (xs > min(a[0], b[0]) + 1e-12) & (xs < max(a[0], b[0]) - 1e-12)
                ]
                for x in cuts[::-1] if a[0] > b[0] else cuts:
                    ring.append(
                        vid(x, a[1] + (x - a[0]) / (b[0] - a[0]) * (b[1] - a[1]))
                    )
        rings.append(ring)
    P = np.array(pts)
    slabs = {}
    for ring in rings:
        for u, v in zip(ring, ring[1:] + ring[:1]):
            if P[u, 0] != P[v, 0]:
                l, r = (u, v) if P[u, 0] < P[v, 0] else (v, u)
                slabs.setdefault(int(np.searchsorted(xs, P[l, 0])), []).append(
                    ((P[l, 1] + P[r, 1]) / 2, l, r)
                )
    line = {}
    for i, (x, y) in enumerate(pts):
        line.setdefault(x, []).append((y, i))

    def chain(x, y0, y1):
        return [i for y, i in sorted(line[x]) if y0 - 1e-12 <= y <= y1 + 1e-12]

    tris = []
    for edges in slabs.values():
        edges.sort()
        for (_, ll, lr), (_, ul, ur) in zip(edges[::2], edges[1::2]):
            Lc = chain(P[ll, 0], P[ll, 1], P[ul, 1])
            Rc = chain(P[lr, 0], P[lr, 1], P[ur, 1])
            i = j = 0
            m, n = len(Lc) - 1, len(Rc) - 1
            while i < m or j < n:
                if j == n or (i < m and i * n <= j * m):
                    t = (Lc[i], Rc[j], Lc[i + 1])
                    i += 1
                else:
                    t = (Lc[i], Rc[j], Rc[j + 1])
                    j += 1
                if len(set(t)) == 3:
                    tris.append(t)
    return P, np.array(tris, np.int64), rings


def extrude(outer, z0, z1, holes=()) -> Mesh:
    """Prism of a 2D polygon (+ holes) between z0 and z1."""
    P, T, rings = cap(outer, holes)
    n = len(P)
    v = np.vstack([np.c_[P, np.full(n, z0)], np.c_[P, np.full(n, z1)]])
    f = [(a, c, b) for a, b, c in T] + [(a + n, b + n, c + n) for a, b, c in T]
    for ring in rings:  # walls face away from the solid (outer ccw, holes cw)
        for u, w in zip(ring, ring[1:] + ring[:1]):
            f += [(u, w, w + n), (u, w + n, u + n)]
    return Mesh(v, f)


def circle(r, n=24, centre=(0.0, 0.0), start=0.0):
    t = start + np.arange(n) * (2 * math.pi / n)
    return np.c_[centre[0] + r * np.cos(t), centre[1] + r * np.sin(t)]


def rrect(w, h, r, n=6, centre=(0.0, 0.0)):
    """Rounded-rectangle outline, counter-clockwise (r = 0 gives a plain rectangle)."""
    r = min(r, w / 2, h / 2)
    if r <= 1e-9:
        pts = [(w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2), (-w / 2, -h / 2)]
    else:
        pts = []
        for cx, cy, a0 in (
            (w / 2 - r, h / 2 - r, 0),
            (-w / 2 + r, h / 2 - r, 90),
            (-w / 2 + r, -h / 2 + r, 180),
            (w / 2 - r, -h / 2 + r, 270),
        ):
            for k in range(n + 1):
                a = math.radians(a0 + 90 * k / n)
                pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.asarray(pts) + np.asarray(centre)


def loft(sections, caps=True) -> Mesh:
    """Closed surface through a list of equal-length 3D rings (each counter-clockwise seen along the loft direction,
    i.e. seen from the end with the last section)."""
    S = np.asarray(sections, float)
    k, m = S.shape[:2]
    f = []
    for i in range(k - 1):
        for j in range(m):
            a, b = i * m + j, i * m + (j + 1) % m
            f += [(a, b, b + m), (a, b + m, a + m)]
    v = S.reshape(-1, 3)
    if caps:
        c0, c1 = S[0].mean(0), S[-1].mean(0)
        v = np.vstack([v, c0, c1])
        for j in range(m):
            f.append(
                (k * m, (j + 1) % m, j)
            )  # start cap faces against the loft direction
            f.append(
                (k * m + 1, (k - 1) * m + j, (k - 1) * m + (j + 1) % m)
            )  # end cap faces along it
    return _outward(Mesh(v, f))


def write_stl(path, m: Mesh) -> None:
    a, b, c = (m.v[m.f[:, i]] for i in range(3))
    nrm = np.cross(b - a, c - a)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.where(ln > 0, nrm / np.maximum(ln, 1e-30), 0)
    rec = np.zeros(
        len(m.f), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]
    )
    rec["n"], rec["v"] = nrm, np.stack([a, b, c], 1)
    tmp = f"{path}.{os.getpid()}.tmp"  # atomic: parallel builders sharing a mesh cache never see a partial file
    with open(tmp, "wb") as fh:
        fh.write(b"dvsim visual".ljust(80, b"\0"))
        fh.write(struct.pack("<I", len(m.f)))
        fh.write(rec.tobytes())
    os.replace(tmp, path)


def check_closed(m: Mesh) -> str | None:
    """None when every undirected edge is shared by exactly two faces with opposite directions and the volume is
    positive (after welding coincident vertices); else a message."""
    key = np.round(m.v, 9)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    f = inv.reshape(-1)[m.f]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    d = {}
    for a, b in map(tuple, e):
        d[(a, b)] = d.get((a, b), 0) + 1
    bad = [
        k for k, c in d.items() if d.get((k[1], k[0]), 0) != c
    ]  # touching solids may share an edge: count, not 1
    if bad:
        return f"{len(bad)} edge(s) without a matching opposite edge (open or inconsistently wound)"
    if m.volume() <= 0:
        return "non-positive volume"
    return None
