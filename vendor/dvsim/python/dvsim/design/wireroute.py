"""3D routing of the wire tubes (visual only; the physics never reads it).

A wire is routed once as a bundle: its centreline goes from the connector mouth of one end to the connector mouth of the other
(`END_ZONE` out along the connector axis) through free space; the conductors run around that centreline as a ring that is
twisted to meet the pins at both ends. "Free" is a voxel field of the distance to every part geometry (everything in the
assembly's body geoms except the wires themselves): a voxel is usable when its centre is farther from every part than the
bundle radius + `CLEAR` + the half voxel diagonal, so the continuous path through usable voxels keeps `CLEAR` between the tube
surface and any part. Steps:

    1. distance field (analytic distance for boxes / spheres / cylinders / capsules, surface samples for STL meshes), capped
    2. A* over the 26-neighbourhood of free voxels, ties broken by insertion order (deterministic)
    3. string pulling: the longest straight runs that are still free
    4. fillets: every corner becomes an arc of the gauge's minimum bend radius (6 x the conductor OD, + the ring radius for the
       bundle centreline so the inner conductor still keeps the radius), shrunk while it would leave free space or the segments
    5. conductors: ring offsets in a parallel-transported frame, twisted linearly so that the ring meets the pins of the far end

Results are cached on a digest of (endpoints, gauge, geometry), so rebuilding an unchanged design costs a dictionary lookup.
`clearance()` is the exact distance from points to the geoms; tests/python/test_wire_routing.py checks every route with it.
"""

import hashlib
import heapq
import math
import os
import pickle
from pathlib import Path

import numpy as np

from .mass import _STL

END_ZONE = 0.012  # m: the tube leaves the connector along its axis for this long; clearance is only promised beyond it
CLEAR = (
    0.002  # m: free space kept between a tube surface and any part (beyond END_ZONE)
)
BEND_FACTOR = 6.0  # minimum bend radius of one conductor = BEND_FACTOR x its outer diameter (silicone hook-up wire rule of thumb)
STUB = 0.005  # m: each conductor first runs straight out of its own pin for this long
CAP = 0.014  # m: the distance field is exact up to this distance (anything farther is "far")
LEAD_STEPS = (
    36  # the lead-out along the connector axis is searched up to END_ZONE + 70 mm
)
MAX_VOXELS = 4_500_000
EXPAND_LIMIT = 800_000
# (clearance, grid margin in voxel pitches): the first rung that routes and verifies wins
LADDER = ((CLEAR, 0.87), (CLEAR, 0.5), (0.001, 0.5), (0.001, 0.25), (0.0005, 0.1))

_cache: dict = {}
_field_cache: dict = {}
_item_cache: dict = {}
_mesh_cache: dict = {}


def clear_caches():
    """Forget every route and field in memory (tests that time or compare fresh routing; the disk cache is only used when DVSIM_ROUTE_CACHE is set)."""
    _cache.clear()
    _field_cache.clear()
    _item_cache.clear()


def _disk_path(ikey):
    d = os.environ.get("DVSIM_ROUTE_CACHE")
    return (
        Path(d) / (hashlib.sha1(repr(ikey).encode()).hexdigest() + ".pkl")
        if d
        else None
    )


def _disk_get(ikey):
    """A route routed by an earlier process (a long recording warms it up headlessly): DVSIM_ROUTE_CACHE=<dir>."""
    p = _disk_path(ikey)
    try:
        return pickle.loads(p.read_bytes()) if p and p.exists() else None
    except Exception:
        return None


def _disk_put(ikey, r):
    p = _disk_path(ikey)
    if p:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(pickle.dumps(r))


# ---- geometry ------------------------------------------------------------------------------------------------------------------
def _tris(path: str) -> np.ndarray:
    t = _mesh_cache.get(path)
    if t is None:
        raw = Path(path).read_bytes()
        n = int.from_bytes(raw[80:84], "little")
        t = np.frombuffer(raw, dtype=_STL, count=n, offset=84)["v"].astype(np.float64)
        _mesh_cache[path] = t
    return t


def world_geoms(body_geoms: dict, T: dict) -> list:
    """The assembly's part geometry in the world frame: [{kind, size, T (4x4), tris (world triangles of a mesh), lo, hi}].
    The wires' own mass capsules and connector spheres (names starting 'wire:') are not obstacles."""
    out = []
    for inst in sorted(body_geoms):
        if inst not in T:
            continue
        for g in body_geoms[inst]:
            if g["name"].startswith("wire:") or "fromto" in g:
                continue
            Tg = np.eye(4)
            Tg[:3, :3], Tg[:3, 3] = g["R"], g["pos"]
            W = T[inst] @ Tg
            d = {
                "kind": g["kind"],
                "size": [float(x) for x in g["size"]] if g.get("size") else [],
                "T": W,
                "owner": inst,
                "name": g["name"],
            }
            if g["kind"] == "mesh":
                tris = _tris(g["path"]) @ W[:3, :3].T + W[:3, 3]
                d["tris"], d["lo"], d["hi"] = (
                    tris,
                    tris.reshape(-1, 3).min(0),
                    tris.reshape(-1, 3).max(0),
                )
            else:
                ext = _prim_extent(d["kind"], d["size"], W[:3, :3])
                d["lo"], d["hi"] = W[:3, 3] - ext, W[:3, 3] + ext
            out.append(d)
    return out


def geoms_from_world(world) -> list:
    """The same obstacle list from a parsed MJCF (drc._World): every geom that is not a wire's. Used to check a built model."""
    out = []
    for g in world.geoms:
        if str(g["name"]).startswith("wire:"):
            continue
        W = world.bodies[g["body"]]["T"] @ g["T"]
        d = {
            "kind": g["kind"],
            "size": [float(x) for x in g["size"]],
            "T": W,
            "owner": world.bodies[g["body"]]["name"],
            "name": g["name"],
        }
        if g["kind"] == "mesh":
            tris = _tris(g["mesh"]) @ W[:3, :3].T + W[:3, 3]
            d["tris"], d["lo"], d["hi"] = (
                tris,
                tris.reshape(-1, 3).min(0),
                tris.reshape(-1, 3).max(0),
            )
        else:
            ext = _prim_extent(d["kind"], d["size"], W[:3, :3])
            d["lo"], d["hi"] = W[:3, 3] - ext, W[:3, 3] + ext
        out.append(d)
    return out


def _prim_extent(kind, size, R):
    A = np.abs(R)
    if kind == "box":
        return A @ np.array(size)
    if kind == "sphere":
        return np.full(3, size[0])
    r, hl = size[0], size[1]
    if kind == "capsule":  # the end caps add r along every axis
        return A[:, 2] * hl + r
    return A[:, 2] * hl + np.sqrt(np.maximum(0.0, 1 - A[:, 2] ** 2)) * r  # cylinder


def _prim_sdf(g, P):
    T = g["T"]
    L = (P - T[:3, 3]) @ T[:3, :3]
    k, s = g["kind"], g["size"]
    if k == "box":
        q = np.abs(L) - np.array(s)
        return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)
    if k == "sphere":
        return np.linalg.norm(L, axis=1) - s[0]
    if k == "cylinder":
        d = np.stack([np.hypot(L[:, 0], L[:, 1]) - s[0], np.abs(L[:, 2]) - s[1]], 1)
        return np.linalg.norm(np.maximum(d, 0), axis=1) + np.minimum(d.max(1), 0)
    z = np.clip(L[:, 2], -s[1], s[1])  # capsule
    return (
        np.linalg.norm(L - np.stack([np.zeros_like(z), np.zeros_like(z), z], 1), axis=1)
        - s[0]
    )


def _seg_dist(P, A, B):
    AB = B - A
    t = np.clip(((P - A) * AB).sum(-1) / np.maximum((AB * AB).sum(-1), 1e-30), 0, 1)
    return np.linalg.norm(P - (A + t[..., None] * AB), axis=-1)


def _tri_dist(P, tris):
    """Exact distance from points (N,3) to the nearest of triangles (M,3,3)."""
    best = np.full(len(P), np.inf)
    step = max(1, 1_500_000 // max(1, len(P)))
    for i in range(0, len(tris), step):
        t = tris[i : i + step]
        A, B, C = t[:, 0][None], t[:, 1][None], t[:, 2][None]  # (1,M,3)
        p = P[:, None, :]
        n = np.cross(B - A, C - A)
        nn = np.linalg.norm(n, axis=-1, keepdims=True)
        n = n / np.maximum(nn, 1e-30)
        dp = ((p - A) * n).sum(-1)
        q = p - dp[..., None] * n
        inside = np.ones(dp.shape, bool)
        for U, V in ((A, B), (B, C), (C, A)):
            inside &= (np.cross(V - U, q - U) * n).sum(-1) >= -1e-12
        e = np.minimum(
            np.minimum(_seg_dist(p, A, B), _seg_dist(p, B, C)), _seg_dist(p, C, A)
        )
        d = np.where(inside & (nn[..., 0] > 1e-18), np.abs(dp), e)
        best = np.minimum(best, d.min(1))
    return best


def clearance(P, geoms, exclude=()) -> np.ndarray:
    """Distance from each point of P (N,3) to the nearest part geometry (surface distance, 0 inside a primitive). Exact."""
    P = np.asarray(P, float).reshape(-1, 3)
    best = np.full(len(P), np.inf)
    for g in geoms:
        if g["owner"] in exclude:
            continue
        near = np.all((P >= g["lo"] - 0.1) & (P <= g["hi"] + 0.1), axis=1)
        if not near.any():
            continue
        Q = P[near]
        d = (
            _tri_dist(Q, g["tris"])
            if g["kind"] == "mesh"
            else np.maximum(_prim_sdf(g, Q), 0.0)
        )
        best[near] = np.minimum(best[near], d)
    return best


# ---- the voxel field -----------------------------------------------------------------------------------------------------------
def _extent(geoms, pts):
    lo = np.min([g["lo"] for g in geoms] + [np.min(pts, 0)], axis=0) - 0.05
    hi = np.max([g["hi"] for g in geoms] + [np.max(pts, 0)], axis=0) + 0.05
    return lo, hi


class Field:
    def __init__(self, geoms, pts, pitch=None):
        lo, hi = _extent(geoms, pts)
        ext = hi - lo
        p = pitch or max(0.003, (np.prod(ext) / MAX_VOXELS) ** (1 / 3))
        self.p, self.lo = p, lo
        self.n = np.ceil(ext / p).astype(int) + 1
        self.d = np.full(self.n, CAP, dtype=np.float32)
        for g in geoms:
            if g["kind"] == "mesh":
                self._mesh(g)
            else:
                self._prim(g)

    def _range(self, lo, hi):
        i0 = np.maximum(np.floor((lo - CAP - self.lo) / self.p).astype(int), 0)
        i1 = np.minimum(np.ceil((hi + CAP - self.lo) / self.p).astype(int) + 1, self.n)
        return i0, i1

    def centers(self, i0, i1):
        ax = [self.lo[k] + self.p * np.arange(i0[k], i1[k]) for k in range(3)]
        X, Y, Z = np.meshgrid(*ax, indexing="ij")
        return np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)

    def _prim(self, g):
        i0, i1 = self._range(g["lo"], g["hi"])
        if np.any(i1 <= i0):
            return
        P = self.centers(i0, i1)
        d = np.maximum(_prim_sdf(g, P), 0.0).reshape(i1 - i0)
        sl = tuple(slice(a, b) for a, b in zip(i0, i1))
        self.d[sl] = np.minimum(self.d[sl], d.astype(np.float32))

    def _mesh(self, g):
        t = g["tris"]
        e1, e2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
        area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
        k = np.minimum(np.ceil(area / (0.4 * self.p) ** 2).astype(int) + 1, 400)
        pts = [t[:, 0]]
        # deterministic stratified samples: every triangle gets k barycentric grid points (corners included)
        idx = np.repeat(np.arange(len(t)), k)
        m = np.concatenate([np.arange(c) for c in k])
        u = (m + 0.5) / np.repeat(k, k)
        s = np.mod(m * 0.6180339887498949, 1.0)
        flip = u + s > 1
        u, s = np.where(flip, 1 - u, u), np.where(flip, 1 - s, s)
        pts.append(t[idx, 0] + u[:, None] * e1[idx] + s[:, None] * e2[idx])
        P = np.concatenate(pts)
        vox = np.unique(np.floor((P - self.lo) / self.p).astype(int), axis=0)
        vox = vox[np.all((vox >= 0) & (vox < self.n), axis=1)]
        r = int(math.ceil(CAP / self.p))
        offs = sorted(
            (
                (dx, dy, dz)
                for dx in range(-r, r + 1)
                for dy in range(-r, r + 1)
                for dz in range(-r, r + 1)
            ),
            key=lambda o: -(o[0] ** 2 + o[1] ** 2 + o[2] ** 2),
        )
        flat = self.d.reshape(-1)
        st = np.array([self.n[1] * self.n[2], self.n[2], 1])
        for o in offs:  # farthest first: nearer offsets overwrite with a smaller value
            val = max(
                0.0,
                math.sqrt(o[0] ** 2 + o[1] ** 2 + o[2] ** 2) * self.p - 0.87 * self.p,
            )  # a sample may sit anywhere in its voxel
            if val >= CAP:
                continue
            tgt = vox + np.array(o)
            ok = np.all((tgt >= 0) & (tgt < self.n), axis=1)
            fi = tgt[ok] @ st
            flat[fi] = np.minimum(flat[fi], val)

    def idx(self, p):
        return np.clip(
            np.rint((np.asarray(p) - self.lo) / self.p).astype(int), 0, self.n - 1
        )

    def at(self, p) -> float:
        """Trilinear distance at p (nearest voxel would be off by up to 0.87 pitch)."""
        f = (np.asarray(p, float) - self.lo) / self.p
        i0 = np.clip(np.floor(f).astype(int), 0, self.n - 2)
        t = np.clip(f - i0, 0.0, 1.0)
        c = self.d[i0[0] : i0[0] + 2, i0[1] : i0[1] + 2, i0[2] : i0[2] + 2]
        c = c[0] * (1 - t[0]) + c[1] * t[0]
        c = c[0] * (1 - t[1]) + c[1] * t[1]
        return float(c[0] * (1 - t[2]) + c[1] * t[2])

    def center(self, i):
        return self.lo + self.p * np.asarray(i)


def _astar(free, start, goal, p):
    nx, ny, nz = free.shape
    sx, sy = ny * nz, nz
    offs = [
        (a, b, c)
        for a in (-1, 0, 1)
        for b in (-1, 0, 1)
        for c in (-1, 0, 1)
        if (a, b, c) != (0, 0, 0)
    ]
    stp = [a * sx + b * sy + c for a, b, c in offs]
    cst = [math.sqrt(a * a + b * b + c * c) * p for a, b, c in offs]
    ff = free.reshape(-1)
    si, gi = (
        (start[0] * sx + start[1] * sy + start[2]),
        (goal[0] * sx + goal[1] * sy + goal[2]),
    )
    gx, gy, gz = goal

    def h(i):
        x, r = divmod(i, sx)
        y, z = divmod(r, sy)
        return math.sqrt((x - gx) ** 2 + (y - gy) ** 2 + (z - gz) ** 2) * p

    g, par, heap, cnt = {si: 0.0}, {}, [(h(si), 0, si)], 0
    closed = set()
    while heap and len(closed) < EXPAND_LIMIT:
        _, _, u = heapq.heappop(heap)
        if u in closed:
            continue
        if u == gi:
            path = [u]
            while path[-1] in par:
                path.append(par[path[-1]])
            path.reverse()
            return [(i // sx, (i % sx) // sy, i % sy) for i in path]
        closed.add(u)
        gu = g[u]
        for s, c in zip(stp, cst):
            v = u + s
            if v < 0 or v >= ff.size or not ff[v] or v in closed:
                continue
            nv = gu + c
            if nv < g.get(v, 1e30):
                g[v] = nv
                par[v] = u
                cnt += 1
                heapq.heappush(heap, (nv + h(v), cnt, v))
    return None


# ---- smoothing -----------------------------------------------------------------------------------------------------------------
def _visible(F, a, b, need):
    n = max(2, int(np.linalg.norm(b - a) / (0.5 * F.p)) + 1)
    for t in np.linspace(0, 1, n):
        if F.at(a + (b - a) * t) < need:
            return False
    return True


def _pull(F, pts, need):
    out, i = [pts[0]], 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not _visible(F, pts[i], pts[j], need):
            j -= 1
        out.append(pts[j])
        i = j
    return out


def _fillet(F, pts, R, need, per_arc=5):
    """Corners -> arcs of radius R (shrunk by half until the arc is free or R < R/4: then the corner stays sharp)."""
    out, bends = [pts[0]], []
    for i in range(1, len(pts) - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        u, w = a - b, c - b
        lu, lw = np.linalg.norm(u), np.linalg.norm(w)
        if lu < 1e-9 or lw < 1e-9:
            continue
        u, w = u / lu, w / lw
        phi = math.acos(float(np.clip(u @ w, -1, 1)))  # interior angle
        if phi > math.pi - 1e-3:
            continue  # straight on
        if phi < 1e-3:
            out.append(b)
            continue
        r = R
        placed = False
        while r >= R / 4:
            t = r / math.tan(phi / 2)
            if t <= 0.5 * min(lu, lw) + 1e-12:
                p1, p2 = b + u * t, b + w * t
                m = u + w
                m = m / np.linalg.norm(m)
                cen = b + m * (r / math.sin(phi / 2))
                v1, v2 = (p1 - cen) / r, (p2 - cen) / r
                om = math.acos(float(np.clip(v1 @ v2, -1, 1)))
                arc = [
                    cen
                    + r
                    * (math.sin((1 - s) * om) * v1 + math.sin(s * om) * v2)
                    / max(math.sin(om), 1e-9)
                    for s in np.linspace(0, 1, per_arc)
                ]
                if all(F.at(q) >= need for q in arc):
                    out.extend(arc)
                    bends.append(r)
                    placed = True
                    break
            r *= 0.5
        if not placed:
            out.append(b)
            bends.append(0.0)
    out.append(pts[-1])
    return np.array(out), bends


def _verify(F, C, need, A0, B0):
    """Dense walk of the centreline: every sample (outside the mouths, which already left the owning part) keeps `need`."""
    for a, b in zip(C[:-1], C[1:]):
        n = max(2, int(np.linalg.norm(b - a) / (0.25 * F.p)) + 1)
        for t in np.linspace(0, 1, n):
            q = a + (b - a) * t
            if min(np.linalg.norm(q - A0), np.linalg.norm(q - B0)) <= F.p:
                continue
            if F.at(q) < need:
                return False
    return True


def _lead(F, P, n, need):
    """Where the tube joins the router: on the connector axis, END_ZONE out, pushed further (2 mm steps, <= LEAD_STEPS) until the
    field there is clear (a connector on a plate or inside an arm tube leaves along its axis through its own part first)."""
    for k in range(LEAD_STEPS):
        q = P + n * (END_ZONE + 0.002 * k)
        if all(F.at(q + n * s * F.p) >= need for s in (0.0, 1.0, 2.0)):
            return q  # clear here and one to two voxels on, so the router can leave the joint
    return P + n * END_ZONE


def _frames(C, u0):
    """Parallel-transported (u, v) along the polyline C (n,3); tangent from neighbours."""
    n = len(C)
    T = np.zeros((n, 3))
    for i in range(n):
        d = C[min(i + 1, n - 1)] - C[max(i - 1, 0)]
        T[i] = d / max(np.linalg.norm(d), 1e-12)
    U, V = np.zeros((n, 3)), np.zeros((n, 3))
    u = u0 - T[0] * (u0 @ T[0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(T[0], [0.0, 0.0, 1.0] if abs(T[0][2]) < 0.9 else [1.0, 0.0, 0.0])
    u = u / np.linalg.norm(u)
    for i in range(n):
        if i:
            ax = np.cross(T[i - 1], T[i])
            s, c = np.linalg.norm(ax), float(T[i - 1] @ T[i])
            if s > 1e-9:
                ax = ax / s
                u = u * c + np.cross(ax, u) * s + ax * (ax @ u) * (1 - c)  # Rodrigues
            u = u - T[i] * (u @ T[i])
            u = u / np.linalg.norm(u)
        U[i], V[i] = u, np.cross(T[i], u)
    return U, V


# ---- one wire ------------------------------------------------------------------------------------------------------------------
def _digest(geoms, items) -> str:
    h = hashlib.sha1()
    for g in geoms:
        h.update(
            repr(
                (
                    g["kind"],
                    [round(x, 5) for x in g["size"]],
                    np.round(g["T"], 5).tolist(),
                    len(g.get("tris", [])),
                )
            ).encode()
        )
    for it in items:
        h.update(
            repr(
                (
                    it["key"],
                    np.round(it["a"], 6).tolist(),
                    np.round(it["b"], 6).tolist(),
                    np.round(it["na"], 4).tolist(),
                    np.round(it["nb"], 4).tolist(),
                    round(it["d"], 6),
                )
            ).encode()
        )
    return h.hexdigest()


def route_wires(geoms: list, items: list) -> dict:
    """items: [{key, a (n,3), b (n,3) world pin positions of the n conductors, na, nb unit outward axes, ua, ub row directions, d
    conductor outer diameter}] -> {key: {"paths": [(m,3) per conductor], "centre": (k,3), "min_bend": m, "ok": bool, "bend_min": r}}."""
    sig = _digest(geoms, items)
    if sig in _cache:
        return _cache[sig]
    pts = (
        np.concatenate([np.concatenate([it["a"], it["b"]]) for it in items])
        if items
        else np.zeros((1, 3))
    )
    # A wire is routed against the field alone, so a design that gains a wire only routes that wire: the field is kept per
    # (geometry, grid extent) and every route per (field, wire). The extent is that of all parts and pins, which a new wire
    # between existing parts leaves alone.
    fkey = None
    if geoms:
        lo, hi = _extent(geoms, pts)
        fkey = (_digest(geoms, []), lo.round(6).tobytes(), hi.round(6).tobytes())
    res = {}
    for it in items:
        ikey = (fkey, _digest([], [it]))
        r = _item_cache.get(ikey) or _disk_get(ikey)
        if r is None:  # the field (seconds) is only built when a wire has to be routed
            F = _field_cache.get(fkey) if geoms else None
            if geoms and F is None:
                F = Field(geoms, pts)
                if len(_field_cache) > 1:
                    _field_cache.pop(next(iter(_field_cache)))
                _field_cache[fkey] = F
            r = _route(F, it)
            _disk_put(ikey, r)
        if len(_item_cache) > 256:
            _item_cache.pop(next(iter(_item_cache)))
        _item_cache[ikey] = res[it["key"]] = r
    if len(_cache) > 32:
        _cache.pop(next(iter(_cache)))
    _cache[sig] = res
    return res


def ring_radius(n: int, d: float) -> float:
    return d / (2 * math.sin(math.pi / n)) if n > 1 else 0.0


def _route(F, it):
    a, b = np.asarray(it["a"], float), np.asarray(it["b"], float)
    n = len(a)
    d = it["d"]
    ring = ring_radius(n, d)
    rb = ring + d / 2
    na, nb = np.asarray(it["na"], float), np.asarray(it["nb"], float)
    Pa, Pb = a.mean(0), b.mean(0)
    A0, B0 = Pa + na * END_ZONE, Pb + nb * END_ZONE
    r_min = BEND_FACTOR * d + ring
    need = rb + CLEAR
    cen, ok, bends, why = None, False, [], "no geometry"
    if F is not None:
        # ladder: full margin first, then tighter grid margins / clearance until a path verifies against the field
        for clear, half in LADDER:
            need = rb + clear
            A0, B0 = _lead(F, Pa, na, need), _lead(F, Pb, nb, need)
            free = F.d >= need + half * F.p
            # carve the mouths: the corridor from the connector out to END_ZONE (+ one voxel) may touch the owning part
            for P0, P1 in ((Pa, A0 + na * F.p), (Pb, B0 + nb * F.p)):
                lo, hi = np.minimum(P0, P1) - rb - F.p, np.maximum(P0, P1) + rb + F.p
                i0, i1 = np.maximum(F.idx(lo), 0), np.minimum(F.idx(hi) + 1, F.n)
                X = F.centers(i0, i1)
                dd = _seg_dist(X, P0, P1).reshape(i1 - i0)
                sl = tuple(slice(p, q) for p, q in zip(i0, i1))
                free[sl] |= dd <= rb + F.p
            free[0], free[-1] = False, False
            free[:, 0], free[:, -1] = False, False
            free[:, :, 0], free[:, :, -1] = False, False
            s, g = F.idx(A0), F.idx(B0)
            why = (
                "start mouth blocked"
                if not free[tuple(s)]
                else "end mouth blocked"
                if not free[tuple(g)]
                else "no free path"
            )
            if not (free[tuple(s)] and free[tuple(g)]):
                continue
            path = _astar(free, tuple(s), tuple(g), F.p)
            if not path:
                continue
            vox = [F.center(i) for i in path]
            vox[0], vox[-1] = A0, B0
            relaxed = need * 0.999
            C = _pull(F, vox, relaxed)
            C, bends = _fillet(F, C, r_min, relaxed)
            if _verify(F, C, relaxed, A0, B0):
                cen, ok, why = C, True, ""
                break
            why = "path failed verification"
    if (
        cen is None
    ):  # no clear path (or no geometry): the straight chord with the mouths, as before
        cen = (
            np.array([A0, B0])
            if np.linalg.norm(B0 - A0) > 1e-9
            else np.array([A0, A0 + 1e-6])
        )
    paths = _conductors(cen, a, b, na, nb, it, ring)
    return {
        "paths": paths,
        "centre": cen,
        "ok": ok,
        "why": why,
        "bends": bends,
        "r_min": r_min,
    }


def _conductors(C, a, b, na, nb, it, ring):
    n = len(a)
    ua = np.asarray(it.get("ua", [0.0, 0.0, 1.0]), float)
    U, V = _frames(C, ua)
    # slots: pins sorted along the row direction get angles pi, pi + 2pi/n, ... ; the far end decides a linear twist
    ra = (a - a.mean(0)) @ ua
    order = np.argsort(ra, kind="stable")
    theta = np.zeros(n)
    theta[order] = math.pi + 2 * math.pi * np.arange(n) / n
    twist = 0.0
    if n > 1:
        q = np.stack([(b - b.mean(0)) @ U[-1], (b - b.mean(0)) @ V[-1]], 1)
        best = None
        for k in range(72):
            tw = 2 * math.pi * k / 72 - math.pi
            pos = ring * np.stack([np.cos(theta + tw), np.sin(theta + tw)], 1)
            cost = float(((pos - q) ** 2).sum())
            if best is None or cost < best[0] - 1e-15:
                best = (cost, tw)
        twist = best[1]
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    s = s / s[-1] if s[-1] > 1e-12 else np.zeros_like(s)
    stub = min(STUB, 0.4 * END_ZONE)
    out = []
    for c in range(n):
        ang = theta[c] + twist * s
        off = ring * (np.cos(ang)[:, None] * U + np.sin(ang)[:, None] * V)
        mid = C + off
        path = np.vstack([a[c], a[c] + na * stub, mid, b[c] + nb * stub, b[c]])
        out.append(path)
    return out
