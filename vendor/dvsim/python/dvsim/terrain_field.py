"""Small-scale terrain fields for legged robots (SP2): 100 x 100 m at 5 cm cells, tiled into patches that are each their own MuJoCo hfield geom
with the patch's material (friction / contact softness), plus box obstacles (boulders, ruin walls) and an exact ground-truth height function.

    f = field("rubble", size=100.0, res=0.05, seed=3)       # kinds: rubble, slopes, scree, ruins
    f.height(x, y)                                           # ground truth: highest surface under (x, y), hfield (MuJoCo's own triangulation) and boxes
    f.material_at(x, y)                                      # "rock" | "gravel" | "grass" | "wet_rock" | "mud" (a box top: the box's material)
    names = add_field(xml_text_or_mjcf_root, f)              # emits <hfield> assets + one hfield geom per patch + the boxes; returns the geom names

Everything is a pure function of (kind, size, res, seed, parameters): `f.digest()` is equal across processes. Patches share their border rows /
columns of the one global height array, so heights are continuous across patch edges by construction (no seams). docs/models.md "Small-scale terrain".
"""

import hashlib
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from dataclasses import field as _dc_field

import numpy as np

KINDS = ("rubble", "slopes", "scree", "ruins")


@dataclass(frozen=True)
class Material:
    """MuJoCo geom friction (sliding, torsional, rolling) and solref (time constant s, damping ratio) of a surface. All friction values are
    ESTIMATES for a rubber foot (the projects' quadruped), bracketed by the cited ranges; none is a measurement of this foot."""

    friction: tuple
    solref: tuple
    fidelity: str
    note: str


# solref time constants are >= 4 dt for dt = 2 ms (MuJoCo wants timeconst >= 2 dt); damping ratio 1 = critical (no bounce).
MATERIALS = {
    "rock": Material(
        (0.8, 0.005, 0.0001),
        (0.02, 1.0),
        "estimated",
        "dry rubber on rough rock; the University of Kentucky rubber-on-rock tribometer study (uknowledge.uky.edu/ktc_researchreports/1245) reports"
        " high dry friction on rough surfaces; 0.8 is a round value inside that regime",
    ),
    "gravel": Material(
        (0.55, 0.005, 0.0001),
        (0.025, 1.0),
        "estimated",
        "Rocscience RocFall friction table (rocscience.com/help/rocfall, dynamic friction): compacted gravel and cobbles 0.50-0.60; midpoint 0.55",
    ),
    "grass": Material(
        (0.6, 0.005, 0.0001),
        (0.03, 1.0),
        "estimated",
        "Rocscience RocFall friction table: earth with grass and some vegetation 0.55-0.65; midpoint 0.6; softer ground (longer solref time constant)",
    ),
    "wet_rock": Material(
        (0.35, 0.005, 0.0001),
        (0.02, 1.0),
        "estimated",
        "University of Kentucky study: wet rubber on coarsely abraded stone 0.6-0.7 but on polished limestone as low as 0.01; 0.35 is a weathered-wet"
        " estimate between the two",
    ),
    "mud": Material(
        (0.25, 0.005, 0.0001),
        (0.04, 1.0),
        "estimated",
        "no rubber-on-mud measurement found; a slick-surface estimate (sliding friction a fraction of wet rock), soft contact (0.04 s)",
    ),
}
RGBA = {
    "rock": "0.45 0.43 0.40 1",
    "gravel": "0.55 0.50 0.42 1",
    "grass": "0.30 0.42 0.22 1",
    "wet_rock": "0.28 0.28 0.31 1",
    "mud": "0.30 0.22 0.15 1",
}
# Cell diagonal of MuJoCo's hfield triangulation, in this module's (x right, y up) grid frame; set by tests against mj_ray.
DIAG = "main"


@dataclass
class Patch:
    x0: float
    y0: float
    x1: float
    y1: float
    material: str
    i0: int  # first / last (shared) grid column and row of the patch
    i1: int
    j0: int
    j1: int


@dataclass
class Box:
    pos: tuple  # centre
    size: tuple  # half extents
    rpy: tuple  # (0, 0, yaw): only yaw is supported (the ground truth assumes a flat top)
    material: str


@dataclass
class Field:
    kind: str
    size: float
    res: float
    seed: int
    heights: (
        np.ndarray
    )  # (ny, nx) world z, row j at y = -size/2 + j res, column i at x = -size/2 + i res
    patches: list = _dc_field(default_factory=list)
    boxes: list = _dc_field(default_factory=list)
    base: np.ndarray = None  # the surface without the rock bumps (rubble / scree); heights == base elsewhere
    rocks: list = _dc_field(
        default_factory=list
    )  # (x, y, radius, height) of the hfield bumps
    patch: float = 10.0
    params: dict = _dc_field(default_factory=dict)
    grid: object = (
        None  # (npatch, npatch) material names, row = y patch, column = x patch
    )

    @property
    def n(self):
        return self.heights.shape[0]

    def _cell(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        u = np.clip((x + self.size / 2) / self.res, 0, self.n - 1 - 1e-9)
        v = np.clip((y + self.size / 2) / self.res, 0, self.n - 1 - 1e-9)
        i, j = u.astype(int), v.astype(int)
        return i, j, u - i, v - j

    def hfield_height(self, x, y):
        """Height of the triangulated hfield surface (what MuJoCo's contacts and rays see)."""
        i, j, fu, fv = self._cell(x, y)
        h = self.heights
        h00, h10, h01, h11 = h[j, i], h[j, i + 1], h[j + 1, i], h[j + 1, i + 1]
        if DIAG == "main":  # diagonal (i, j) - (i+1, j+1)
            out = np.where(
                fu >= fv,
                h00 + fu * (h10 - h00) + fv * (h11 - h10),
                h00 + fv * (h01 - h00) + fu * (h11 - h01),
            )
        else:  # diagonal (i+1, j) - (i, j+1)
            out = np.where(
                fu + fv <= 1,
                h00 + fu * (h10 - h00) + fv * (h01 - h00),
                h11 + (1 - fu) * (h01 - h11) + (1 - fv) * (h10 - h11),
            )
        return out

    def _box_top(self, x, y):
        """(top z, material) of the highest box under each point (-inf, '' where none)."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        top = np.full(x.shape, -np.inf)
        mat = np.full(x.shape, "", dtype=object)
        for b in self.boxes:
            c, s = math.cos(b.rpy[2]), math.sin(b.rpy[2])
            dx, dy = x - b.pos[0], y - b.pos[1]
            lx, ly = c * dx + s * dy, -s * dx + c * dy
            inside = (np.abs(lx) <= b.size[0]) & (np.abs(ly) <= b.size[1])
            z = b.pos[2] + b.size[2]
            hit = inside & (z > top)
            top = np.where(hit, z, top)
            mat = np.where(hit, b.material, mat)
        return top, mat

    def height(self, x, y):
        """Ground truth: highest surface (hfield or box top) under (x, y); numpy-broadcast."""
        h = self.hfield_height(x, y)
        if self.boxes:
            h = np.maximum(h, self._box_top(x, y)[0])
        return float(h) if np.ndim(h) == 0 else h

    def _patch_material(self, x, y):
        pc = round(self.patch / self.res)
        i, j, _, _ = self._cell(x, y)
        npatch = (self.n - 1) // pc
        a, b = np.minimum(i // pc, npatch - 1), np.minimum(j // pc, npatch - 1)
        return np.asarray(self.grid, dtype=object)[b, a]

    def material_at(self, x, y):
        """Material name under (x, y): the box's if a box top is the ground truth there, else the patch's."""
        m = self._patch_material(x, y)
        if self.boxes:
            top, bm = self._box_top(x, y)
            m = np.where(top > self.hfield_height(x, y), bm, m)
        return str(m) if np.ndim(m) == 0 else m

    def digest(self):
        """Hex digest of everything that defines the field (heights bit for bit, patch materials, boxes)."""
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.heights, dtype=np.float64).tobytes())
        h.update(repr([(p.material, p.i0, p.j0) for p in self.patches]).encode())
        h.update(
            repr([(b.pos, b.size, b.rpy, b.material) for b in self.boxes]).encode()
        )
        return h.hexdigest()


# ---- generation ---------------------------------------------------------------------------------------------------------------
def _fbm(n, res, rng, hurst=0.8):
    """Zero-mean unit-std surface with power spectrum ~ k^-(2 hurst + 2) (random phases), periodic."""
    ky = np.fft.fftfreq(n, d=res)[:, None]
    kx = np.fft.rfftfreq(n, d=res)[None, :]
    k = np.hypot(kx, ky)
    k[0, 0] = 1.0
    amp = k ** -(hurst + 1.0)
    amp[0, 0] = 0.0
    spec = (
        rng.standard_normal((n, n // 2 + 1)) + 1j * rng.standard_normal((n, n // 2 + 1))
    ) * amp
    f = np.fft.irfft2(spec, s=(n, n))
    return (f - f.mean()) / f.std()


def _bumps(h, res, size, rng, count, rmin, rmax, hmin, hmax):
    """Raise `h` in place by `count` smooth rock bumps, profile peak (1 - (d/r)^2)^2, combined with max (a bump never digs). -> rocks list."""
    n = h.shape[0]
    base = h.copy()
    rocks = []
    for _ in range(count):
        x, y = rng.uniform(-size / 2, size / 2, 2)
        r = rng.uniform(rmin, rmax)
        ht = rng.uniform(hmin, hmax)
        i, j = int(round((x + size / 2) / res)), int(round((y + size / 2) / res))
        w = int(math.ceil(r / res)) + 1
        i0, i1, j0, j1 = (
            max(i - w, 0),
            min(i + w + 1, n),
            max(j - w, 0),
            min(j + w + 1, n),
        )
        xs = -size / 2 + res * np.arange(i0, i1)[None, :]
        ys = -size / 2 + res * np.arange(j0, j1)[:, None]
        d2 = ((xs - x) ** 2 + (ys - y) ** 2) / r**2
        prof = np.where(d2 < 1, (1 - d2) ** 2, 0.0)
        # the bump rises prof x ht above the pre-bump ground; where bumps overlap the higher one wins
        h[j0:j1, i0:i1] = np.maximum(h[j0:j1, i0:i1], base[j0:j1, i0:i1] + prof * ht)
        rocks.append((float(x), float(y), float(r), float(ht)))
    return rocks


def _slopes_profile(
    n, res, rng, theta_deg=(15.0, 30.0), step=(0.10, 0.18), ramp=(2.5, 5.0), cap=1.5
):
    """1-D terraced profile along x: ramps at 15-30 degrees alternate up / down, each ended by a vertical-ish step (one cell) of 10-18 cm."""
    z = np.zeros(n)
    i, level, up = 0, 0.0, True
    while i < n - 1:
        th = math.radians(rng.uniform(*theta_deg))
        length = int(round(rng.uniform(*ramp) / res))
        for k in range(1, length + 1):
            if i + k >= n:
                break
            z[i + k] = level + (1 if up else -1) * math.tan(th) * k * res
        i = min(i + length, n - 1)
        level = z[i]
        up = not up
        s = rng.uniform(*step)
        sgn = (
            -1.0
            if level > cap
            else 1.0
            if level < -cap
            else (1.0 if rng.random() < 0.5 else -1.0)
        )
        if i + 1 < n:
            level += sgn * s
            z[i + 1] = level
            i += 1
    return z


def _material_grid(kind, npatch, rng):
    """(npatch, npatch) array of material names, row = y patch, column = x patch."""
    opts = {
        "rubble": (["rock", "gravel", "wet_rock"], [0.6, 0.3, 0.1]),
        "slopes": (["rock", "grass", "gravel"], [0.5, 0.3, 0.2]),
        "scree": (["gravel", "wet_rock", "mud", "rock"], [0.55, 0.15, 0.15, 0.15]),
        "ruins": (["rock", "grass", "gravel"], [0.6, 0.25, 0.15]),
    }[kind]
    idx = rng.choice(len(opts[0]), size=(npatch, npatch), p=opts[1])
    return np.array(opts[0], dtype=object)[idx]


def _maze(size, cell, rng, wall_t=0.3):
    """Ruin walls on a `cell`-spaced grid: a random spanning tree (+ ~15 % extra) of door edges, solid walls elsewhere. A door is a gap of
    0.8-1.6 m between two wall boxes on the same line; a wall may be 'collapsed' (lower). -> list of Box."""
    m = int(size // cell)
    half = m * cell / 2
    parent = list(range(m * m))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    edges = [(i, j, i + 1, j) for i in range(m - 1) for j in range(m)] + [
        (i, j, i, j + 1) for i in range(m) for j in range(m - 1)
    ]
    order = rng.permutation(len(edges))
    door = set()
    for e in order:
        i, j, k, l = edges[e]
        a, b = find(j * m + i), find(l * m + k)
        if a != b:
            parent[a] = b
            door.add(e)
        elif rng.random() < 0.15:
            door.add(e)
    boxes = []
    for e, (i, j, k, l) in enumerate(edges):
        # the wall lies on the grid line between the two cells, over the full cell length
        if k == i + 1:  # neighbours in x: wall along y at x = (i + 1) cell
            cx, cy0, along = -half + (i + 1) * cell, -half + j * cell, 1
        else:  # neighbours in y: wall along x
            cx, cy0, along = -half + i * cell, -half + (j + 1) * cell, 0
        hgt = rng.uniform(0.3, 2.5) if rng.random() < 0.3 else rng.uniform(1.8, 2.5)
        spans = [(0.0, cell)]
        if e in door:
            g = rng.uniform(0.8, 1.6)
            c = rng.uniform(0.2 * cell + g / 2, 0.8 * cell - g / 2)
            spans = [(0.0, c - g / 2), (c + g / 2, cell)]
        for s0, s1 in spans:
            if s1 - s0 < 0.05:
                continue
            mid, ln = (s0 + s1) / 2, (s1 - s0) / 2
            if along:
                pos, sz = (cx, cy0 + mid, hgt / 2), (wall_t / 2, ln, hgt / 2)
            else:
                pos, sz = (cx + mid, cy0, hgt / 2), (ln, wall_t / 2, hgt / 2)
            boxes.append(Box(pos, sz, (0.0, 0.0, 0.0), "rock"))
    return boxes


def field(kind, size=100.0, res=0.05, seed=0, patch=10.0, **p):
    """Seeded small-scale terrain field. Kinds: rubble (5-20 cm rock bumps + boulder boxes), slopes (15-30 degree ramps with 10-18 cm steps),
    scree (small loose stones, low-friction patches), ruins (flat ground + a maze of walls with 0.8-1.6 m doors). Keywords: rocks / boulders
    (rubble), walls_cell (ruins, m)."""
    if kind not in KINDS:
        raise ValueError(f"terrain field kind {kind!r}: choose from {KINDS}")
    n = int(round(size / res)) + 1
    pc = int(round(patch / res))
    if (
        abs(size / patch - round(size / patch)) > 1e-9
        or abs(patch / res - pc) > 1e-9
        or (n - 1) % pc
    ):
        raise ValueError(
            "size must be a whole number of patches and patch a whole number of cells"
        )
    rng = np.random.default_rng(seed)
    h = 0.012 * _fbm(n, res, rng)  # 1.2 cm rms ground roughness everywhere
    rocks, boxes, base = [], [], None
    if kind == "rubble":
        base = h.copy()  # the ground without the rock bumps
        rocks = _bumps(h, res, size, rng, p.get("rocks", 2500), 0.12, 0.45, 0.05, 0.20)
    elif kind == "scree":
        h += 0.02 * _fbm(n, res, rng, 0.5)
        base = h.copy()
        rocks = _bumps(h, res, size, rng, p.get("rocks", 6000), 0.05, 0.18, 0.02, 0.07)
    elif kind == "slopes":
        h = h + _slopes_profile(n, res, rng)[None, :]
    if kind == "rubble":
        for _ in range(p.get("boulders", 40)):
            x, y = rng.uniform(-size / 2 + 2, size / 2 - 2, 2)
            sz = (rng.uniform(0.15, 0.4), rng.uniform(0.15, 0.4), rng.uniform(0.1, 0.3))
            yaw = rng.uniform(0, math.pi)
            i, j = int(round((x + size / 2) / res)), int(round((y + size / 2) / res))
            w = int(max(sz[0], sz[1]) * 1.5 / res)
            top = h[
                max(j - w, 0) : j + w + 1, max(i - w, 0) : i + w + 1
            ].max()  # rests on the highest ground under it, 2 cm embedded
            boxes.append(
                Box(
                    (float(x), float(y), float(top + sz[2] - 0.02)),
                    tuple(map(float, sz)),
                    (0.0, 0.0, float(yaw)),
                    "rock",
                )
            )
    elif kind == "ruins":
        boxes = _maze(size, p.get("walls_cell", 5.0), rng)
    npatch = (n - 1) // pc
    grid = _material_grid(kind, npatch, rng)
    patches = []
    for b in range(npatch):
        for a in range(npatch):
            x0, y0 = -size / 2 + a * patch, -size / 2 + b * patch
            patches.append(
                Patch(
                    x0,
                    y0,
                    x0 + patch,
                    y0 + patch,
                    str(grid[b, a]),
                    a * pc,
                    (a + 1) * pc,
                    b * pc,
                    (b + 1) * pc,
                )
            )
    f = Field(
        kind,
        float(size),
        float(res),
        int(seed),
        h,
        patches,
        boxes,
        base,
        rocks,
        float(patch),
        dict(p),
    )
    f.grid = grid
    return f


# ---- MuJoCo ---------------------------------------------------------------------------------------------------------------------
def _fmt(v):
    return " ".join(f"{x:.9g}" for x in v)


def add_field(spec, f, name="fld", group=0):
    """Add the field to an MJCF (<mujoco> ElementTree root, edited in place, or XML text -> returns (xml text, names)). One <hfield> asset and one
    static hfield geom per patch with the patch material's friction / solref (priority 1: the surface's friction replaces the foot's, it is not
    the max of both), box geoms for boulders / walls. -> the list of geom names (patches first). Contact filters: contype / conaffinity 1."""
    if isinstance(spec, str):
        root = ET.fromstring(spec)
        names = add_field(root, f, name, group)
        return ET.tostring(root, encoding="unicode"), names
    asset = spec.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        spec.insert(0, asset)
    wb = spec.find("worldbody")
    names = []
    pc = int(round(f.patch / f.res))
    for p in f.patches:
        sub = f.heights[p.j0 : p.j1 + 1, p.i0 : p.i1 + 1]
        zlo, zhi = float(sub.min()), float(sub.max())
        elev = max(zhi - zlo, 1e-3)
        norm = np.flipud(
            (sub - zlo) / elev
        )  # MuJoCo's data row 0 is the +y edge (as terrain.add_terrain)
        a, b = p.i0 // pc, p.j0 // pc
        hn, gn = f"{name}_h{b}_{a}", f"{name}_p{b}_{a}"
        ET.SubElement(
            asset, "hfield", name=hn, nrow=str(pc + 1), ncol=str(pc + 1), size=_fmt([f.patch / 2, f.patch / 2, elev, 0.1]),
            elevation=" ".join(f"{v:.7g}" for v in norm.ravel()),
        )  # fmt: skip
        m = MATERIALS[p.material]
        ET.SubElement(
            wb, "geom", name=gn, type="hfield", hfield=hn, pos=_fmt([(p.x0 + p.x1) / 2, (p.y0 + p.y1) / 2, zlo]), contype="1",
            conaffinity="1", group=str(group), friction=_fmt(m.friction), solref=_fmt(m.solref), priority="1", rgba=RGBA[p.material],
        )  # fmt: skip
        names.append(gn)
    for k, bx in enumerate(f.boxes):
        m = MATERIALS[bx.material]
        c, s = math.cos(bx.rpy[2] / 2), math.sin(bx.rpy[2] / 2)
        gn = f"{name}_b{k}"
        ET.SubElement(
            wb, "geom", name=gn, type="box", size=_fmt(bx.size), pos=_fmt(bx.pos), quat=_fmt([c, 0, 0, s]), contype="1", conaffinity="1",
            group=str(group), friction=_fmt(m.friction), solref=_fmt(m.solref), priority="1", rgba=RGBA[bx.material],
        )  # fmt: skip
        names.append(gn)
    return names
