"""Seeded terrain for landing scenes: MuJoCo heightfields (mountain ridge with a summit pad, mesa with a cliff edge, a ledge pad below a
crest, rolling high moorland), flattened landing pads, boulders, and the terrain-aware wind device that goes with them.

    t = generate("ridge", size=(1600, 1200), n=257, relief=260, roughness=2.5, seed=3,
                 pads=[PadSpec("summit", at="summit", radius=4)], rocks=40)
    t.pad("summit")                         # Pad(name, x, y, z, radius): z is exact (the pad is flat), world frame
    t.height_at(x, y)                       # bilinear surface height (numpy broadcast), world z
    add_terrain(mjcf_root, t, wind=WindField(z0=0.03, ridge_gain=1.0), origin_alt=3200.0)   # edits an ElementTree MJCF in place
    xml = add_terrain_xml(xml_text, t, wind=WindField())                                    # the same on MJCF text

World frame: z up; the datum `zero_at` pad (default: the first pad) is at z = 0, so a vehicle built "on the ground" at (pad.x, pad.y, 0) stands on
the pad; `origin_alt` (m MSL of world z = 0, Environment device) then gives the ISA density of the site (2500-4000 m for the mountain scenes).

Physics: ONE static `hfield` geom named "terrain" (contype / conaffinity 1, geom group 0): bodies with contype 1 rest on it, mj_ray / rangefinders /
ground-effect rays see it (vehicle geoms are group 4, outside the default ray mask). Pad markers (a painted disc and an H, group 4, no contact) are
invisible to rays, so a downward range at a pad is exactly the height above the flat pad. Boulders are static ellipsoids (contact, group 0).
The WindField device (src/mujoco/devices/windfield.cpp, docs/models.md "Terrain and wind field") makes the wind follow the ground.

Terrain model (all synthetic, seeded, deterministic): large-scale shape (Gaussian ridge, tanh cliff, rolling hills) + spectral fBm detail with power
spectrum ~ k^-(2 hurst + 2) (random phases; Voss / Peitgen-Saupe fractional Brownian surface) of RMS `roughness` metres; pads are flattened by blending the
surface to the pad height over `blend` metres outside the pad radius (smoothstep): cut and fill, so a pad on a steep face becomes a carved ledge.
"""

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

KINDS = ("ridge", "mesa", "ledge", "moorland")
GEOM_NAME = "terrain"
HFIELD_NAME = "terrain"
WIND_NAME = "wind"


@dataclass
class PadSpec:
    """A landing pad: `at` = an anchor name of the terrain kind or an (x, y) tuple (m, terrain-centred frame, before the datum shift); `radius` = the flat disc (m);
    `blend` = width of the smooth cut/fill ring around it (m); `z` = pad height (m above the datum before the shift; None = the surface mean over the disc)."""

    name: str
    at: object = "center"
    radius: float = 3.0
    blend: float = 20.0
    z: float | None = None


@dataclass
class Pad:
    name: str
    x: float
    y: float
    z: float  # world z of the flat pad surface (exact, up to float32 storage ~ 1e-4 m at 300 m relief)
    radius: float  # the requested (marked) radius
    flat_radius: float = 0.0  # the really flat disc: radius + 1.5 grid cells


@dataclass
class Rock:
    """A boulder: static ellipsoid with semi-axes (rx, ry, rz) m at (x, y, z), yawed about z (rad)."""

    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float
    yaw: float


@dataclass
class WindField:
    """Config of the WindField device (schema WindField); see docs/models.md. `Environment.wind` is the steady wind at `z_ref` m above ground."""

    z0: float = 0.03  # roughness length m: short grass / moorland 0.03, bare rock and snow 0.001-0.01; 0 = no log profile
    z_ref: float = 10.0
    h_max: float = 300.0
    ridge_gain: float = 1.0  # 0 = no terrain-induced flow
    ridge_len: float = 80.0  # m, smoothing length and decay height of the ridge flow
    lee_turb: float = 2.0
    lee_down: float = 0.2

    def device(self, hfield_geom=GEOM_NAME, name=WIND_NAME):
        d = dict(name=name, z0=self.z0, z_ref=self.z_ref, h_max=self.h_max, ridge_gain=self.ridge_gain,
                 ridge_len=self.ridge_len, lee_turb=self.lee_turb, lee_down=self.lee_down)  # fmt: skip
        if hfield_geom:
            d["hfield_geom"] = hfield_geom
        return d


@dataclass
class Terrain:
    kind: str
    size: tuple  # (sx, sy) m, full extents, centred on the world origin in x and y
    heights: np.ndarray  # (ny, nx) world z of the surface; row r at y = -sy/2 + r dy, column c at x = -sx/2 + c dx
    seed: int
    pads: dict = field(default_factory=dict)
    rocks: list = field(default_factory=list)
    anchors: dict = field(
        default_factory=dict
    )  # name -> (x, y, z) of the kind's anchor points (world frame)
    params: dict = field(default_factory=dict)

    @property
    def nx(self):
        return self.heights.shape[1]

    @property
    def ny(self):
        return self.heights.shape[0]

    @property
    def zmin(self):
        return float(self.heights.min())

    @property
    def zmax(self):
        return float(self.heights.max())

    def pad(self, name) -> Pad:
        return self.pads[name]

    def height_at(self, x, y):
        """Bilinear surface height (world z); clamped to the grid edge outside. (MuJoCo's contact surface is the triangulated cell, which
        differs from this inside a cell by at most the local curvature; exact on the flat pads.)"""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        sx, sy = self.size
        u = np.clip((x + sx / 2) / sx * (self.nx - 1), 0, self.nx - 1)
        v = np.clip((y + sy / 2) / sy * (self.ny - 1), 0, self.ny - 1)
        i = np.minimum(u.astype(int), self.nx - 2)
        j = np.minimum(v.astype(int), self.ny - 2)
        fu, fv = u - i, v - j
        h = self.heights
        out = (1 - fv) * ((1 - fu) * h[j, i] + fu * h[j, i + 1]) + fv * (
            (1 - fu) * h[j + 1, i] + fu * h[j + 1, i + 1]
        )
        return float(out) if out.ndim == 0 else out

    def slope_at(self, x, y, step=None):
        """Terrain slope (rise/run) magnitude by central differences over `step` m (default 2 cells)."""
        s = step or 2 * self.size[0] / (self.nx - 1)
        gx = (self.height_at(x + s, y) - self.height_at(x - s, y)) / (2 * s)
        gy = (self.height_at(x, y + s) - self.height_at(x, y - s)) / (2 * s)
        return np.hypot(gx, gy)


# ---- generation ---------------------------------------------------------------------------------------------------------------------


def _fbm(ny, nx, dx, dy, rng, hurst):
    """Zero-mean, unit-std surface with power spectrum ~ k^-(2 hurst + 2) (k in cycles/m), periodic."""
    ky = np.fft.fftfreq(ny, d=dy)[:, None]
    kx = np.fft.rfftfreq(nx, d=dx)[None, :]
    k = np.hypot(kx, ky)
    k[0, 0] = 1.0
    amp = k ** -(hurst + 1.0)
    amp[0, 0] = 0.0
    spec = (
        rng.standard_normal((ny, nx // 2 + 1))
        + 1j * rng.standard_normal((ny, nx // 2 + 1))
    ) * amp
    f = np.fft.irfft2(spec, s=(ny, nx))
    return (f - f.mean()) / f.std()


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _shape(kind, X, Y, sx, sy, relief, rng, p):
    """Large-scale shape (m above the lowland datum) and the anchor points (x, y) of the kind."""
    anchors = {"center": (0.0, 0.0)}
    if kind in ("ridge", "ledge"):
        w = p.get("width", 0.14 * sx)
        skew = p.get(
            "skew", 0.0 if kind == "ridge" else 0.45
        )  # >0: the +x (lee) face is steeper than the -x face
        ws = np.where(X >= 0, w * (1 - skew), w * (1 + 0.3 * skew))
        crest = 1.0 + 0.18 * np.sin(
            2 * np.pi * (Y / (0.9 * sy) + rng.uniform())
        )  # crest height varies along the ridge
        ys = p.get("summit_y", 0.12 * sy)
        peak = 0.3 * np.exp(
            -(((X + 0.1 * w) ** 2 + (Y - ys) ** 2) / (2 * (0.5 * w) ** 2))
        )  # a summit above the crest line
        h = relief * (crest * np.exp(-((X / ws) ** 2) / 2) + peak)
        anchors["summit"] = (-0.1 * w, ys)
        # ledge: on the +x face at 30 % of the relief below the summit (where the profile has that height), same y
        xl = float(np.sqrt(-2 * np.log(0.7)) * w * (1 - skew))
        anchors["ledge"] = (xl, ys)
        anchors["windward"] = (-1.2 * w, 0.0)
        return h, anchors
    if kind == "mesa":
        edge = p.get("edge", 0.12 * sx)  # cliff line x (plateau on the -x side)
        wob = p.get("wobble", 0.05 * sy)
        ph = rng.uniform(0, 2 * np.pi, 3)
        edge_y = edge + wob * (
            np.sin(2 * np.pi * Y / sy * 1.0 + ph[0])
            + 0.5 * np.sin(2 * np.pi * Y / sy * 2.3 + ph[1])
            + 0.25 * np.sin(2 * np.pi * Y / sy * 5.1 + ph[2])
        )
        cw = p.get("cliff_width", 25.0)
        step = 0.5 * (1 - np.tanh((X - edge_y) / cw))
        h = relief * step - 0.08 * relief * np.clip(
            (X - edge) / (0.5 * sx), 0, 1
        )  # lowland falls away gently
        anchors["plateau"] = (edge - 0.2 * sx, 0.0)
        anchors["brink"] = (edge - 4 * cw, 0.0)
        anchors["lowland"] = (edge + 0.3 * sx, 0.0)
        return h, anchors
    if kind == "moorland":
        # broad rolling swells of ~relief height
        h = np.zeros_like(X)
        for _ in range(5):
            a, b = rng.uniform(-1, 1, 2)
            lam = rng.uniform(0.3, 0.8) * sx
            ang = rng.uniform(0, np.pi)
            u = X * np.cos(ang) + Y * np.sin(ang)
            h += (
                relief
                * rng.uniform(0.08, 0.2)
                * np.sin(2 * np.pi * u / lam + rng.uniform(0, 6.28))
            )
        h += (
            0.5
            * relief
            * np.exp(
                -((X - 0.2 * sx) ** 2 + (Y + 0.1 * sy) ** 2) / (2 * (0.18 * sx) ** 2)
            )
        )  # a hill to sit on
        anchors["hilltop"] = (0.2 * sx, -0.1 * sy)
        return h, anchors
    raise ValueError(f"unknown terrain kind {kind!r}; one of {KINDS}")


def generate(
    kind="ridge",
    size=(1600.0, 1200.0),
    n=257,
    relief=250.0,
    roughness=2.0,
    hurst=0.8,
    seed=1,
    pads=(),
    rocks=0,
    zero_at=None,
    rock_size=(0.4, 1.6),
    **shape,
):
    """Seeded terrain. `size` (sx, sy) m; `n` samples along x (ny follows the aspect, both odd-ish; <= 513 keeps MuJoCo's contact fast and the
    renderer's triangle count below ~500k); `relief` m = height of the main feature above the lowland; `roughness` m RMS of the fBm detail
    (hurst 0.5 rough .. 0.95 smooth); `pads` = PadSpec list (flattened in order); `rocks` boulders (seeded, kept off the pads);
    `zero_at` = pad name placed at z = 0 (default: first pad; none: the lowland datum is z = 0). `shape` overrides kind parameters:
    ridge/ledge `width`, `skew`, `summit_y`; mesa `edge`, `cliff_width`, `wobble`. Returns Terrain."""
    if kind not in KINDS:
        raise ValueError(f"unknown terrain kind {kind!r}; one of {KINDS}")
    sx, sy = float(size[0]), float(size[1])
    nx = int(n)
    ny = max(3, int(round((nx - 1) * sy / sx)) + 1)
    dx, dy = sx / (nx - 1), sy / (ny - 1)
    rng = np.random.default_rng(seed)
    x = -sx / 2 + dx * np.arange(nx)
    y = -sy / 2 + dy * np.arange(ny)
    X, Y = np.meshgrid(x, y)
    h, anchors = _shape(kind, X, Y, sx, sy, float(relief), rng, shape)
    if roughness > 0:
        detail = _fbm(ny, nx, dx, dy, rng, hurst)
        taper = (
            0.35 + 0.65 * np.clip(h / max(float(relief), 1e-9), 0, 1)
            if kind != "moorland"
            else 1.0
        )  # alpine detail rougher on the heights
        h = h + roughness * detail * taper
    # the datum is the lowland: shift so that the minimum of the base shape is not far below 0 (informational; pads fix the real datum)
    pad_out = {}
    anchor_xyz = {}
    specs = list(pads)
    xy = np.stack([X, Y], -1)
    for sp in specs:
        px, py = anchors[sp.at] if isinstance(sp.at, str) else tuple(sp.at)
        # Snap the centre to (0.31, 0.17) cells off the grid lattice: a vertical ray exactly on a cell edge or diagonal can miss in MuJoCo's
        # mj_ray (seen with a rangefinder at a pad centre on a grid line), and vehicles are placed at the pad centre.
        px = -sx / 2 + (round((px + sx / 2) / dx - 0.31) + 0.31) * dx
        py = -sy / 2 + (round((py + sy / 2) / dy - 0.17) + 0.17) * dy
        d = np.hypot(X - px, Y - py)
        # every grid triangle that touches the disc must be flat: the flat area reaches 1.5 cells beyond the requested radius
        flat = sp.radius + 1.5 * max(dx, dy)
        disc = d <= flat
        z = float(h[disc].mean()) if sp.z is None else float(sp.z)
        w = 1.0 - _smoothstep((d - flat) / max(sp.blend, 1e-6))
        h = h * (1 - w) + z * w
        pad_out[sp.name] = [px, py, z, sp.radius, flat]
    shift = 0.0
    if specs:
        ref = zero_at or specs[0].name
        shift = pad_out[ref][2]
    elif zero_at:
        raise ValueError("zero_at needs pads")
    h = h - shift
    pads_o = {
        k: Pad(k, v[0], v[1], v[2] - shift, v[3], v[4]) for k, v in pad_out.items()
    }
    # exact pad heights after the shift (the discs are constant, so any node inside is the value)
    t = Terrain(
        kind,
        (sx, sy),
        h,
        seed,
        pads_o,
        [],
        {},
        dict(relief=relief, roughness=roughness, hurst=hurst, n=nx, **shape),
    )
    for k, (ax, ay) in anchors.items():
        t.anchors[k] = (ax, ay, t.height_at(ax, ay))
    # boulders
    if rocks:
        keep = [(p.x, p.y, p.radius + 25.0 + 3.0) for p in pads_o.values()]
        tries = 0
        while len(t.rocks) < rocks and tries < 50 * rocks:
            tries += 1
            rx_ = rng.uniform(-0.45 * sx, 0.45 * sx)
            ry_ = rng.uniform(-0.45 * sy, 0.45 * sy)
            if any(np.hypot(rx_ - kx, ry_ - ky) < kr for kx, ky, kr in keep):
                continue
            r = float(np.exp(rng.uniform(np.log(rock_size[0]), np.log(rock_size[1]))))
            a, b, c = (
                r * rng.uniform(0.8, 1.4),
                r * rng.uniform(0.7, 1.2),
                r * rng.uniform(0.5, 0.9),
            )
            t.rocks.append(
                Rock(
                    rx_,
                    ry_,
                    t.height_at(rx_, ry_) + 0.15 * c,
                    a,
                    b,
                    c,
                    float(rng.uniform(0, np.pi)),
                )
            )
    return t


# ---- MJCF ---------------------------------------------------------------------------------------------------------------------------


def _f(v):
    return " ".join(f"{x:.9g}" for x in v)


def add_terrain(
    root,
    t,
    wind=None,
    origin_alt=None,
    markers=True,
    rgba=(0.45, 0.43, 0.36, 1.0),
    replace_floor=True,
    wind_name=WIND_NAME,
):
    """Put terrain `t` into an MJCF ElementTree (<mujoco> root) in place: <asset><hfield> + a static hfield geom "terrain" (+ pad markers, boulders), removes
    the root-level ground plane(s) (replace_floor), and, in the dvsim.devices JSON, adds the WindField device (`wind`: WindField or None) and sets the
    environment's origin_alt (m MSL of world z = 0) when given. The scene must have an environment device for `wind`."""
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(0, asset)
    wb = root.find("worldbody")
    if replace_floor:
        for g in [g for g in wb.findall("geom") if g.get("type", "sphere") == "plane"]:
            wb.remove(g)
    zlo, zhi = t.zmin, t.zmax
    elev = max(zhi - zlo, 1e-3)
    norm = np.flipud((t.heights - zlo) / elev)  # MuJoCo's data row 0 is the +y edge
    ET.SubElement(asset, "hfield", name=HFIELD_NAME, nrow=str(t.ny), ncol=str(t.nx), size=_f([t.size[0] / 2, t.size[1] / 2, elev, 5.0]),
                  elevation=" ".join(f"{v:.9g}" for v in norm.ravel()))  # fmt: skip
    g = ET.Element("geom", name=GEOM_NAME, type="hfield", hfield=HFIELD_NAME, pos=_f([0, 0, zlo]), contype="1", conaffinity="1",
                   group="0", rgba=_f(rgba))  # fmt: skip
    wb.insert(0, g)
    for i, r in enumerate(t.rocks):
        c, s = np.cos(r.yaw / 2), np.sin(r.yaw / 2)
        ET.SubElement(wb, "geom", name=f"rock{i}", type="ellipsoid", size=_f([r.rx, r.ry, r.rz]), pos=_f([r.x, r.y, r.z]),
                      quat=_f([c, 0, 0, s]), contype="1", conaffinity="1", group="0", rgba="0.42 0.40 0.38 1")  # fmt: skip
    if markers:
        for p in t.pads.values():
            _marker(wb, p)
    dev = root.find("custom/text[@name='dvsim.devices']")
    if dev is not None and (wind is not None or origin_alt is not None):
        d = json.loads(dev.get("data"))
        if origin_alt is not None:
            for e in d.get("environments", []):
                e["origin_alt"] = float(origin_alt)
        if wind is not None:
            if not d.get("environments"):
                raise ValueError(
                    "add_terrain(wind=...): the scene has no environment device"
                )
            d["wind_fields"] = [wind.device(GEOM_NAME, wind_name)]
        dev.set("data", json.dumps(d, sort_keys=True, separators=(",", ":")))
    return root


def _marker(wb, p):
    """Painted pad: orange disc, grey inner disc, white H. Group 4 (shown by the renderer, outside the default ray mask), no contact."""
    z = p.z + 0.004
    common = dict(contype="0", conaffinity="0", group="4")
    n = p.name
    ET.SubElement(
        wb,
        "geom",
        name=f"pad_{n}_ring",
        type="cylinder",
        size=_f([p.radius, 0.002]),
        pos=_f([p.x, p.y, z]),
        rgba="0.95 0.45 0.08 1",
        **common,
    )
    ET.SubElement(
        wb,
        "geom",
        name=f"pad_{n}_disc",
        type="cylinder",
        size=_f([p.radius * 0.88, 0.002]),
        pos=_f([p.x, p.y, z + 0.003]),
        rgba="0.40 0.41 0.43 1",
        **common,
    )
    a, w = 0.42 * p.radius, 0.07 * p.radius
    for k, (cx, cy, hx, hy) in enumerate(
        ((-a * 0.6, 0, w, a), (a * 0.6, 0, w, a), (0, 0, a * 0.6, w))
    ):
        ET.SubElement(
            wb,
            "geom",
            name=f"pad_{n}_h{k}",
            type="box",
            size=_f([hx, hy, 0.002]),
            pos=_f([p.x + cx, p.y + cy, z + 0.007]),
            rgba="0.96 0.96 0.96 1",
            **common,
        )


def add_terrain_xml(xml, t, **kw):
    """add_terrain on MJCF text -> MJCF text."""
    root = ET.fromstring(xml)
    add_terrain(root, t, **kw)
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


# Small-scale legged-robot fields (dvsim/terrain_field.py, docs/models.md "Small-scale terrain"). `field` is rebound here, after the dataclasses
# above used dataclasses.field: dvsim.terrain.field(kind, size, res, seed, ...) -> Field; add_field(mjcf, field) -> geom names.
from . import terrain_field as _tf

field, add_field, Field, Patch, Box, Material, MATERIALS = (
    _tf.field,
    _tf.add_field,
    _tf.Field,
    _tf.Patch,
    _tf.Box,
    _tf.Material,
    _tf.MATERIALS,
)
