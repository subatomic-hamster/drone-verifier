"""Scene helper for landing on a ship: the 12.6 m offshore support vessel (assemblies/osv.toml) plus a lander in ONE World with ONE sea and ONE air device.

    sc = ship_scene(outdir, lander="skydio", lander_start="deck", hs=1.5, seed=3)       # or lander_start="air", air_height=3.0
    sim = dvsim.Sim.load(sc["model"])
    sc["ship"], sc["lander"]    # vehicle names (device prefix "osv_" / "<lander>_"; bodies /world/osv_hull, /world/<lander>_<root>)
    pad_pose(sc, hull_pos, hull_quat) -> (pad centre world, pad up normal)

What it does on top of `dvsim.cuas.scene.compose`:
  - the vessel is lowered by its equilibrium draft (the 4 hull sections against the assembly's total mass, sea water 1025 kg/m3), so it floats at rest from tick 0;
  - ONE sea device (the vessel's `osv_sea`): a seeded JONSWAP sea of significant wave height `hs` (peak period from the fully developed Pierson-Moskowitz relation
    Tp = 5 sqrt(Hs) s, at least 3.5 s unless `tp` is given), travelling towards `wave_dir` (math angle, deg from East), plus an optional steady `current`;
  - ONE air device (`air`, compose's): `wind`, `w20` (Dryden), `origin_alt`; the lander's own environment device is dropped by compose;
  - the water plane `water` (rendered by the viewer / renderer as the LIVE wave mesh of the sea device), the seabed at -40 m, no other scenery;
  - the lander either standing on the helipad (`lander_start="deck"`, its lowest collision point 2 mm over the pad surface) or hovering `air_height` m over the pad centre;
  - the helipad is the `landing` geom of the vessel design: contype 0 / conaffinity 1, geom group 3 (ray-visible, so the lander's depth cameras, lidars and rotor ground-effect rays
    see it), friction 0.9; the lander's skids / legs collide with it (design-builder vehicle geoms are contype 1).
Everything is deterministic in (arguments, seed).
"""

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .scene import REPO, Environment, Vehicle, compose

SEA_RHO = 1025.0  # kg/m3, as the vessel's sea device
SEABED = -40.0  # m
# Points of the vessel in its body frame (assemblies/osv.toml, parts/marine/generic-osv-hull-1260.toml): origin mid-length on the centreline at the keel, x forward, y to port, z up.
PAD_CENTRE = np.array(
    [-3.9, 0.0, 1.75]
)  # centre of the helipad's top surface (deck 1.7 m + 0.05 m plate)
PAD_SIZE = 3.0  # m, square
POINTS = dict(
    pad=PAD_CENTRE,
    mru=np.array([-0.6, 0.0, 1.7]),  # VN-100 on the main deck
    rtk_base=np.array(
        [2.0, 0.0, 5.04 + 0.0225]
    ),  # ANN-MB-00 phase centre (centre of the antenna), top of the 1.2 m mast on the wheelhouse roof
    gps=np.array([1.0, 0.5, 3.84 + 0.01]),
    bow=np.array([6.3, 0.0, 1.7]),
    stern=np.array([-6.3, 0.0, 1.7]),
)


def peak_period(hs, tp=None):
    return float(tp) if tp else max(3.5, 5.0 * math.sqrt(max(hs, 0.0)))


def jonswap_waves(hs, tp, direction, seed, n=40, w_lo=0.5, w_hi=2.2, gamma=3.3):
    """JONSWAP sea as the Sea device's regular components (amplitude, period, direction rad, phase), the kernel's own recipe (S(w) Hasselmann form, one frequency drawn per bin,
    a = sqrt(2 S dw), random phases, S scaled so that Hm0 = hs) over [w_lo, w_hi] x wp, with the upper limit 2.2 wp instead of the kernel's 3.5 wp: the buoyancy-cell hull
    integrates the wave SURFACE, with no decay of the wave pressure with depth (Smith effect), so waves of a few hull widths (2-3 s periods) force the 3.7 m beam roll many times
    harder than real ones and ring the 2.1 s roll resonance (9.8 deg rms in Hs 1.5 m beam seas against 2-4 deg for a real boat). The omitted tail holds ~1 % of the variance."""
    rng = np.random.default_rng([int(seed), 8821])
    wp = 2.0 * math.pi / tp
    w = w_lo * wp + (w_hi - w_lo) * wp * (np.arange(n) + rng.random(n)) / n
    dw = (w_hi - w_lo) * wp / n
    sig = np.where(w <= wp, 0.07, 0.09)
    S = (
        w**-5
        * np.exp(-1.25 * (wp / w) ** 4)
        * gamma ** np.exp(-((w - wp) ** 2) / (2 * sig**2 * wp**2))
    )
    S *= (hs / 4.0) ** 2 / (S.sum() * dw)  # m0 = (Hm0 / 4)^2 over the kept band
    a = np.sqrt(2.0 * S * dw)
    ph = rng.uniform(0.0, 2.0 * math.pi, n)
    return [
        dict(
            amplitude=float(a[i]),
            period=float(2.0 * math.pi / w[i]),
            direction=float(direction),
            phase=float(ph[i]),
        )
        for i in range(n)
    ]


def equilibrium_draft(mass, parts=REPO / "parts", rho=SEA_RHO):
    """Keel-to-still-water distance at which the hull sections of generic-osv-hull-1260 displace `mass` kg (bisection on the stacked boxes)."""
    from .. import design

    lib = design.Library(parts)
    shapes = lib.get("generic-osv-hull-1260").geometry["shapes"]
    boxes = [(np.array(s["pos"], float), np.array(s["size"], float)) for s in shapes]

    def vol(T):
        v = 0.0
        for c, h in boxes:
            z0, z1 = c[2] - h[2], c[2] + h[2]
            v += (
                4 * h[0] * h[1] * max(0.0, min(z1, T) - z0)
            )  # 2hx * 2hy * submerged height
        return v

    lo, hi = 0.0, 1.7
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if rho * vol(mid) < mass else (lo, mid)
    return 0.5 * (lo + hi)


def _fmt(v):
    return " ".join(f"{x:.9g}" for x in v)


def rot_z(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def ship_scene(
    outdir,
    lander=None,
    lander_start="deck",
    air_height=3.0,
    lander_name=None,
    hs=1.5,
    tp=None,
    wave_dir=0.0,
    current=(0.0, 0.0),
    wind=(0.0, 0.0, 0.0),
    w20=0.0,
    origin_alt=0.0,
    seed=1,
    ship_pos=(0.0, 0.0),
    ship_yaw=0.0,
    lander_offset=(0.0, 0.0),
    lander_yaw=None,
    visual=True,
    extra_vehicles=(),
    w_hi=2.2,
):
    """Compose the vessel (and a lander) into `outdir`/scene.xml -> dict(model, manifest, ship, lander, draft, pad (world, at tick 0), ...). `lander` = an assembly name
    ("skydio", "x500v2", "hopper") or None for the vessel alone; `lander_offset` shifts the lander (x, y) in the vessel frame from the pad centre. `ship_yaw` / `lander_yaw`
    in degrees (math angle; the lander defaults to the ship's heading). `extra_vehicles` = more `Vehicle`s (e.g. a second lander).
    The sea is a seeded JONSWAP of `hs` / `tp` (default Pierson-Moskowitz Tp = 5 sqrt(Hs)) travelling towards `wave_dir` (deg, math angle) built by `jonswap_waves` up to `w_hi` x wp
    (2.2; see there; None = the kernel's JONSWAP up to 3.5 wp)."""
    out = Path(outdir)
    R = rot_z(ship_yaw)
    sx, sy = ship_pos
    pad_xy = np.array([sx, sy, 0.0]) + R @ PAD_CENTRE
    veh = [Vehicle("osv", "osv", pos=(sx, sy, 0.0), yaw=ship_yaw)]
    name = None
    if lander:
        name = lander_name or lander.split("-")[0].replace("-", "_")
        off = R @ np.array([lander_offset[0], lander_offset[1], 0.0])
        veh.append(
            Vehicle(
                name,
                lander,
                pos=(pad_xy[0] + off[0], pad_xy[1] + off[1], 0.0),
                yaw=ship_yaw if lander_yaw is None else lander_yaw,
            )
        )
    veh += list(extra_vehicles)
    res = compose(
        out,
        veh,
        env=Environment(wind=tuple(wind), w20=w20, origin_alt=origin_alt, seed=seed),
        seed=seed,
        visual=visual,
    )
    mass = res["manifest"]["vehicles"]["osv"]["mass"]
    draft = equilibrium_draft(mass)
    pad_w = pad_xy + np.array([0.0, 0.0, -draft])
    path = Path(res["model"])
    root = ET.parse(path).getroot()
    wb = root.find("worldbody")
    # the vessel floats at its draft; the lander sits on the pad (2 mm clearance) or hovers over it: shift the top-level bodies (compose placed everything at z = 0)
    for b in wb.findall("body"):
        n = b.get("name", "")
        dz = (
            -draft
            if n.startswith("osv_")
            else (
                pad_w[2] + (0.002 if lander_start == "deck" else float(air_height))
                if name and n.startswith(name + "_")
                else 0.0
            )
        )
        if dz:
            p = [float(x) for x in b.get("pos", "0 0 0").split()]
            b.set("pos", _fmt((p[0], p[1], p[2] + dz)))
    txt = root.find("custom/text")
    d = json.loads(txt.get("data"))
    sea = next(s for s in d["seas"] if s["name"] == "osv_sea")
    sea.pop("jonswap", None)
    if hs > 0 and w_hi is None:  # the kernel's own JONSWAP (components up to 3.5 wp)
        sea["jonswap"] = dict(
            hs=float(hs),
            tp=peak_period(hs, tp),
            direction=math.radians(wave_dir),
            seed=1000 + int(seed),
        )
    elif hs > 0:
        sea["waves"] = jonswap_waves(
            hs, peak_period(hs, tp), math.radians(wave_dir), 1000 + int(seed), w_hi=w_hi
        )
    if tuple(current) != (0.0, 0.0):
        sea["current"] = [float(current[0]), float(current[1]), 0.0]
    d["seas"] = [sea]
    txt.set("data", json.dumps(d, sort_keys=True, separators=(",", ":")))
    for g in wb.findall("geom"):
        if g.get("name") == "floor":
            g.set("pos", _fmt((0, 0, SEABED)))
            g.set("rgba", "0.30 0.27 0.20 1")
    ET.SubElement(
        wb,
        "geom",
        name="water",
        type="plane",
        size="600 600 0.1",
        pos="0 0 0",
        rgba="0.02 0.15 0.22 1",
        group="2",
        contype="0",
        conaffinity="0",
    )
    ET.indent(root)
    path.write_text(ET.tostring(root, encoding="unicode"))
    man = res["manifest"]
    return dict(
        model=str(path),
        manifest=man,
        ship="osv",
        lander=name,
        draft=draft,
        mass=mass,
        pad=pad_w,
        tp=peak_period(hs, tp),
        hs=hs,
        seed=seed,
        ship_yaw=ship_yaw,
    )


def body_to_world(hull_pos, hull_quat, local):
    """World position of a point given in the vessel's body frame (`POINTS[...]` or any local vector) from the hull body's pos / quat (w, x, y, z)."""
    from .relnav import quat_to_mat

    return np.asarray(hull_pos, float) + quat_to_mat(hull_quat) @ np.asarray(
        local, float
    )


def pad_pose(hull_pos, hull_quat):
    """-> (pad centre in the world, pad surface normal in the world) from the hull pose."""
    from .relnav import quat_to_mat

    R = quat_to_mat(hull_quat)
    return np.asarray(hull_pos, float) + R @ PAD_CENTRE, R @ np.array([0.0, 0.0, 1.0])
