"""Generators for marine hardware: the twin-hull catamaran (BlueBoat style: two deep demi-hulls with flat inner walls, a
hatch lid on each and aluminium cross-members between them). Frame: centred in x, y; keel at z = 0, deck at z = hull_depth;
bow towards +x."""

import math

from . import gen
from .mesh import loft, merge, rbox, revolve, rot, tube


def _hull(L, W, H, n=36, stations=28):
    """One demi-hull centred on the origin (x stern -> bow): superelliptic sections, elliptical bow plan, keel rising to a raked
    bow, a slightly narrower transom."""
    rings = []
    for i in range(stations):
        s = (
            1 - (1 - i / (stations - 1)) ** 1.5
        )  # stations bunch up towards the bow, where the plan curves
        u = max(0.0, (s - 0.55) / 0.45)
        w = (0.08 + 0.92 * math.sqrt(max(1 - u * u, 0.0))) * (
            0.9 + 0.1 * min(1.0, s / 0.12)
        )
        zb = 0.45 * H * ((max(0.0, s - 0.45) / 0.55) ** 2)
        h = H - zb
        ring = []
        for k in range(n):
            t = (
                -2 * math.pi * k / n
            )  # clockwise in (y, z) seen from +x = counter-clockwise seen along +x
            cs, sn = math.cos(t), math.sin(t)
            ring.append(
                (
                    -L / 2 + s * L,
                    w * W / 2 * math.copysign(abs(cs) ** 0.5, cs),
                    zb + h / 2 + h / 2 * math.copysign(abs(sn) ** 0.45, sn),
                )
            )
        rings.append(ring)
    return loft(rings)


@gen(
    "catamaran_hull",
    hull="hull-orange",
    lid="plastic-dark",
    beam="alu-anodised-grey",
    skeg="plastic-dark",
)
def catamaran_hull(c, **_):
    L = c.p("hull_length", "length", default=1.2)
    W = c.p("hull_width", default=0.17)
    H = c.p("hull_depth", default=0.20)
    B = c.p("beam", default=0.93)
    off = (B - W) / 2
    hull = _hull(L, W, H)
    hulls = merge([hull.at((0, off, 0)), hull.at((0, -off, 0))])
    lids = merge(
        rbox((0.30, 0.6 * W, 0.008), 0.003).at((0.05, s * off, H)) for s in (1, -1)
    )
    beams = merge(
        rbox((0.04, 2 * off, 0.035), 0.004).at((x, 0, 0.55 * H)) for x in (-0.15, 0.30)
    )
    out = {"hull": hulls, "lid": lids, "beam": beams}
    chord, depth = c.p("skeg_chord"), c.p("skeg_depth")
    if (
        chord and depth
    ):  # fixed skeg under the stern of each hull (params skeg_chord / skeg_depth / skeg_x)
        x = c.p("skeg_x", default=-0.45)
        out["skeg"] = merge(
            rbox((chord, 0.008, depth + 0.012), 0.002).at(
                (x, s * off, -depth / 2 + 0.006)
            )
            for s in (1, -1)
        )
    return out


@gen("ducted_thruster", can="alu-anodised-grey", duct="plastic-dark")
def ducted_thruster(c, **_):
    """Flooded thruster (T200 style): motor can with a rounded nose, short shroud ring held by four stator struts. Mount face at
    z = 0, prop side +z (the part's `diameter` over the shroud, `length_body` tall)."""
    D = c.p("diameter", default=0.10)
    Lb = c.p("length_body", default=0.113)
    rc = 0.24 * D
    can = revolve(
        [
            (0, 0),
            (rc, 0),
            (rc, 0.78 * Lb),
            (0.85 * rc, 0.9 * Lb),
            (0.5 * rc, 0.985 * Lb),
            (0, Lb),
        ],
        48,
    )
    ro = D / 2
    z0 = 0.4 * Lb
    shroud = tube(ro, ro - 0.005, Lb - z0, 64, z0=z0)
    struts = merge(
        rbox((ro - rc, 0.004, 0.03), 0.001)
        .at((rc + (ro - rc) / 2 - 0.002, 0, 0.55 * Lb))
        .at(R=rot([0, 0, 1], k * math.pi / 2))
        for k in range(4)
    )
    return {"can": can, "duct": merge([shroud, struts])}


@gen(
    "esc_sealed",
    body="alu-anodised-black",
    label="label-white",
    fins="alu-anodised-grey",
)
def esc_sealed(c, **_):
    """Waterproof marine ESC: sealed rounded case with a label on top and cooling ribs along both long sides. Underside at
    z = 0 (size_length x size_width x size_height)."""
    L = c.p("size_length", default=0.06)
    W = c.p("size_width", default=0.038)
    H = c.p("size_height", default=0.025)
    body = rbox((L, W, H), 0.003).at((0, 0, H / 2))
    label = rbox((0.6 * L, 0.6 * W, 0.0006), 0.0001).at((0, 0, H + 0.0001))
    xs = [(i - 3.5) * 0.09 * L for i in range(8)]
    fins = merge(
        rbox((0.0025, 0.003, 0.7 * H), 0.0005).at((x, s * (W / 2 + 0.0008), 0.5 * H))
        for x in xs
        for s in (1, -1)
    )
    return {"body": body, "label": label, "fins": fins}


def _osv_taper(s):
    """Plan-form factor of the half-beam along the hull (s = 0 transom .. 1 bow): full beam to 62 % of the length, then a
    superelliptic bow."""
    u = max(0.0, (s - 0.62) / 0.38)
    return (1.0 - u**2.2) ** 0.55 if u < 1 else 0.0


def _osv_rings(L, Bd, depth, z_w, stations=30):
    """Deep-V sections of the 12 m work boat, transom to bow: keel (0, 0), waterline chine (half-beam 1.2 m at z_w, 0.85 m up),
    flared topsides to the deck edge Bd / 2 at `depth`. Returns lower (keel .. chine) and upper (chine .. deck) rings plus the
    deck outline [(x, half-beam)]. The keel rises towards the bow (raked stem)."""
    y_w = 0.63 * Bd  # waterline half-beam
    low, up, outline = [], [], []
    for i in range(stations):
        s = (i / (stations - 1)) ** 0.8 if i < stations - 1 else 1.0
        x = -L / 2 + s * L
        f = max(_osv_taper(s), 0.03)
        zb = 0.75 * depth * s**3  # keel rise
        k = lambda z: zb + z * (1 - zb / depth)
        yw, yd = y_w * f, 0.5 * Bd * f
        low.append([(x, 0.0, zb), (x, yw, k(z_w)), (x, -yw, k(z_w))])
        up.append([(x, yw, k(z_w)), (x, yd, depth), (x, -yd, depth), (x, -yw, k(z_w))])
        outline.append((x, yd))
    return low, up, outline


@gen(
    "osv_hull",
    bottom="alu-anodised-blue",
    side="hull-orange",
    deck="gunmetal",
    fender="rubber",
)
def osv_hull(c, **_):
    """12.6 m aluminium crew-boat hull (Damen FCS 1204 class): deep-V bottom (dark blue), orange topsides, grey non-skid deck,
    rubber D-fender along the gunwale. Frame: mid-length on the centreline, keel at z = 0, bow +x, deck at `depth`."""
    from .mesh import extrude

    L = c.p("length", default=12.6)
    Bd = c.p("beam", default=3.8)
    depth = c.p("depth", default=1.7)
    z_w = c.p("chine_z", default=0.85)
    low, up, outline = _osv_rings(L, Bd, depth, z_w)
    bottom = loft(low)
    side = loft(up)
    poly = [(x, y) for x, y in outline] + [(x, -y) for x, y in reversed(outline)]
    deck = extrude(
        poly[::-1], depth - 0.006, depth
    )  # ccw outline; the top is the deck plane, so parts and wires lying on the deck are not inside the slab
    fenders = []
    for sg in (1, -1):
        rings = []
        for x, y in outline[2:-1]:
            ring = [
                (x, sg * (y + 0.04 + r * math.cos(t)), depth - 0.12 + r * math.sin(t))
                for r, t in ((0.06, 2 * math.pi * k / 10) for k in range(10))
            ]
            rings.append(ring[::sg])
        fenders.append(loft(rings))
    return {"bottom": bottom, "side": side, "deck": deck, "fender": merge(fenders)}


@gen(
    "osv_deckhouse",
    house="plastic-white",
    glass="glass-dark",
    roof="plastic-white",
    mast="alu-anodised-grey",
)
def osv_deckhouse(c, **_):
    """Wheelhouse of the work boat: a long cabin with a raked front, dark window bands, a flat roof with a short instrument
    mast. Frame: centre of the footprint at the deck level (z = 0), bow +x."""
    from .mesh import extrude

    L, W, H = (
        c.p("length", default=3.6),
        c.p("width", default=2.8),
        c.p("height", default=2.1),
    )
    hw, hl = W / 2, L / 2
    plan = [
        (-hl, -hw),
        (hl - 0.9, -hw),
        (hl, -hw + 0.5),
        (hl, hw - 0.5),
        (hl - 0.9, hw),
        (-hl, hw),
    ]
    house = extrude(plan, 0.0, H - 0.05)
    roof = extrude(
        [
            (
                x * 1.0 + (0.08 if x > 0 else -0.08),
                y * 1.02 + (0.05 if y > 0 else -0.05),
            )
            for x, y in plan
        ],
        H - 0.05,
        H + 0.04,
    )
    glass = []
    z0, z1 = 1.0, 1.75
    for sg in (1, -1):  # side window bands, proud of the wall by 5 mm
        glass.append(
            rbox((L - 1.6, 0.012, z1 - z0), 0.0).at(
                (-0.45, sg * (hw + 0.004), (z0 + z1) / 2)
            )
        )
    glass.append(
        rbox((0.012, W - 1.3, z1 - z0), 0.0).at((hl + 0.004, 0.0, (z0 + z1) / 2))
    )  # front windows
    mast = merge([tube(0.06, 0.05, 1.2, 24, z0=H + 0.04).at((0.2, 0.0, 0.0))])
    return {"house": house, "roof": roof, "glass": merge(glass), "mast": mast}


@gen(
    "helipad_deck",
    plate="gunmetal",
    border="plastic-yellow",
    mark="plastic-white",
    ring="plastic-white",
)
def helipad_deck(c, **_):
    """Square landing pad with markings (render only: the physics is the `landing` shape of the part): yellow perimeter band,
    white touchdown circle and a white H. Frame: centre of the underside at the origin, +z up, H reads from the stern (-x)."""
    S = c.p("pad_size", default=3.0)
    T = c.p("pad_thickness", default=0.05)
    m = 0.002  # marking height
    plate = rbox((S, S, T), 0.004).at((0, 0, T / 2))
    b = 0.14  # perimeter band width
    bands = merge(
        [
            rbox((S - 0.02, b, m), 0.0).at((0, sg * (S / 2 - b / 2 - 0.01), T + m / 2))
            for sg in (1, -1)
        ]
        + [
            rbox((b, S - 0.02 - 2 * b, m), 0.0).at(
                (sg * (S / 2 - b / 2 - 0.01), 0, T + m / 2)
            )
            for sg in (1, -1)
        ]
    )
    r = 0.42 * S
    ring = tube(r, r - 0.09, m, 64, z0=T)
    h = 0.5 * r  # H: two legs and a crossbar
    hmark = merge(
        [
            rbox((1.1 * h, 0.16 * h, m), 0.0).at((0, 0, T + m / 2)),
            rbox((0.16 * h, 1.1 * h * 1.6, m), 0.0).at((0.55 * h, 0, T + m / 2)),
            rbox((0.16 * h, 1.1 * h * 1.6, m), 0.0).at((-0.55 * h, 0, T + m / 2)),
        ]
    )
    return {"plate": plate, "border": bands, "mark": hmark, "ring": ring}
