"""Servo families: Dynamixel X-series, Feetech bus servos, hobby PWM servos. One shared frame (the Koch / SO-101 servo
frame): origin on the output axis at the outer face of the horn, +z out of the horn, +x along the case's long axis towards
the case centre, body towards -z. Horn offset from the end of the case is a fixed fraction of its length per family."""

import math

import numpy as np

from . import gen
from .mesh import circle, cylinder, extrude, merge, rbox, tube

PI = math.pi


def _ring(r, n, rad, start=0.0):
    return [
        circle(
            r,
            10,
            (
                rad * math.cos(start + 2 * PI * k / n),
                rad * math.sin(start + 2 * PI * k / n),
            ),
        )
        for k in range(n)
    ]


@gen(
    "dxl_servo",
    body="dxl-black",
    seam="plastic-black",
    horn="alu-champagne",
    screw="steel-dark",
    idler="plastic-dark",
    label="label-white",
    port="plastic-white",
)
def dxl_servo(c, off=None, hub=None, depth=None, **_):
    """ROBOTIS DYNAMIXEL X-series (XL330 / XL430 / XC430 / XM430 / XM540): black shell with a seam, a round horn with a ring of
    eight screw holes and a centre screw on one face, four corner screws, the model label below the horn, an idler ring on the
    opposite face. Appearance: ROBOTIS e-Manual product photos (emanual.robotis.com, xl430_product_new.png). `off` = output axis
    from the end of the case, `hub` = horn protrusion, `depth` = case depth (defaults: XL430 proportions)."""
    W = c.p("width") or 0.0285  # across the horn face (y)
    H = c.p("height") or 0.0465  # long axis (x)
    D = depth or c.p("depth") or 0.034  # along the output axis (z)
    s = W / 0.0285
    hub = hub or 0.115 * D  # horn protrusion above the case face
    off = off or 0.242 * H  # output axis from the end of the case
    xc, zt = H / 2 - off, -hub
    zb = zt - D
    body = rbox([H, W, D], 0.0016 * s, 3).at([xc, 0, (zt + zb) / 2])
    seam = rbox([0.0006, W + 0.0003, D + 0.0003], 0.0002, 1).at(
        [xc + 0.12 * H, 0, (zt + zb) / 2]
    )
    small = W < 0.024  # XL330: four holes round a larger hub
    rh = (0.4 if small else 0.35) * W
    horn = merge(
        [
            extrude(
                circle(rh, 64),
                -hub,
                0,
                _ring(0.0011 * s, 4 if small else 8, 0.74 * rh, PI / 4 if small else 0),
            ),
            cylinder(0.3 * rh, 0.0008 * s, 40, 0),
        ]
    )
    cs = 0.0013 * s
    sx = [-off + 0.0048 * s, H - off - 0.0048 * s]
    sy = W / 2 - 0.0048 * s
    screws = merge(
        [cylinder(cs, 0.0004, 16, zt).at([x, y, 0]) for x in sx for y in (-sy, sy)]
        + [cylinder(0.0017 * s, 0.0006, 20, 0.0008 * s)]
    )
    ro = 0.34 * W
    idler = extrude(
        circle(ro, 48),
        zb - 0.0014,
        zb,
        [circle(0.17 * W, 24)] + _ring(0.0011 * s, 8, 0.26 * W),
    )
    label = rbox([0.15 * H, 0.62 * W, 0.0004], 0.0001, 1).at(
        [H - off - 0.17 * H, 0, zt + 0.0002]
    )
    ports = rbox(
        [0.0014, 0.0075 * s, 0.0055 * s], 0.0002, 1
    ).at(  # JST EH housing in the end face opposite the horn
        [H - off + 0.0003, 0, zt - D / 2]
    )
    return {
        "body": body,
        "seam": seam,
        "horn": horn,
        "screw": screws,
        "idler": idler,
        "label": label,
        "port": ports,
    }


@gen(
    "bus_servo",
    body="plastic-black",
    cover="plastic-dark",
    horn="alu-brushed",
    screw="steel-dark",
    label="label-white",
    port="plastic-white",
    boss="alu-brushed",
)
def bus_servo(c, **_):
    """Feetech STS / SCS / HL bus servo (STS3215 proportions): black case, silver output horn disc with four holes and a centre
    screw, inset black cover on the horn face, white label on the side, two three-pin ports in the end face opposite the horn,
    bearing boss on the far face. Appearance: Feetech product photos / SO-ARM100 build photos (github.com/TheRobotStudio/SO-ARM100)."""
    L = c.p("length") or 0.0452
    W = c.p("width") or 0.0247
    H = c.p("height") or 0.035
    if L < W:
        L, W = W, L
    s = W / 0.0247
    hub = 0.0030 * s
    off = 0.2257 * L
    xc, zt = L / 2 - off, -hub
    zb = zt - H
    body = rbox([L, W, H], 0.0015 * s, 3).at([xc, 0, (zt + zb) / 2])
    cover = rbox([L - 0.006 * s, W - 0.004 * s, 0.0005], 0.0004, 2).at(
        [xc + 0.0, 0, zt + 0.00025]
    )
    rh = 0.40 * W
    horn = merge(
        [
            extrude(circle(rh, 64), -hub, 0, _ring(0.0010 * s, 4, 0.62 * rh, PI / 4)),
            cylinder(0.3 * rh, 0.0008 * s, 40, 0),
        ]
    )
    screw = cylinder(0.0017 * s, 0.0008 * s, 20, 0.0008 * s)
    label = rbox([0.4 * L, 0.0005, 0.2 * H], 0.0001, 1).at(
        [xc + 0.1 * L, W / 2 + 0.0001, zt - 0.5 * H]
    )
    ports = merge(
        [
            rbox([0.0012, 0.0062 * s, 0.0045 * s], 0.0002, 1).at(
                [L - off + 0.0002, y, zt - 0.55 * H]
            )
            for y in (-0.0045 * s, 0.0045 * s)
        ]
    )
    boss = tube(0.0085 * s, 0.0035 * s, 0.0016, 36, zb - 0.0016)
    return {
        "body": body,
        "cover": cover,
        "horn": horn,
        "screw": screw,
        "label": label,
        "port": ports,
        "boss": boss,
    }


@gen(
    "hobby_servo",
    case="servo-black",
    horn="plastic-white",
    boss="plastic-dark",
    screw="steel",
    label="label-white",
    grommet="rubber",
)
def hobby_servo(c, **_):
    """Standard / mini / micro PWM hobby servo (Hitec HS, TowerPro, Savox, DS): case with a mounting flange carrying one
    grommeted hole at each end, output boss and a smaller idle boss on top, a four-arm white horn, a label on top. Size from the
    part's length / width / height, else scaled from mass off the standard 40.7 x 19.8 x 36 mm servo. Appearance: Hitec HS-645MG and
    TowerPro MG996R / SG92R product photos."""
    m = c.p("mass") or 0.045
    sc = min(1.4, max(0.55, (m / 0.045) ** (1 / 3)))
    L = c.p("length", "envelope_length") or 0.0407 * sc
    W = c.p("width", "envelope_width") or 0.0198 * sc
    Ht = c.p("height", "envelope_height") or 0.0366 * sc
    if L < W:
        L, W = W, L
    s = W / 0.0198
    off = 0.255 * L
    xc = L / 2 - off
    zt = -0.0065 * s  # case top; the output boss and horn sit above it
    Hb = max(Ht - 0.0065 * s, 0.6 * Ht)
    zb = zt - Hb
    body = rbox([L, W, Hb], 0.0012 * s, 2).at([xc, 0, (zt + zb) / 2])
    zf = zt - 0.27 * Hb  # flange plane
    Lf = 1.34 * L
    ex = Lf / 2 - 0.0034 * s
    flange = extrude(
        [(-Lf / 2, -W / 2), (Lf / 2, -W / 2), (Lf / 2, W / 2), (-Lf / 2, W / 2)],
        zf - 0.0013 * s,
        zf + 0.0013 * s,
        [circle(0.0022 * s, 14, (sx * ex, 0)) for sx in (-1, 1)],
    ).at([xc, 0, 0])
    grom = merge(
        [
            tube(0.0034 * s, 0.0019 * s, 0.0036 * s, 20, zf - 0.0018 * s).at(
                [xc + sx * ex, 0, 0]
            )
            for sx in (-1, 1)
        ]
    )
    boss = merge(
        [
            cylinder(0.0058 * s, 0.0040 * s, 40, zt),
            cylinder(0.0036 * s, 0.0013 * s, 28, zt).at([0.0135 * s, 0, 0]),
            cylinder(0.0034 * s, 0.0030 * s, 24, zt),
        ]
    )
    # four-arm horn: disc + two crossed arms, holes along the arms
    th = 0.0018 * s
    z0 = -th
    arms = merge(
        [
            extrude(
                [
                    (-0.0155 * s, -0.0022 * s),
                    (0.0155 * s, -0.0022 * s),
                    (0.0155 * s, 0.0022 * s),
                    (-0.0155 * s, 0.0022 * s),
                ],
                z0,
                0,
                [
                    circle(0.0007 * s, 8, (sx * k * 0.0035 * s, 0))
                    for sx in (-1, 1)
                    for k in (2, 3, 4)
                ],
            ).at(
                [0, 0, 0],
                R=np.array(
                    [
                        [math.cos(a), -math.sin(a), 0],
                        [math.sin(a), math.cos(a), 0],
                        [0, 0, 1],
                    ]
                ),
            )
            for a in (0.0, PI / 2)
        ]
        + [cylinder(0.0062 * s, th, 36, z0)]
    )
    screw = cylinder(0.0016 * s, 0.0007 * s, 16, 0)
    label = rbox([0.45 * L, 0.7 * W, 0.0003], 0.0001, 1).at(
        [L - off - 0.3 * L, 0, zt + 0.00015]
    )
    return {
        "case": merge([body, flange]),
        "horn": arms,
        "boss": boss,
        "screw": screw,
        "label": label,
        "grommet": grom,
    }
