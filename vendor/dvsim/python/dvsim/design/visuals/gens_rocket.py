"""Generators for rocketry hardware: solid motor casing, airframe tube, tangent-ogive nose cone, swept fin set, splayed landing
leg and the three printed TVC gimbal parts. Frames follow each part's own: motor nozzle face at z = 0 and +z towards the nose;
tube / nose base / fin root at z = 0; landing-leg hinge at the origin; gimbal parts in their part frame."""

import math

import numpy as np

from . import gen
from .mesh import (
    cylinder,
    extrude,
    merge,
    rbox,
    revolve,
    rot,
    rot_to,
    sphere,
    tube,
)

PI = math.pi


@gen(
    "solid_motor",
    casing="kraft-paper",
    label="label-white",
    nozzle="ceramic-grey",
    cap="kraft-dark",
)
def solid_motor(c, **_):
    """Model-rocket motor: paper casing cup (closed forward end), printed label band, recessed ceramic nozzle with a bell
    bore at the aft face (z = 0). Dimensions from `diameter` and `length`."""
    D = c.p("diameter", default=0.029)
    L = c.p("length", default=0.114)
    R = D / 2
    wall = max(0.0006, 0.04 * R)
    ri = R - wall
    cup = revolve(
        [(ri, 0), (R, 0), (R, L), (0, L), (0, L - 0.0015), (ri, L - 0.0015)], 48
    )
    zn, nl = 0.001, max(0.005, 0.5 * R)  # nozzle plug: recessed 1 mm, length nl
    rp, re = ri - 0.0001, 0.42 * R
    d = 0.8 * nl
    nozzle = revolve(
        [
            (0, zn + d),
            (0.15 * R, zn + 0.8 * d),
            (0.28 * R, zn + 0.4 * d),
            (re, zn),
            (rp, zn),
            (rp, zn + nl),
            (0, zn + nl),
        ],
        48,
    )
    label = tube(R + 0.0004, R - 0.0003, 0.78 * L, 48, z0=0.08 * L)
    cap = cylinder(
        ri - 0.0001, 0.0008, 48, z0=L - 0.0023
    )  # forward cap disc, darker paper
    return {"casing": cup, "label": label, "nozzle": nozzle, "cap": cap}


@gen("body_tube", tube="fiberglass")
def body_tube(c, **_):
    """Hollow airframe tube from z = 0 to `length` (outer_diameter, inner_diameter or wall_thickness)."""
    ro = c.p("outer_diameter", default=0.1016) / 2
    ri = c.p("inner_diameter")
    ri = ri / 2 if ri else ro - c.p("wall_thickness", default=0.00125)
    L = c.p("length", default=0.45)
    return {"tube": tube(ro, ri, L, 64)}


@gen("ogive_nose", shell="pla-white")
def ogive_nose(c, **_):
    """Tangent-ogive nose cone, hollow printed shell open at the base (z = 0), tip at z = length."""
    R = c.p("base_diameter", default=0.1016) / 2
    L = c.p("length", default=0.30)
    t = c.p("wall_thickness", default=0.0012)
    rho = (R * R + L * L) / (2 * R)
    zs = L * (
        1 - (1 - np.linspace(0, 1, 41)) ** 1.6
    )  # denser near the tip where the curvature is
    ro = np.sqrt(np.maximum(rho * rho - zs * zs, 0.0)) + R - rho
    ro[-1] = 0.0
    keep = ro - t > 0.0004
    zi, ri = zs[keep], ro[keep] - t
    prof = [(r, z) for r, z in zip(ro, zs)]  # base -> tip along the outside
    prof += [(0.0, zi[-1] + 0.0004)] + [(r, z) for r, z in zip(ri[::-1], zi[::-1])]
    return {"shell": revolve(prof, 64)}


@gen("fin_set", fin="pla-white", collar="pla-white")
def fin_set(c, tube_radius=None, **_):
    """`fin_count` swept trapezoidal fins on a thin printed collar around the tube (root chord from z = 0 to root_chord, trailing
    edge aft; the tip chord's leading edge is `sweep` behind the root's)."""
    n = int(c.p("fin_count", default=4))
    rc, tc = c.p("root_chord", default=0.16), c.p("tip_chord", default=0.08)
    span, sweep = c.p("span", default=0.12), c.p("sweep", default=0.09)
    th = c.p("thickness", default=0.0015)
    r0 = (
        tube_radius or 0.0508
    )  # radius of the tube the collar slides over (the 4 in tube's outer radius)
    le = rc - sweep  # tip leading edge z
    outline = np.array(
        [(r0 - 0.001, 0.0), (r0 + span, le - tc), (r0 + span, le), (r0 - 0.001, rc)]
    )
    fin = extrude(outline, -th / 2, th / 2).at(
        R=rot([1, 0, 0], PI / 2)
    )  # (radial, chord, thickness) -> (x, z, -y)
    fins = merge(fin.at(R=rot([0, 0, 1], 2 * PI * k / n)) for k in range(n))
    return {"fin": fins, "collar": tube(r0 + th, r0 - 0.0003, rc, 64)}


@gen("landing_leg", rod="carbon-fibre", foot="rubber", bracket="alu-anodised-grey")
def landing_leg(c, **_):
    """Carbon leg tube (10 x 8 mm) from the hinge at the origin outwards (+x) and down, `splay` from the vertical, with a TPU
    ball foot at `length` and an aluminium hinge block on the tube side (-x)."""
    L = c.p("length", default=0.25)
    a = c.p("splay", default=0.349)
    ax = np.array([math.sin(a), 0.0, -math.cos(a)])
    rod = tube(0.005, 0.004, L, 24).at(R=rot_to(ax))
    foot = sphere(0.012, 24, centre=ax * L)
    block = rbox((0.010, 0.024, 0.026), 0.002).at((-0.005, 0, 0))
    barrel = cylinder(0.0065, 0.022, 24, z0=-0.011).at(R=rot_to([0, 1, 0]))
    return {"rod": rod, "foot": foot, "bracket": merge([block, barrel])}


@gen("tvc_gimbal", part="pla-matte", boss="steel-dark")
def tvc_gimbal(c, kind="plate", **_):
    """Printed gimbal parts in their own frames. plate: tube bulkhead + stepped servo arm on +y; ring: pitch ring with
    trunnion bosses on +-y and the yaw-servo bracket on +x; cradle: motor sleeve with a flange and the horn boss on +x."""
    if kind == "plate":
        r, th = c.p("disc_radius", default=0.0495), c.p("disc_thickness", default=0.004)
        b = c.bounds()
        top = float(b[1][2]) if b is not None else 0.05
        disc = revolve(
            [
                (0, top - th),
                (r - 0.001, top - th),
                (r, top - th + 0.001),
                (r, top),
                (0, top),
            ],
            64,
        )
        arm = merge(
            [
                rbox((0.024, 0.0027, 0.035), 0.001).at((0, 0.04825, 0.0075)),
                rbox((0.020, 0.0027, 0.021), 0.001).at((0, 0.04825, 0.0355)),
            ]
        )
        hub = cylinder(0.0075, 0.004, 32, z0=0.0).at(
            (0, 0.0469, 0.0075), rot_to([0, -1, 0])
        )
        return {"part": merge([disc, arm]), "boss": hub}
    if kind == "ring":
        ro, ri = (
            c.p("outer_radius", default=0.021),
            c.p("inner_radius", default=0.01775),
        )
        h = c.p("height", default=0.03)
        ring = tube(ro, ri, h, 64, z0=-h / 2)
        bracket = merge(
            [
                rbox((0.0027, 0.024, 0.035), 0.001).at((0.04825, 0, 0.0075)),
                rbox((0.026, 0.005, 0.004), 0.001).at((0.034, 0, 0.013)),
            ]
        )
        pins = merge(
            cylinder(0.006, 0.0045, 24, z0=0.0).at(
                (0, s * (ro - 0.0005), 0), rot_to([0, s, 0])
            )
            for s in (1, -1)
        )
        return {"part": merge([ring, bracket]), "boss": pins}
    bore = c.p("bore", default=0.0295)
    ro, h = c.p("outer_radius", default=0.017), c.p("height", default=0.07)
    z0 = -0.04
    sleeve = merge(
        [tube(ro, bore / 2, h, 64, z0=z0), tube(ro + 0.002, bore / 2, 0.003, 64, z0=z0)]
    )
    horn = cylinder(0.0065, 0.0065, 24, z0=0.0).at(
        (ro - 0.0015, 0, 0), rot_to([1, 0, 0])
    )
    return {"part": sleeve, "boss": horn}
