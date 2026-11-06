"""Generators for drone parts: outrunner motor, propeller, LiPo / Li-ion pack, 18650 cell. Canonical frames (see parts/README.md):
motor mounting face at z = 0, shaft towards +z; prop hub underside at z = 0; pack centred in x, y, z from 0 up."""

import math

import numpy as np

from . import gen
from .mesh import (
    Mesh,
    circle,
    cylinder,
    extrude,
    loft,
    merge,
    rbox,
    revolve,
    rot,
    rrect,
    tube,
)

TAU = 2 * math.pi


def _sector(ro, ri, a0, a1, z0, z1, n=6):
    a = np.linspace(a0, a1, n)
    outer = np.c_[ro * np.cos(a), ro * np.sin(a)]
    inner = np.c_[ri * np.cos(a[::-1]), ri * np.sin(a[::-1])]
    return extrude(np.vstack([outer, inner]), z0, z1)


@gen(
    "bldc_outrunner",
    base="alu-anodised-black",
    bell="alu-anodised-grey",
    stator="steel-dark",
    copper="copper",
    magnet="steel",
    shaft="steel",
    adapter="alu-anodised-red",
)
def bldc_outrunner(c, slots=12, mount_a=None, mount_b=None, adapter=False, vents=True):
    """Vented-bell outrunner: cross base with mount holes, 12-tooth stator with copper windings, magnet ring, bell with
    wall windows and a vented top, shaft, optional prop adapter. Sizes: diameter, length_body, shaft_diameter, stator_*."""
    d = c.size("diameter", "body_diameter", axis=0, default=0.028)
    L = c.size("length_body", "body_length", axis=2, default=0.4 * d + 0.012)
    sd = c.p("shaft_diameter", default=0.005 if d < 0.03 else 0.006)
    R = d / 2
    stub = min(max(0.15 * L, 0.003), 0.012)
    top = L - stub  # bell top
    hb = max(0.0025, 0.13 * top)  # base thickness
    a = mount_a or c.p("mount_pattern_a") or 0.6 * d * 0.7
    b = mount_b or c.p("mount_pattern_b") or a
    ms = c.p("mount_circle")
    if ms and not c.p("mount_pattern_a"):
        a = b = ms / math.sqrt(2)
    hole_r = 0.0016
    rs = (c.p("stator_diameter") or 0.78 * d) / 2
    sh = c.p("stator_height") or 0.42 * top
    wall = max(0.0012, 0.06 * R)
    # base: four mounting arms + hub disc
    parts = [cylinder(0.42 * R, hb)]
    for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        hx, hy = sx * a / 2, sy * b / 2
        ln = math.hypot(hx, hy)
        ang = math.atan2(hy, hx)
        w = 0.0065
        arm = rrect(ln + 0.007, w, w / 2 - 1e-4, 6, centre=(ln / 2, 0))
        Rz = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
        arm = arm @ Rz.T
        hole = circle(hole_r, 12, (hx, hy))
        parts.append(extrude(arm, 0, hb, [hole]))
    base = merge(parts)
    # stator
    z_s = hb + 0.012 * top + 0.0005
    core, cu = [cylinder(0.55 * rs, sh, 36, z_s)], []
    tw = 0.85 * TAU * 0.62 * rs / slots / 1.6
    for k in range(slots):
        ang = TAU * k / slots
        Rz = rot([0, 0, 1], ang)
        tooth = rbox([0.5 * rs, tw, sh], 0.0004, 1).at(
            [0.55 * rs + 0.25 * rs, 0, z_s + sh / 2]
        )
        head = rbox([0.1 * rs, 1.7 * tw, sh], 0.0004, 1).at(
            [0.96 * rs, 0, z_s + sh / 2]
        )
        core += [tooth.at(R=Rz), head.at(R=Rz)]
        wnd = rbox([0.30 * rs, tw + 0.0016, sh + 0.0026], 0.0008, 2).at(
            [0.74 * rs, 0, z_s + sh / 2]
        )
        cu.append(wnd.at(R=Rz))
    # bell: bottom band, top band, pillars between windows, vented top plate
    z_b0, z_b1 = hb + 0.004, hb + 0.004 + 0.62 * (top - hb - 0.004)  # window span
    ro, ri = R, R - wall
    bell = [
        tube(ro, ri, z_b0 - (hb + 0.0035) + 0.0005, 48, hb + 0.0035),
        tube(ro, ri, top - z_b1, 48, z_b1),
    ]
    nwin = slots
    for k in range(nwin if vents else 0):
        a0 = TAU * (k + 0.16) / nwin
        a1 = TAU * (k + 0.84) / nwin
        bell.append(_sector(ro, ri, a0, a1, z_b0 - 0.0005, z_b1 + 0.0005, 3))
    if not vents:  # closed unibell: one smooth cup with a rounded shoulder
        bell = [
            revolve(
                [
                    (ri, hb + 0.0035),
                    (ro, hb + 0.0035),
                    (ro, top - 0.002),
                    (ro - 0.002, top),
                    (sd / 2 + 0.001, top),
                    (sd / 2 + 0.001, top - wall),
                    (ri, top - wall),
                ],
                64,
            )
        ]
    holes = [
        circle(
            0.13 * R,
            14,
            (
                0.58 * R * math.cos(TAU * k / 6 + 0.5),
                0.58 * R * math.sin(TAU * k / 6 + 0.5),
            ),
        )
        for k in range(6)
    ]
    plate = extrude(
        circle(ro, 48),
        top - wall,
        top,
        [h for h in holes if (0.58 * R) > 0.13 * R + 0.0004],
    )
    bell.append(plate)
    bell.append(
        revolve(
            [
                (0.30 * R, top),
                (0.30 * R, top + 0.0012),
                (sd / 2 + 0.0015, top + 0.0012),
                (sd / 2 + 0.0015, top),
            ],
            32,
        )
    )
    mag = tube(ri, ri - max(0.0010, 0.05 * R), sh * 1.02, 48, z_s - 0.0002)
    out = {
        "base": base,
        "bell": merge(bell),
        "stator": merge(core),
        "copper": merge(cu),
        "magnet": mag,
        "shaft": cylinder(sd / 2, L - 0.5 * hb, 24, 0.5 * hb),
    }
    if adapter:
        out["adapter"] = cylinder(0.26 * R, stub * 0.8, 36, top + 0.0012)
    return out


def _section(chord, thick, camber, n=12):
    """Closed airfoil ring (u along chord 0..1, w up), counter-clockwise seen from +x (u to the right, w up -> lower first)."""
    t = np.linspace(0, math.pi, n)
    x = (1 - np.cos(t)) / 2
    yt = (
        5
        * thick
        * (
            0.2969 * np.sqrt(x)
            - 0.126 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1036 * x**4
        )
    )
    yc = camber * 4 * x * (1 - x)
    low = np.c_[x, yc - yt]
    up = np.c_[x[::-1], (yc + yt)[::-1]]
    ring = np.vstack(
        [low, up[1:-1]]
    )  # LE ... TE (lower) then back along the upper surface
    return ring


@gen("propeller", blade="plastic-black", hub="plastic-dark")
def propeller(c, cw=True, chord=None, thick=0.07, hub_d=None):
    """Twisted blades: chord / pitch-angle along the radius from diameter + pitch (+ blade_count, hub_diameter)."""
    D = c.size("diameter", axis=0, default=0.25)
    if c.bounds() is not None and c.p("diameter") is None:
        D = max(D, 2 * c.bounds()[1][0])
    pitch = c.p("pitch", default=0.45 * D)
    nb = int(c.p("blade_count", default=2))
    hub_d = hub_d or c.p("hub_diameter") or max(0.0055, 0.09 * D)
    hub_t = c.p("hub_thickness") or max(0.003, 0.025 * D)
    R = D / 2
    rh = hub_d / 2 * 0.9
    cmax = chord or 0.115 * D * (2.0 / nb) ** 0.5
    nst = 14
    rings = []
    for i in range(nst):
        t = i / (nst - 1)
        r = rh + (R - rh) * t
        cr = (
            cmax
            * (
                0.55
                + 0.45 * math.sin(math.pi * min(1.0, (0.12 + 0.88 * t) ** 0.85) * 0.92)
            )
            * 0.85
        )
        if t > 0.9:
            cr *= math.sqrt(max(0.0, 1 - ((t - 0.9) / 0.1) ** 2)) * 0.9 + 0.1
        beta = math.atan2(pitch, TAU * r)
        beta = min(beta, math.radians(55))
        sec = _section(cr, thick, 0.04)
        u = (sec[:, 0] - 0.3) * cr
        w = sec[:, 1] * cr
        y = u * math.cos(beta) - w * math.sin(beta)
        z = u * math.sin(beta) + w * math.cos(beta)
        rings.append(np.c_[np.full(len(u), r), y, z + hub_t / 2])
    blade = loft(rings)
    parts = [blade.at(R=rot([0, 0, 1], TAU * k / nb)) for k in range(nb)]
    if not cw:
        parts = [Mesh(m.v * [1, -1, 1], m.f[:, ::-1]) for m in parts]
    hub = revolve(
        [
            (0, 0),
            (hub_d / 2, 0),
            (hub_d / 2, hub_t * 0.8),
            (hub_d * 0.32, hub_t),
            (0, hub_t),
        ],
        36,
    )
    return {"blade": merge(parts), "hub": hub}


@gen(
    "lipo_pack",
    body="shrink-blue",
    label="label-white",
    lead_red="wire-red",
    lead_black="wire-black",
    connector="plastic-yellow",
    balance="plastic-white",
    pins="gold",
)
def lipo_pack(c, connector="xt60", lead=0.03):
    """Rounded shrink-wrapped pack (x = length), label panel on top, power lead + XT connector and balance lead at the +x end."""
    L = c.size("length", axis=0, default=0.07)
    W = c.size("width", axis=1, default=0.03)
    H = c.size("height", axis=2, default=0.02)
    r = min(0.0035, 0.2 * min(W, H))
    body = rbox([L, W, H], r, 3).at([0, 0, H / 2])
    label = rbox([0.6 * L, 0.84 * W, 0.0007], 0.0002, 1).at([-0.04 * L, 0, H + 0.0001])
    ex = L / 2
    wr = 0.0017 if H > 0.012 else 0.0012
    yoff = 0.35 * min(W, 0.016)
    lead_r = (
        cylinder(wr, lead + 0.003)
        .at([0, 0, 0], R=rot([0, 1, 0], math.pi / 2))
        .at([ex - 0.003, yoff, H / 2])
    )
    lead_b = (
        cylinder(wr, lead + 0.003)
        .at([0, 0, 0], R=rot([0, 1, 0], math.pi / 2))
        .at([ex - 0.003, -yoff, H / 2])
    )
    cw, ch, cl = {
        "xt30": (0.0105, 0.0105, 0.0100),
        "xt60": (0.0158, 0.0083, 0.0156),
        "xt90": (0.0225, 0.0105, 0.0240),
    }.get(connector, (0.0158, 0.0083, 0.0156))
    if H < cw * 1.05 and connector != "xt30":
        pass
    conn = rbox([cl, cw, ch], 0.0008, 1).at([ex + lead + cl / 2, 0, H / 2])
    # XT shape: chamfer the housing via a pair of cut-corner wedge boxes is overkill; pin holes read as dark pads
    pins = [
        cylinder(0.0017, 0.0008).at(
            [ex + lead + cl, s * cw * 0.22, H / 2], R=rot([0, 1, 0], math.pi / 2)
        )
        for s in (-1, 1)
    ]
    bal = rbox([0.004, 0.0105, 0.0042], 0.0004, 1).at([ex - 0.004, 0, 0.0021 + 0.0005])
    return {
        "body": body,
        "label": label,
        "lead_red": lead_r,
        "lead_black": lead_b,
        "connector": conn,
        "pins": merge(pins),
        "balance": bal,
    }


@gen("cell_18650", body="shrink-blue", cap="steel", label="label-white")
def cell_18650(c, **_):
    d = c.size("diameter", axis=0, default=0.0185)
    L = c.size("length", axis=2, default=0.065)
    wrap = revolve(
        [
            (0, 0),
            (d / 2 - 0.0004, 0),
            (d / 2, 0.0004),
            (d / 2, L - 0.0004),
            (d / 2 - 0.0004, L),
            (0, L),
        ],
        36,
    )
    cap = cylinder(d * 0.22, 0.0012, 24, L)
    ring = tube(d / 2 - 0.0006, d * 0.30, 0.0004, 36, L - 0.0004)
    return {
        "body": wrap,
        "cap": merge([cap, ring]),
        "label": tube(d / 2 + 0.0001, d / 2 - 0.0004, L * 0.4, 36, L * 0.3),
    }


@gen("cf_tube", tube="carbon-fibre", end="carbon-fibre")
def cf_tube(c, **_):
    """Hollow carbon tube along z, centred on its length; od / length from the part, else the geometry cylinder."""
    od = c.p("outer_diameter", "cad_outer_diameter", "nominal_outer_diameter")
    ln = c.p("length", "cad_length")
    from ..build import shape_frame

    g = c.part.geometry
    s = (g.get("shapes") or [g])[0] if (g.get("shapes") or "shape" in g) else {}
    pos, R = (
        (shape_frame(s, c.part.id))
        if s.get("shape") == "cylinder"
        else ([0, 0, 0], None)
    )
    if s.get("shape") == "cylinder":
        od = od or 2 * s["size"][0]
        ln = ln or 2 * s["size"][1]
    od = od or 0.016
    ln = ln or 0.2
    ido = c.p("inner_diameter", "cad_inner_diameter") or od - 0.002
    return {"tube": tube(od / 2, ido / 2, ln, 36, -ln / 2).at(pos, R)}
