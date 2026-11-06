"""Generators for structure and tooling: aluminium extrusion profile, T-nuts, brackets and joining plates, GT2 belts, flanged
nuts, and a generic PCP-class air-rifle launcher. Frames are documented per generator."""

import math

import numpy as np

from . import gen
from .gens_mech import Y90
from .mesh import (
    Mesh,
    circle,
    cylinder,
    extrude,
    merge,
    rbox,
    revolve,
    rot,
    rrect,
    sphere,
    tube,
)

PI = math.pi
MM = 1e-3


def _rot2(pts, a):
    c, s = math.cos(a), math.sin(a)
    return [(c * x - s * y, s * x + c * y) for x, y in pts]


def _tslot_side(d, length, n, tile=20.0):
    """+x face of a T-slot profile (units mm) from y = -length/2 to +length/2: n slots at `tile` pitch, outer face at x = d.
    Slot geometry of the common 20-series T-slot 6: 6.2 mm opening, 1.8 mm lip, 11 mm wide cavity, 5.8 mm deep."""
    pts = [(d, -length / 2 + 0.4)]
    for i in range(n):
        y = -length / 2 + tile / 2 + tile * i
        pts += [
            (d, y - 3.1),
            (d - 1.8, y - 3.1),
            (d - 1.8, y - 5.5),
            (d - 2.9, y - 5.5),
            (d - 5.8, y - 2.8),
            (d - 5.8, y + 2.8),
            (d - 2.9, y + 5.5),
            (d - 1.8, y + 5.5),
            (d - 1.8, y + 3.1),
            (d, y + 3.1),
        ]
    pts.append((d, length / 2 - 0.4))
    return pts


@gen("extrusion_profile", body="alu-extrusion")
def extrusion_profile(c, **_):
    """T-slot aluminium extrusion (2020, 2040, 2060 ... one 20 mm tile per slot) with its true cross-section: 6.2 mm slot
    openings, T cavities, chamfered corners and a centre bore per tile. Frame: profile in x, y centred; length along +z from 0."""
    w = (c.p("section_width", "width") or 0.020) / MM
    h = (c.p("section_height", "height") or 0.020) / MM
    L = c.p("length") or 0.1
    nx, ny = max(1, round(w / 20)), max(1, round(h / 20))
    outline = []
    for ang, d, ls, n in (
        (0.0, w / 2, h, ny),
        (PI / 2, h / 2, w, nx),
        (PI, w / 2, h, ny),
        (3 * PI / 2, h / 2, w, nx),
    ):
        outline += _rot2(_tslot_side(d, ls, n), ang)
    bore = (c.p("center_hole_diameter") or 4.2 * MM) / MM / 2
    holes = [
        circle(bore, 20, (-w / 2 + 10 + 20 * i, -h / 2 + 10 + 20 * j))
        for i in range(nx)
        for j in range(ny)
    ]
    return {
        "body": extrude(
            np.asarray(outline) * MM, 0, L, [np.asarray(x) * MM for x in holes]
        )
    }


@gen("t_nut", body="steel")
def t_nut(c, **_):
    """T-nut for 20-series slots: slim / oval plate with a threaded hole, drop-in = taller block with a spring ball. Frame: base
    at z = 0, thread axis +z, length along x."""
    t = c.part.id.lower()
    td = c.p("thread_diameter") or 0.004
    dropin = "dropin" in t or "drop-in" in t
    L, W, H = (
        (0.0105, 0.0100, 0.0042)
        if dropin
        else (0.0125, 0.0070, 0.0030)
        if "oval" in t
        else (0.0105, 0.0062, 0.0028)
    )
    body = extrude(rrect(L, W, 0.0008, 3), 0, H, [circle(td / 2, 16)])
    ball = cylinder(0.0012, 0.0006, 16, H).at([0, W * 0.3, 0]) if dropin else Mesh()
    return {"body": merge([body, ball])}


def _arm(x0, x1, y0, y1, th, hd, sp):
    """Flat rectangular arm with 5.3 mm-class holes at `sp` pitch along its long direction, centred."""
    horiz = (x1 - x0) >= (y1 - y0)
    long_, wide = (x1 - x0, y1 - y0) if horiz else (y1 - y0, x1 - x0)
    n = int(round((long_ - wide) / sp)) + 1
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    pts = [
        ((k - (n - 1) / 2) * sp + cx, cy)
        if horiz
        else (cx, (k - (n - 1) / 2) * sp + cy)
        for k in range(n)
    ]
    return extrude(
        [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        0,
        th,
        [circle(hd, 18, p) for p in pts],
    )


@gen("bracket_plate", plate="alu-extrusion", rib="alu-extrusion")
def bracket_plate(c, **_):
    """Joining plates and brackets for 20-series extrusion, by name: coupling / L / T / cross plate (flat, 5.3 mm holes at 20 mm
    pitch), corner brace (two legs and a gusset), NEMA 17 L-bracket. Frame: plates on z = 0 .. t with the hole row along x through
    the origin; corner brace and L-bracket: foot on z = 0 .. t, upright at y = 0 .. t."""
    t = (c.part.id + " " + c.part.raw.get("name", "")).lower()
    th = c.p("thickness") or 0.004
    wd = c.p("width") or 0.018
    hd = (c.p("hole_diameter") or 0.0053) / 2
    sp = c.p("hole_spacing") or 0.02
    if "nema17" in t or "nema 17" in t or "l-bracket" in t:
        a, tt = 0.0423, 0.0032
        up = (
            extrude(
                rrect(a, a, 0.003, 4),
                0,
                tt,
                [circle(0.011, 40)]
                + [
                    circle(0.0017, 12, (sx * 0.0155, sy * 0.0155))
                    for sx in (-1, 1)
                    for sy in (-1, 1)
                ],
            )
            .at([0, 0, 0], R=rot([1, 0, 0], PI / 2))
            .at([0, tt, a / 2])
        )
        foot = extrude(
            rrect(a, 0.0225, 0.002, 3),
            0,
            tt,
            [circle(0.0022, 14, (sx * 0.0125, 0.0)) for sx in (-1, 1)],
        ).at([0, 0.0225 / 2, 0])
        return {"plate": merge([up, foot])}
    if "corner" in t:
        leg, tt, bw = 0.0195, 0.003, 0.0185
        per = max(1, int(c.p("hole_count", default=2)) // 2)
        hs = [(k - (per - 1) / 2) * 0.0095 for k in range(per)]
        holes = [circle(0.0027, 14, (x, 0.0)) for x in hs]
        foot = extrude(rrect(bw, leg, 0.001, 2), 0, tt, holes).at([0, leg / 2, 0])
        up = (
            extrude(rrect(bw, leg, 0.001, 2), 0, tt, holes)
            .at([0, 0, 0], R=rot([1, 0, 0], PI / 2))
            .at([0, tt, leg / 2])
        )
        g = 0.6 * leg
        gusset = extrude([(0, 0), (g, 0), (0, g)], -0.0012, 0.0012).at(
            [0, tt, tt], R=np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        )
        return {"plate": merge([foot, up]), "rib": gusset}
    hw = wd / 2
    L = c.p("length") or 0.06
    if "cross" in t:
        parts = [
            _arm(-L / 2, L / 2, -hw, hw, th, hd, sp),
            _arm(-hw, hw, -0.03, 0.03, th, hd, sp),
        ]
    elif "l-plate" in t:
        parts = [
            _arm(-hw, 2 * sp + hw, -hw, hw, th, hd, sp),
            _arm(-hw, hw, -hw, 2 * sp + hw, th, hd, sp),
        ]
    elif "t-plate" in t:
        parts = [
            _arm(-sp - hw, sp + hw, -hw, hw, th, hd, sp),
            _arm(-hw, hw, -2 * sp - hw, hw, th, hd, sp),
        ]
    else:
        parts = [_arm(-L / 2, L / 2, -hw, hw, th, hd, sp)]
    return {"plate": merge(parts)}


@gen("belt_loop", belt="rubber")
def belt_loop(c, **_):
    """Closed GT2 timing belt laid out as a stadium with 20 mm radius ends: smooth back outside, trapezoid teeth inside. Frame:
    loop in the x-y plane, centred; belt width along z."""
    Lp = c.p("pitch_length", "length") or 0.4
    pitch = c.p("pitch") or 0.002
    W = c.p("width") or 0.006
    R = min(0.02, Lp / (4 * PI))
    S = max((Lp - 2 * PI * R) / 2, 0.0)
    t, h = 0.0008, 0.0007
    per = 2 * S + 2 * PI * R
    n = max(int(round(per / pitch)), 8)

    def at(s, off):  # point at arclength s on the stadium centreline, `off` outwards
        s %= per
        if s < S:
            return (-S / 2 + s, -(R + off))
        s -= S
        if s < PI * R:
            a = -PI / 2 + s / R
            return (S / 2 + (R + off) * math.cos(a), (R + off) * math.sin(a))
        s -= PI * R
        if s < S:
            return (S / 2 - s, R + off)
        s -= S
        a = PI / 2 + s / R
        return (-S / 2 + (R + off) * math.cos(a), (R + off) * math.sin(a))

    outer = [at(k * per / (4 * n), t / 2) for k in range(4 * n)]
    inner = [
        at((i + u) * per / n, -t / 2 - o)
        for i in range(n)
        for u, o in ((0.0, 0.0), (0.3, h), (0.7, h))
    ]
    return {"belt": extrude(np.asarray(outer), -W / 2, W / 2, [np.asarray(inner)])}


@gen("flanged_nut", body="brass")
def flanged_nut(c, **_):
    """Flanged lead-screw / traveling nut: cylindrical barrel, round flange with four holes, threaded bore. Frame: flange face at
    z = 0, barrel towards +z."""
    d = c.p("screw_diameter") or 0.008
    r, rf, L, tf = 0.0085, 0.0125, 0.016, 0.004
    holes = [
        circle(
            0.0012,
            10,
            (
                0.0095 * math.cos(k * PI / 2 + PI / 4),
                0.0095 * math.sin(k * PI / 2 + PI / 4),
            ),
        )
        for k in range(4)
    ]
    flange = extrude(circle(rf, 48), 0, tf, [circle(d / 2, 24)] + holes)
    return {"body": merge([flange, tube(r, d / 2, L, 40, tf)])}


@gen(
    "pcp_launcher",
    receiver="alu-anodised-grey",
    reservoir="steel-dark",
    shroud="gunmetal",
    barrel="steel",
    rail="gunmetal",
    handle="steel",
    gauge="label-white",
    mount="alu-brushed",
    knob="plastic-black",
)
def pcp_launcher(c, **_):
    """Generic compact PCP-class air-rifle module (no maker's styling): receiver block with a Picatinny-style top rail, air
    reservoir bottle under the barrel with a pressure gauge on its front, a barrel shroud with a muzzle crown, a side bolt handle
    with a ball knob, and a clamp plate towards the pan-tilt horn. Frame = the part frame: tilt axis at the origin, barrel axis
    along +x at y = yc, z = 0, plate towards -y. Extents follow the part's [geometry] (receiver box + barrel)."""
    g = c.part.geometry.get("shapes") or [{}, {}]
    box, bar = g[0], g[1] if len(g) > 1 else {}
    bs, bp = box.get("size", [0.10, 0.025, 0.035]), box.get("pos", [-0.02, 0.025, 0.0])
    xr, yc = bp[0] - bs[0], bp[1]  # receiver box: rear face and axis line
    zt, zb = bp[2] + bs[2], bp[2] - bs[2]
    xm = bar.get("pos", [0.21])[0] + bar.get("size", [0.008, 0.13])[1]  # muzzle
    xs = xr + 0.14  # the shroud starts where the receiver ends

    def alongx(m, y=yc, z=0.0):
        return m.at([0, y, z], R=Y90)

    rec = rbox([0.14, 0.032, 0.042], 0.003, 3).at([xr + 0.07, yc, 0.007])
    rail_len = 0.155
    rail = rbox([rail_len, 0.0215, 0.0035], 0.0006, 1).at(
        [xr + 0.005 + rail_len / 2, yc, 0.028 + 0.00175]
    )  # base; the cross ridges are the Picatinny slot pitch (9.5 mm)
    rail_slots = merge(
        [
            rbox([0.0055, 0.0215, zt - 0.0315], 0.0004, 1).at(
                [xr + 0.012 + k * 0.0095, yc, (zt + 0.0315) / 2]
            )
            for k in range(int((rail_len - 0.012) / 0.0095))
        ]
    )
    rr = 0.0115
    zres = zb + rr
    reservoir = alongx(
        revolve(
            [
                (0, xr + 0.004),
                (rr * 0.8, xr + 0.004),
                (rr, xr + 0.012),
                (rr, xm - 0.19),
                (rr * 0.85, xm - 0.175),
                (0, xm - 0.175),
            ],
            48,
        ),
        yc,
        zres,
    )
    gauge = merge([alongx(cylinder(0.0085, 0.0045, 36, xm - 0.1795), yc, zres)])
    gauge_ring = alongx(tube(0.0095, 0.0078, 0.0035, 36, xm - 0.1795), yc, zres)
    ro, ri = 0.0125, 0.0045
    shroud = alongx(
        revolve(
            [
                (0, xs - 0.002),
                (ro, xs - 0.002),
                (ro, xm - 0.045),
                (0.0105, xm - 0.044),
                (0.0105, xm - 0.04),
                (0.0065, xm - 0.04),
                (0, xm - 0.04),
            ],
            48,
        )
    )
    barrel = merge(
        [
            alongx(cylinder(ri, 0.042, 24, xm - 0.04)),
            alongx(tube(ri + 0.0012, ri - 0.0015, 0.0012, 24, xm - 0.0012)),
        ]
    )
    stem = (
        cylinder(0.002, 0.022, 14, 0)
        .at([0, 0, 0], R=rot([1, 0, 0], -PI / 2))
        .at([xr + 0.07, yc + 0.012, 0.008])
    )
    ball = sphere(0.0055, 16, (xr + 0.07, yc + 0.037, 0.008))
    plate = rbox([0.07, 0.010, 0.05], 0.002, 2).at([-0.01, 0.005, 0.0])
    bolts = merge(
        [
            cylinder(0.0022, 0.0015, 14, 0)
            .at([0, 0, 0], R=rot([1, 0, 0], PI / 2))
            .at([x, 0.0, z])
            for x in (-0.03, 0.01)
            for z in (-0.018, 0.018)
        ]
    )
    return {
        "receiver": rec,
        "reservoir": merge([reservoir, gauge_ring]),
        "gauge": gauge,
        "shroud": shroud,
        "barrel": barrel,
        "rail": merge([rail, rail_slots]),
        "handle": stem,
        "knob": ball,
        "mount": merge([plate, bolts]),
    }
