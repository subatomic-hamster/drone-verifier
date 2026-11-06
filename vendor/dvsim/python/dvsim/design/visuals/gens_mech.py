"""Generators for mechanical parts: bus servos, gearmotors, steppers, QDD actuators, wheels, bearings, standoffs, screws,
pulleys, collars, plates and a geometry-following fallback. Canonical frames are documented per generator."""

import math

import numpy as np

from . import gen
from .gens_board import _dims, _holes
from .mesh import (
    Mesh,
    box,
    circle,
    cylinder,
    extrude,
    merge,
    rbox,
    revolve,
    rot,
    rrect,
    tube,
)

PI = math.pi
Y90 = rot([0, 1, 0], PI / 2)


def _hex(af):
    r = af / math.sqrt(3)
    return np.array(
        [
            (r * math.cos(PI / 6 + k * PI / 3), r * math.sin(PI / 6 + k * PI / 3))
            for k in range(6)
        ]
    )


@gen(
    "servo_case",
    body="plastic-black",
    horn="alu-brushed",
    screw="steel",
    port="plastic-white",
    cover="plastic-dark",
)
def servo_case(c, **_):
    """Bus servo (STS3215 / Dynamixel style): body block with mounting ears, horn disc + screws on top at one end, cable
    ports on the side. Frame: centred in x, y; bottom z = 0; length along x, width along y, horn on +z at x = +L/2 - W/2."""
    L = c.p("length", "depth") or 0.045
    W = c.p("width") or 0.025
    H = c.p("height") or 0.035
    if L < W:
        L, W = W, L
    body = rbox([L - 0.008 if L > 0.03 else L, W, H], 0.0015, 2).at([0, 0, H / 2])
    ear_t = 0.0024
    ears = (
        extrude(
            rrect(L + 0.0, W, 0.002, 4),
            0,
            ear_t,
            [
                circle(0.0011, 12, (sx * (L / 2 - 0.0025), sy * W * 0.28))
                for sx in (-1, 1)
                for sy in (-1, 1)
            ],
        )
        if L > 0.03
        else Mesh()
    )
    hx = L / 2 - W / 2 if L > W * 1.2 else 0.0
    hz = H
    horn = merge(
        [
            cylinder(W * 0.36, 0.0035, 48, hz),
            cylinder(W * 0.17, 0.0013, 36, hz + 0.0035),
        ]
    ).at([hx, 0, 0])
    scr = merge(
        [
            cylinder(0.0008, 0.0008, 12, hz + 0.0035).at(
                [hx + 0.0075 * math.cos(k * PI / 2), 0.0075 * math.sin(k * PI / 2), 0]
            )
            for k in range(4)
        ]
    )
    ports = merge(
        [
            rbox([0.0035, 0.0085, 0.0045], 0.0003, 1).at(
                [-L / 2 + 0.0015 + 0.004, s * 0.0, H * 0.55]
            )
            for s in (0,)
        ]
    )
    cover = rbox(
        [L - 0.012 if L > 0.03 else L - 0.004, W - 0.004, 0.0006], 0.0002, 1
    ).at([-0.002 if L > 0.03 else 0, 0, H + 0.0003])
    return {
        "body": merge([body, ears]),
        "horn": horn,
        "screw": scr,
        "port": ports,
        "cover": cover,
    }


@gen(
    "gearmotor",
    gearbox="alu-brushed",
    can="alu-brushed",
    cap="plastic-black",
    shaft="steel",
    pcb="soldermask-green",
    magnet="steel",
    bolt="steel-dark",
)
def gearmotor(c, **_):
    """Brushed DC gearmotor with metal gearbox and rear encoder (Pololu 37D). Frame: mounting face at z = 0, body towards
    -z, output shaft towards +z."""
    d = c.p("body_diameter", "diameter")
    Lb = c.p("body_length", "length") or 0.068
    sd = c.p("shaft_diameter") or 0.006
    sl = c.p("shaft_length") or 0.0156
    if d is None and c.p(
        "body_width"
    ):  # rectangular micro metal gearmotor: steel gearbox block, rounded can behind it
        w, h = c.p("body_width"), c.p("body_height") or c.p("body_width")
        lg = 0.42 * Lb
        gearbox = rbox([w, h, lg], 0.0004, 1).at([0, 0, -lg / 2])
        can = rbox([w * 0.97, h * 0.97, Lb - lg], 0.3 * min(w, h), 5).at(
            [0, 0, -lg - (Lb - lg) / 2]
        )
        shaft = revolve(
            [(0, -0.002), (sd / 2, -0.002), (sd / 2, sl), (0, sl)], 24
        )  # (the real shaft is D-shaped; round here)
        return {
            "gearbox": gearbox,
            "can": can,
            "cap": Mesh(),
            "pcb": Mesh(),
            "magnet": Mesh(),
            "shaft": shaft,
            "bolt": Mesh(),
        }
    d = d or 0.037
    r = d / 2
    lg = 0.40 * Lb  # gearbox length
    gearbox = merge(
        [
            revolve(
                [
                    (0, 0),
                    (r * 0.62 + 0.002, 0),
                    (r * 0.62 + 0.002, 0.0015),
                    (0, 0.0015),
                ],
                36,
            ).at([0, 0, 0.0]),  # pilot boss
            cylinder(r, lg, 48, -lg),
            tube(r + 0.0003, r - 0.0004, 0.0012, 48, -lg + 0.0),
        ]
    )
    can_r = r * 0.97
    enc = 0.0135 if Lb > 0.05 else 0.0
    can = cylinder(can_r, Lb - lg - enc, 48, -Lb + enc)
    cap = (
        cylinder(can_r * 1.0, enc, 48, -Lb) if enc else Mesh()
    )  # black encoder housing
    pcb = Mesh()
    magnet = Mesh()
    bolts = (
        merge(
            [
                cylinder(0.0013, 0.0003, 12, 0.0).at(
                    [
                        0.0158 * math.cos(2 * PI * k / 6 + 0.3),
                        0.0158 * math.sin(2 * PI * k / 6 + 0.3),
                        0,
                    ]
                )
                for k in range(6)
            ]
        )
        if r > 0.015
        else Mesh()
    )
    shaft = revolve(
        [
            (0, -lg * 0.2),
            (sd / 2, -lg * 0.2),
            (sd / 2, sl),
            (sd / 2 - 0.0004, sl),
            (0, sl),
        ],
        24,
    )
    can = merge([can, tube(can_r + 0.0002, can_r - 0.0006, 0.0008, 48, -lg - 0.0008)])
    return {
        "gearbox": gearbox,
        "can": can,
        "cap": cap,
        "pcb": pcb,
        "magnet": magnet,
        "shaft": shaft,
        "bolt": bolts,
    }


@gen("wheel_tyre", tyre="rubber", hub="plastic-dark", rib="rubber")
def wheel_tyre(c, **_):
    """Tyre with tread lugs on a hub with lightening holes and a bore. Frame: centred, axle along z."""
    d = c.p("diameter") or 0.09
    w = c.p("width") or 0.01
    R = d / 2
    rt = 0.82 * R
    lug = min(0.0015, 0.04 * R)
    tyre = revolve(
        [
            (rt, -w / 2),
            (R - lug - 0.001, -w / 2),
            (R - lug, -w / 2 + 0.001),
            (R - lug, w / 2 - 0.001),
            (R - lug - 0.001, w / 2),
            (rt, w / 2),
        ],
        72,
    )
    n = 96
    lugs = merge(
        [
            box([lug * 2, 0.5 * 2 * PI * R / n, w * 0.94])
            .at([R - lug, 0, 0])
            .at(R=rot([0, 0, 1], 2 * PI * k / n))
            for k in range(n)
        ]
    )
    bore = c.p("bore") or 0.006
    r1, r2, nsp = (
        0.27 * R,
        rt - 0.0035,
        6,
    )  # six spokes: the openings between them are sector holes

    def opening(k):
        a0, a1 = 2 * PI * (k + 0.17) / nsp, 2 * PI * (k + 0.83) / nsp
        a = np.linspace(a0, a1, 7)
        return np.vstack(
            [
                np.c_[r2 * np.cos(a), r2 * np.sin(a)],
                np.c_[r1 * np.cos(a[::-1]), r1 * np.sin(a[::-1])],
            ]
        )

    hub = merge(
        [
            extrude(
                circle(rt + 0.0004, 96),
                -0.3 * w,
                0.3 * w,
                [opening(k) for k in range(nsp)],
            ),
            tube(r1, bore / 2, 0.8 * w, 36, -0.4 * w),
        ]
    )
    return {"tyre": tyre, "rib": lugs, "hub": hub}


@gen(
    "stepper_nema",
    body="steel-dark",
    front="alu-brushed",
    back="alu-brushed",
    shaft="steel",
)
def stepper_nema(c, **_):
    """NEMA stepper: chamfered square stack with end plates, pilot boss and a D shaft. Frame: front face at z = 0, body
    towards -z, shaft towards +z."""
    f = c.p("flange_size") or 0.0423
    Lb = c.p("body_length") or 0.04
    sd = (
        c.p("shaft_diameter", "screw_diameter") or 0.005
    )  # lead-screw steppers: the screw is the shaft
    sl = c.p("shaft_length", "screw_length") or 0.022
    ch = 0.07 * f
    h = f / 2
    sq = np.array(
        [
            (h - ch, -h),
            (h, -h + ch),
            (h, h - ch),
            (h - ch, h),
            (-h + ch, h),
            (-h, h - ch),
            (-h, -h + ch),
            (-h + ch, -h),
        ]
    )
    pat = 0.737 * f
    holes = [
        circle(0.0016, 14, (sx * pat / 2, sy * pat / 2))
        for sx in (-1, 1)
        for sy in (-1, 1)
    ]
    tp = min(0.008, 0.2 * Lb)
    front = extrude(sq, -tp, 0, holes)
    back = extrude(sq, -Lb, -Lb + tp * 0.8, holes)
    stack = extrude(sq * 0.975, -Lb + tp * 0.8, -tp)
    boss = (
        merge([cylinder(0.011 * f / 0.0423, 0.002, 48), cylinder(0.0035, 0.0)])
        if False
        else cylinder(0.011 * f / 0.0423, 0.002, 48)
    )
    shaft = revolve([(0, -0.003), (sd / 2, -0.003), (sd / 2, sl), (0, sl)], 24)
    return {"body": stack, "front": merge([front, boss]), "back": back, "shaft": shaft}


@gen(
    "qdd_actuator", body="alu-anodised-black", flange="alu-brushed", cap="plastic-black"
)
def qdd_actuator(c, **_):
    """Quasi-direct-drive actuator (CubeMars AK): finned cylindrical housing, bolt-circle output flange, rear cable cap.
    Frame: output face at z = 0, body towards -z."""
    d = c.p("body_diameter") or 0.07
    L = c.p("body_length") or 0.04
    r = d / 2
    fl = min(0.006, 0.15 * L)
    fins = [(r, -fl - k * (L - fl - 0.006) / 6) for k in range(1, 6)]
    prof = [(0, -L), (r * 0.9, -L), (r * 0.9, -L + 0.004)]
    z = -L + 0.004
    while z < -fl - 0.002:
        prof += [
            (r, z),
            (r, z + 0.0032),
            (r * 0.93, z + 0.0032),
            (r * 0.93, z + 0.0045),
        ]
        z += 0.0045
    prof += [(r, -fl), (r, 0), (0, 0)]
    prof = [(a, b) for a, b in prof]
    body = revolve(prof, 64)
    bc = 0.62 * r
    flange = extrude(
        circle(0.82 * r, 64),
        -0.0012,
        0.0016,
        [
            circle(
                0.0016,
                12,
                (bc * math.cos(2 * PI * k / 6), bc * math.sin(2 * PI * k / 6)),
            )
            for k in range(6)
        ]
        + [circle(0.22 * r, 32)],
    )
    cap = merge(
        [
            cylinder(0.34 * r, 0.0035, 36, -L - 0.0035),
            rbox([0.012, 0.006, 0.004], 0.0005, 1).at([0.0, r * 0.55, -L - 0.002]),
        ]
    )
    return {"body": body, "flange": flange, "cap": cap}


@gen("bearing", ring="steel", shield="steel-dark")
def bearing(c, **_):
    """Deep-groove ball bearing as two rings and shields. Frame: centred, axis z."""
    d = c.p("bore_diameter") or 0.006
    D = c.p("outer_diameter") or 0.019
    w = c.p("width") or 0.006
    t = (D - d) / 2
    ro_i, ri_o = d / 2 + 0.28 * t, D / 2 - 0.28 * t
    outer = revolve(
        [(ri_o, -w / 2), (D / 2, -w / 2), (D / 2, w / 2), (ri_o, w / 2)], 48
    )
    inner = revolve(
        [(d / 2, -w / 2), (ro_i, -w / 2), (ro_i, w / 2), (d / 2, w / 2)], 48
    )
    sh = merge(
        [
            tube(
                ri_o + 0.0001,
                ro_i - 0.0001,
                0.0003,
                48,
                s * (w / 2 - 0.00025) - 0.00015,
            )
            for s in (-1, 1)
        ]
    )
    return {"ring": merge([outer, inner]), "shield": sh}


@gen("standoff", body="brass", stud="steel")
def standoff(c, **_):
    """Hex standoff, with a threaded stud on one end when the part has stud_length. Frame: base at z = 0, along +z."""
    af = c.p("across_flats") or 0.005
    L = c.p("body_length", "overall_length", "length") or 0.01
    td = c.p("thread_diameter") or 0.003
    body = extrude(
        _hex(af), 0, L, [circle(td / 2, 16)] if not c.p("stud_length") else []
    )
    stud = c.p("stud_length")
    out = {"body": body}
    if stud:
        out["stud"] = cylinder(td / 2, stud, 16, L)
    return out


@gen("screw", head="steel", shank="steel", nut="steel")
def screw(c, **_):
    """Socket-head cap screw (head down to z = 0 face, shank towards -z), or a hex nut when only across_flats is given."""
    td = c.p("thread_diameter") or 0.003
    af = c.p("across_flats", "width_across_flats")
    L = c.p("length")
    fd = c.p("flange_diameter")
    if (af or fd) and not L:
        h = c.p("height", "height_max") or 0.8 * td
        nut = extrude(_hex(af or 1.75 * td), 0, h, [circle(td / 2, 16)])
        if fd:  # flange nut: round flange under the hex
            tf = 0.25 * h
            nut = merge([nut, extrude(circle(fd / 2, 40), 0, tf, [circle(td / 2, 16)])])
        return {"nut": nut}
    L = L or 4 * td
    hd = c.p("head_diameter") or 1.6 * td
    hh = c.p("head_height") or td
    head = revolve(
        [
            (0, 0),
            (hd / 2, 0),
            (hd / 2, hh * 0.9),
            (hd / 2 - 0.1 * hd, hh),
            (0.28 * hd, hh),
            (0.28 * hd, hh * 0.45),
            (0, hh * 0.45),
        ],
        32,
    )
    return {"head": head, "shank": cylinder(td / 2, L, 20, -L)}


@gen("diabolo_pellet", body="steel")
def diabolo_pellet(c, **_):
    """Diabolo airgun pellet: domed head, narrow waist, flared hollow skirt. Proportions of a .177 (4.5 mm) pellet scaled by the
    head diameter (the part publishes only the head size). Frame: skirt base at z = 0, nose towards +z."""
    d = c.p("head_diameter_max", "head_diameter_min", "caliber") or 0.00452
    k = d / 0.0045
    r = d / 2
    ri = 0.36 * d  # skirt cavity
    prof = [
        (ri, 0),
        (1.02 * r, 0),
        (1.0 * r, 0.0014 * k),
        (0.82 * r, 0.0020 * k),
        (0.78 * r, 0.0027 * k),  # waist
        (1.0 * r, 0.0033 * k),
        (1.0 * r, 0.0044 * k),
        (0.78 * r, 0.0052 * k),  # domed nose
        (0.4 * r, 0.0056 * k),
        (0.0, 0.0057 * k),
        (0.0, 0.0030 * k),  # cavity apex
        (ri, 0.0006 * k),
    ]
    return {"body": revolve(prof, 36)}


@gen("heat_insert", body="brass")
def heat_insert(c, **_):
    """Threaded heat-set insert (CNC Kitchen / Ruthex): brass barrel with alternating knurl bands and a threaded bore. Frame: top
    face at z = 0, body towards -z."""
    od = c.p("outer_diameter") or 0.005
    cd = c.p("centre_diameter") or 0.92 * od
    L = c.p("length") or 1.5 * od
    hd = (
        0.62 * od
    )  # threaded bore (the part's hole_diameter is the hole to print, not the bore)
    ro, rc = od / 2, min(cd, od) / 2
    rr = [ro, rc, ro, rc, ro, rc, ro]  # bands from the top down
    nb = len(rr)
    prof = [(hd / 2, -L)]
    for i in range(nb - 1, -1, -1):
        z0, z1 = -L * (i + 1) / nb, -L * i / nb
        prof += [(rr[i], z0), (rr[i], z1)]
    prof.append((hd / 2, 0))
    return {"body": revolve(prof, 36)}


@gen("pulley", body="alu-brushed", flange="alu-brushed")
def pulley(c, **_):
    """Toothed pulley / hub: teeth band between two flanges around a bore. Frame: centred, axis z."""
    bore = c.p("bore_diameter", "bore") or 0.005
    D = (
        c.p("outer_diameter", "diameter", "tooth_tip_diameter", "pitch_diameter")
        or 0.03
    )
    w = c.p("length", "thickness") or 0.012
    fd = c.p("flange_diameter") or D + 0.004
    n = int(c.p("teeth", default=0))
    body = revolve(
        [
            (bore / 2, -w / 2),
            (D / 2 * 0.92, -w / 2),
            (D / 2 * 0.92, w / 2),
            (bore / 2, w / 2),
        ],
        48,
    )
    if n >= 8:
        t = [(D / 2 * (1 - 0.06 * (k % 2)), 0) for k in range(2 * n)]
    return {
        "body": body,
        "flange": merge(
            [
                tube(fd / 2, bore / 2, 0.0015, 48, -w / 2),
                tube(fd / 2, bore / 2, 0.0015, 48, w / 2 - 0.0015),
            ]
        )
        if fd > D
        else Mesh(),
    }


@gen("shaft_collar", body="steel-dark")
def shaft_collar(c, **_):
    bore = c.p("bore_diameter") or 0.006
    D = c.p("outer_diameter") or 0.016
    w = c.p("width") or 0.007
    return {
        "body": merge(
            [
                tube(D / 2, bore / 2, w, 40, -w / 2),
                cylinder(0.0014, D / 2 * 0.55, 12)
                .at([0, 0, 0], R=Y90)
                .at([D / 2 - 0.0005, 0, 0]),
            ]
        )
    }


@gen("hub_disc", body="alu-anodised-black")
def hub_disc(c, **_):
    """Universal mounting hub (Pololu): short disc with a centre bore and four threaded mounting holes in the face. Frame: shaft-side
    face at z = 0, mounting face at +z, axis z."""
    D = c.p("outer_diameter") or 0.019
    t = c.p("thickness", "width") or 0.005
    bore = c.p("bore_diameter") or 0.005
    holes = [
        circle(0.0016, 12, (0.31 * D * math.cos(a), 0.31 * D * math.sin(a)))
        for a in (0, PI / 2, PI, 3 * PI / 2)
    ]
    return {"body": extrude(circle(D / 2, 48), 0, t, [circle(bore / 2, 24)] + holes)}


@gen("plate_outline", plate="carbon-fibre")
def plate_outline(c, **_):
    """Flat plate from its geometry box (rounded corners, holes at the part's bolt-pattern ports). Frame: part frame."""
    b = c.bounds()
    if b is None:
        L, W, H = _dims(
            c,
            0.1,
            0.1,
            c.p("thickness", "thickness_actual", "thickness_nominal") or 0.002,
        )
        lo, hi = np.array([-L / 2, -W / 2, -H / 2]), np.array([L / 2, W / 2, H / 2])
    else:
        lo, hi = b
    ctr = (lo + hi) / 2
    L, W, T = hi - lo
    holes = [(x - ctr[0], y - ctr[1], r) for x, y, r in _holes(c, 10, 10)]
    holes = [
        h
        for h in holes
        if abs(h[0]) + h[2] < L / 2 - 0.0015 and abs(h[1]) + h[2] < W / 2 - 0.0015
    ]  # fully inside
    return {
        "plate": extrude(
            rrect(L, W, min(0.006, 0.12 * min(L, W)), 6),
            -T / 2,
            T / 2,
            [circle(r, 14, (x, y)) for x, y, r in holes],
        ).at(ctr)
    }


@gen("shapes", **{"*": "plastic-black"})
def shapes(c, **_):
    """Fallback: each [geometry] shape redrawn with soft edges (box -> rounded box, cylinder -> chamfered, sphere);
    slot s<k> = the k-th shape, so `colors = { s2 = ... }` paints one of them."""
    from ..build import shape_frame

    out = {}
    g = c.part.geometry
    for i, raw in enumerate(g.get("shapes") or ([g] if "shape" in g else [])):
        k, sz = raw.get("shape"), raw.get("size")
        s = dict(zip(("pos", "R"), shape_frame(raw, c.part.id)))
        if k == "box":
            m = rbox([2 * x for x in sz], min(0.0015, 0.3 * min(sz) * 2 * 0.5), 2)
        elif k == "cylinder":
            ch = min(0.0008, 0.2 * sz[0])
            m = revolve(
                [
                    (0, -sz[1]),
                    (sz[0] - ch, -sz[1]),
                    (sz[0], -sz[1] + ch),
                    (sz[0], sz[1] - ch),
                    (sz[0] - ch, sz[1]),
                    (0, sz[1]),
                ],
                40,
            )
        elif k == "sphere":
            from .mesh import sphere

            m = sphere(sz[0], 24)
        elif k == "mesh":
            from .stl import read_stl

            m = read_stl(c.root / raw["mesh"])
        else:
            continue
        out[f"s{i}"] = m.at(s["pos"], s["R"])
    return out
