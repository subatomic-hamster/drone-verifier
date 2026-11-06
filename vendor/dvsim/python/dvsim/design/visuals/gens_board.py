"""Generators for boards and small electronics: ESC / FC / power boards, SBC, GNSS puck, camera module, sensor module,
enclosed box devices, connectors. Canonical frame: centred in x (length) and y (width), PCB underside at z = 0."""

import math
import re

import numpy as np

from . import gen
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
)

PI = math.pi


def _dims(c, L=0.03, W=0.03, H=0.006):
    b = c.bounds()
    ext = (b[1] - b[0]) if b is not None else None
    lw = [
        c.p("length", "geom_length", "size_length", "board_length"),
        c.p("width", "geom_width", "size_width", "board_width"),
        c.p("height", "geom_height", "size_height", "board_height"),
    ]
    if lw[0] is None and c.p("board_size_xy"):
        lw[0] = lw[1] = c.p("board_size_xy")
    d = [
        lw[i]
        if lw[i] is not None
        else (float(ext[i]) if ext is not None else (L, W, H)[i])
        for i in range(3)
    ]
    return d


def _holes(c, L, W):
    """Mounting hole centres (x, y, r) from the part's bolt-pattern ports (`bolt-pattern.AxB.mN`) in the board plane."""
    out = []
    for p in c.part.ports:
        m = re.match(r"bolt-pattern\.([\d.]+)x([\d.]+)\.m(\d)", p.get("type", ""))
        if (
            not m
            or p.get("kind") != "mech"
            or abs(abs((p.get("axis") or [0, 0, 0])[2]) - 1) > 1e-6
        ):
            continue
        a, b, th = float(m[1]) / 1000, float(m[2]) / 1000, int(m[3])
        ox, oy = (p.get("pos") or [0, 0, 0])[:2]
        for sx in (-1, 1):
            for sy in (-1, 1):
                h = (
                    round(ox + sx * a / 2, 6),
                    round(oy + sy * b / 2, 6),
                    0.00055 * th + 0.0003,
                )
                if abs(h[0]) < L / 2 and abs(h[1]) < W / 2 and h not in out:
                    out.append(h)
    return out


def _pcb(L, W, T, holes, corner=0.0015):
    return extrude(
        rrect(L, W, corner, 4), 0, T, [circle(r, 14, (x, y)) for x, y, r in holes]
    )


def _scatter(rng, L, W, holes, specs, margin=0.0012):
    """specs = [(sx, sy, h, count)] -> list of (x, y, sx, sy, h) rectangles that overlap neither each other nor a hole keep-out."""
    placed, out = [], []
    keep = [(x, y, r + 0.0025) for x, y, r in holes]
    for sx, sy, h, n in specs:
        tries = 0
        k = 0
        while k < n and tries < 400:
            tries += 1
            x = rng.uniform(-L / 2 + sx / 2 + margin, L / 2 - sx / 2 - margin)
            y = rng.uniform(-W / 2 + sy / 2 + margin, W / 2 - sy / 2 - margin)
            if any(
                math.hypot(max(abs(x - hx) - sx / 2, 0), max(abs(y - hy) - sy / 2, 0))
                < r
                for hx, hy, r in keep
            ):
                continue
            if any(
                abs(x - px) < (sx + psx) / 2 + 0.0004
                and abs(y - py) < (sy + psy) / 2 + 0.0004
                for px, py, psx, psy, _ in placed
            ):
                continue
            placed.append((x, y, sx, sy, h))
            out.append((x, y, sx, sy, h))
            k += 1
    return out


def _board(
    c, kind, mask, conns=(), caps=(), big=0, heat=False, pads=0, usb=False, label=False
):
    L, W, H = _dims(c)
    T = min(0.0016, 0.5 * H) if H > 0 else 0.0016
    holes = _holes(c, L, W)
    rng = c.rng()
    out = {
        "pcb": _pcb(L, W, T, holes),
        "chip": Mesh(),
        "cap": Mesh(),
        "conn": Mesh(),
        "pin": Mesh(),
        "metal": Mesh(),
        "sink": Mesh(),
    }
    sc = min(1.0, 0.6 * L / 0.03 + 0.4) if L < 0.03 else 1.0
    hmax = max(H - T, 0.001)
    specs = []
    if kind in ("fc", "esc", "generic", "power"):
        specs += [
            (0.007 * sc, 0.007 * sc, min(0.0012, hmax), 1 if kind != "power" else 0),
            (0.005 * sc, 0.004 * sc, min(0.001, hmax), 3 if kind != "power" else 1),
            (0.003 * sc, 0.003 * sc, min(0.0008, hmax), 6 if L > 0.02 else 3),
        ]
    if kind == "esc":
        specs += [(0.0055 * sc, 0.0065 * sc, min(0.0014, hmax), 8)]  # MOSFET packages
    chips = _scatter(rng, L, W, holes, specs)
    for x, y, sx, sy, h in chips:
        out["chip"] = out["chip"] + rbox([sx, sy, h], 0.0002, 1).at([x, y, T + h / 2])
    capspecs = (
        [(0.0016, 0.0012, min(0.0009, hmax), cc) for cc in caps]
        if caps
        else [(0.0016, 0.0009, 0.0008, 10 if L > 0.02 else 4)]
    )
    for x, y, sx, sy, h in _scatter(
        rng, L, W, holes + [(a, b, 0.002) for a, b, *_ in chips], capspecs
    ):
        out["cap"] = out["cap"] + rbox([sx, sy, h], 0.0001, 1).at([x, y, T + h / 2])
    big_caps = (
        _scatter(
            rng,
            L,
            W,
            holes,
            [(0.0075 * sc + 0.002, 0.0075 * sc + 0.002, min(hmax, 0.0105), big)],
        )
        if big
        else []
    )
    for x, y, sx, sy, h in big_caps:
        out["metal"] = out["metal"] + cylinder(sx / 2, h, 24, T).at([x, y, 0])
    # connectors along the long edges
    for edge, w_, d_, h_, n in conns:
        h_ = min(h_, max(hmax, 0.002))
        for i in range(n):
            u = (i + 0.5) / n - 0.5
            if edge in "+x-x":
                pass
            ex = (
                (L / 2 - d_ / 2 - 0.0004) * (1 if edge == "+x" else -1)
                if edge in ("+x", "-x")
                else (u * (L - 0.012)) * 0.9
            )
            ey = (
                (W / 2 - d_ / 2 - 0.0004) * (1 if edge == "+y" else -1)
                if edge in ("+y", "-y")
                else (u * (W - 0.012)) * 0.9
            )
            sx_, sy_ = (d_, w_) if edge in ("+x", "-x") else (w_, d_)
            out["conn"] = out["conn"] + rbox([sx_, sy_, h_], 0.0003, 1).at(
                [ex, ey, T + h_ / 2]
            )
    if usb:
        out["metal"] = out["metal"] + rbox([0.0073, 0.0088, 0.0032], 0.0006, 2).at(
            [L / 2 - 0.0036, 0, T + 0.0016]
        )
    if heat:
        hs = min(0.45 * min(L, W), 0.02)
        hh = min(hmax, 0.006)
        sink = rbox([hs, hs, 0.0012], 0.0003, 1).at([0, 0, T + hh - 0.0006])
        fins = (
            [
                rbox([hs, 0.0007, hh - 0.0012], 0.0001, 1).at(
                    [0, (k - 3) * hs / 7, T + (hh - 0.0012) / 2]
                )
                for k in range(7)
            ]
            if hs > 0.012
            else []
        )
        out["sink"] = merge([sink] + fins)
    for k in range(pads):  # gold solder pads
        a = 2 * PI * k / max(pads, 1)
        out["pin"] = out["pin"] + rbox([0.0045, 0.003, 0.0004], 0.0001, 1).at(
            [
                (L / 2 - 0.004)
                * math.copysign(1, math.cos(a) + 1e-9)
                * (1 if k % 2 else 0.6),
                (W / 2 - 0.0035) * (1 if k < pads / 2 else -1),
                T + 0.0002,
            ]
        )
    if label:
        out["label"] = rbox([0.5 * L, 0.5 * W, 0.0004], 0.0001, 1).at(
            [0, 0, T + 0.0002 + hmax * 0.0]
        )
    return out


_BOARD_SLOTS = dict(
    pcb="soldermask-black",
    chip="ic-black",
    cap="brass",
    conn="plastic-cream",
    pin="gold",
    metal="steel",
    sink="alu-brushed",
    label="label-white",
)


@gen("board_generic", **dict(_BOARD_SLOTS, pcb="soldermask-green"))
def board_generic(c, big=0, **_):
    return _board(c, "generic", None, big=big, conns=(("+y", 0.005, 0.004, 0.0035, 2),))


@gen("board_esc", **dict(_BOARD_SLOTS, pcb="soldermask-black"))
def board_esc(c, **_):
    """ESC: 4-in-1 stack board (large caps, MOSFET rows, gold motor pads) or a single small ESC."""
    ch = int(c.p("channels", default=1))
    L, W, H = _dims(c)
    if ch >= 4 or L >= 0.028:
        return _board(
            c, "esc", None, big=2, pads=12, conns=(("+y", 0.008, 0.004, 0.0035, 1),)
        )
    return _board(c, "esc", None, pads=6)


@gen("board_fc", **dict(_BOARD_SLOTS, pcb="soldermask-black"))
def board_fc(c, case=False, **_):
    """Flight controller: stack board with MCU, USB-C and JST-SH sockets; case=True = enclosed Pixhawk-style box."""
    if case:
        L, W, H = _dims(c, 0.0848, 0.044, 0.0124)
        xs = 0.05 * L  # the connector end is a lower step (Pixhawk 6C case)
        hl = 0.62 * H
        upper = rbox([xs + L / 2, W, H], 0.0015, 2).at([(xs - L / 2) / 2, 0, H / 2])
        lower = rbox([L / 2 - xs, W, hl], 0.0015, 2).at([(xs + L / 2) / 2, 0, hl / 2])
        lab = rbox([0.4 * L, 0.5 * W, 0.0004], 0.0001, 1).at([-0.2 * L, 0, H + 0.0001])
        cw = (L / 2 - xs) / 3.3
        conn = merge(
            [
                rbox([cw * 0.7, 0.0105 if k < 2 else 0.0165, 0.0032], 0.0003, 1).at(
                    [xs + cw * (0.9 + 1.1 * r), (k - 1) * 0.0125, hl + 0.0016]
                )
                for r in range(3)
                for k in range(3)
            ]
        )
        return {"pcb": merge([upper, lower]), "label": lab, "conn": conn}
    return _board(
        c,
        "fc",
        None,
        conns=(("-y", 0.0058, 0.0035, 0.0035, 2), ("+y", 0.0058, 0.0035, 0.0035, 2)),
        usb=True,
        pads=0,
    )


@gen("board_power", **dict(_BOARD_SLOTS, pcb="soldermask-black"))
def board_power(c, **_):
    return _board(
        c,
        "power",
        None,
        big=1 if (c.bounds() or _dims(c)[0] > 0.02) else 0,
        pads=8,
        conns=(("-x", 0.0105, 0.0085, 0.0085, 1),),
    )


@gen("board_sbc", **dict(_BOARD_SLOTS, pcb="soldermask-green"))
def board_sbc(c, **_):
    """Single-board computer (Raspberry Pi form factor): SoC lid, USB / Ethernet stacks on one end, GPIO header."""
    L, W, H = _dims(c, 0.085, 0.056, 0.017)
    T = 0.0016
    holes = (
        [
            (x, y, 0.00135)
            for x in (-L / 2 + 0.0035 + 0.0, -L / 2 + 0.0035 + 0.058)
            for y in (-0.0245, 0.0245)
        ]
        if W > 0.05 and L > 0.07
        else []
    )
    out = {
        "pcb": _pcb(L, W, T, holes, 0.003),
        "chip": Mesh(),
        "cap": Mesh(),
        "conn": Mesh(),
        "pin": Mesh(),
        "metal": Mesh(),
        "sink": Mesh(),
        "label": Mesh(),
    }
    out["metal"] = rbox([0.014, 0.014, 0.0025], 0.0005, 1).at(
        [-L / 2 + 0.0295, 0.0, T + 0.00125]
    )  # SoC lid
    for k, (w_, d_, h_) in enumerate(
        ((0.0135, 0.0175, 0.0155), (0.0135, 0.0175, 0.0155), (0.0155, 0.021, 0.0135))
    ):
        out["metal"] = out["metal"] + rbox([d_, w_, h_], 0.0004, 1).at(
            [
                L / 2 - d_ / 2 + 0.0008,
                W / 2 - 0.0085 - k * 0.0195 + (0.0025 if k == 2 else 0),
                T + h_ / 2,
            ]
        )
    out["conn"] = rbox([0.051, 0.005, 0.0085], 0.0003, 1).at(
        [0.0, W / 2 - 0.0035, T + 0.00425]
    )  # GPIO header
    out["pin"] = merge(
        [
            rbox([0.0006, 0.0006, 0.0035], 0, 0).at(
                [
                    -0.0245 + 0.00254 * i,
                    W / 2 - 0.0035 + 0.00127 * (j * 2 - 1),
                    T + 0.0085 + 0.00175,
                ]
            )
            for i in range(20)
            for j in (0, 1)
        ]
    )
    out["chip"] = rbox([0.01, 0.01, 0.001], 0.0002, 1).at(
        [-L / 2 + 0.014, -0.012, T + 0.0005]
    ) + rbox([0.0065, 0.0065, 0.001], 0.0002, 1).at([-L / 2 + 0.015, 0.012, T + 0.0005])
    return out


@gen(
    "gnss_puck",
    base="plastic-black",
    ring="plastic-black",
    top="plastic-white",
)
def gnss_puck(c, **_):
    """Round GNSS module: lipped base, bevelled cap, patch-antenna top."""
    b = c.bounds()
    if c.p("diameter", "diameter_v1") is None and c.p("length") and c.p("width"):
        # square module (Matek M9N / M10Q-5883): small board under a ceramic patch antenna
        L, W, H = c.p("length"), c.p("width"), c.p("height") or 0.01
        a = c.p("antenna_size_xy") or 0.8 * min(L, W)
        ta = c.p("antenna_thickness") or 0.004
        tb = max(0.0016, H - ta)
        return {
            "base": _pcb(L, W, tb, [], 0.001),
            "ring": Mesh(),
            "top": rbox([a, a, ta], 0.0004, 1).at([0, 0, tb + ta / 2]),
        }
    d = c.p("diameter", "diameter_v1") or (
        2 * min(b[1][0], b[1][1]) if b is not None else 0.04
    )
    h = c.p("height", "height_v1") or (b[1][2] - b[0][2] if b is not None else 0.012)
    r = d / 2
    base = revolve(
        [
            (0, 0),
            (r, 0),
            (r, 0.25 * h),
            (0.97 * r, 0.62 * h),
            (0.85 * r, 0.9 * h),
            (0.62 * r, h),
            (0, h),
        ],
        64,
    )
    ring = Mesh()
    arrow = np.array(
        [(0.5 * r, 0), (-0.25 * r, 0.2 * r), (-0.1 * r, 0), (-0.25 * r, -0.2 * r)]
    )
    top = extrude(arrow, h - 0.0001, h + 0.0003)
    return {"base": base, "ring": ring, "top": top}


@gen("box_device", body="plastic-black", label="label-white", conn="steel")
def box_device(c, **_):
    """Enclosed device (PSU brick, DC-DC): rounded box with a label and a barrel / terminal at one end."""
    L, W, H = _dims(c, 0.05, 0.03, 0.02)
    body = rbox([L, W, H], min(0.002, 0.12 * min(L, W, H)), 2).at([0, 0, H / 2])
    lab = rbox([0.6 * L, 0.7 * W, 0.0005], 0.0001, 1).at([0, 0, H + 0.00005])
    return {
        "body": body,
        "label": lab,
        "conn": cylinder(0.0028, 0.006)
        .at([0, 0, 0], R=rot([0, 1, 0], PI / 2))
        .at([L / 2, 0, H / 2]),
    }
