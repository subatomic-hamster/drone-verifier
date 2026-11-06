"""Sensors and computers: single-board computers (Raspberry Pi 4 / 5 / Zero, Jetson module), camera module with a lens barrel,
rangefinder / lidar / flow modules, bare IC packages and breakout carriers. Frame: centred in x (length) and y (width), PCB
underside at z = 0, lens / sensor facing +z (the same convention as the board generators)."""

import math
import re

import numpy as np

from . import gen
from .gens_board import _dims, _pcb
from .mesh import Mesh, cylinder, merge, rbox, rot, tube

PI = math.pi
MM = 1e-3
T = 1.6 * MM  # PCB thickness
SLOTS = dict(
    pcb="soldermask-green",
    chip="ic-black",
    conn="plastic-cream",
    pin="gold",
    metal="steel",
    sink="alu-brushed",
    label="label-white",
    usb="plastic-blue",
    black="plastic-black",
    lens="plastic-black",
    glass="glass-dark",
    ring="steel",
)


def _blk(x, y, w, d, h, z0=T, rad=0.3 * MM):
    """Box at board position (x, y) (centre), size w (along x) x d (along y) x h, standing on z0."""
    return rbox([w, d, h], rad, 1).at([x, y, z0 + h / 2])


def _gpio(L, W, y, x=0.0):
    hdr = _blk(x, y, 51 * MM, 5 * MM, 8.5 * MM, T, 0.2 * MM)
    pins = merge(
        [
            rbox([0.6 * MM, 0.6 * MM, 3.5 * MM], 0, 0).at(
                [
                    x - 24.5 * MM + 2.54 * MM * i,
                    y + 1.27 * MM * (2 * j - 1),
                    T + 8.5 * MM + 1.75 * MM,
                ]
            )
            for i in range(20)
            for j in (0, 1)
        ]
    )
    return hdr, pins


def _pi_large(c, L, W, pi5):
    x0, y0 = -L / 2, -W / 2
    holes = [
        (x0 + 3.5 * MM + dx, y0 + 3.5 * MM + dy, 1.35 * MM)
        for dx in (0, 58 * MM)
        for dy in (0, 49 * MM)
    ]
    out = {
        k: Mesh()
        for k in (
            "pcb",
            "chip",
            "conn",
            "pin",
            "metal",
            "sink",
            "label",
            "usb",
            "black",
        )
    }
    out["pcb"] = _pcb(L, W, T, holes, 3 * MM)
    soc = 15 * MM if pi5 else 14 * MM
    out["metal"] = _blk(x0 + 29 * MM, 0.0, soc, soc, 2.4 * MM)  # SoC heat-spreader lid
    out["chip"] = _blk(x0 + 15 * MM, 4 * MM, 10 * MM, 10 * MM, 1.0 * MM) + _blk(
        x0 + 48 * MM, -4 * MM, 8 * MM, 8 * MM, 1.0 * MM
    )
    if pi5:
        out["chip"] = out["chip"] + _blk(
            x0 + 13 * MM, -12 * MM, 7 * MM, 7 * MM, 1.0 * MM
        )  # RP1 I/O chip
    # right end: Ethernet + two USB stacks
    stack = [
        (15.6 * MM, 21.5 * MM, 13.5 * MM, y0 + 10.25 * MM + 0.0),
        (13.1 * MM, 17.5 * MM, 15.6 * MM, y0 + 29.0 * MM),
        (13.1 * MM, 17.5 * MM, 15.6 * MM, y0 + 47.0 * MM),
    ]
    for k, (w_, d_, h_, yy) in enumerate(stack):
        out["metal"] = out["metal"] + _blk(L / 2 - d_ / 2 + 1.0 * MM, yy, d_, w_, h_)
        if k:
            out["usb"] = out["usb"] + _blk(
                L / 2 - 1.2 * MM,
                yy,
                0.6 * MM,
                w_ - 3 * MM,
                h_ - 3 * MM,
                T + 1.5 * MM,
                0.1 * MM,
            )
    # bottom edge: USB-C power, two micro-HDMI, audio jack (Pi 4) / power button side (Pi 5)
    out["metal"] = out["metal"] + _blk(
        x0 + 11.2 * MM,
        y0 + 3.65 * MM - 0.6 * MM,
        9.0 * MM,
        7.3 * MM,
        3.2 * MM,
        T,
        1.0 * MM,
    )
    for hx in (26.0, 39.5):
        out["metal"] = out["metal"] + _blk(
            x0 + hx * MM,
            y0 + 3.0 * MM - 0.5 * MM,
            6.6 * MM,
            6.0 * MM,
            3.1 * MM,
            T,
            0.5 * MM,
        )
    if not pi5:
        out["black"] = out["black"] + _blk(
            x0 + 53.5 * MM, y0 + 7.5 * MM, 7.0 * MM, 15.0 * MM, 5.6 * MM, T, 1.0 * MM
        )  # 3.5 mm AV jack
    out["conn"] = (
        _blk(x0 + 45.0 * MM, y0 + 3.5 * MM, 22 * MM, 2.6 * MM, 5.0 * MM, T, 0.2 * MM)
        if not pi5
        else Mesh()
    )
    if pi5:  # CSI / DSI FFC connectors, PCIe FFC, fan header, power button
        out["conn"] = _blk(
            x0 + 46.0 * MM, y0 + 3.0 * MM, 17 * MM, 2.8 * MM, 3.0 * MM, T, 0.2 * MM
        ) + _blk(x0 + 3.0 * MM, y0 + 28 * MM, 2.5 * MM, 12 * MM, 3.0 * MM, T, 0.2 * MM)
        out["black"] = out["black"] + _blk(
            x0 + 21.0 * MM, y0 + 4.0 * MM, 3.5 * MM, 3.5 * MM, 1.5 * MM, T, 0.2 * MM
        )
    hdr, pins = _gpio(L, W, y0 + W - 3.5 * MM, x0 + 3.5 * MM + 29 * MM)
    out["black"] = out["black"] + hdr
    out["pin"] = pins
    return out


def _pi_zero(c, L, W):
    x0, y0 = -L / 2, -W / 2
    holes = [
        (x0 + 3.5 * MM + dx, y0 + 3.5 * MM + dy, 1.35 * MM)
        for dx in (0, 58 * MM)
        for dy in (0, 23 * MM)
    ]
    out = {
        k: Mesh()
        for k in (
            "pcb",
            "chip",
            "conn",
            "pin",
            "metal",
            "sink",
            "label",
            "usb",
            "black",
        )
    }
    out["pcb"] = _pcb(L, W, T, holes, 3 * MM)
    out["metal"] = _blk(x0 + 33 * MM, 0.0, 13 * MM, 13 * MM, 1.4 * MM) + _blk(
        x0 + 16 * MM, 4 * MM, 10 * MM, 8 * MM, 1.2 * MM
    )  # SiP shield, RF shield
    out["metal"] = (
        out["metal"]
        + _blk(x0 + 12.4 * MM, y0 + 2.8 * MM - 0.4 * MM, 11.2 * MM, 5.6 * MM, 3.2 * MM)
        + _blk(x0 + 41.4 * MM, y0 + 2.8 * MM - 0.4 * MM, 8 * MM, 5.6 * MM, 2.6 * MM)
        + _blk(x0 + 54.0 * MM, y0 + 2.8 * MM - 0.4 * MM, 8 * MM, 5.6 * MM, 2.6 * MM)
    )
    out["metal"] = out["metal"] + _blk(
        x0 + 6 * MM, 0.0, 14 * MM, 12 * MM, 1.8 * MM, T
    )  # microSD slot
    out["conn"] = _blk(
        L / 2 - 2.5 * MM, 0.0, 3.5 * MM, 17 * MM, 2.0 * MM, T, 0.2 * MM
    )  # CSI
    out["pin"] = merge(
        [
            rbox([1.7 * MM, 1.7 * MM, 0.3 * MM], 0.2 * MM, 1).at(
                [
                    x0 + 7.5 * MM + 2.54 * MM * i,
                    y0 + W - 3.6 * MM + 2.54 * MM * (j - 0.5),
                    T + 0.15 * MM,
                ]
            )
            for i in range(20)
            for j in (0, 1)
        ]
    )
    return out


def _jetson(c, L, W):
    x0, y0 = -L / 2, -W / 2
    out = {
        k: Mesh()
        for k in (
            "pcb",
            "chip",
            "conn",
            "pin",
            "metal",
            "sink",
            "label",
            "usb",
            "black",
        )
    }
    out["pcb"] = _pcb(
        L,
        W,
        T,
        [
            (x0 + 2.5 * MM, y0 + 2.5 * MM, 1.1 * MM),
            (x0 + 2.5 * MM, y0 + W - 2.5 * MM, 1.1 * MM),
        ],
        1 * MM,
    )
    # SO-DIMM edge: gold fingers along the long edge at -y
    out["pin"] = merge(
        [
            rbox([0.3 * MM, 3.0 * MM, 0.14 * MM], 0, 0).at(
                [x0 + 6 * MM + 0.5 * MM * i, y0 + 1.5 * MM, T + 0.07 * MM]
            )
            for i in range(120)
        ]
    )
    out["chip"] = merge(
        [
            _blk(x0 + 10 * MM + 16 * MM * k, y0 + 33 * MM, 12 * MM, 9 * MM, 0.9 * MM)
            for k in range(3)
        ]
    )
    out["metal"] = _blk(
        x0 + 36 * MM, 0.0, 28 * MM, 28 * MM, 3.0 * MM
    )  # thermal transfer plate over the SoC
    fins = merge(
        [
            rbox([26 * MM, 0.8 * MM, 9 * MM], 0.1 * MM, 1).at(
                [x0 + 36 * MM, (k - 8.5) * 1.6 * MM, T + 3.0 * MM + 4.5 * MM]
            )
            for k in range(18)
        ]
    )
    out["sink"] = merge(
        [
            _blk(x0 + 36 * MM, 0.0, 30 * MM, 30 * MM, 1.5 * MM, T + 3.0 * MM, 0.3 * MM),
            fins,
        ]
    )
    out["label"] = _blk(x0 + 14 * MM, y0 + 12 * MM, 14 * MM, 7 * MM, 0.2 * MM, T, 0.0)
    return out


@gen("sbc_board", **dict(SLOTS, sink="alu-anodised-black"))
def sbc_board(c, **_):
    """Single-board computer from its footprint: Raspberry Pi 4 / 5 (85 x 56: USB-C, two micro-HDMI, Ethernet + two USB stacks,
    SoC lid, 40-pin header, CSI), Raspberry Pi Zero (65 x 30: mini-HDMI, two micro-USB, microSD, CSI) or a Jetson Orin Nano module
    (69.6 x 45: SO-DIMM fingers, thermal plate with a fin stack). Appearance: Raspberry Pi product photos; NVIDIA Jetson Orin Nano
    module datasheet drawing."""
    t = (c.part.id + c.part.raw.get("name", "")).lower()
    L, W, H = _dims(c, 0.085, 0.056, 0.017)
    if "jetson" in t:
        return _jetson(c, L, W)
    if L < 0.075:
        return _pi_zero(c, L, W)
    return _pi_large(c, L, W, pi5=bool(re.search(r"pi[- ]?5\b", t)))


@gen("sensor_ic", **SLOTS)
def sensor_ic(c, size=None, **_):
    """Bare sensor package (IMU, barometer, magnetometer: black body, pin-1 dot, gold lands), a GNSS module (metal shield on a small
    board) or, when the part is a carrier board (Pololu), a breakout PCB with the chip, decoupling parts and a 0.1 in header."""
    t = (c.part.id + c.part.raw.get("name", "")).lower()
    L = c.p("length", "package_length") or (size[0] if size else 3 * MM)
    W = c.p("width", "package_width") or (size[1] if size else 3 * MM)
    H = c.p("height", "package_height") or (size[2] if size else 1.0 * MM)
    if "gnss" in c.part.category or "neo-" in t:
        L, W, H = (
            (c.p("length") or 16 * MM),
            (c.p("width") or 12.2 * MM),
            (c.p("height") or 2.4 * MM),
        )
        pcb = _pcb(L, W, 0.8 * MM, [], 0.3 * MM)
        shield = rbox([L - 2 * MM, W - 2 * MM, H - 0.8 * MM], 0.2 * MM, 1).at(
            [0, 0, 0.8 * MM + (H - 0.8 * MM) / 2]
        )
        lands = merge(
            [
                rbox([1.0 * MM, 0.5 * MM, 0.1 * MM], 0, 0).at(
                    [x, s * (W / 2 - 0.4 * MM), 0.85 * MM]
                )
                for x in np.linspace(-L / 2 + 1.5 * MM, L / 2 - 1.5 * MM, 8)
                for s in (-1, 1)
            ]
        )
        return {"pcb": pcb, "metal": shield, "pin": lands}
    if L >= 0.012 or "carrier" in t or "pololu" in t:  # breakout board
        L, W = (c.p("length") or 0.02), (c.p("width") or 0.013)
        holes = (
            [
                (x * (L / 2 - 1.8 * MM), y * (W / 2 - 1.8 * MM), 0.7 * MM)
                for x in (-1, 1)
                for y in (-1, 1)
            ]
            if W > 0.011
            else []
        )
        pcb = _pcb(L, W, T, holes, 1 * MM)
        chip = _blk(L / 6, 0.0, 4 * MM, 4 * MM, 1.0 * MM) + _blk(
            -L / 6, 0.0, 3 * MM, 3 * MM, 0.9 * MM
        )
        pins = merge(
            [
                rbox([0.9 * MM, 0.9 * MM, 0.12 * MM], 0, 0).at(
                    [
                        -L / 2 + 2.0 * MM + 2.54 * MM * i,
                        -W / 2 + 1.5 * MM,
                        T + 0.06 * MM,
                    ]
                )
                for i in range(int((L - 4 * MM) / (2.54 * MM)))
            ]
        )
        return {"pcb": pcb, "chip": chip, "pin": pins}
    H = max(H, 0.5 * MM)
    body = rbox([L, W, H], 0.1 * MM, 1).at([0, 0, H / 2])
    dot = cylinder(0.25 * MM, 0.05 * MM, 12, H).at(
        [-L / 2 + 0.6 * MM, -W / 2 + 0.6 * MM, 0]
    )
    lands = merge(
        [
            rbox([0.4 * MM, 0.25 * MM, 0.1 * MM], 0, 0).at([x, s * (W / 2), 0.05 * MM])
            for x in np.linspace(-L / 2 + 0.5 * MM, L / 2 - 0.5 * MM, 5)
            for s in (-1, 1)
        ]
    )
    return {"chip": merge([body, dot]), "pin": lands}


@gen("sensor_module", **dict(SLOTS, body="plastic-black"))
def sensor_module(c, **_):
    """Rangefinder / lidar / optical-flow module: a ToF breakout carrier (Pololu VL53: small PCB, shiny sensor window, header
    row), a lidar housing (TF-Luna / TFmini-S: black shell, two round windows, mounting ears) or an optical-flow board with
    a lens. Appearance: Benewake and Pololu product photos."""
    t = (c.part.id + c.part.raw.get("name", "")).lower()
    L = c.p("length") or 0.04
    W = c.p("width") or 0.02
    H = c.p("height") or 0.015
    if "vl53" in t or "carrier" in t:
        holes = [
            (x * (L / 2 - 1.8 * MM), y * (W / 2 - 1.8 * MM), 0.7 * MM)
            for x in (-1, 1)
            for y in (-1, 1)
        ]
        pcb = _pcb(L, W, T, holes, 1 * MM)
        win = _blk(L / 2 - 6 * MM, 0.0, 4.9 * MM, 2.5 * MM, 1.56 * MM, T, 0.2 * MM)
        pins = merge(
            [
                rbox([0.9 * MM, 0.9 * MM, 0.12 * MM], 0, 0).at(
                    [
                        -L / 2 + 1.8 * MM + 2.54 * MM * i,
                        -W / 2 + 1.4 * MM,
                        T + 0.06 * MM,
                    ]
                )
                for i in range(5)
            ]
        )
        chip = _blk(-L / 6, W / 5, 3.2 * MM, 3.0 * MM, 0.9 * MM)
        return {"pcb": pcb, "glass": win, "chip": chip, "pin": pins}
    if "flow" in c.part.category or "3901" in t:
        pcb = _pcb(L, W, T, [], 1 * MM)
        lens = cylinder(2.6 * MM, 3.0 * MM, 24, T).at([-L / 6, 0, 0])
        tof = _blk(L / 4, 0.0, 4.9 * MM, 2.5 * MM, 1.56 * MM, T, 0.2 * MM)
        return {
            "pcb": pcb,
            "lens": lens,
            "glass": tof,
            "chip": _blk(0.0, W / 3, 3 * MM, 3 * MM, 0.9 * MM),
        }
    r = 0.28 * min(W, H * 1.3)
    shell = rbox([L, W, H], min(0.002, 0.15 * min(L, W, H)), 2).at([0, 0, H / 2])
    ring = merge(
        [
            tube(r + 0.0006, r, 0.0006, 24)
            .at([0, 0, 0], R=rot([0, 1, 0], PI / 2))
            .at([L / 2 - 0.0006, s * 0.52 * (W / 2), H / 2])
            for s in (-1, 1)
        ]
    )
    win = merge(
        [
            cylinder(r, 0.0006, 24)
            .at([0, 0, 0], R=rot([0, 1, 0], PI / 2))
            .at([L / 2 - 0.0006, s * 0.52 * (W / 2), H / 2])
            for s in (-1, 1)
        ]
    )
    return {"body": shell, "glass": win, "ring": ring}


@gen("camera_module", **dict(SLOTS, lens="plastic-black"))
def camera_module(c, **_):
    """Camera board with a lens barrel and a flex-cable connector (Raspberry Pi Camera Module 3 proportions: 25 x 24 mm green
    PCB, square sensor block with a round barrel and a bright front ring, FFC socket on one edge, two mounting holes per side).
    Appearance: Raspberry Pi Camera Module 3 product photo."""
    L = c.p("length", default=0.025)
    W = c.p("width", default=0.024)
    H = c.p("height", default=0.0115)
    holes = [
        (x * (L / 2 - 2 * MM), y * (W / 2 - 2 * MM), 1.1 * MM)
        for x in (-1, 1)
        for y in (-1, 1)
    ]
    pcb = _pcb(L, W, T, holes, 2 * MM)
    zb = T + 2.2 * MM
    block = rbox([8.5 * MM, 8.5 * MM, 2.2 * MM], 0.4 * MM, 1).at(
        [0.0, 0.0, T + 1.1 * MM]
    )
    barrel = cylinder(4.1 * MM, max(H - zb - 0.8 * MM, 1 * MM), 36, zb)
    ring = tube(3.6 * MM, 2.7 * MM, 0.8 * MM, 36, H - 0.8 * MM)
    glass = cylinder(2.7 * MM, 0.3 * MM, 28, H - 0.5 * MM)
    conn = rbox([2.5 * MM, 16.5 * MM, 2.7 * MM], 0.3 * MM, 1).at(
        [
            -L / 2 + 3.5 * MM,
            0,
            -1.35 * MM,
        ]  # on the back, opposite the lens (Raspberry Pi docs)
    )
    chip = _blk(L / 2 - 5 * MM, 4 * MM, 4 * MM, 4 * MM, 0.9 * MM) + _blk(
        L / 2 - 5 * MM, -5 * MM, 3 * MM, 3 * MM, 0.9 * MM
    )
    return {
        "pcb": pcb,
        "lens": merge([block, barrel]),
        "ring": ring,
        "glass": glass,
        "conn": conn,
        "chip": chip,
    }


@gen(
    "lidar_2d",
    body="plastic-black",
    head="plastic-dark",
    window="glass-dark",
    ring="alu-anodised-grey",
)
def lidar_2d(c, **_):
    """Spinning 2-D lidar (RPLIDAR style): a square base with rounded corners and mounting flange, and a round rotating head on
    top with a dark optical window band at the scan plane (`scan_height` above the underside). Underside at z = 0, centred."""
    L = c.p("size_length", default=0.077)
    W = c.p("size_width", default=0.077)
    H = c.p("size_height", default=0.03885)
    zs = c.p("scan_height", default=0.65 * H)
    base_h = 0.42 * H
    r = 0.36 * min(L, W)
    base = rbox([L, W, base_h], 0.006, 2).at([0, 0, base_h / 2])
    head = cylinder(r, H - base_h, 48, base_h)
    window = tube(
        r + 0.0006, r - 0.001, 0.30 * (H - base_h), 48, zs - 0.15 * (H - base_h)
    )
    ring = tube(r + 0.0004, r - 0.001, 0.0015, 48, base_h)
    return {"body": base, "head": head, "window": window, "ring": ring}
