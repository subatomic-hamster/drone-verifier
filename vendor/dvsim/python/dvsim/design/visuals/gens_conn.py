"""Connector family: XT / AS / EC / MR / T-plug power housings, bullets, JST wire-to-board housings, pin headers and the DC jack.
One generator, family chosen from the part id / name. Frame: mating direction +x (the mating face at x = L), the pin row along
y, up = +z, housing centred in y and z; a PCB type has its pins down (-z)."""

import math
import re

import numpy as np

from . import gen
from .mesh import Mesh, circle, cylinder, extrude, merge, rbox, revolve, rot, tube

PI = math.pi
MM = 1e-3
Y90 = rot([0, 1, 0], PI / 2)
UVT = np.array(
    [[0, 0, 1], [1, 0, 0], [0, 1, 0]], float
)  # (u, v, t) -> (y, z, x): section in y-z, extruded along x

# family: (width across the pin row, height, length, pin pitch, pin diameter) in mm
_POWER = {
    "xt30": (10.2, 5.0, 10.3, 5.0, 2.0),
    "xt60": (15.8, 8.0, 16.0, 7.2, 3.5),
    "xt90": (22.0, 10.5, 24.0, 10.0, 4.5),
    "xt150": (28.0, 13.0, 28.0, 12.0, 6.0),
    "as150": (28.0, 13.0, 28.0, 12.0, 7.0),
    "ec5": (15.5, 9.5, 14.0, 7.5, 5.0),
    "tplug": (14.0, 8.0, 13.0, 7.0, 3.5),
}


def _section(w, h, ch):
    """Housing cross-section in (u, v): rectangle with the two corners of the +v side chamfered (the polarity key)."""
    return [
        (-w / 2, -h / 2),
        (w / 2, -h / 2),
        (w / 2, h / 2 - ch),
        (w / 2 - ch, h / 2),
        (-w / 2 + ch, h / 2),
        (-w / 2, h / 2 - ch),
    ]


def _to_xyz(m):
    return m.at([0, 0, 0], R=UVT)


def _power(c, fam, t):
    w, h, L, pitch, pd = (v * MM for v in _POWER[fam])
    female = t.endswith(("-f", "-fm"))
    pcb = bool(re.search(r"(pw|aw)-[mf]$", t))
    sy = [-pitch / 2, pitch / 2]
    holes = [circle(pd * 0.6, 20, (y, 0.0)) for y in sy] if female else []
    body = _to_xyz(extrude(np.asarray(_section(w, h, 0.28 * h)), 0, L, holes))
    pins = []
    for y in sy:
        if female:
            pins.append(
                tube(pd * 0.62, pd * 0.35, 0.55 * L, 24, 0)
                .at([0, 0, 0], R=Y90)
                .at([0.18 * L, y, 0])
            )
        else:
            pins.append(
                cylinder(pd / 2, 1.0 * L + 0.004, 24, 0.12 * L)
                .at([0, 0, 0], R=Y90)
                .at([0, y, 0])
            )
        if not pcb:
            pins.append(
                tube(pd * 0.62, pd * 0.4, 0.4 * L, 20, 0)
                .at([0, 0, 0], R=Y90)
                .at([-0.38 * L, y, 0])
            )  # solder cup
        else:
            pins.append(
                cylinder(0.0006, 0.004 + h / 2, 12, 0).at([0.35 * L, y, -h / 2 - 0.004])
            )
    return {"body": body, "pins": merge(pins)}


def _jst(c, t, pitch):
    m = re.search(r"(\d+)p\b", t)
    n = int(m[1]) if m else 4
    W, H, L = (n - 1) * pitch + 2.0 * pitch, 2.8 * pitch, 2.3 * pitch
    body = rbox([L, W, H], 0.08 * pitch, 1).at([0, 0, 0])
    latch = rbox([0.55 * L, 0.5 * W, 0.12 * H], 0.05 * pitch, 1).at(
        [0.1 * L, 0, H / 2 + 0.06 * H]
    )
    pins = merge(
        [
            rbox([0.5 * pitch, 0.34 * pitch, 0.5 * pitch], 0.02 * pitch, 1).at(
                [L / 2 + 0.2 * pitch, (k - (n - 1) / 2) * pitch, -0.12 * H]
            )
            for k in range(n)
        ]
    )
    return {"body": merge([body, latch]), "pins": pins}


def _bullet(c, t):
    m = re.search(r"(\d+(?:\.\d+)?)mm", t)
    d = float(m[1]) * MM if m else 0.004
    L = 3.0 * d + 0.002
    prof = [
        (0, 0),
        (d / 2 * 0.8, 0),
        (d / 2 * 0.8, 0.12 * L),
        (d / 2, 0.12 * L),
        (d / 2, 0.55 * L),
        (d / 2 * 0.92, 0.6 * L),
        (d / 2 * 0.92, L),
        (0, L),
    ]
    return {"pins": revolve(prof, 36).at([0, 0, 0], R=Y90).at([-L / 2, 0, 0])}


def _header(c, t):
    socket = "socket" in t
    p = 2.54 * MM
    n = int(c.p("pins", default=3)) or 3
    W = n * p
    if socket:
        body = extrude(
            np.array(
                [(-p / 2, -W / 2), (p / 2, -W / 2), (p / 2, W / 2), (-p / 2, W / 2)]
            ),
            0,
            8.5 * MM,
            [
                np.array(circle(0.4 * MM, 8, (0, (k - (n - 1) / 2) * p)))
                for k in range(n)
            ],
        )
        return {"body": body, "pins": Mesh()}
    base = rbox([p, W, 2.5 * MM], 0.1 * MM, 1).at([0, 0, 1.25 * MM])
    pins = merge(
        [
            rbox([0.64 * MM, 0.64 * MM, 11.5 * MM], 0.05 * MM, 1).at(
                [0, (k - (n - 1) / 2) * p, 5.75 * MM + 0.0]
            )
            for k in range(n)
        ]
    )
    return {"body": base, "pins": pins}


def _jack(c):
    body = rbox([14.5 * MM, 9 * MM, 11 * MM], 0.5 * MM, 2).at([0, 0, 5.5 * MM])
    barrel = (
        tube(3.1 * MM, 1.8 * MM, 12 * MM, 28, 0)
        .at([0, 0, 0], R=Y90)
        .at([-1 * MM, 0, 5.5 * MM])
    )
    pin = (
        cylinder(0.75 * MM, 9 * MM, 14, 0)
        .at([0, 0, 0], R=Y90)
        .at([-1 * MM, 0, 5.5 * MM])
    )
    return {"body": body, "pins": merge([barrel, pin])}


@gen("connector", body="plastic-yellow", pins="gold")
def connector(c, **_):
    """Connector by family: XT30 / XT60 / XT90 / XT150 / AS150 (yellow, polarity chamfer, male pins or female sockets, solder
    cups or PCB pins), EC5 (blue), T-plug, bullets (gold), JST XH / PH / GH / SH housings with their pin count, 2.54 mm pin /
    socket headers, DC jack; anything else a plain block. Dimensions are the maker's published ones (Amass / JST catalogue
    drawings), chosen by part id; see the family table."""
    t = (c.part.id + " " + c.part.raw.get("name", "")).lower()
    for key, fam in (
        ("as150", "as150"),
        ("xt150", "xt150"),
        ("xt90", "xt90"),
        ("xt60", "xt60"),
        ("xt30", "xt30"),
        ("ec5", "ec5"),
        ("t-plug", "tplug"),
        ("tplug", "tplug"),
    ):
        if key in t:
            return _power(c, fam, c.part.id.lower())
    if "mr30" in t or "mr60" in t:
        return _power(c, "xt30" if "mr30" in t else "xt60", c.part.id.lower())
    if "bullet" in t:
        return _bullet(c, t)
    for key, p in (("jst-gh", 1.25), ("jst-sh", 1.0), ("jst-ph", 2.0), ("jst-xh", 2.5)):
        if key in t:
            return _jst(c, c.part.id.lower(), p * MM)
    if "header" in t:
        return _header(c, t)
    if "jack" in t or "pj-102" in t:
        return _jack(c)
    return {"body": rbox([0.012, 0.008, 0.006], 0.0008, 1).at([0, 0, 0.003])}
