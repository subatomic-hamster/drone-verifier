"""The viewer's data-driven HUD: build a JSON document of screen panels and world-anchored marks, hand it to a `Hud`, and the
window draws the latest one every frame (src/viewer/hud.h has the full schema). Nothing here is specific to one project: a counter-UAS
turret (dvsim.cuas.hudview), a rocket's flight phases, a boat's depth readout or a drone's obstacle map are all just items.

    hud = Hud()
    threading.Thread(target=feed, args=(hud,)).start()      # any thread: hud.set(...) is thread-safe
    sim.show(hud=hud, clean=True)                           # sim = Sim.load(..., host=True); clean = no panes, the view fills the window

    def feed(hud):
        hud.set(doc([tabs(["BOOST", "COAST", "APOGEE"], 1, at=(16, 16), status="coasting"),
                     nudge(kv([("ALT", "412 m"), ("SPEED", "88 m/s")], at=(1, 0), a="tr", f=True), -16, 16),
                     wtext((10, 0, 50), "ROCKET 412 m", off=(0, -12))], seq=1))

Positions are window pixels from the viewport's top-left (`f=True`: fractions of the viewport), measured from the anchor `a` of the item
("tl", "t", "tr", "l", "c", "r", "bl", "b", "br"); colours are [r, g, b] or [r, g, b, a], 0-255. `camera=dict(target=, yaw=, pitch=, dist=)` in the
document drives the orbit camera (omit it and the mouse orbits as usual).
"""

import json
import math

try:
    from . import _dvsim

    Hud = (
        _dvsim.Hud if hasattr(_dvsim, "Hud") else None
    )  # absent when the build has no viewer
except ImportError:  # no built kernel: the builders below still work
    Hud = None

BLACK = (0, 0, 0, 228)  # the HUD boxes: small, black, slightly translucent
WHITE = (255, 255, 255)
DIM = (160, 165, 175)
CYAN = (60, 220, 255)
RED = (255, 90, 80)
GREEN = (110, 235, 140)
AMBER = (255, 200, 80)


def _put(d, **kw):
    d.update({k: v for k, v in kw.items() if v is not None})
    return d


def _pos(at, a, f):
    return dict(at=list(at), a=a, f=1 if f else None)


def nudge(item, dx, dy):
    """The item shifted by (dx, dy) window pixels after its position is resolved (a 16 px margin from a fractional anchor)."""
    return dict(item, m=[dx, dy])


def text(
    s, at, a="tl", size=14, col=WHITE, bg=None, pad=None, font=None, f=False, r=None
):
    return _put(
        dict(k="text", s=s),
        **_pos(at, a, f),
        size=size,
        col=list(col),
        bg=list(bg) if bg else None,
        pad=pad,
        font=font,
        r=r,
    )


def kv(
    rows, at, a="tl", size=13, col=WHITE, key_col=DIM, bg=BLACK, pad=10, gap=26, f=False
):
    """Label / value rows in one box (`rows` = [(label, value), ...]); labels left, values right-aligned."""
    return _put(
        dict(k="kv", rows=[[str(x), str(y)] for x, y in rows]),
        **_pos(at, a, f),
        size=size,
        col=list(col),
        key_col=list(key_col),
        bg=list(bg),
        pad=pad,
        gap=gap,
    )


def tabs(
    items,
    active,
    at,
    a="tl",
    size=13,
    status="",
    bg=BLACK,
    pad=10,
    gap=22,
    col=WHITE,
    dim=DIM,
    f=False,
):
    """A row of text tabs in a box, a thin underline under items[active], and an optional status line below."""
    return _put(
        dict(k="tabs", items=list(items), active=active, s=status),
        **_pos(at, a, f),
        size=size,
        bg=list(bg),
        pad=pad,
        gap=gap,
        col=list(col),
        dim=list(dim),
    )


def rect(at, size, a="tl", fill=None, line=None, lw=1.0, r=None, f=False, ar=None):
    return _put(
        dict(k="rect", size=list(size)),
        **_pos(at, a, f),
        fill=list(fill) if fill else None,
        line=list(line) if line else None,
        lw=lw,
        r=r,
        ar=ar,
    )


def line(pts, col=WHITE, lw=1.0):
    return dict(k="line", pts=[list(p) for p in pts], col=list(col), lw=lw)


def image(path, at, size, a="tl", uv=None, id="img", tint=None, f=False, ar=None):
    return _put(
        dict(k="image", path=str(path), size=list(size)),
        **_pos(at, a, f),
        uv=list(uv) if uv else None,
        id=id,
        tint=list(tint) if tint else None,
        ar=ar,
    )


def group(
    items, at, size, space=None, a="tl", fill=None, line=None, lw=1.0, f=False, ar=None
):
    """Children use their own pixel space (`space`, e.g. a camera frame's size), mapped onto the rectangle and clipped to it."""
    return _put(dict(k="group", size=list(size), items=items), **_pos(at, a, f), space=list(space) if space else None,
                fill=list(fill) if fill else None, line=list(line) if line else None, lw=lw, ar=ar)  # fmt: skip


def wtext(w, s, off=(0, 0), a="b", size=12, col=WHITE, bg=BLACK, pad=4, font=None):
    return _put(
        dict(k="wtext", w=[float(x) for x in w], s=s),
        off=list(off),
        a=a,
        size=size,
        col=list(col),
        bg=list(bg) if bg else None,
        pad=pad,
        font=font,
    )


def wline(pts, col=WHITE, lw=1.0):
    return dict(
        k="wline", pts=[[float(x) for x in p] for p in pts], col=list(col), lw=lw
    )


def wring(w, r=10, col=WHITE, lw=1.5):
    return dict(k="wring", w=[float(x) for x in w], r=r, col=list(col), lw=lw)


def wdot(w, r=3, col=WHITE):
    return dict(k="wdot", w=[float(x) for x in w], r=r, col=list(col))


def doc(items, seq=0, tick=None, camera=None):
    """The document `Hud.set` takes (a JSON string). `tick`: the sim tick this describes -- a producer stepping a paused sim in lockstep
    with the window waits until `Hud.progress()["stable"]` shows the window drew this seq at this tick."""
    d = dict(seq=int(seq), items=items)
    if tick is not None:
        d["tick"] = int(tick)
    if camera:
        d["camera"] = camera
    return json.dumps(d, separators=(",", ":"))


def project(cam_pos, cam_look, fovy_deg, width, height, p):
    """Where the viewer's renderer puts world point p in a width x height window: (x, y) pixels from the top-left, None behind the camera.
    The analytic form of the camera maths (render/renderer.cpp make_view, viewer/build3d.cpp Cam): z up, vertical fov `fovy_deg`."""
    sub = lambda a, b: [a[i] - b[i] for i in range(3)]
    dot = lambda a, b: sum(a[i] * b[i] for i in range(3))
    cross = lambda a, b: [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
    unit = lambda a: [x / math.sqrt(dot(a, a)) for x in a]
    f = unit(sub(cam_look, cam_pos))
    r = unit(cross(f, [0, 1, 0] if abs(f[2]) > 0.9999 else [0, 0, 1]))
    u = cross(r, f)
    rel = sub(p, cam_pos)
    z = dot(f, rel)
    if z < 0.02:
        return None
    t = 1 / math.tan(math.radians(fovy_deg) / 2)
    return (
        (t * height / width * dot(r, rel) / z * 0.5 + 0.5) * width,
        (-t * dot(u, rel) / z * 0.5 + 0.5) * height,
    )


def orbit_camera(target, yaw, pitch, dist):
    """(position, look-at) of the viewer's orbit camera: position = target + dist * (cos p cos y, cos p sin y, sin p)."""
    c = math.cos(pitch)
    return [
        target[0] + dist * c * math.cos(yaw),
        target[1] + dist * c * math.sin(yaw),
        target[2] + dist * math.sin(pitch),
    ], list(target)
