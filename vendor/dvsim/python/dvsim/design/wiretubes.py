"""Wires as 3D tubes (visual only). Each conductor of each assembly wire becomes a MuJoCo spatial tendon through a chain of sites
that follow the bodies; the renderer draws tendons as tubes (Solver::tubes). The chain is the conductor's route from its pin on
one end to its pin on the other, planned around the part geometry by wireroute.py (it leaves the connector along the connector
axis and keeps the tube clear of every part beyond the connector mouth); a wire the router cannot place, or a build without
geometry, falls back to the straight chord with the old sag. The tendons have no stiffness, damping, limit or actuator: they add
no force, the dynamics are unchanged (tests/python/test_wire_tubes.py compares state hashes with and without)."""

import numpy as np

from . import pins as PN
from . import wireroute as WR

# Insulated (silicone hook-up wire) outer diameter by AWG, mm. Compiled from vendor listings of silicone hook-up wire,
# +-0.2 mm between manufacturers: 30 .. 16 AWG and 14 / 10 AWG from the Adafruit 26 AWG silicone page and the openelab /
# Progressive RC silicone wire listings (https://www.progressiverc.com/prc-silicone-wire-10-awg.html: 10 AWG ~5 mm);
# 12 and 8 AWG are interpolated / extrapolated from their neighbours (fidelity: estimated). Cosmetic only: no physics reads it.
AWG_OD_MM = {
    30: 1.1,
    28: 1.2,
    26: 1.3,
    24: 1.4,
    22: 1.6,
    20: 1.8,
    18: 2.0,
    16: 2.4,
    14: 3.5,
    12: 4.2,
    10: 5.0,
    8: 6.2,
}
MAX_TUBES = 16  # conductors drawn per wire (a ribbon of more is drawn as 16)

RED, BLACK = (0.82, 0.08, 0.08), (0.04, 0.04, 0.04)
GREY = (0.45, 0.45, 0.48)
PHASES = [
    (0.95, 0.78, 0.10),
    (0.10, 0.62, 0.25),
    (0.15, 0.40, 0.90),
]  # motor phases A B C: yellow, green, blue
SIGNAL = [
    (0.95, 0.78, 0.10),
    (0.92, 0.92, 0.92),
    (0.15, 0.40, 0.90),
    (0.10, 0.62, 0.25),
    (0.95, 0.45, 0.10),
    (0.55, 0.25, 0.75),
]


def od_m(awg, conductor_d=None) -> float:
    """Insulated outer diameter in metres: the AWG table (nearest gauge), else conductor diameter + 2 x 0.4 mm of wall."""
    if awg is not None:
        return AWG_OD_MM[min(AWG_OD_MM, key=lambda a: abs(a - awg))] * 1e-3
    return (conductor_d + 0.8e-3) if conductor_d else 1.3e-3


def colors(kind: str, n: int, port_name: str) -> list:
    """Function colours of the n conductors of a wire: power pair red / black, a 3-conductor power cable carries motor
    phases, a lone conductor is black on a ground-like port else red, signal wires cycle yellow / white / blue / ..."""
    n = min(n, MAX_TUBES)
    if kind == "power":
        if n == 1:
            return [
                BLACK
                if any(
                    s in port_name.lower() for s in ("gnd", "neg", "minus", "return")
                )
                else RED
            ]
        if n == 2:
            return [RED, BLACK]
        if n == 3:
            return PHASES
        return [RED, BLACK] + SIGNAL[: n - 2]
    return [SIGNAL[i % len(SIGNAL)] for i in range(n)]


def pin_colors(kind: str, pins: list) -> list:
    """Colours from the pins each conductor lands on: ground black, power red, signal cycling, no-connect grey; three bare power
    pins without a voltage are motor phases."""
    if (
        kind == "power"
        and len(pins) == 3
        and all(p["kind"] == "power" and p.get("v") is None for p in pins)
    ):
        return PHASES
    out, k = [], 0
    for p in pins:
        if p["kind"] == "ground":
            out.append(BLACK)
        elif p["kind"] == "power":
            out.append(RED)
        elif p["kind"] == "nc":
            out.append(GREY)
        else:
            out.append(SIGNAL[k % len(SIGNAL)])
            k += 1
    return out


def _unit(v, default):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.asarray(default, float)


def _end(inst, port, refs, Tm, centre):
    """(pin world positions (n,3) | None, connector mouth position, outward axis, row direction) of one wire end."""
    R, t = Tm[:3, :3], Tm[:3, 3]
    conns = inst.part.connectors
    pins = [PN.find_pin(inst.part, r) for r in refs]
    if refs and all(p is not None for p in pins):
        pos = np.array([R @ p["pos"] + t for p in pins])
        c = conns[PN.split_ref(refs[0])[0]]
        return pos, pos.mean(0), _unit(R @ c["axis"], [0, 0, 1]), R @ c["ref"], pins
    p0 = R @ np.array(port.get("pos", [0, 0, 0]), dtype=float) + t
    if "axis" in port:
        ax = R @ np.array(port["axis"], dtype=float)
    else:  # no declared direction: away from the part's middle, snapped to the dominant world axis
        d = p0 - centre
        ax = np.zeros(3)
        ax[int(np.argmax(np.abs(d)))] = np.sign(d[int(np.argmax(np.abs(d)))]) or 1.0
    ref = R @ np.array(port.get("ref", [1.0, 0.0, 0.0]), dtype=float)
    return None, p0, _unit(ax, [0, 0, 1]), ref, None


def wire_tubes(wires: list, T: dict, sites: dict, geoms=None):
    """Adds the conductor sites to `sites` ({inst: {name: (pos, R)}}, instance-local) and returns the <tendon> lines.
    `geoms` (wireroute.world_geoms) switches the routing on; each wire gets `_route` (the router's result) for the tests."""
    boxes = {}
    for g in geoms or []:
        lo, hi = boxes.get(g["owner"], (g["lo"], g["hi"]))
        boxes[g["owner"]] = (np.minimum(lo, g["lo"]), np.maximum(hi, g["hi"]))
    prep, items = [], []
    for w in wires:
        ia, ib = w["_ia"], w["_ib"]
        if ia.id not in T or ib.id not in T:
            continue  # an offboard end has no pose: nothing to draw
        pairs = [p for p in w.get("_pairs", []) if p[0] and p[1]]
        awg = w["_part"].value("awg")
        d = od_m(awg, w["_part"].value("conductor_diameter"))
        ends = []
        for inst, port, refs in (
            (ia, w["_pa"], [p[0] for p in pairs]),
            (ib, w["_pb"], [p[1] for p in pairs]),
        ):
            lo, hi = boxes.get(inst.id, (T[inst.id][:3, 3],) * 2)
            ends.append(_end(inst, port, refs, T[inst.id], (lo + hi) / 2))
        (pa, ma, na, ua, qa), (pb, mb, nb, ub, qb) = ends
        if pa is not None and pb is not None:
            n = min(len(pairs), MAX_TUBES)
            a, b = pa[:n], pb[:n]
            cols = pin_colors(w["kind"], qa[:n])
        else:  # a port-level end: every conductor starts at the port position and the ring separates them
            n = min(w["conductors"], MAX_TUBES)
            a, b = np.tile(ma, (n, 1)), np.tile(mb, (n, 1))
            cols = colors(w["kind"], n, w["_pa"]["name"])
        if n < 1 or (np.linalg.norm(a[0] - b[0]) < 1e-9 and ia.id == ib.id):
            continue
        prep.append((w, ia, ib, a, b, cols, d))
        items.append(
            {
                "key": w["index"],
                "a": a,
                "b": b,
                "na": na,
                "nb": nb,
                "ua": ua,
                "ub": ub,
                "d": d,
            }
        )
    routes = WR.route_wires(geoms, items) if geoms is not None and items else {}
    lines = []
    for w, ia, ib, a, b, cols, d in prep:
        r = routes.get(w["index"])
        w["_route"] = r
        n = len(a)
        if r is not None and r["ok"]:
            paths = r["paths"]
        else:  # the chord, as before the router existed
            chord = b.mean(0) - a.mean(0)
            ln = np.linalg.norm(chord)
            ax = chord / ln if ln > 1e-9 else np.array([1.0, 0, 0])
            u = np.cross(ax, [0.0, 0.0, 1.0])
            u = (
                u / np.linalg.norm(u)
                if np.linalg.norm(u) > 1e-6
                else np.array([0.0, 1.0, 0.0])
            )
            v = np.cross(ax, u)
            ring = WR.ring_radius(n, d)
            paths = []
            for c in range(n):
                ang = 2 * np.pi * c / n + np.pi / 2
                off = ring * (np.cos(ang) * u + np.sin(ang) * v)
                paths.append(
                    np.array(
                        [
                            a[c] + off - (a[c] - a.mean(0)),
                            b[c] + off - (b[c] - b.mean(0)),
                        ]
                    )
                )
        for c, path in enumerate(paths):
            seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
            total = float(seg.sum())
            s = np.concatenate([[0.0], np.cumsum(seg)])
            names = []
            for k, p in enumerate(path):
                inst = (
                    ia if s[k] <= 0.5 * total else ib
                )  # the first half follows end A's body, the second half end B's
                Tm = T[inst.id]
                name = f"wire:{w['index']}:{c}:{k}"
                sites.setdefault(inst.id, {})[name] = (
                    Tm[:3, :3].T @ (p - Tm[:3, 3]),
                    np.eye(3),
                )
                names.append(name)
            rgba = " ".join(f"{x:.3f}" for x in (*cols[c], 1.0))
            routed = r is not None and r["ok"]
            L = (
                total if routed else max(w["length"], total)
            )  # a routed tube is drawn taut: sagging would leave the clear path
            lines.append(
                f'    <spatial name="wire:{w["index"]}:{c}|{ia.id}|{ib.id}" width="{d / 2:.6g}" rgba="{rgba}"'  # stiffness / damping default to 0: visual only
                f' springlength="{L:.6g} {L:.6g}">'
            )
            lines += [f'      <site site="{nm}"/>' for nm in names]
            lines.append("    </spatial>")
    return lines
