"""Orthogonal wire routing + label placement for the 2D wiring canvas (wire2d.py puts the result into the canvas state; the viewer
only draws the polylines). A wire leaves its pin horizontally outwards, runs on a grid of free tracks (card edges, gaps between
cards, pin rows) and enters the target pin horizontally; the path is the cheapest one (length + a cost per bend + a cost for
sharing a track with a wire routed before it) that stays outside every card (inflated by MARGIN). Dijkstra on a Hanan-style grid:
a few thousand states for a dozen cards."""

import heapq

HEAD_H, ROW_H, PAD = 30, 22, 10  # mirror wire2d.py / src/viewer/wire2d.cpp
STUB, MARGIN, TURN, SHARE = 16.0, 8.0, 40.0, 0.6
LABEL_H, LABEL_CW, LABEL_PAD = 16.0, 6.2, 10.0


def pin_xy(card, pin):
    x = card["x"] + (card["w"] if pin["side"] == "right" else 0.0)
    return x, card["y"] + HEAD_H + PAD * 0.5 + (pin["row"] + 0.5) * ROW_H


def _gaps(intervals, min_gap):
    """Free gaps of the union of [lo, hi] intervals wider than min_gap -> [(lo, hi)]."""
    out, end = [], None
    for lo, hi in sorted(intervals):
        if end is not None and lo - end > min_gap:
            out.append((end, lo))
        end = hi if end is None else max(end, hi)
    return out


def _grid(cards, ends=None):
    """Track coordinates. `ends` = {(card id, terminal name)} that a wire lands on: only those pin rows (and stubs) get a track, so a
    40-pin header costs the router three rows, not forty."""
    xs, ys = set(), set()
    for c in cards:
        for p in c["pins"]:
            if ends is not None and (c["id"], p["name"]) not in ends:
                continue
            _, y = pin_xy(c, p)
            ys.add(y)
            xs.add(c["x"] + (c["w"] + STUB if p["side"] == "right" else -STUB))
        xs.update((c["x"] - 12, c["x"] + c["w"] + 12))
        ys.update((c["y"] - 12, c["y"] + c["h"] + 12))
    if cards:
        xs.update(
            (min(c["x"] for c in cards) - 40, max(c["x"] + c["w"] for c in cards) + 40)
        )
        ys.update(
            (min(c["y"] for c in cards) - 40, max(c["y"] + c["h"] for c in cards) + 40)
        )
    for lo, hi in _gaps(
        [(c["x"] - MARGIN, c["x"] + c["w"] + MARGIN) for c in cards], 30
    ):
        xs.update(
            m
            for m in ((lo + hi) / 2 + 7 * k for k in range(-9, 10))
            if lo + 4 < m < hi - 4
        )
    for lo, hi in _gaps(
        [(c["y"] - MARGIN, c["y"] + c["h"] + MARGIN) for c in cards], 20
    ):
        ys.update(
            m
            for m in ((lo + hi) / 2 + 6 * k for k in range(-6, 7))
            if lo + 4 < m < hi - 4
        )
    # between vertically stacked cards of one column (the union above hides those gaps when columns are staggered)
    for a in cards:
        for b in cards:
            if (
                b["y"] > a["y"] + a["h"] + 18
                and b["x"] < a["x"] + a["w"]
                and a["x"] < b["x"] + b["w"]
            ):
                ys.add((a["y"] + a["h"] + b["y"]) / 2)
    return sorted(round(v, 2) for v in xs), sorted(round(v, 2) for v in ys)


def _blocked_h(rects, y, x0, x1):
    return any(t < y < b and x0 < r and x1 > l for l, t, r, b in rects)


def _blocked_v(rects, x, y0, y1):
    return any(l < x < r and y0 < b and y1 > t for l, t, r, b in rects)


def route_wires(cards, wires):
    """cards: [{x, y, w, h, id, pins: [{name, side, row}]}], wires: [{name, from, to, ...}] -> {wire name: [[x, y], ...]}, canvas units,
    the polyline from the `from` pin to the `to` pin (orthogonal, corners only). A wire with no free path gets a plain Z."""
    by = {c["id"]: c for c in cards}
    rects = [
        (
            c["x"] - MARGIN,
            c["y"] - MARGIN,
            c["x"] + c["w"] + MARGIN,
            c["y"] + c["h"] + MARGIN,
        )
        for c in cards
    ]
    ends = set()
    for w in wires:
        for e in ("from", "to"):
            i_, _, t_ = w[e].partition(".")
            ends.add((i_, t_))
    xs, ys = _grid(cards, ends)
    xi, yi = {v: i for i, v in enumerate(xs)}, {v: i for i, v in enumerate(ys)}
    bh = {}  # (horizontal?, i, j) -> that grid edge is blocked by a card
    used = set()
    out = {}

    def edge_blocked(horiz, i, j):
        k = (horiz, i, j)
        if k not in bh:
            bh[k] = (
                _blocked_h(rects, ys[j], xs[i], xs[i + 1])
                if horiz
                else _blocked_v(rects, xs[i], ys[j], ys[j + 1])
            )
        return bh[k]

    for w in wires:
        (ia, _, pa), (ib, _, pb) = w["from"].partition("."), w["to"].partition(".")
        ca, cb = by[ia], by[ib]
        qa = next(p for p in ca["pins"] if p["name"] == pa)
        qb = next(p for p in cb["pins"] if p["name"] == pb)
        a, b = pin_xy(ca, qa), pin_xy(cb, qb)
        sa = STUB if qa["side"] == "right" else -STUB
        sb = STUB if qb["side"] == "right" else -STUB
        a2, b2 = (
            (round(a[0] + sa, 2), round(a[1], 2)),
            (round(b[0] + sb, 2), round(b[1], 2)),
        )
        # direction of the last move: 0 = +x, 1 = -x (into a left pin the wire moves right)
        want_end = 0 if qb["side"] == "left" else 1
        start_dir = 0 if qa["side"] == "right" else 1
        path = None
        if a2[0] in xi and a2[1] in yi and b2[0] in xi and b2[1] in yi:
            s = (xi[a2[0]], yi[a2[1]], start_dir)
            goal = (xi[b2[0]], yi[b2[1]], want_end)
            dist, prev, heap = {s: 0.0}, {}, [(0.0, s)]
            while heap:
                d, u = heapq.heappop(heap)
                if d > dist.get(u, 1e18):
                    continue
                if u == goal:
                    break
                i, j, dr = u
                for nd in range(4):  # 0 +x, 1 -x, 2 +y, 3 -y
                    if nd // 2 == dr // 2 and nd != dr:
                        continue  # no reversing
                    ni, nj = (
                        i + (1 if nd == 0 else -1 if nd == 1 else 0),
                        j + (1 if nd == 2 else -1 if nd == 3 else 0),
                    )
                    if not (0 <= ni < len(xs) and 0 <= nj < len(ys)):
                        continue
                    horiz = nd < 2
                    ei, ej = (min(i, ni), j) if horiz else (i, min(j, nj))
                    if edge_blocked(horiz, ei, ej):
                        continue
                    ln = abs(xs[ni] - xs[i]) + abs(ys[nj] - ys[j])
                    cost = (
                        d
                        + ln * (1 + (SHARE if (horiz, ei, ej) in used else 0))
                        + (TURN if nd != dr else 0)
                    )
                    v = (ni, nj, nd)
                    if cost < dist.get(v, 1e18):
                        dist[v] = cost
                        prev[v] = u
                        heapq.heappush(heap, (cost, v))
            if goal in dist:
                seq, u = [], goal
                while u != s:
                    seq.append(u)
                    u = prev[u]
                seq.append(s)
                seq.reverse()
                for (i, j, _), (k, m, _) in zip(seq, seq[1:]):
                    horiz = j == m
                    used.add((horiz, min(i, k), j) if horiz else (horiz, i, min(j, m)))
                path = [(xs[i], ys[j]) for i, j, _ in seq]
        if (
            path is None
        ):  # no free path (cards pushed together by hand): a plain Z through the middle
            mx = (a2[0] + b2[0]) / 2
            path = [a2, (mx, a2[1]), (mx, b2[1]), b2]
        pts = [a] + path + [b]
        simp = [pts[0]]
        for k in range(1, len(pts) - 1):  # drop collinear / duplicate vertices
            p, q, r = simp[-1], pts[k], pts[k + 1]
            if q == p or (p[0] == q[0] == r[0]) or (p[1] == q[1] == r[1]):
                continue
            simp.append(q)
        simp.append(pts[-1])
        out[w["name"]] = [[round(x, 2), round(y, 2)] for x, y in simp]
    return out


def label_text(w):
    awg = w.get("awg")
    return f"{awg:.0f} AWG" if awg else str(w.get("part", ""))[:14]


def place_labels(cards, wires, routes):
    """-> {wire name: {"text", "x", "y", "w", "h"}} (centre, canvas units): on the wire's longest free stretch, clear of cards and of the
    labels placed before. A wire whose stretches are all crowded gets none (the viewer shows it on hover)."""
    rects = [
        (c["x"] - 2, c["y"] - 2, c["x"] + c["w"] + 2, c["y"] + c["h"] + 2)
        for c in cards
    ]
    placed, out = [], {}
    for w in wires:
        text = label_text(w)
        lw, lh = len(text) * LABEL_CW + LABEL_PAD, LABEL_H
        pts = routes[w["name"]]
        segs = []
        for p, q in zip(pts, pts[1:]):
            ln = abs(q[0] - p[0]) + abs(q[1] - p[1])
            segs.append((ln, p, q))
        for ln, p, q in sorted(segs, key=lambda s: -s[0]):
            horiz = abs(q[1] - p[1]) < 1e-6
            if ln < (lw if horiz else lh) + 8:
                continue
            for f in (0.5, 0.35, 0.65, 0.2, 0.8):
                cx, cy = p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f
                box = (cx - lw / 2, cy - lh / 2, cx + lw / 2, cy + lh / 2)
                if any(
                    box[0] < r and box[2] > l and box[1] < b and box[3] > t
                    for l, t, r, b in rects + placed
                ):
                    continue
                placed.append(box)
                out[w["name"]] = {
                    "text": text,
                    "x": round(cx, 2),
                    "y": round(cy, 2),
                    "w": lw,
                    "h": lh,
                }
                break
            if w["name"] in out:
                break
    return out
