"""Partial scoring, on hfbox: `python tests/check_partial.py`. The reference is exactly 100, do-nothing earns no flight points, open-loop is plausible, an over-budget design flies nothing, a rerun
gives identical numbers, and the mission score behaves on hand-made metrics."""

import json
from pathlib import Path

from dv import design as D
from dv import heldout
from dv import partial as P

R = Path(__file__).resolve().parent.parent / "reference"
REF = json.loads((R / "design.json").read_text())
CODE = (R / "controller.py").read_text()
SEED = heldout.heldout_seed()


def show(name, r):
    print(name, r["partial_score"], [(g["id"], g["points"]) for g in r["partial_gates"]], flush=True)  # fmt: skip


def main():
    good = dict(z_max_m=3.1, hold_s=5.0, touchdown_m_s=0.5, land_dist_m=0.2, final_tilt_deg=1.0, t_end_s=18.0, tilt_max_deg=8.0)  # fmt: skip
    assert P.mission_score({}, True) == 1.0
    assert P.mission_score(dict(z_max_m=0.0, t_end_s=15.0), False) == 0.0  # no lift-off
    assert P.mission_score(good, False) == 0.95  # failed for something else: capped
    assert P.parts(dict(good, land_dist_m=5.0))["accuracy"] == 0.0
    assert P.parts(dict(good, touchdown_m_s=4.5))["landing"] == 0.0
    assert P.parts(dict(good, final_tilt_deg=80.0))["landing"] == 0.0  # tipped over
    assert P.parts(dict(good, final_tilt_deg=80.0))["accuracy"] == 0.0
    assert P.parts(dict(good, z_max_m=6.3))["altitude"] == 0.0
    assert P.parts(dict(good, z_max_m=1.35))["altitude"] == 0.5
    assert P.mission_score(dict(good, tilt_max_deg=60.0), False) == 0.5

    b = D.apply(REF)
    r = P.score(b, CODE, SEED)
    show("reference", r)
    assert r["partial_score"] == 100.0, r
    again = P.score(b, CODE, SEED)
    assert again == r, "rerun differs"
    print("rerun identical")

    r = P.score(b, (R / "do_nothing.py").read_text(), SEED)
    show("do_nothing", r)
    assert r["partial_score"] == 20.0 and all(g["points"] == 0 for g in r["partial_gates"][2:])  # fmt: skip

    r = P.score(b, (R / "open_loop.py").read_text(), SEED)
    show("open_loop", r)
    assert 20.0 < r["partial_score"] < 100.0

    r = P.score(D.apply(dict(REF, gnss="sparkfun-gps-rtk-sma-16481")), CODE, SEED)
    show("stock over budget", r)
    assert r["partial_score"] < 20.0 and "not flown" in r["partial_gates"][2]["reason"]

    r = P.score(D.apply({}), CODE, SEED)
    show("empty design", r)
    assert r["partial_score"] == 0.0

    # the hidden exam: frozen on the reference, in its band, deterministic; no flight reaches 1.0
    row = dict(z=3.0)
    assert P.hard_mission(good, [row, dict(z=2.9), row]) < 1.0
    assert P.hard_mission(dict(z_max_m=0.0, t_end_s=15.0), [dict(z=0.0)]) == 0.0
    assert P.hard_mission(dict(good, touchdown_m_s=0.2), [row]) > P.hard_mission(
        good, [row]
    )
    assert P.hard_mission(dict(good, land_dist_m=0.1), [row]) > P.hard_mission(
        good, [row]
    )

    def show_hard(name, h):
        print(
            name,
            h["hard_score"],
            [(g["id"], g["points"]) for g in h["hard_gates"]],
            flush=True,
        )

    h = P.score(b, CODE, SEED, hard=True)
    show_hard("hard reference", h)
    assert 85.0 <= h["hard_score"] <= 92.0, h["hard_score"]
    assert P.score(b, CODE, SEED, hard=True) == h, "hard rerun differs"
    print("hard rerun identical")
    fl = [f for g in h["hard_gates"][2:] for f in ([g["flight"]] if "flight" in g else g["flights"])]  # fmt: skip
    print("hard reference flights passed", sum(f["pass"] for f in fl), "of", len(fl))
    assert all(f["mission"] < 1.0 for f in fl)

    h = P.score(b, (R / "do_nothing.py").read_text(), SEED, hard=True)
    show_hard("hard do_nothing", h)
    assert h["hard_score"] == 20.0 and all(g["points"] == 0 for g in h["hard_gates"][2:])  # fmt: skip

    h = P.score(b, (R / "open_loop.py").read_text(), SEED, hard=True)
    show_hard("hard open_loop", h)
    assert 20.0 < h["hard_score"] < 86.0
    print("OK")


main()
