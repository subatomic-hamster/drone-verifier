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
    print("OK")


main()
