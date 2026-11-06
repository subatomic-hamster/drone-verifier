"""Offline checks for the feedback sweep plan, the layered leaderboard and baseline's Pareto pick. python harness/tests/test_layers.py"""

import json
import sys
import tempfile
from pathlib import Path

H = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(H))
import baseline
import bench
import leaderboard


def test_plan():
    t = bench.plan(["fable", "opus", "sonnet", "haiku"], ["whatif"], 1, sweep=True)
    assert [(m, l) for m, l, _ in t] == [(m, "whatif") for m in bench.COST_ORDER] + [
        (m, l) for m in ("haiku", "sonnet") for l in bench.LEVELS[:3]
    ]
    assert bench.plan(
        ["haiku"],
        ["whatif", "diag"],
        2,
        done=lambda n: n in ("haiku-diag-0", "haiku-1"),
    ) == [("haiku", "whatif", 0), ("haiku", "diag", 1)]
    assert (
        bench.runname("haiku", "whatif") == "haiku"
        and bench.runname("haiku", "none") == "haiku-none"
    )


def test_leaderboard_layers():
    d = Path(tempfile.mkdtemp())
    gates = [{"id": "G1", "pass": True, "points": 10, "max_points": 10}]
    runs = [
        ("haiku-0", "haiku", "whatif", 100, True, 600, 1.2),
        ("haiku-diag-0", "haiku-diag", "diag", 30, False, None, None),
        ("haiku-none-0", "haiku-none", "none", 0, False, None, None),
        ("haiku-none-1", "haiku-none", "none", 100, True, 650, 1.1),
        ("human-1", "human", None, 100, True, 574, 1.26),
    ]
    for name, model, fb, score, ok, cost, mass in runs:
        (d / name).mkdir()
        r = {
            "model": model,
            "score": score,
            "gates": gates,
            "verify_calls": 2,
            "pass_within_budget": ok,
            "cost_usd_bom": cost,
            "mass_kg": mass,
            "robustness_gap": 5.0 if ok else None,
            "prediction_error": {"hover_throttle_pct": 3.0, "flight_time_pct": 8.0},
        }
        if fb:
            r["feedback"] = fb
        else:
            r["baseline"] = True
        (d / name / "results.json").write_text(json.dumps(r))
    lb = leaderboard.build(d)
    assert {e["model"] for e in lb["entries"]} == {
        "haiku",
        "human",
    }  # diag and none runs stay off the main table
    h = next(e for e in lb["entries"] if e["model"] == "haiku")
    assert (
        h["trials"] == 1
        and "not a rate" in h["n_note"]
        and h["pareto"] == [{"cost_usd": 600, "mass_kg": 1.2}]
    )
    assert (
        h["prediction_error"]["flight_time_pct"] == 8.0
        and next(e for e in lb["entries"] if e["model"] == "human")["baseline"]
    )
    assert (
        lb["feedback_curve"]["haiku"]
        == {
            "none": {"runs": 2, "passed": 1, "rate": 0.5},
            "diag": {"runs": 1, "passed": 0, "rate": 0.0},
            "whatif": {"runs": 1, "passed": 1, "rate": 1.0},
        }
        and "human" not in lb["feedback_curve"]
    )


def test_pareto():
    rows = [
        {"cost": 500, "mass": 1.3},
        {"cost": 550, "mass": 1.2},
        {"cost": 560, "mass": 1.25},
        {"cost": 600, "mass": 1.1},
    ]
    assert [r["cost"] for r in baseline.nondominated(rows)] == [500, 550, 600]
    assert leaderboard.pareto(
        [{"cost_usd": 1, "mass_kg": 2}, {"cost_usd": 2, "mass_kg": 3}]
    ) == [{"cost_usd": 1, "mass_kg": 2}]


if __name__ == "__main__":
    test_plan()
    test_leaderboard_layers()
    test_pareto()
    print("layers ok")
