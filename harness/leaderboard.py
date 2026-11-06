"""results/*/results.json -> results/leaderboard.json.  leaderboard.py [--pick]  (--pick prints the lowest and highest scorer, baselines excluded).

Main entries: whatif runs (and baselines, flagged baseline: true). feedback_curve covers every level. Run counts are always carried so
one trial is not read as a rate. Metric shapes follow docs/LAYERS.md; `viable_designs` (list of {cost_usd, mass_kg}) feeds the Pareto
front, falling back to first_viable and the final design of a passing run.
"""

import json
import statistics
import sys
import time
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
LEVELS = ["none", "passfail", "diag", "whatif"]


def base_model(r, dirname):
    m = r.get("model") or dirname.rsplit("-", 1)[0]
    fb = r.get("feedback")
    return m[: -len(fb) - 1] if fb and m.endswith("-" + fb) else m


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 2) if xs else None


def passed(r):
    return bool(r.get("pass_within_budget", r.get("score", 0) >= 100))


def pareto(points):
    """Cost-vs-mass non-dominated set (lower is better on both), sorted by cost."""
    pts = sorted({(round(p["cost_usd"], 2), round(p["mass_kg"], 3)) for p in points})
    front, best_mass = [], float("inf")
    for c, m in pts:
        if m < best_mass:
            front.append({"cost_usd": c, "mass_kg": m})
            best_mass = m
    return front


def viable(r):
    v = list(r.get("viable_designs") or [])
    fv = r.get("first_viable")
    if fv and fv.get("cost_usd") is not None and fv.get("mass_kg") is not None:
        v.append(fv)
    if passed(r) and r.get("cost_usd_bom") is not None and r.get("mass_kg") is not None:
        v.append({"cost_usd": r["cost_usd_bom"], "mass_kg": r["mass_kg"]})
    return v


def entry(model, rs):
    best = max(rs, key=lambda r: r.get("score", 0))
    g = lambda k: mean(r.get(k) for r in rs)
    ok = [x["id"] for x in best.get("gates", []) if x.get("pass")]
    tok = lambda r: sum((r.get("tokens") or {}).values())
    fv = [r["first_viable"] for r in rs if r.get("first_viable")]
    pe = [r.get("prediction_error") or {} for r in rs]
    cc = [r["crosscheck"] for r in rs if r.get("crosscheck")]
    n_pass = sum(passed(r) for r in rs)
    return {
        "model": model,
        "baseline": any(r.get("baseline") for r in rs),
        "trials": len(rs),
        "n_note": f"n={len(rs)}" + (" (one run: not a rate)" if len(rs) == 1 else ""),
        "score": g("score"),
        "best_gate": ok[-1] if ok else None,
        "gates": [
            {k: x.get(k) for k in ("id", "pass", "points", "max_points")}
            for x in best.get("gates", [])
        ],
        "verify_calls": g("verify_calls"),
        "flights": g("flights"),
        "wall_s": g("wall_s"),
        "cost_usd": g("cost_usd"),
        "tokens": round(statistics.mean(tok(r) for r in rs)),
        "failure_reasons": [x for r in rs for x in r.get("failure_reasons", [])],
        "pass_within_budget": {
            "runs": len(rs),
            "passed": n_pass,
            "rate": round(n_pass / len(rs), 2),
        },
        "first_viable": {
            "runs": len(rs),
            "reached": len(fv),
            "mean_cost_usd": mean(f.get("cost_usd") for f in fv),
            "mean_iterations": mean(f.get("iteration") for f in fv),
            "mean_verify_calls": mean(f.get("verify_calls") for f in fv),
        },
        "robustness_gap": mean(r.get("robustness_gap") for r in rs),
        "prediction_error": {
            k: mean(p.get(k) for p in pe)
            for k in ("hover_throttle_pct", "flight_time_pct")
        },
        "pareto": pareto([p for r in rs for p in viable(r)]),
        "crosscheck": {
            "runs": len(cc),
            "mean_score_change": mean(c.get("score_change") for c in cc),
        }
        if cc
        else None,
    }


def build(results_dir=RESULTS):
    runs = []
    for f in sorted(Path(results_dir).glob("*/results.json")):
        r = json.loads(f.read_text())
        a = (
            f.parent / "agent.json"
        )  # the final usage event arrives after finish: the harness's own totals are the complete ones
        if a.exists():
            a = json.loads(a.read_text())
            if a.get("claude_cost_usd") is not None:
                r["cost_usd"] = a["claude_cost_usd"]
            if a.get("tokens", {}).get("input"):
                r["tokens"] = a["tokens"]
        runs.append((base_model(r, f.parent.name), r))
    main, curve = {}, {}
    for m, r in runs:
        if r.get("baseline") or r.get("feedback", "whatif") == "whatif":
            main.setdefault(m, []).append(r)
        if not r.get("baseline"):
            c = curve.setdefault(m, {}).setdefault(
                r.get("feedback", "whatif"), {"runs": 0, "passed": 0}
            )
            c["runs"] += 1
            c["passed"] += passed(r)
    for lv in curve.values():
        for c in lv.values():
            c["rate"] = round(c["passed"] / c["runs"], 2)
    entries = [entry(m, rs) for m, rs in main.items()]
    # ties: fewer verify calls, then less wall time, then fewer tokens
    entries.sort(
        key=lambda e: (
            -e["score"],
            e["verify_calls"] or 0,
            e["wall_s"] or 0,
            e["tokens"],
        )
    )
    curve = {m: {l: lv[l] for l in LEVELS if l in lv} for m, lv in curve.items()}
    return {
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "entries": entries,
        "feedback_curve": curve,
    }


if __name__ == "__main__":
    lb = build()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "leaderboard.json").write_text(json.dumps(lb, indent=1))
    e = lb["entries"]
    if "--pick" in sys.argv:
        e = [x for x in e if not x["baseline"]]
        if not e:
            sys.exit("ERROR: no results yet")
        print(
            f"lowest {e[-1]['model']} {e[-1]['score']}\nhighest {e[0]['model']} {e[0]['score']}"
        )
    else:
        for x in e:
            p = x["pass_within_budget"]
            print(
                f"{x['model']:12} {x['score']:6} {x['n_note']} gate={x['best_gate']} pass={p['passed']}/{p['runs']} verify={x['verify_calls']} ${x['cost_usd']}{' [baseline]' if x['baseline'] else ''}"
            )
        for m, lv in lb["feedback_curve"].items():
            print(m, {l: f"{c['passed']}/{c['runs']}" for l, c in lv.items()})
