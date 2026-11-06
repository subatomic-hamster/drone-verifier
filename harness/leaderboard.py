"""results/*-*/results.json -> results/leaderboard.json.  leaderboard.py [--pick]  (--pick prints lowest and highest scorer)."""

import json
import statistics
import sys
import time
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def build(results_dir=RESULTS):
    by = {}
    for f in sorted(Path(results_dir).glob("*/results.json")):
        r = json.loads(f.read_text())
        by.setdefault(r.get("model") or f.parent.name.rsplit("-", 1)[0], []).append(r)
    entries = []
    for model, rs in by.items():
        best = max(rs, key=lambda r: r.get("score", 0))
        mean = lambda k: round(statistics.mean(r.get(k) or 0 for r in rs), 2)
        passed = [g["id"] for g in best.get("gates", []) if g.get("pass")]
        tok = lambda r: sum((r.get("tokens") or {}).values())
        entries.append(
            {
                "model": model,
                "trials": len(rs),
                "score": mean("score"),
                "best_gate": passed[-1] if passed else None,
                "gates": [
                    {k: g.get(k) for k in ("id", "pass", "points", "max_points")}
                    for g in best.get("gates", [])
                ],
                "verify_calls": mean("verify_calls"),
                "flights": mean("flights"),
                "wall_s": mean("wall_s"),
                "cost_usd": mean("cost_usd"),
                "tokens": round(statistics.mean(tok(r) for r in rs)),
                "failure_reasons": [
                    x for r in rs for x in r.get("failure_reasons", [])
                ],
            }
        )
    # ties: fewer verify calls, then less wall time, then fewer tokens
    entries.sort(
        key=lambda e: (-e["score"], e["verify_calls"], e["wall_s"], e["tokens"])
    )
    return {"updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "entries": entries}


if __name__ == "__main__":
    lb = build()
    (RESULTS).mkdir(exist_ok=True)
    (RESULTS / "leaderboard.json").write_text(json.dumps(lb, indent=1))
    e = lb["entries"]
    if "--pick" in sys.argv:
        if not e:
            sys.exit("ERROR: no results yet")
        print(
            f"lowest {e[-1]['model']} {e[-1]['score']}\nhighest {e[0]['model']} {e[0]['score']}"
        )
    else:
        for x in e:
            print(
                f"{x['model']:8} {x['score']:6} n={x['trials']} gate={x['best_gate']} verify={x['verify_calls']} ${x['cost_usd']}"
            )
