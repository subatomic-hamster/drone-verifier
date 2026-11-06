"""`python -m dv.heldout RUN_DIR`: flies the design and controller left in a run dir on held-out seeds (env DV_HELDOUT_SEED, default in task.toml) and writes results.json. Also what finish() calls."""

import json
import os
import sys
from pathlib import Path

from dv import design as D
from dv import verify as V


def heldout_seed():
    return int(os.environ.get("DV_HELDOUT_SEED") or D.TASK["heldout"]["seed"])


def summarise(run_dir):
    """Figures of the run from its own event log: the numbers on the leaderboard can be recomputed from the run's files."""
    p = Path(run_dir) / "events.jsonl"
    out = dict(
        verify_calls=0,
        flights=0,
        tokens=dict(input=0, output=0),
        cost_usd=None,
        first_pass_flight_s=None,
        failure_reasons=[],
        wall_s=None,
    )
    t0 = None
    t_last = None
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        k = e.get("kind")
        if k == "final":
            break
        if k == "tool_call":
            t0 = t0 if t0 is not None else e["t"]
            t_last = e["t"]
        elif k == "tool_result":
            t_last = e["t"]
        elif k == "flight_start":
            out["flights"] += 1
        elif (
            k == "flight_end"
            and e.get("pass")
            and out["first_pass_flight_s"] is None
            and t0 is not None
        ):
            out["first_pass_flight_s"] = round(e["t"] - t0, 1)
        elif k == "verify":
            out["verify_calls"] += 1
            bad = next((g for g in e["gates"] if not g["pass"]), None)
            if bad:
                out["failure_reasons"].append(f"{bad['id']}: {bad['reason']}")
        elif k == "agent_usage":
            # the harness posts running totals, so the last one is the total
            out["tokens"]["input"] = int(
                e.get("input_tokens") or out["tokens"]["input"]
            )
            out["tokens"]["output"] = int(
                e.get("output_tokens") or out["tokens"]["output"]
            )
            if e.get("cost_usd") is not None:
                out["cost_usd"] = float(e["cost_usd"])
    if t0 is not None:
        out["wall_s"] = round(t_last - t0, 1)
    return out


def run(run_dir, model=None, trial=None, emit=None, seed=None):
    """Held-out rerun of what is in run_dir. -> the results dict (also written to results.json)."""
    rd = Path(run_dir)
    choices = (
        json.loads((rd / "design.json").read_text())
        if (rd / "design.json").exists()
        else {}
    )
    code = (rd / "controller.py").read_text() if (rd / "controller.py").exists() else ""
    summary = summarise(rd)
    b = D.apply(choices)
    r = V.run_gates(b, code, heldout_seed() if seed is None else seed, emit=emit)
    meta = (
        json.loads((rd / "meta.json").read_text())
        if (rd / "meta.json").exists()
        else {}
    )
    res = dict(
        model=model if model is not None else meta.get("model"),
        trial=trial if trial is not None else meta.get("trial"),
        score=r["score"],
        gates=r["gates"],
        **summary,
        design={s: choices.get(s) for s in D.SLOTS}
        | {k: choices[k] for k in ("battery_x_mm", "battery_y_mm") if k in choices},
        cost_usd_bom=round(b.cost, 2),
        mass_kg=round(b.mass, 4),
        cost_by_kind=b.cost_by_kind,
        heldout_flights=r["flights"],
    )
    (rd / "results.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1]), indent=1))
