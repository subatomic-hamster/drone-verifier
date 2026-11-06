"""`python -m dv.heldout RUN_DIR`: flies the design and controller left in a run dir on held-out seeds (env DV_HELDOUT_SEED, default in task.toml) and writes results.json. Also what finish() calls."""

import json
import os
import sys
from pathlib import Path

from dv import analytic
from dv import design as D
from dv import flight as F
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
        feedback=None,
        pass_within_budget=False,
        first_viable=None,
        public_score=None,
        viable_designs=[],
    )
    t0 = None
    t_last = None
    if not p.exists():
        return out
    iteration, cost, mass = 0, None, None
    for line in p.read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        k = e.get("kind")
        if k == "final":
            break
        if k == "config":
            out["feedback"] = e.get("feedback")
        elif k == "meters":
            cost, mass = e.get("cost_usd"), e.get("mass_kg")
        elif k == "flight_end" and e.get("pass") and e.get("scenario") == "nominal":
            # a flight only happens once L0 and L1 pass, so this design passed L0, L1 and G3
            seen = dict(cost_usd=cost, mass_kg=mass)
            if seen not in out["viable_designs"]:
                out["viable_designs"].append(seen)
            if out["first_viable"] is None:
                out["first_viable"] = dict(iteration=iteration, verify_calls=out["verify_calls"], **seen)  # fmt: skip
        if k == "tool_call":
            t0 = t0 if t0 is not None else e["t"]
            t_last = e["t"]
            iteration += e.get("tool") == "design_set"
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
            out["verify_calls"] += e.get("charged", True)
            out["public_score"] = e["score"]
            flight_gates = [g for g in e["gates"] if g["id"] in ("G3", "G4", "G5", "G6")]  # fmt: skip
            out["pass_within_budget"] |= all(g["pass"] for g in flight_gates)
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
    public = summary.pop("public_score")
    summary["feedback"] = summary["feedback"] or meta.get("feedback")
    res = dict(
        model=model if model is not None else meta.get("model"),
        trial=trial if trial is not None else meta.get("trial"),
        score=r["score"],
        gates=r["gates"],
        **summary,
        robustness_gap=None if public is None else round(public - r["score"], 1),
        prediction_error=prediction_error(b, r["flown"]),
        thrust_l1_vs_static_sim_pct=_thrust_error(b),
        energy_invariant=dict(
            ok=all(f["metrics"]["energy"]["ok"] for f in r["flown"]),
            flights=[
                dict(scenario=f["scenario"], seed=f["seed"], **f["metrics"]["energy"])
                for f in r["flown"]
            ],
        ),  # fmt: skip
        design={s: choices.get(s) for s in D.SLOTS}
        | {k: choices[k] for k in ("battery_x_mm", "battery_y_mm") if k in choices},
        cost_usd_bom=round(b.cost, 2),
        mass_kg=round(b.mass, 4),
        cost_by_kind=b.cost_by_kind,
        heldout_flights=r["flights"],
    )
    (rd / "results.json").write_text(json.dumps(res, indent=1))
    return res


def prediction_error(b, flown):
    """L1 against the nominal flight (model against sim, never real flight): {hover_throttle_pct, flight_time_pct}, signed (L1 - sim) / sim, null without a hover to compare."""
    nom = next((f["metrics"] for f in flown if f["scenario"] == "nominal"), None)
    if not (nom and b.l1 and nom.get("hover_duty_pct") and nom.get("hover_i_a")):
        return dict(hover_throttle_pct=None, flight_time_pct=None)
    spec = analytic.power_specs(b)
    usable = spec["capacity_ah"] * (1 - D.LIMITS["soc_min_pct"] / 100.0) * 3600.0
    sim_time = usable / nom["hover_i_a"]
    return dict(
        hover_throttle_pct=round(100.0 * (b.l1["hover_throttle_pct"] / nom["hover_duty_pct"] - 1), 1),
        flight_time_pct=round(100.0 * (b.l1["flight_time_s"] / sim_time - 1), 1),
    )  # fmt: skip


def _thrust_error(b):
    """L1's full-throttle thrust against a static thrust run in the sim."""
    if not (b.xml and b.l1):
        return None
    sim = F.static_thrust(b.xml)["thrust_n"]
    return round(100.0 * (b.l1["thrust_max_n"] / sim - 1), 1)


def crosscheck(run_dir, sim_dt=0.0005, seed=None):
    """Re-fly a run's final design and controller at half the timestep, on the held-out seeds. After the benchmark, not during agent runs. -> {score, gates, sim_dt, score_was, change}. Does not touch results.json."""
    rd = Path(run_dir)
    choices = json.loads((rd / "design.json").read_text())
    code = (rd / "controller.py").read_text()
    r = V.run_gates(D.apply(choices), code, heldout_seed() if seed is None else seed, sim_dt=sim_dt)  # fmt: skip
    was = None
    if (rd / "results.json").exists():
        was = json.loads((rd / "results.json").read_text()).get("score")
    return dict(
        score=r["score"],
        gates=r["gates"],
        sim_dt=sim_dt,
        score_was=was,
        change=None if was is None else round(r["score"] - was, 1),
    )


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1]), indent=1))
