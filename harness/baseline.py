"""Baselines with no LLM cost (docs/LAYERS.md). Same results.json shape, flagged baseline: true.

  python harness/baseline.py [--fly N]      laptop: runs the remote part on hfbox through scripts/hf.sh, pulls results/human-1 and results/brute_force-1
  python harness/baseline.py --remote ...   on hfbox (what the line above runs); writes $DV_RUNS/baseline-<name>/results.json

human: reference/design.json + reference/controller.py, public and held-out seeds.
brute_force: every catalog combination through L0 (dv.design) and L1 (dv/analytic.py once CORE commits it, else G2's static thrust run),
then the cheapest and lightest non-dominated few fly with the reference controller. The result is the best flown candidate, plus
enumerated / passed_l0 / passed_l1 counts and every flown candidate under `candidates`.
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF = str(ROOT / "scripts" / "hf.sh")
REMOTE_RUNS = "/workspace/wt/dv/runs"
FIXED = ("battery_x_mm", "battery_y_mm")


def l1(b):
    """-> (ok, reason, kind). CORE's dv.analytic.l1 (None while the design has no pack or ESC counts as failed)."""
    from dv import analytic

    r = getattr(b, "l1", None) or analytic.l1(b)
    return (
        (bool(r["ok"]), r["reason"], "analytic")
        if r
        else (False, "no pack or ESC to analyse", "analytic")
    )


def screen(choices):
    """L0 then L1 for one combination. -> dict(choices, l0, l1, cost, mass)."""
    from dv import design as D
    from dv import verify as V

    try:
        b = D.apply(choices)
    except Exception as e:  # an invalid combination is a failed L0
        return dict(choices=choices, l0=False, l1=False, why=str(e)[:120])
    g = V.g1(b)
    out = dict(
        choices=choices,
        l0=g["ok"],
        l1=False,
        cost=round(b.cost, 2),
        mass=round(b.mass, 4),
        why=g["reason"],
    )
    if g["ok"]:
        out["l1"], out["why"], out["l1_kind"] = l1(b)
    return out


def nondominated(rows):
    rows = sorted(rows, key=lambda r: (r["cost"], r["mass"]))
    front, best = [], float("inf")
    for r in rows:
        if r["mass"] < best:
            front.append(r)
            best = r["mass"]
    return front


def result(model, choices, code, extra):
    """The same figures results.json carries for an agent run, from public and held-out flights of one design + controller."""
    from dv import design as D
    from dv import heldout
    from dv import verify as V

    b = D.apply(choices)
    pub = V.run_gates(b, code, D.TASK["public"]["seed"])
    held = V.run_gates(b, code, heldout.heldout_seed())
    passed = all(g["pass"] for g in held["gates"])
    return dict(
        model=model,
        trial=1,
        baseline=True,
        score=held["score"],
        gates=held["gates"],
        verify_calls=0,
        flights=pub["flights"] + held["flights"],
        tokens=dict(input=0, output=0),
        cost_usd=0.0,
        wall_s=None,
        failure_reasons=[
            f"{g['id']}: {g['reason']}" for g in held["gates"] if not g["pass"]
        ],
        pass_within_budget=passed,
        robustness_gap=round(pub["score"] - held["score"], 1),
        first_viable=dict(
            iteration=1,
            verify_calls=0,
            cost_usd=round(b.cost, 2),
            mass_kg=round(b.mass, 4),
        ),
        design={s: choices.get(s) for s in D.SLOTS}
        | {k: choices[k] for k in FIXED if k in choices},
        cost_usd_bom=round(b.cost, 2),
        mass_kg=round(b.mass, 4),
        **extra,
    )


def remote(n_fly):
    from dv import design as D

    ref = json.loads((ROOT / "reference" / "design.json").read_text())
    code = (ROOT / "reference" / "controller.py").read_text()
    runs = Path(os.environ.get("DV_RUNS", "runs"))

    def save(name, res, choices, code):
        d = runs / f"baseline-{name}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "results.json").write_text(json.dumps(res, indent=1))
        (d / "design.json").write_text(json.dumps(choices, indent=1))
        (d / "controller.py").write_text(code)

    h = result("human", ref, code, {})
    save("human", h, ref, code)
    print(f"human: score {h['score']}", flush=True)

    slots = list(D.SLOTS)
    combos = [
        dict(zip(slots, c), **{k: ref[k] for k in FIXED if k in ref})
        for c in itertools.product(
            *[[o["part"] for o in D.SLOTS[s]["options"]] for s in slots]
        )
    ]
    with ProcessPoolExecutor(4) as ex:  # OMP 1 thread each; far under the 8-flight cap
        rows = list(ex.map(screen, combos, chunksize=16))
    p0, p1 = [r for r in rows if r["l0"]], [r for r in rows if r["l1"]]
    print(
        f"enumerated {len(rows)}, passed L0 {len(p0)}, passed L1 {len(p1)}", flush=True
    )
    picks = nondominated(p1)[:n_fly]
    flown = []
    for r in picks:
        x = result("brute_force", r["choices"], code, {})
        flown.append(x)
        print(f"flew ${r['cost']} {r['mass']} kg: score {x['score']}", flush=True)
    if not flown:
        sys.exit("ERROR: no candidate passed L0 and L1")
    best = max(flown, key=lambda x: (x["score"], -x["cost_usd_bom"]))
    best["enumerated"], best["passed_l0"], best["passed_l1"] = (
        len(rows),
        len(p0),
        len(p1),
    )
    best["l1_kind"] = next((r.get("l1_kind") for r in p0 if r.get("l1_kind")), None)
    best["candidates"] = [
        dict(
            cost_usd=x["cost_usd_bom"],
            mass_kg=x["mass_kg"],
            score=x["score"],
            design=x["design"],
        )
        for x in flown
    ]
    best["viable_designs"] = [
        dict(cost_usd=x["cost_usd_bom"], mass_kg=x["mass_kg"])
        for x in flown
        if x["gates"][2]["pass"]
    ]  # passed G3
    save("brute_force", best, best["design"], code)
    print(f"brute_force: best score {best['score']}")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--remote", action="store_true")
    a.add_argument(
        "--pull",
        action="store_true",
        help="only copy the finished baseline results from hfbox",
    )
    a.add_argument("--fly", type=int, default=4, help="candidates to fly")
    a = a.parse_args()
    if a.remote:
        sys.path.insert(0, str(ROOT))
        return remote(a.fly)
    if not a.pull:
        r = subprocess.run(
            [HF, "run", f"python harness/baseline.py --remote --fly {a.fly}"]
        )
        if r.returncode:
            sys.exit("ERROR: baseline run on hfbox failed")
    for name, local in (("human", "human-1"), ("brute_force", "brute_force-1")):
        (ROOT / "results" / local).mkdir(parents=True, exist_ok=True)
        for f in (
            "design.json",
            "controller.py",
            "results.json",
        ):  # results.json last: marks it done; the others feed crosscheck.py
            out = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    os.environ.get("DV_HOST", "hfbox"),
                    f"cat {REMOTE_RUNS}/baseline-{name}/{f}",
                ],
                capture_output=True,
                text=True,
            )
            (ROOT / "results" / local / f).write_text(out.stdout)
    subprocess.run([sys.executable, str(ROOT / "harness" / "leaderboard.py")])


if __name__ == "__main__":
    main()
