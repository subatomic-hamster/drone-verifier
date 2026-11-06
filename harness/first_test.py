"""The benchmark's score: how the verifier rates an agent's very first test. Replays a run's events.jsonl to the state at its first call that reached the simulator (first `fly` that flew or
`verify` that flew; L0/L1 refusals fly nothing, so they do not count), flies that design and controller on the held-out seed through V.run_gates, and adds `first_test` and `first_test_score`
to results.json. `score` stays the final score. Nothing else in the run dir is written. Safe to rerun; a finished run only, so live sessions are left alone.

On hfbox: python harness/first_test.py [--force] RUN_DIR...    or --all ($DV_RUNS/*)    --dump prints what is done, as JSON
On the laptop: python harness/first_test.py --pull             merges --dump into results/*/results.json, matched by the event log's first timestamp"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def last_run(path):
    """The events of the last session in a log: a restarted session appends to the file, and each one opens with `config`."""
    evs = []
    for line in Path(path).read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("kind") == "config":
            evs = []
        evs.append(e)
    return evs


def replay(evs):
    """The state at the first simulator-touching call. -> {choices, code, trigger, scenario, attempt_flown}, or {reason} if the agent never tested."""
    from dv import design as D
    from dv.sandbox import check_source

    choices, code, tool, out = {}, "", None, None
    for e in evs:
        k = e.get("kind")
        if k == "final":
            break
        if out is not None:
            if k == "flight_end":
                out["attempt_flown"] = bool(e.get("pass"))
                return out
        elif k == "tool_call":
            tool = e.get("tool")
            if tool == "design_set":
                # as Session.design_set: slots merge into the current choices, and a call that raises changes nothing
                merged = {**choices, **(e.get("args", {}).get("choices") or {})}
                try:
                    D.apply(merged)
                    choices = merged
                except (D.DesignError, ValueError, TypeError, KeyError):
                    pass
        elif k == "code" and (e.get("ok") or not check_source(e.get("text", ""))):
            # as Session.write_controller: code past the source check is the current code even when its trial step crashed (ok false)
            code = e["text"]
        elif k == "flight_start":
            if tool not in ("fly", "verify"):
                break  # finish's own held-out flights: the agent never tested
            out = dict(choices=choices, code=code, trigger=tool, scenario=e.get("scenario"), attempt_flown=None)  # fmt: skip
    return out or dict(reason="the agent never reached the simulator")


def first_test(run_dir, seed=None):
    """-> the first_test dict for a run dir. No file is written here; the flights use their own temp dirs."""
    from dv import design as D
    from dv import heldout
    from dv import verify as V

    st = replay(last_run(Path(run_dir) / "events.jsonl"))
    if "reason" in st:
        return dict(score=0.0, gates=[], trigger=None, scenario=None, attempt_flown=None, design=None, cost_usd=None, mass_kg=None, reason=st["reason"])  # fmt: skip
    b = D.apply(st["choices"])
    r = V.run_gates(b, st["code"], heldout.heldout_seed() if seed is None else seed)
    return dict(
        score=r["score"],
        gates=r["gates"],
        trigger=st["trigger"],
        scenario=st["scenario"],
        attempt_flown=st["attempt_flown"],
        design=st["choices"],
        cost_usd=round(b.cost, 2),
        mass_kg=round(b.mass, 4),
    )


def add_fields(path, first):
    """Adds the two fields to a results.json, atomically: nothing else in it changes."""
    path = Path(path)
    res = json.loads(path.read_text())
    res["first_test"], res["first_test_score"] = first, first["score"]
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(res, indent=1))
    os.replace(tmp, path)
    return res


def finished(run_dir):
    """A live session has no `final` yet in its last run (and its results.json, if any, would be an older run's)."""
    rd = Path(run_dir)
    if not (rd / "results.json").exists() or not (rd / "events.jsonl").exists():
        return False
    return any(e.get("kind") == "final" for e in last_run(rd / "events.jsonl"))


def process(run_dir, force=False):
    """-> results dict, or None for a run that is not finished."""
    rd = Path(run_dir)
    if not finished(rd):
        return None
    res = json.loads((rd / "results.json").read_text())
    if "first_test" in res and not force:
        return res
    return add_fields(rd / "results.json", first_test(rd))


def t0(run_dir):
    with open(Path(run_dir) / "events.jsonl") as f:
        return json.loads(f.readline())["t"]


def dump(runs):
    """Finished run dirs that have the fields -> {first event time: {run, first_test, first_test_score}}."""
    out = {}
    for rd in runs:
        if finished(rd):
            res = json.loads((rd / "results.json").read_text())
            if "first_test" in res:
                out[str(t0(rd))] = dict(run=rd.name, first_test=res["first_test"], first_test_score=res["first_test_score"])  # fmt: skip
    return out


def pull():
    """Laptop: merges hfbox's fields into results/*/results.json, for the runs whose event log is the same one."""
    r = subprocess.run([str(ROOT / "scripts" / "hf.sh"), "run", "python harness/first_test.py --dump"], capture_output=True, text=True)  # fmt: skip
    if r.returncode:
        sys.exit(f"ERROR: --dump failed on hfbox: {r.stderr[-300:]}")
    done = json.loads(r.stdout)
    print(f"{'run':24} {'first_test_score':>16} {'final score':>12}")
    for f in sorted((ROOT / "results").glob("*/results.json")):
        if not (f.parent / "events.jsonl").exists():
            continue
        got = done.get(str(t0(f.parent)))
        if got is None:
            print(f"{f.parent.name:24} {'not scored':>16}")
            continue
        res = add_fields(f, got["first_test"])
        print(f"{f.parent.name:24} {res['first_test_score']:16g} {res['score']:12g}")


def main():
    args = sys.argv[1:]
    force = "--force" in args
    args = [a for a in args if a != "--force"]
    if "--pull" in args:
        return pull()
    runs = Path(os.environ.get("DV_RUNS", "runs"))
    every = sorted(p for p in runs.iterdir() if p.is_dir()) if ("--all" in args or "--dump" in args) else []  # fmt: skip
    if "--dump" in args:
        return print(json.dumps(dump(every)))
    print(f"{'run':24} {'first_test_score':>16} {'final score':>12}  first test")
    for rd in every or [Path(a) for a in args]:
        res = process(rd, force)
        if res is None:
            print(f"{rd.name:24} {'not finished':>16}", flush=True)
            continue
        ft = res["first_test"]
        print(f"{rd.name:24} {res['first_test_score']:16g} {res['score']:12g}  {ft['trigger']} {ft['scenario']} flown pass {ft['attempt_flown']}{' ' + ft['reason'] if ft.get('reason') else ''}", flush=True)  # fmt: skip
    return None


if __name__ == "__main__":
    main()
