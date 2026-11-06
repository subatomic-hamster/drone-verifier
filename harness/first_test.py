"""The benchmark's score: how the verifier rates an agent's very first test. Replays a run's events.jsonl to the state at its first call that reached the simulator (first `fly` that flew or
`verify` that flew; L0/L1 refusals fly nothing, so they do not count), flies that design and controller on the held-out seed through V.run_gates, and adds `first_test` and `first_test_score`
to results.json. `score` stays the final score. Nothing else in the run dir is written. Safe to rerun; a finished run only, so live sessions are left alone.

Also adds partial credit (dv/partial.py) for the final and first-test states: partial_score, partial_gates, first_test.partial_score, first_test.partial_gates; baselines too.

On hfbox: python harness/first_test.py [--force | --repartial] RUN_DIR...    or --all ($DV_RUNS/*)    --dump prints what is done, as JSON
On the laptop: python harness/first_test.py --pull             merges --dump into results/*/results.json, matched by the event log's first timestamp (a baseline by model)"""

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APPLY = threading.Lock()
WORKERS = 4


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
                    _apply(merged)
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
    from dv import heldout
    from dv import verify as V

    st = replay(last_run(Path(run_dir) / "events.jsonl"))
    if "reason" in st:
        return dict(score=0.0, gates=[], trigger=None, scenario=None, attempt_flown=None, design=None, cost_usd=None, mass_kg=None, reason=st["reason"])  # fmt: skip
    b = _apply(st["choices"])
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


def _write(path, res):
    """results.json, atomically."""
    tmp = Path(path).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(res, indent=1))
    os.replace(tmp, path)
    return res


def add_fields(path, first):
    """Adds the two fields to a results.json: nothing else in it changes."""
    res = json.loads(Path(path).read_text())
    res["first_test"], res["first_test_score"] = first, first["score"]
    return _write(path, res)


def _apply(choices):
    from dv import design as D

    with APPLY:  # one design at a time: --all scores runs in threads
        return D.apply(choices)


def partial_fields(rd, res):
    """Partial credit for the final state (the run dir's design and controller, as heldout.run) and the first-test state. -> the fields to add; strict fields are not touched."""
    from dv import heldout
    from dv import partial as P

    seed = heldout.heldout_seed()
    choices = json.loads((rd / "design.json").read_text()) if (rd / "design.json").exists() else {}  # fmt: skip
    code = (rd / "controller.py").read_text() if (rd / "controller.py").exists() else ""
    final = P.score(_apply(choices), code, seed)
    if res.get(
        "baseline"
    ):  # one shot: leaderboard.py reads its final as its first test
        return final, None
    st = replay(last_run(rd / "events.jsonl"))
    if "reason" in st:
        return final, dict(partial_score=0.0, partial_gates=[])
    same = st["choices"] == choices and st["code"] == code
    return final, final if same else P.score(_apply(st["choices"]), st["code"], seed)


def add_partial(rd, res):
    final, first = partial_fields(rd, res)
    res = json.loads((rd / "results.json").read_text())
    res.update(final)
    if first is not None:
        res["first_test"].update(first)
    return _write(rd / "results.json", res)


def finished(run_dir):
    """A live session has no `final` yet in its last run (and its results.json, if any, would be an older run's). A baseline has no session at all."""
    rd = Path(run_dir)
    if not (rd / "results.json").exists():
        return False
    if json.loads((rd / "results.json").read_text()).get("baseline"):
        return True
    if not (rd / "events.jsonl").exists():
        return False
    return any(e.get("kind") == "final" for e in last_run(rd / "events.jsonl"))


def process(run_dir, force=False, repartial=False):
    """-> results dict, or None for a run that is not finished. Skips what is already there: force redoes everything, repartial the partial fields only."""
    rd = Path(run_dir)
    if not finished(rd):
        return None
    res = json.loads((rd / "results.json").read_text())
    if not res.get("baseline") and ("first_test" not in res or force):
        res = add_fields(rd / "results.json", first_test(rd))
    if "partial_score" not in res or force or repartial:
        res = add_partial(rd, res)
    return res


def t0(run_dir):
    with open(Path(run_dir) / "events.jsonl") as f:
        return json.loads(f.readline())["t"]


def key(run_dir, res):
    """What identifies a run on both machines: the event log's first timestamp; a baseline's model."""
    return f"baseline:{res['model']}" if res.get("baseline") else str(t0(run_dir))


def dump(runs):
    """Finished run dirs that have the fields -> {key: {run, first_test, first_test_score, partial_score, partial_gates}}."""
    out = {}
    for rd in runs:
        if finished(rd):
            res = json.loads((rd / "results.json").read_text())
            got = {k: res[k] for k in ("first_test", "first_test_score", "partial_score", "partial_gates") if k in res}  # fmt: skip
            if got:
                out[key(rd, res)] = dict(run=rd.name, **got)
    return out


def pull():
    """Laptop: merges hfbox's fields into results/*/results.json, for the runs whose event log is the same one."""
    r = subprocess.run([str(ROOT / "scripts" / "hf.sh"), "run", "python harness/first_test.py --dump"], capture_output=True, text=True)  # fmt: skip
    if r.returncode:
        sys.exit(f"ERROR: --dump failed on hfbox: {r.stderr[-300:]}")
    done = json.loads(r.stdout)
    print(HEAD)
    for f in sorted((ROOT / "results").glob("*/results.json")):
        res = json.loads(f.read_text())
        if not res.get("baseline") and not (f.parent / "events.jsonl").exists():
            continue
        got = done.get(key(f.parent, res))
        if got is None:
            print(f"{f.parent.name:24} not scored")
            continue
        res.update({k: v for k, v in got.items() if k != "run"})
        print(row(f.parent.name, _write(f, res)))


HEAD = f"{'run':24} {'first partial':>13} {'final partial':>13} {'first strict':>12} {'final strict':>12}"


def row(name, res):
    ft = res.get("first_test") or {}
    first = ft.get("partial_score", res.get("partial_score"))
    strict = res.get("first_test_score", res["score"] if res.get("baseline") else None)
    return f"{name:24} {first!s:>13} {res.get('partial_score')!s:>13} {strict!s:>12} {res['score']!s:>12}"


def main():
    args = sys.argv[1:]
    force, repartial = "--force" in args, "--repartial" in args
    args = [a for a in args if a not in ("--force", "--repartial")]
    if "--pull" in args:
        return pull()
    runs = Path(os.environ.get("DV_RUNS", "runs"))
    every = sorted(p for p in runs.iterdir() if p.is_dir()) if ("--all" in args or "--dump" in args) else []  # fmt: skip
    if "--dump" in args:
        return print(json.dumps(dump(every)))
    todo = [rd for rd in (every or [Path(a) for a in args]) if finished(rd)]
    # the flights queue on the machine's flock slots, whatever else is flying
    with ThreadPoolExecutor(WORKERS) as ex:
        done = list(ex.map(lambda rd: process(rd, force, repartial), todo))
    print(HEAD)
    for rd, res in zip(todo, done):
        print(row(rd.name, res), flush=True)
    return None


if __name__ == "__main__":
    main()
