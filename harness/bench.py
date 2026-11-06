"""Benchmark driver: bench.py --models haiku,sonnet,opus,fable --trials 1 [--parallel 4] [--feedback whatif,diag] [--sweep] [--wall-min N]

Per (model, level, trial): headless session on hfbox (hf.sh bg, ports 8801-8804), agent via run_agent, pull results.json + events.jsonl into
results/<runname>-<trial>/, stop the session. runname = model for whatif (the default level), model-LEVEL otherwise, so run dirs stay distinct
by level. A run dir with results.json is skipped (resume-safe).
--sweep: haiku and sonnet at none,passfail,diag,whatif; opus and fable at whatif only; cheapest model first so a cut-off run leaves usable data.
"""

import argparse
import hashlib
import json
import os
import queue
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_agent import ROOT, run_agent

HF = str(ROOT / "scripts" / "hf.sh")
RESULTS = ROOT / "results"
REMOTE_RUNS = "/workspace/wt/dv/runs"
PORTS = [8801, 8802, 8803, 8804]
SESSIONS = dict(zip(PORTS, "ABCD"))
LEVELS = ["none", "passfail", "diag", "whatif"]
COST_ORDER = ["haiku", "sonnet", "opus", "fable"]
SWEEP = {"haiku": LEVELS, "sonnet": LEVELS, "opus": ["whatif"], "fable": ["whatif"]}


def runname(model, level):
    return model if level == "whatif" else f"{model}-{level}"


def plan(models, levels, trials, sweep=False, done=lambda name: False):
    """[(model, level, trial)] still to run, cheapest model first, then trial, then level."""
    combos = [(m, l) for m in models for l in (SWEEP[m] if sweep else levels)]
    combos.sort(
        key=lambda ml: (
            COST_ORDER.index(ml[0]) if ml[0] in COST_ORDER else 9,
            LEVELS.index(ml[1]),
        )
    )
    return [
        (m, l, t)
        for t in range(trials)
        for m, l in combos
        if not done(f"{runname(m, l)}-{t}")
    ]


def sh(*args, timeout=300):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def ssh(cmd):
    return sh(
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        os.environ.get("DV_HOST", "hfbox"),
        cmd,
    )


def ssh_cat(path):
    r = ssh(f"cat {path}")
    return r.stdout if r.returncode == 0 else None


def wait_state(port, secs=120):
    end = time.time() + secs
    while time.time() < end:
        try:
            return json.load(
                urllib.request.urlopen(f"http://127.0.0.1:{port}/state", timeout=3)
            )
        except Exception:
            time.sleep(2)
    return None


def render_prompt(level):
    """CORE's prompt for a feedback level; falls back to the plain call when render_prompt has no --feedback yet."""
    r = sh(HF, "run", f"python -m dv.render_prompt --feedback {level}")
    if r.returncode or not r.stdout.strip():
        r = sh(HF, "run", "python -m dv.render_prompt")
    if r.returncode or not r.stdout.strip():
        sys.exit(f"ERROR: dv.render_prompt failed on hfbox: {r.stderr[-300:]}")
    return r.stdout


def one(model, level, trial, port, prompt, wall_s):
    rn = runname(model, level)
    tag = f"{rn}-{trial}"
    out = RESULTS / tag
    sess, name = SESSIONS[port], f"bench-{tag}"
    run = f"{REMOTE_RUNS}/{sess}-{rn}-{trial}"
    try:
        ssh(
            f"rm -rf {run}"
        )  # leftovers of a failed attempt must not pass for this run's results
        r = sh(
            HF,
            "bg",
            name,
            f"python -m dv.server --session {sess} --port {port} --model {rn} --trial {trial} --feedback {level}",
        )
        if r.returncode:
            return f"{tag}: ERROR session start failed: {r.stderr[-200:]}"
        if not wait_state(port):
            return f"{tag}: ERROR no /state on port {port} (bridge or server down); log: hf.sh log {name}"
        run_agent(model, port, out, prompt, wall_s=wall_s, quiet=True)
        res = None
        for _ in range(
            90
        ):  # held-out rerun runs inside finish; wait up to 15 min for results.json
            res = ssh_cat(f"{run}/results.json")
            if res:
                break
            time.sleep(10)
        if not res:
            return f"{tag}: ERROR no results.json in {run}"
        (out / "events.jsonl").write_text(ssh_cat(f"{run}/events.jsonl") or "")
        for f in ("design.json", "controller.py"):  # kept for crosscheck.py
            (out / f).write_text(ssh_cat(f"{run}/{f}") or "")
        (out / "results.json").write_text(
            res
        )  # written last: its presence marks the run done
        return f"{tag}: score {json.loads(res).get('score')}"
    finally:
        sh(HF, "stop", name)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--models", default="haiku,sonnet,opus,fable")
    a.add_argument("--trials", type=int, default=1)
    a.add_argument("--parallel", type=int, default=4)
    a.add_argument(
        "--feedback", default="whatif", help="comma list of none,passfail,diag,whatif"
    )
    a.add_argument("--sweep", action="store_true")
    a.add_argument(
        "--wall-min",
        type=float,
        help="agent wall limit in minutes (default: task.toml)",
    )
    a = a.parse_args()
    levels = a.feedback.split(",")
    bad = [l for l in levels if l not in LEVELS]
    if bad:
        sys.exit(f"ERROR: unknown feedback level {bad}")
    RESULTS.mkdir(exist_ok=True)
    todo = plan(
        a.models.split(","),
        levels,
        a.trials,
        a.sweep,
        lambda n: (RESULTS / n / "results.json").exists(),
    )
    print(f"{len(todo)} to run: {todo}")
    if not todo:
        return
    prompts = {}
    for l in sorted({l for _, l, _ in todo}):
        prompts[l] = render_prompt(l)
        h = hashlib.sha256(prompts[l].encode()).hexdigest()[:16]
        (RESULTS / f"prompt-{l}.txt").write_text(prompts[l])
        (RESULTS / f"prompt-{l}.sha256").write_text(h + "\n")
        print(f"prompt[{l}] sha256 {h} ({len(prompts[l])} chars)")
    n = min(a.parallel, len(PORTS))
    ports = queue.Queue()
    [ports.put(p) for p in PORTS[:n]]
    # sshd forbids port forwarding: the bridge speaks to dv/pipe.py over plain ssh exec
    tun = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).parent / "bridge.py"),
            "--port",
            ",".join(map(str, PORTS[:n])),
        ]
    )
    time.sleep(4)
    if tun.poll() is not None:
        sys.exit("ERROR: bridge did not start (port in use?)")
    wall_s = a.wall_min * 60 if a.wall_min else None

    def job(t):
        p = ports.get()
        try:
            return one(*t, p, prompts[t[1]], wall_s)
        finally:
            ports.put(p)

    try:
        with ThreadPoolExecutor(n) as ex:
            for line in ex.map(job, todo):
                print(line, flush=True)
    finally:
        tun.terminate()
    subprocess.run([sys.executable, str(Path(__file__).parent / "leaderboard.py")])


if __name__ == "__main__":
    main()
