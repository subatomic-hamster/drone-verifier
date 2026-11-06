"""Benchmark driver: bench.py --models haiku,sonnet,opus,fable --trials 1 [--parallel 4]

Per (model, trial): headless session on hfbox (hf.sh bg, ports 8801-8804), agent via run_agent, pull results.json + events.jsonl into
results/<model>-<trial>/, stop the session. A (model, trial) with results/<model>-<trial>/results.json is skipped (resume-safe).
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


def sh(*args, timeout=300):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def ssh_cat(path):
    r = sh(
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        os.environ.get("DV_HOST", "hfbox"),
        f"cat {path}",
    )
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


def render_prompt():
    r = sh(HF, "run", "python -m dv.render_prompt")
    if r.returncode or not r.stdout.strip():
        sys.exit(f"ERROR: dv.render_prompt failed on hfbox: {r.stderr[-300:]}")
    return r.stdout


def one(model, trial, port, prompt):
    out = RESULTS / f"{model}-{trial}"
    sess, name = SESSIONS[port], f"bench-{model}-{trial}"
    run = f"{REMOTE_RUNS}/{sess}-{model}-{trial}"
    try:
        r = sh(
            HF,
            "bg",
            name,
            f"python -m dv.server --session {sess} --port {port} --model {model} --trial {trial}",
        )
        if r.returncode:
            return f"{model}-{trial}: ERROR session start failed: {r.stderr[-200:]}"
        if not wait_state(port):
            return f"{model}-{trial}: ERROR no /state on port {port} (tunnel or server down); log: hf.sh log {name}"
        run_agent(model, port, out, prompt, quiet=True)
        res = None
        for _ in range(
            90
        ):  # held-out rerun runs inside finish; wait up to 15 min for results.json
            res = ssh_cat(f"{run}/results.json")
            if res:
                break
            time.sleep(10)
        if not res:
            return f"{model}-{trial}: ERROR no results.json in {run}"
        ev = ssh_cat(f"{run}/events.jsonl") or ""
        (out / "events.jsonl").write_text(ev)
        (out / "results.json").write_text(
            res
        )  # written last: its presence marks the trial done
        return f"{model}-{trial}: score {json.loads(res).get('score')}"
    finally:
        sh(HF, "stop", name)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--models", default="haiku,sonnet,opus,fable")
    a.add_argument("--trials", type=int, default=1)
    a.add_argument("--parallel", type=int, default=4)
    a = a.parse_args()
    RESULTS.mkdir(exist_ok=True)
    todo = [
        (m, t)
        for t in range(a.trials)
        for m in a.models.split(",")
        if not (RESULTS / f"{m}-{t}" / "results.json").exists()
    ]
    print(f"{len(todo)} to run: {todo}")
    if not todo:
        return
    prompt = render_prompt()
    h = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    (RESULTS / "prompt.txt").write_text(prompt)
    (RESULTS / "prompt.sha256").write_text(h + "\n")
    print(f"prompt sha256 {h} ({len(prompt)} chars)")
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
        sys.exit(
            "ERROR: ssh tunnel to hfbox did not come up (check ssh agent / funnel)"
        )

    def job(mt):
        p = ports.get()
        try:
            return one(*mt, p, prompt)
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
