"""Demo driver.  demo.py single MODEL | versus LOW HIGH | replay
single/versus: GUI sessions on hfbox, tunnel, web/ served locally, agents started live, URL printed.
replay: serves web/ plus results/ (saved events.jsonl + leaderboard.json) with no hfbox link at all.
URL convention (GUI worker must honour, or change URL_* below): live ?live=8801[,8802]  replay ?replay=1
"""

import argparse
import functools
import http.server
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bench import HF, sh, wait_state
from run_agent import ROOT, run_agent

# served from the repo root in both modes: the page reads ../results/ too
URL_LIVE = "http://127.0.0.1:{web}/web/index.html?live={ports}"
URL_REPLAY = "http://127.0.0.1:{web}/web/index.html?replay=1"


def die(msg):
    sys.exit(f"ERROR: {msg}")


def serve(directory):
    """Static server on a free port; returns the port. Directory = web/ (live) or repo root (replay)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    h = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(directory)
    )
    h.log_message = lambda *a: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return port


def check(port):
    """One-line failure for each piece that is down: tunnel, /state, /events stream, GUI stream."""
    try:
        socket.create_connection(("127.0.0.1", port), 3).close()
    except OSError:
        die(
            f"tunnel down: nothing listening on 127.0.0.1:{port} (python harness/bridge.py --port {port})"
        )
    st = wait_state(port, 150)
    if not st:
        die(
            f"/state on port {port} does not answer (session not up: hf.sh log demo-{port})"
        )
    for path in ("/events", "/stream"):
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=60)
            r.read(1)
            r.close()
        except Exception as e:
            die(f"{path} on port {port} does not answer ({e})")


def live(models):
    prompt = sh(HF, "run", "python -m dv.render_prompt").stdout
    if not prompt.strip():
        die("dv.render_prompt printed nothing on hfbox")
    ports = [8801, 8802][: len(models)]  # GUI sessions; the benchmark uses 8803-8810
    for m, p, s in zip(models, ports, "AB"):
        r = sh(
            HF,
            "bg",
            f"demo-{p}",
            f"python -m dv.server --session {s} --port {p} --gui --model {m} --trial 0",
        )
        if r.returncode:
            die(f"could not start session on hfbox: {r.stderr[-200:]}")
    # sshd forbids port forwarding: the bridge speaks to dv/pipe.py over plain ssh exec
    tun = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).parent / "bridge.py"),
            "--port",
            ",".join(map(str, ports)),
        ]
    )
    try:
        time.sleep(4)
        if tun.poll() is not None:
            die("ssh tunnel to hfbox did not come up (ssh agent / funnel)")
        for p in ports:
            check(p)
        web = serve(ROOT)
        print(
            "OPEN",
            URL_LIVE.format(web=web, ports=",".join(map(str, ports))),
            flush=True,
        )
        ts = [
            threading.Thread(
                target=run_agent,
                args=(m, p, ROOT / "results" / f"demo-{m}-{p}", prompt),
                kwargs={"quiet": True},
            )
            for m, p in zip(models, ports)
        ]
        [t.start() for t in ts]
        [t.join() for t in ts]
        print("agents done; Ctrl-C to close the demo", flush=True)
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass
    finally:
        tun.terminate()
        for p in ports:
            sh(HF, "stop", f"demo-{p}")


def replay():
    if not (ROOT / "results" / "leaderboard.json").exists():
        die("no results/leaderboard.json: run bench.py first")
    print("OPEN", URL_REPLAY.format(web=serve(ROOT)), flush=True)
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("mode", choices=["single", "versus", "replay"])
    a.add_argument("models", nargs="*")
    a = a.parse_args()
    if a.mode == "replay":
        replay()
    elif (
        a.mode == "single"
        and len(a.models) == 1
        or a.mode == "versus"
        and len(a.models) == 2
    ):
        live(a.models)
    else:
        die("usage: demo.py single MODEL | versus LOW HIGH | replay")
