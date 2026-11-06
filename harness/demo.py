"""Demo driver.  demo.py serve [--port 8790] | single MODEL | versus LOW HIGH | replay
serve: the Start panel at /web/index.html on 127.0.0.1 (POST /api/start, POST /api/stop, GET /api/status), reports and leaderboard beside it.
single/versus: GUI sessions on hfbox, bridge, agents started live, URL printed.
replay: serves web/ plus results/ (saved events.jsonl + leaderboard.json) with no hfbox link at all.
URL convention (GUI worker must honour, or change URL_* below): live ?live=8801[,8802]  replay ?replay=1
"""

import argparse
import http.server
import json
import os
import posixpath
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import hosts
from bench import (
    HF,
    LEVELS,
    PORTS,
    REMOTE_RUNS,
    RESULTS,
    plan,
    render_prompt,
    runname,
    sh,
    ssh,
    ssh_cat,
    wait_state,
)
from run_agent import ROOT, run_agent, task_limits

# served from the repo root in both modes: the page reads ../results/ too
URL_LIVE = "http://127.0.0.1:{web}/web/index.html?live={ports}"
URL_REPLAY = "http://127.0.0.1:{web}/web/index.html?replay=1"

# The one list of models anything here will start. `fake` starts the sessions and no agent, for checks that spend nothing.
MODELS = {
    "haiku": dict(paid=True, note=""),
    "sonnet": dict(paid=True, note=""),
    "opus": dict(paid=True, note=""),
    "fable": dict(paid=True, note=""),
    "nemotron": dict(paid=False, note="free"),
    "deepseek-flash": dict(paid=True, note="untested"),
    "deepseek-pro": dict(paid=True, note="untested"),
}
FAKE = dict(paid=False, note="no agent")
LIVE_PORTS = [8801, 8802, 8803, *range(8812, 8820)]  # never the bench's PORTS
MAX_TRIALS = 5


def die(msg):
    sys.exit(f"ERROR: {msg}")


class Static(http.server.SimpleHTTPRequestHandler):
    """The page and the results, not the rest of the repo."""

    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(ROOT), **k)

    def log_message(self, *a):
        pass

    def _shown(self):
        # as the file server will read it: /web/../harness is not under /web/
        p = posixpath.normpath(urllib.parse.unquote(self.path.split("?")[0]))
        return p.startswith(("/web/", "/results/"))

    def do_GET(self):
        return super().do_GET() if self._shown() else self.send_error(404)

    def do_HEAD(self):
        return super().do_HEAD() if self._shown() else self.send_error(404)


def serve(handler=Static, port=0):
    """Static server on 127.0.0.1 only; returns the port."""
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv.server_address[1]


def free_ports(n):
    """The first n of LIVE_PORTS with nothing listening on them, here or on hfbox (the bridge binds the same number on both ends)."""
    r = ssh("cat /proc/net/tcp")
    if r.returncode:
        die(f"cannot reach hfbox: {r.stderr[-200:]}")
    busy = {
        int(f[1].split(":")[1], 16)
        for f in (ln.split() for ln in r.stdout.splitlines()[1:])
        if f[3] == "0A"
    }
    out = []
    for p in LIVE_PORTS:
        if p in busy:
            continue
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
            except OSError:
                continue
        out.append(p)
        if len(out) == n:
            return out
    die(f"no {n} free ports in {LIVE_PORTS}")


def bridge(ports):
    """sshd forbids port forwarding: the bridge speaks to dv/pipe.py over plain ssh exec. None when the sessions run on this machine."""
    if hosts.LOCAL:
        return None
    return subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).parent / "bridge.py"),
            "--port",
            ",".join(map(str, ports)),
        ]
    )


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


def live(models, level="whatif"):
    prompt = render_prompt(level)
    ports = free_ports(len(models))
    print("PORTS", ",".join(map(str, ports)), flush=True)
    # not the bench's session letters: its run dirs and scores stay its own
    for m, p, s in zip(models, ports, "LR"):
        name = runname(m, level)
        ssh(f"rm -rf {REMOTE_RUNS}/{s}-{name}-0")  # names come from MODELS and LEVELS
        r = sh(
            HF,
            "bg",
            f"demo-{p}",
            f"python -m dv.server --session {s} --port {p} --gui --model {name} --trial 0 --feedback {level}",
        )
        if r.returncode:
            die(f"could not start session on hfbox: {r.stderr[-200:]}")
    tun = bridge(ports)
    try:
        time.sleep(4)
        if tun and tun.poll() is not None:
            die("ssh tunnel to hfbox did not come up (ssh agent / funnel)")
        for p in ports:
            check(p)
        print(
            "OPEN",
            URL_LIVE.format(web=serve(), ports=",".join(map(str, ports))),
            flush=True,
        )
        ts = [
            threading.Thread(
                target=run_agent,
                args=(m, p, RESULTS / f"demo-{runname(m, level)}-{p}", prompt),
                kwargs={"quiet": True},
            )
            for m, p in zip(models, ports)
            if m != "fake"
        ]
        [t.start() for t in ts]
        [t.join() for t in ts]
        print("agents done; Ctrl-C to close the demo", flush=True)
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass
    finally:
        if tun:
            tun.terminate()
        for p in ports:
            sh(HF, "stop", f"demo-{p}")


def replay():
    if not (RESULTS / "leaderboard.json").exists():
        die("no results/leaderboard.json: run bench.py first")
    print("OPEN", URL_REPLAY.format(web=serve()), flush=True)
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass


def refly(runs, scenario, wait_s=20):
    """Two finished runs' saved designs and controllers, flown again side by side on the same scenario and seed, for the picture. No agent, no finish:
    their results.json stay as they are. stream.mp4, stream.json and refly.jsonl land in results/<run>/, which the versus replay plays."""
    dirs = [RESULTS / r for r in runs]
    if runs[0] == runs[1]:
        die("two different runs")
    for r, d in zip(runs, dirs):
        if not re.fullmatch(r"[\w.-]+-\d+", r):
            die(f"bad run name: {r}")
        for f in ("design.json", "controller.py"):
            if not (d / f).exists():
                die(f"no {d / f}")
    scores = {d: (d / "results.json").read_bytes() for d in dirs if (d / "results.json").exists()}  # fmt: skip
    ports = free_ports(2)
    print("PORTS", ",".join(map(str, ports)), flush=True)
    route = f"DVSIM_ROUTE_CACHE={hosts.DIR}/route-cache"
    remote = [f"{hosts.DIR}/refly/{r}" for r in runs]
    for d, rd in zip(dirs, remote):
        if ssh(f"mkdir -p {rd}").returncode:
            die("cannot reach the session host")
        for f in ("design.json", "controller.py"):
            hosts.run(f"cat > {rd}/{f}", input=(d / f).read_text())
    # the wires first, once per geometry: a window asking for them would freeze for minutes
    designs = {(d / "design.json").read_bytes(): rd for d, rd in zip(dirs, remote)}

    def warm(rd):
        r = sh(
            HF,
            "run",
            f"{route} python -m dv.refly --warm --design {rd}/design.json 2>&1 | tail -1",
            timeout=1800,
        )
        print("wires", rd.rsplit("/", 1)[1], r.stdout.strip()[-200:], flush=True)

    ts = [threading.Thread(target=warm, args=(rd,)) for rd in designs.values()]
    [t.start() for t in ts]
    [t.join() for t in ts]
    run_dirs = []
    for r, d, rd, p, s in zip(runs, dirs, remote, ports, "RS"):
        name, trial = r.rsplit("-", 1)
        res = json.loads(scores[d]) if d in scores else {}
        level = next((v for v in LEVELS if v == res.get("feedback")), "whatif")
        run = f"{REMOTE_RUNS}/{s}-{name}-{trial}"
        ssh(f"rm -rf {run}")  # R and S are this command's letters: not a bench run
        run_dirs.append(run)
        if sh(HF, "bg", f"refly-{p}", f"{route} python -X faulthandler -m dv.refly --session {s} --port {p} --model {name} --trial {trial} --feedback {level} --design {rd}/design.json --code {rd}/controller.py --scenario {scenario}").returncode:  # fmt: skip
            die("could not start a session on hfbox")
    tun = bridge(ports)
    try:
        time.sleep(4)
        if tun and tun.poll() is not None:
            die("bridge did not start")
        for p in ports:
            check(p)
        print("OPEN", URL_LIVE.format(web=serve(), ports=",".join(map(str, ports))), flush=True)  # fmt: skip
        print(f"flying in {wait_s} s", flush=True)
        time.sleep(wait_s)
        ssh("touch " + " ".join(f"{run}/refly.go" for run in run_dirs))
        end = time.time() + 240
        while time.time() < end:
            done = ssh("ls " + " ".join(f"{run}/refly.done" for run in run_dirs))
            if done.returncode == 0:
                break
            time.sleep(3)
        else:
            die("the flights did not end in 240 s (hf.sh log refly-PORT)")
        for run, d in zip(run_dirs, dirs):
            ev = ssh_cat(f"{run}/events.jsonl") or ""
            verdict = [json.loads(ln) for ln in ev.splitlines() if '"flight_end"' in ln]
            print(d.name, *[(v["scenario"], v["seed"], v["pass"], v["reason"][:80]) for v in verdict], flush=True)  # fmt: skip
            (d / "refly.jsonl").write_text(ev)
            for f in ("stream.mp4", "stream.json"):
                with open(d / f, "wb") as out:
                    if subprocess.run(hosts.argv(f"cat {run}/{f}"), stdout=out, stderr=subprocess.DEVNULL, timeout=600).returncode:  # fmt: skip
                        print(f"could not copy {f} of {d.name}", flush=True)
        for d, before in scores.items():
            if (d / "results.json").read_bytes() != before:
                die(f"{d}/results.json changed")
        print(f"recorded; results.json untouched ({len(scores)} checked). Ctrl-C closes the sessions", flush=True)  # fmt: skip
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass
    finally:
        if tun:
            tun.terminate()
        for p in ports:
            sh(HF, "stop", f"refly-{p}")


# ---- the Start panel --------------------------------------------------------------------------------------------------
MODES = {"design": (1, 1), "versus": (2, 2), "benchmark": (1, None)}
FIELDS = {"mode", "models", "feedback", "trials", "confirm"}
GRACE_S = 20  # a stopped run gets this long to close its own sessions


def validate(body, known):
    """-> (mode, models, level, trials), every string one of ours and none of the client's. ValueError names the field."""
    if not isinstance(body, dict) or set(body) - FIELDS:
        raise ValueError("unknown fields")
    mode = next((m for m in MODES if m == body.get("mode")), None)
    if not mode:
        raise ValueError("mode")
    level = next((v for v in LEVELS if v == body.get("feedback")), None)
    if not level:
        raise ValueError("feedback")
    asked = body.get("models")
    if not isinstance(asked, list) or not all(isinstance(m, str) for m in asked):
        raise ValueError("models")
    if len(set(asked)) != len(asked) or set(asked) - set(known):
        raise ValueError("models")
    models = [m for m in known if m in asked]  # our order, cheapest first in bench
    lo, hi = MODES[mode]
    if len(models) < lo or (hi and len(models) > hi):
        raise ValueError(f"{mode} takes {lo if lo == hi else f'{lo} or more'} model(s)")
    trials = body.get("trials", 1)
    if type(trials) is not int or not 1 <= trials <= MAX_TRIALS:
        raise ValueError("trials")
    if mode != "benchmark":
        trials = 1
    return mode, models, level, trials


def command(mode, models, level, trials):
    """The argv of a run; no shell ever sees it."""
    me = str(Path(__file__).resolve())
    if mode == "benchmark":
        bench = str(Path(__file__).parent / "bench.py")
        return [sys.executable, bench, "--models", ",".join(models), "--trials", str(trials), "--feedback", level]  # fmt: skip
    verb = "single" if mode == "design" else "versus"
    return [sys.executable, me, verb, *models, "--feedback", level]


def descendants(root):
    """Everything below the PID, by parent links: an agent runs in its own session, so no process group has it."""
    kids = {}
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                ppid = int((d / "stat").read_text().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            kids.setdefault(ppid, []).append(int(d.name))
    out, todo = [], [root]
    while todo:
        for c in kids.get(todo.pop(), []):
            out.append(c)
            todo.append(c)
    return out


def signal_all(pids, sig):
    for pid in pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


class Run:
    def __init__(self, rid, mode, models, level, trials, argv):
        self.id, self.mode, self.models = rid, mode, models
        self.level, self.trials = level, trials
        self.t0 = time.time()
        self.stopping = False
        self.tags = [
            f"{runname(m, l)}-{t}"
            for m, l, t in plan(
                models,
                [level],
                trials,
                done=lambda n: (RESULTS / n / "results.json").exists(),
            )
        ]
        self.log = RESULTS / "panel" / f"{int(self.t0)}-{rid}.log"
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log, "w") as f:
            self.p = subprocess.Popen(
                argv,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

    def lines(self):
        return self.log.read_text(errors="replace").splitlines()

    def ports(self):
        m = re.search(r"^PORTS ([\d,]+)$", "\n".join(self.lines()), re.MULTILINE)
        return [int(p) for p in m.group(1).split(",")] if m else []

    def remote(self):
        """The hfbox sessions this run started, by name."""
        if self.mode == "benchmark":
            return [f"bench-{t}" for t in self.tags]
        return [f"demo-{p}" for p in self.ports()]

    def stop(self):
        if self.stopping:
            return
        self.stopping = True

        def work():
            tree = descendants(self.p.pid)
            signal_all(tree, signal.SIGTERM)
            signal_all(
                [self.p.pid], signal.SIGINT
            )  # runs its finally: closes its sessions
            try:
                self.p.wait(GRACE_S)
            except subprocess.TimeoutExpired:
                signal_all([*tree, self.p.pid], signal.SIGKILL)
                self.p.wait()
            for name in self.remote():
                for _ in range(10):  # the link drops for minutes at a time
                    sh(HF, "stop", name)
                    if ssh(f"tmux has-session -t dv-{name}").returncode == 1:
                        break  # the box answered: it is gone
                    time.sleep(30)

        threading.Thread(target=work, daemon=True).start()

    def bench_rows(self):
        rows = []
        for tag in self.tags:
            d = RESULTS / tag
            if (d / "results.json").exists():
                score = json.loads((d / "results.json").read_text()).get("score")
                rows.append(dict(name=tag, state=f"done, score {score}"))
            elif (d / "stream.jsonl").exists() and (
                d / "stream.jsonl"
            ).stat().st_mtime >= self.t0:
                rows.append(dict(name=tag, state="running"))
            else:
                rows.append(dict(name=tag, state="queued"))
        return rows

    def live_rows(self):
        rows = []
        ports = self.ports()
        for i, m in enumerate(self.models):
            state = "starting"
            if i < len(ports):
                try:
                    s = json.load(
                        urllib.request.urlopen(
                            f"http://127.0.0.1:{ports[i]}/state", timeout=1
                        )
                    )
                    state = (
                        f"finished, score {s.get('score')}"
                        if s.get("finished")
                        else f"running, verify left {(s.get('budget') or {}).get('verify_left')}"
                    )
                except Exception:
                    state = "session starting"
            rows.append(dict(name=m, state=state))
        return rows

    def status(self):
        rc = self.p.poll()
        lines = self.lines()
        ports = self.ports()
        ready = any(ln.startswith("OPEN ") for ln in lines)
        return dict(
            id=self.id,
            mode=self.mode,
            feedback=self.level,
            state="running" if rc is None and not self.stopping else "stopping" if rc is None else f"exited {rc}",
            alive=rc is None,
            pid=self.p.pid,
            rows=[] if rc is not None else self.bench_rows() if self.mode == "benchmark" else self.live_rows(),
            url=f"/web/index.html?live={','.join(map(str, ports))}" if ready and rc is None else "/web/index.html?replay=1" if self.mode == "benchmark" else None,
            last=lines[-1][:200] if lines else "",
        )  # fmt: skip


class Panel:
    """Runs started from the page. Each one is a child of this process, known by PID."""

    def __init__(self, fake=False, command=command):
        self.known = {**MODELS, **({"fake": FAKE} if fake else {})}
        self.command = command
        self.runs = {}
        self.lock = threading.Lock()
        self.next = 1

    def alive(self, benchmark):
        return [
            r
            for r in self.runs.values()
            if r.p.poll() is None and (r.mode == "benchmark") == benchmark
        ]

    def start(self, body):
        """-> (http code, reply)."""
        try:
            mode, models, level, trials = validate(body, self.known)
        except ValueError as e:
            return 400, dict(error=f"bad {e}")
        n = (
            len(plan(models, [level], trials, done=lambda n: (RESULTS / n / "results.json").exists()))
            if mode == "benchmark"
            else len(models)
        )  # fmt: skip
        paid = [m for m in models if self.known[m]["paid"]]
        if paid and body.get("confirm") is not True:
            usd = task_limits()[1]
            return 409, dict(
                need_confirm=paid,
                runs=n,
                usd_each=usd,
                text=f"This costs money: {', '.join(paid)}. {n} run(s), each capped at ${usd:g} where the model's CLI enforces a cap.",
            )
        with self.lock:
            if self.alive(mode == "benchmark"):
                return 409, dict(
                    error=f"a {'benchmark' if mode == 'benchmark' else 'live run'} is already running: stop it first"
                )
            if mode == "benchmark" and not port_free(PORTS[0]):
                return 409, dict(error=f"port {PORTS[0]} is in use: another benchmark holds the bench's ports")  # fmt: skip
            run = Run(self.next, mode, models, level, trials, self.command(mode, models, level, trials))  # fmt: skip
            self.runs[self.next] = run
            self.next += 1
        return 200, dict(id=run.id, pid=run.p.pid)

    def stop(self, body):
        rid = body.get("id") if isinstance(body, dict) else None
        run = self.runs.get(rid) if type(rid) is int else None
        if not run:
            return 404, dict(error="no such run")
        run.stop()
        return 200, dict(id=rid)

    def status(self):
        return 200, dict(
            models=[dict(name=k, **v) for k, v in self.known.items()],
            levels=LEVELS,
            max_trials=MAX_TRIALS,
            runs=[r.status() for r in self.runs.values()],
        )

    def close(self):
        """Nothing keeps spending after the panel is gone."""
        for r in self.runs.values():
            if r.p.poll() is None:
                r.stop()
        for r in self.runs.values():
            r.p.wait()


def port_free(p):
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", p))
            return True
        except OSError:
            return False


def panel_handler(panel):
    class H(Static):
        def _json(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)

        def _ours(self):
            """A page elsewhere reaches 127.0.0.1 too: the Host must be this server (rebinding), and a POST's Origin too."""
            port = self.server.server_address[1]
            hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            return self.headers.get("Host") in hosts

        def do_GET(self):
            if not self._ours():
                return self._json(403, dict(error="host"))
            if self.path.split("?")[0] == "/api/status":
                return self._json(*panel.status())
            return super().do_GET()

        def do_HEAD(self):
            if not self._ours():
                return self.send_error(403)
            return super().do_HEAD()

        def do_POST(self):
            origin = self.headers.get("Origin", "")
            if not self._ours() or origin.removeprefix("http://") != self.headers.get(
                "Host"
            ):
                return self._json(403, dict(error="origin"))
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self._json(415, dict(error="content type"))
            try:
                n = int(self.headers.get("Content-Length", ""))
                if not 0 < n <= 4096:
                    raise ValueError
                body = json.loads(self.rfile.read(n))
            except ValueError:
                return self._json(400, dict(error="body"))
            if self.path == "/api/start":
                return self._json(*panel.start(body))
            if self.path == "/api/stop":
                return self._json(*panel.stop(body))
            return self._json(404, dict(error="no such route"))

    return H


def panel_main(port, fake):
    panel = Panel(fake)
    serve(panel_handler(panel), port)
    print(f"OPEN http://127.0.0.1:{port}/web/index.html", flush=True)
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass
    finally:
        panel.close()


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("mode", choices=["serve", "single", "versus", "replay", "refly"])
    a.add_argument("models", nargs="*", help="refly: two run names under results/")
    a.add_argument("--scenario", choices=["nominal", "wind", "rain"], default="wind")
    a.add_argument("--feedback", choices=LEVELS, default="whatif")
    a.add_argument("--port", type=int, default=8790)
    a.add_argument(
        "--fake", action="store_true", help="serve: offer the no-agent model"
    )
    a = a.parse_args()
    # also when started in the background, where SIGINT comes ignored; a SIGTERM closes the sessions the same way
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    if a.mode == "refly":
        if len(a.models) != 2:
            die("usage: demo.py refly RUN_A RUN_B [--scenario wind|rain|nominal]")
        refly(a.models, a.scenario)
    elif set(a.models) - {*MODELS, "fake"}:
        die(f"unknown model: {a.models}; one of {[*MODELS, 'fake']}")
    elif a.mode == "replay":
        replay()
    elif a.mode == "serve":
        panel_main(a.port, a.fake)
    elif (
        a.mode == "single"
        and len(a.models) == 1
        or a.mode == "versus"
        and len(a.models) == 2
    ):
        live(a.models, a.feedback)
    else:
        die("usage: demo.py serve | single MODEL | versus LOW HIGH | replay")
