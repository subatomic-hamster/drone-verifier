"""The viewer's Test / Run button: run.* ops (mixed into `viewer.Backend` as `Backend.runs`).

  run.targets {}  -> {scenarios: [rel path of every *.scenario.toml under the program root], components: [{id, part, kind}]}
                     (instances with a device model first; `kind` = the device types bound to the instance)
  run.start {mode: "bench"|"full", component?, scenario?, files?: [open files, rel]} -> {run_id}
  run.poll {run_id} -> {state: running|done|failed|stopped, out: <new output since the last poll>, result?}
  run.stop {run_id}

mode "bench": the selected component's device model on a short standalone bench (a fresh Sim of the design's own build: ESC / prop
throttle sweep, servo step response, battery sag under load, sensor noise readout), then the pytest files tied to the open file
(test_<name>.py, or a test that imports it). Result {pass, metrics}. A component without a device model says so and only runs the tests.
mode "full": the scenario's program (dvsim program semantics) against the VIEWER's own World, in real time (dvsim.program.run(live=...)):
the scenario model is loaded into the Sim the window shows, the program process drives it (dvsim.attach), the replay log, monitors and
pass / fail are kept. The final scene STAYS in the window (the verdict is shown there); run.restore loads the design's own model
back when the window leaves for the design pages. Result = the run's result.json (+ run_dir).

Paths are relative to the program root (the IDE's: Prog.root) and confined to it. This runs USER CODE (the scenario's program, the tests):
like `dvsim program` it is deliberately not an MCP tool; only the viewer's Backend serves it.
"""

import itertools
import math
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .library import DesignError
from .uiops import REPO

SKIP = {
    ".git",
    ".venv",
    "build",
    "runs",
    "node_modules",
    "remote-out",
    "__pycache__",
    "env",
}
BENCH = (
    "esc",
    "rotor",
    "servo",
    "battery",
    "psu",
    "imu",
    "baro",
    "mag",
    "gnss",
    "rangefinder",
)
TEST_WALL_S = 300
IMPORTER_WALL_S = 60  # tests that merely import the open file, narrowed with -k
THROTTLES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


class Stopped(Exception):
    pass


class Run:
    def __init__(self, rid, mode):
        self.id, self.mode, self.state, self.result = rid, mode, "running", None
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self._out = []
        self.log = None  # program.log / pytest log being tailed
        self._off = 0
        self.proc = None

    def emit(self, text):
        with self.lock:
            self._drain()
            self._out.append(text if text.endswith("\n") else text + "\n")

    def _drain(
        self,
    ):  # under the lock: file output first, so own messages and the child's keep their order
        if self.log and Path(self.log).exists():
            with open(self.log, "rb") as f:
                f.seek(self._off)
                b = f.read()
            if b:
                self._off += len(b)
                self._out.append(b.decode("utf-8", "replace"))

    def take(self):
        with self.lock:
            self._drain()
            out, self._out = "".join(self._out), []
        return out


def table(head, rows):
    w = [max(len(str(x)) for x in col) for col in zip(head, *rows)]
    line = lambda r: "  ".join(str(x).rjust(n) for x, n in zip(r, w))
    return "\n".join([line(head), line(["-" * n for n in w]), *map(line, rows)])


class Runs:
    def __init__(self, backend):
        self.b = backend
        self.runs = {}
        self.ids = itertools.count(1)
        self.held = False  # a full run's final scene is in the window's World

    @property
    def root(self):
        """The program root: the IDE's (Prog.root: examples/<name> beside assemblies/<name>.toml), so paths match the explorer."""
        return self.b.prog.root()

    # ---- paths ------------------------------------------------------------------------------------------------------
    def _rel(self, name, what):
        root = self.root
        p = (root / name).resolve()
        if p != root and root not in p.parents:
            raise DesignError(f"run: {what} {name!r} is outside the program root")
        return p

    def _walk(self, suffix, base=None):
        base = base or self.root
        for dp, dn, fn in os.walk(base):
            dn[:] = sorted(d for d in dn if d not in SKIP)
            for f in sorted(fn):
                if f.endswith(suffix):
                    yield str((Path(dp) / f).relative_to(base))

    # ---- ops --------------------------------------------------------------------------------------------------------
    def handle(self, tool, a):
        if tool == "run.targets":
            return {"ok": True, "result": self.targets()}
        if tool == "run.start":
            return {"ok": True, "result": self.start(a)}
        if tool == "run.restore":
            return {"ok": True, "result": {"restored": self.restore()}}
        r = self.runs.get(a.get("run_id"))
        if r is None:
            raise DesignError(f"{tool}: unknown run_id {a.get('run_id')!r}")
        if tool == "run.poll":
            out = r.take()
            return {
                "ok": True,
                "result": {
                    "state": r.state,
                    "out": out,
                    **({"result": r.result} if r.result is not None else {}),
                },
            }
        if tool == "run.stop":
            r.stop.set()
            if r.proc:  # a pytest child
                from ..program import kill_group

                kill_group(r.proc, 1.0)
            return {"ok": True, "result": {"state": r.state}}
        raise DesignError(f"unknown tool {tool!r}")

    def targets(self):
        d = self.b.get_design()
        comps = []
        if d is not None:
            for i in d.doc["instances"]:
                kinds = [
                    v["type"] for v in self._devs(d.doc, i["id"]) if v["type"] in BENCH
                ]
                comps.append(
                    {
                        "id": i["id"],
                        "part": i["part"],
                        "kind": ",".join(dict.fromkeys(kinds)),
                    }
                )
            comps.sort(
                key=lambda c: not c["kind"]
            )  # stable: device-model components first
        return {"scenarios": list(self._walk(".scenario.toml")), "components": comps}

    @staticmethod
    def _devs(doc, iid):
        return [
            v
            for v in doc.get("devices", [])
            if iid in (v.get("instance"), v.get("motor"), v.get("body"))
        ]

    def start(self, a):
        mode = a.get("mode")
        if mode not in ("bench", "full"):
            raise DesignError("run.start: mode is 'bench' or 'full'")
        files = [
            str(self._rel(f, "file").relative_to(self.root))
            for f in a.get("files") or []
        ]
        if any(r.state == "running" for r in self.runs.values()):
            raise DesignError("run.start: a run is already running (run.stop it first)")
        run = Run(f"r{next(self.ids)}", mode)
        if mode == "bench":
            d = self.b.get_design()
            if d is None:
                raise DesignError("run.start: no design open")
            iid = a.get("component", "")
            d._inst(d.doc, iid)  # DesignError for an unknown instance
            doc = {"devices": [dict(v) for v in d.doc.get("devices", [])]}
            mine = self._devs(doc, iid)
            xml = (
                self.b.state(d)["xml"]
                if any(v["type"] in BENCH for v in mine)
                else None
            )
            target = lambda: self._bench(run, iid, mine, doc["devices"], xml, files)
        else:
            sc = self._rel(a.get("scenario", ""), "scenario")
            if not sc.is_file() or not sc.name.endswith(".toml"):
                raise DesignError(
                    f"run.start: {a.get('scenario')!r} is not a scenario file"
                )
            sim = getattr(self.b, "sim", None)
            if sim is None:
                raise DesignError(
                    "run.start: full needs the viewer's World (open the window with Sim.show(design=backend))"
                )
            target = lambda: self._full(run, sc, sim)
        self.runs[run.id] = run
        threading.Thread(
            target=self._guard, args=(run, target), name=f"dvsim-{run.id}", daemon=True
        ).start()
        return {"run_id": run.id}

    def _guard(self, run, target):
        try:
            target()
            run.state = "stopped" if run.stop.is_set() else "done"
        except Stopped:
            run.state = "stopped"
        except (
            Exception
        ) as e:  # incl. ScenarioError: the user sees the message in the terminal
            run.emit(f"error: {type(e).__name__}: {e}")
            run.result, run.state = {"pass": False, "error": str(e)}, "failed"
        if run.state == "stopped":
            run.emit("stopped")

    # ---- full: the scenario in the viewer's World -----------------------------------------------------------------
    def _full(self, run, sc, sim):
        from .. import program

        run.emit(
            f"[run] {sc.relative_to(self.root)} live in the viewer (real time); Stop kills the program"
        )

        def on_dir(rd):
            run.log = rd / "program.log"

        self._ui(
            "ui.mode", name="simulate"
        )  # the window shows the vehicle fly (a no-op without a window)
        over = threading.Event()
        threading.Thread(target=self._chase, args=(sim, over), daemon=True).start()
        try:
            res, rd = program.run(
                sc,
                self.b.out / "runs",
                live=dict(
                    sim=sim,
                    stop=run.stop,
                    root=self.b.out,  # the viewer's Sim root: its log also holds the design's own loads
                    on_dir=on_dir,
                    rate=getattr(self.b, "run_rate", None),
                ),
            )
        finally:
            over.set()
            self.held = True  # the run's final scene stays in the window; run.restore (the window leaving SIMULATE) loads the design back
            sim.call({"cmd_type": "Pause", "cmd": {}})
        sm, rp = res.get("sim", {}), res.get("replay", {})
        run.emit(
            f"[run] {'PASS' if res['pass'] else 'FAIL'}  {res['name']}  sim {sm.get('sim_s', sm.get('tick', 0) * sm.get('dt', 0)):.1f} s"
            f"  hash {sm.get('state_hash', '-')}  replay {('match' if rp.get('match') else 'MISMATCH') if rp.get('available') else 'n/a'}"
        )
        for m in res["monitors"]:
            run.emit(f"[run]   {'ok  ' if m['pass'] else 'FAIL'} {m['name']}")
        for w in res["fail_reasons"]:
            run.emit(f"[run]   why: {w}")
        run.emit(f"[run] run dir: {rd}")
        run.result = {**res, "run_dir": str(rd)}
        if run.stop.is_set():
            raise Stopped

    def restore(self):
        """Load the design's own (current) model back into the window's World after a full run left its final scene there.
        No-op (False) while a run is live or when nothing is held."""
        sim = getattr(self.b, "sim", None)
        d = self.b.get_design()
        if (
            not self.held
            or sim is None
            or d is None
            or any(r.state == "running" for r in self.runs.values())
        ):
            return False
        xml = self.b.state(d)["xml"]
        if not xml or not Path(xml).exists():
            return False
        r = sim.call({"cmd_type": "LoadModel", "cmd": {"path": str(xml)}})
        self.held = not r["ok"]
        return r["ok"]

    def _ui(self, tool, **a):
        try:
            with self.b.lock:
                self.b._ui_tool(tool, a)
        except DesignError:
            pass

    def _chase(self, sim, over):
        """Chase camera for a live run: the scenario's scene is far bigger than the design, so the window's own framing shows a
        speck. Follows the root part's body (the design's root instance id is the end of its body name) from behind and above."""
        path, st = None, {}
        while not over.wait(0.25):
            if sim.tick < 300:  # the scenario model is not stepping yet
                continue
            try:
                if path is None:
                    d = self.b.get_design()
                    root = (
                        next((i["id"] for i in d.doc["instances"] if i.get("root")), "")
                        if d is not None
                        else ""
                    )
                    path = next(
                        (
                            k
                            for k in sim.body_positions()
                            if root and k.rsplit("/", 1)[-1].endswith(root)
                        ),
                        "",
                    )
                if not path:
                    return
                p = [float(x) for x in sim.query(path, "pos")[0]]
            except (KeyError, ValueError, RuntimeError):
                continue
            eye = self._chase_eye(sim, np.array(p), st)
            look = np.array(p) if st.get("look") is None else st["look"]
            st["look"] = look + 0.5 * (np.array(p) - look)
            self._ui(
                "camera.fly",
                eye=[float(x) for x in eye],
                look=[float(x) for x in st["look"]],
                seconds=0.3,
                chase=True,  # the window drops these while FOLLOW is off (the user took the camera)
            )

    @staticmethod
    def _chase_eye(sim, p, st):
        """The camera position for a vehicle at p: keep the current azimuth while its line of sight (sim.raycast from the vehicle to
        the eye) is clear, else the nearest clear azimuth (then a shorter arm, then a spring arm pulled in before the blocker); eased toward the target."""
        az0 = st.setdefault("az", math.atan2(-4.0, -4.0))
        arms = (
            (4.5, 3.0),
            (3.5, 2.6),
            (2.6, 2.0),
            (2.4, 0.9),
        )  # (horizontal m, height m)

        def free_len(
            e,
        ):  # how far from p towards the eye e the line of sight is clear (0 = the eye is under the ground)
            u = e - p
            n = float(np.linalg.norm(u))
            if e[2] < 0.8:
                return 0.0
            try:
                hit = float(sim.raycast(p + 0.4 * u / n, [u])[0])
            except (ValueError, RuntimeError):
                return n
            return n if hit < 0 else min(n, hit + 0.4)

        def pulled_in(
            e, free
        ):  # a spring arm: the eye brought in to just before the blocker (never closer than 0.8 m)
            return p + (e - p) * max(free - 0.5, 0.8) / np.linalg.norm(e - p)

        tgt, best = None, (-1.0, None, az0)
        for arm in arms:
            for da in (0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6, 2.4, -2.4, math.pi):
                e = p + [
                    arm[0] * math.cos(az0 + da),
                    arm[0] * math.sin(az0 + da),
                    arm[1],
                ]
                free = free_len(e)
                if free > np.linalg.norm(e - p) - 0.3:  # clear all the way to the eye
                    st["az"], tgt = az0 + da, e
                    break
                if free > best[0]:
                    best = (free, e, az0 + da)
            if tgt is not None:
                break
        if (
            tgt is None
        ):  # boxed in: the longest line of sight, the camera pulled in before its blocker
            free, e, st["az"] = best
            tgt = pulled_in(e, free)
        eye = st.get("eye")
        eye = tgt if eye is None else eye + 0.4 * (tgt - eye)
        free = free_len(
            eye
        )  # the eased camera has not been through a tree: check where it actually is
        st["eye"] = (
            eye if free > np.linalg.norm(eye - p) - 0.3 else pulled_in(eye, free)
        )
        return st["eye"]

    # ---- bench: the component's device model, then the tests tied to the open file ---------------------------------
    def _bench(self, run, iid, mine, all_devs, xml, files):
        em = run.emit
        metrics, ok = {}, True
        models = [v for v in mine if v["type"] in BENCH]
        if not models:
            em(
                f"[bench] {iid}: no device model on this component (its devices: {sorted({v['type'] for v in mine}) or 'none'}); tests only"
            )
        else:
            em(
                f"[bench] {iid}: "
                + ", ".join(f"{v['type']} {v['name']}" for v in models)
            )
            import dvsim

            from .simulate import _model

            dev, _, _, dt, _ = _model(xml)
            ctx = dict(
                run=run,
                em=em,
                sim=lambda: dvsim.Sim.load(xml),
                dev=dev,
                dt=dt,
                devs=all_devs,
            )
            fns = dict(
                esc=_motor,
                rotor=_motor,
                servo=_servo,
                battery=_battery,
                psu=_battery,
                imu=_sensor,
                baro=_sensor,
                mag=_sensor,
                gnss=_sensor,
                rangefinder=_sensor,
            )
            done = set()
            for v in models:
                key = (fns[v["type"]], v["name"] if fns[v["type"]] is _sensor else None)
                if key in done:
                    continue
                done.add(key)
                try:
                    metrics.update(fns[v["type"]](ctx, v))
                except Stopped:
                    raise
                except Exception as e:
                    ok = False
                    em(
                        f"[bench] {v['type']} {v['name']} FAILED: {type(e).__name__}: {e}"
                    )
            bad = [
                k
                for k, x in metrics.items()
                if isinstance(x, float) and not math.isfinite(x)
            ]
            if bad:
                ok = False
                em(f"[bench] non-finite metrics: {bad}")
        tests, kexpr, budget = self._tests(files)
        t_ok = True
        if tests:
            em(
                f"[tests] pytest {' '.join(map(self._disp, tests))}"
                + (f" -k {kexpr}" if kexpr else "")
                + f"   (budget {budget:.0f} s; Stop cancels)"
            )
            t_ok = self._pytest(run, tests, kexpr, budget)
        elif files:
            em(f"[tests] no test file is tied to {', '.join(files)}")
        if run.stop.is_set():
            raise Stopped
        run.result = {
            "pass": ok and t_ok,
            "metrics": metrics,
            "tests": [self._disp(t) for t in tests],
            "tests_pass": t_ok if tests else None,
        }
        em(
            f"[bench] {'PASS' if run.result['pass'] else 'FAIL'}"
            + (f"  {len(metrics)} metrics" if metrics else "")
        )

    def _tests(self, files):
        """-> (test files, -k expression or None, wall budget s). Tests that TARGET the open files: the file itself if it is a test,
        else test_<stem>*.py (root or the repo's tests/). Only when there is none: the tests that import <stem>, narrowed to the
        test names containing <stem> (-k) and a short budget (a whole suite that merely imports the module is not the bench)."""
        out, importers, stems = [], [], []
        bases = [self.root] + [
            b for b in (REPO / "tests",) if b.is_dir() and b != self.root
        ]  # the repo's tests/ are read, not editable
        for f in files:
            if not f.endswith(".py"):
                continue
            stem = Path(f).stem
            if stem.startswith("test_"):
                out.append(self.root / f)
                continue
            stems.append(stem)
            imp = re.compile(
                rf"^\s*(?:from|import)\s+[\w.]*\b{re.escape(stem)}\b", re.MULTILINE
            )
            for base in bases:
                for t in self._walk(".py", base):
                    n = Path(t).name
                    if not n.startswith("test_"):
                        continue
                    if n == f"test_{stem}.py" or n.startswith(f"test_{stem}_"):
                        out.append(base / t)
                    elif imp.search((base / t).read_text(errors="replace")):
                        importers.append(base / t)
        if out:
            return list(dict.fromkeys(out))[:8], None, TEST_WALL_S
        if importers and stems:
            return (
                list(dict.fromkeys(importers))[:4],
                " or ".join(stems),
                IMPORTER_WALL_S,
            )
        return [], None, 0

    def _disp(self, p):
        for b in (self.root, REPO):
            if b in Path(p).parents:
                return str(Path(p).relative_to(b))
        return str(p)

    def _pytest(self, run, tests, kexpr=None, budget=TEST_WALL_S):
        log = self.b.out / "runs" / f"{run.id}-pytest.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        run.log, run._off = log, 0
        with open(log, "wb") as lf:
            run.proc = p = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "--no-header",
                    "-p",
                    "no:cacheprovider",
                    *(["-k", kexpr] if kexpr else []),
                    *map(str, tests),
                ],
                cwd=self.root,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                stdin=subprocess.DEVNULL,
                stdout=lf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            t0 = time.monotonic()
            while (
                p.poll() is None
                and not run.stop.is_set()
                and time.monotonic() - t0 < budget
            ):
                time.sleep(0.1)
            if p.poll() is None:
                from ..program import kill_group

                kill_group(p, 1.0)
                if not run.stop.is_set():
                    run.emit(f"[tests] budget of {budget:.0f} s spent: killed")
                    return False
            rc = p.wait()
        with run.lock:
            run._drain()  # the tail of pytest's output, before the verdict line
        return rc in (0, 5) and not run.stop.is_set()  # 5 = nothing collected


# ---- device benches (module functions: ctx, device doc -> metrics) -------------------------------------------------------
def _chk(ctx):
    if ctx["run"].stop.is_set():
        raise Stopped


def _q(sim, dev, attr):
    return float(sim.query(f"/devices/{dev}", attr)[0][0])


def _supply(ctx):
    return next(
        (v["name"] for v in ctx["devs"] if v["type"] in ("battery", "psu")), None
    )


def _motor(ctx, v):
    """Throttle sweep of one ESC + motor + prop: every level is a fresh run from the initial state (the vehicle does not fly off)."""
    em, devs = ctx["em"], ctx["devs"]
    joint = v["joint"]
    esc = next((x for x in devs if x["type"] == "esc" and x["joint"] == joint), None)
    rot = next((x for x in devs if x["type"] == "rotor" and x["joint"] == joint), None)
    if esc is None:
        raise DesignError("no ESC drives this rotor")
    sim, sup, n = ctx["sim"](), _supply(ctx), int(0.6 / ctx["dt"])
    base = sim.save()
    rows, M = (
        [],
        {k: [] for k in ("thrust", "rpm", "i_bus", "v_sup", "duty", "temp", "torque")},
    )
    for u in THROTTLES:
        _chk(ctx)
        sim.restore(base)
        sim.set(f"/devices/{esc['name']}", "throttle", u)
        sim.step(n)
        e = esc["name"]
        th = _q(sim, rot["name"], "thrust") if rot else float("nan")
        rpm = _q(sim, e, "omega") * 60 / (2 * math.pi)
        vs = _q(sim, sup, "voltage") if sup else float("nan")
        for k, x in zip(
            M,
            (
                th,
                rpm,
                _q(sim, e, "bus_current"),
                vs,
                _q(sim, e, "duty"),
                _q(sim, e, "temperature"),
                _q(sim, e, "torque"),
            ),
        ):
            M[k].append(x)
        rows.append(
            [
                f"{u:.1f}",
                f"{rpm:.0f}",
                f"{th:.2f}",
                f"{M['torque'][-1]:.3f}",
                f"{M['i_bus'][-1]:.2f}",
                f"{vs:.2f}",
                f"{M['duty'][-1]:.2f}",
                f"{M['temp'][-1]:.1f}",
            ]
        )
    em(
        f"[bench] {esc['name']}"
        + (f" + {rot['name']}" if rot else "")
        + ": throttle sweep, 0.6 s per level, other ESCs off"
    )
    em(
        table(
            [
                "thr",
                "rpm",
                "thrust N",
                "torque Nm",
                "bus A",
                "supply V",
                "duty",
                "temp C",
            ],
            rows,
        )
    )
    th = np.array(M["thrust"])
    mono = bool(np.all(np.diff(th) > -0.02 * max(th.max(), 1e-9))) if rot else True
    em(
        f"[bench] max thrust {th[-1]:.2f} N ({th[-1] / 9.80665 * 1000:.0f} gf), {M['i_bus'][-1]:.1f} A from the pack, "
        f"sag {M['v_sup'][0] - M['v_sup'][-1]:.2f} V; thrust monotone in throttle: {mono}"
    )
    if not mono:
        raise DesignError("thrust is not monotone in throttle")
    return {
        f"{esc['name']}.max_thrust_N": float(th[-1]),
        f"{esc['name']}.max_bus_current_A": float(M["i_bus"][-1]),
        f"{esc['name']}.max_rpm": float(M["rpm"][-1]),
        f"{esc['name']}.supply_sag_V": float(M["v_sup"][0] - M["v_sup"][-1]),
        f"{esc['name']}.thrust_50_N": float(th[5]),
    }


def _servo(ctx, v):
    """Position step of 0.3 rad from rest: rise (10-90 %), overshoot, 2 % settling, steady-state error, peak current and torque."""
    em, dt, name = ctx["em"], ctx["dt"], v["name"]
    sim = ctx["sim"]()
    for s in ctx[
        "devs"
    ]:  # every servo holds its pose, so the chain does not collapse under gravity
        if s["type"] == "servo":
            sim.set(f"/devices/{s['name']}", "torque_enable", 1)
            sim.set(
                f"/devices/{s['name']}", "goal_position", _q(sim, s["name"], "position")
            )
    sim.step(int(0.3 / dt))
    p0 = _q(sim, name, "position")
    goal = p0 + 0.3
    sim.set(f"/devices/{name}", "goal_position", goal)
    n = int(1.0 / dt)
    t, pos, cur, tq = np.arange(1, n + 1) * dt, np.zeros(n), np.zeros(n), np.zeros(n)
    for k in range(n):
        if k % 50 == 0:
            _chk(ctx)
        sim.step(1)
        pos[k], cur[k], tq[k] = (
            _q(sim, name, "position"),
            _q(sim, name, "current"),
            _q(sim, name, "torque"),
        )
    final = float(pos[-50:].mean())
    d = final - p0
    if abs(d) < 0.05:
        raise DesignError(
            f"the servo did not move (stepped {d:.3f} rad of 0.3): joint limit or load?"
        )
    fr = (pos - p0) / d
    t10, t90 = (t[np.argmax(fr >= f)] for f in (0.1, 0.9))
    over = float(max(fr.max() - 1.0, 0.0) * 100)
    out = np.nonzero(np.abs(fr - 1.0) > 0.02)[0]
    settle = float(t[out[-1]]) if len(out) else 0.0
    rows = [
        [f"{t[k]:.3f}", f"{pos[k]:.4f}", f"{cur[k]:.3f}", f"{tq[k]:.4f}"]
        for k in range(49, n, 100)
    ]
    em(f"[bench] {name}: step {p0:.3f} -> {goal:.3f} rad (all servos torque-on)")
    em(table(["t s", "pos rad", "current A", "torque Nm"], rows))
    em(
        f"[bench] rise {t90 - t10:.3f} s, overshoot {over:.1f} %, settle(2%) {settle:.3f} s, steady error {goal - final:+.4f} rad, "
        f"peak {cur.max():.2f} A / {tq.max():.3f} Nm"
    )
    return {
        f"{name}.rise_s": float(t90 - t10),
        f"{name}.overshoot_pct": over,
        f"{name}.settle_s": settle,
        f"{name}.steady_error_rad": float(goal - final),
        f"{name}.peak_current_A": float(cur.max()),
        f"{name}.peak_torque_Nm": float(tq.max()),
    }


def _battery(ctx, v):
    """Supply under load: every ESC at 0 / 25 / 50 / 75 / 100 % throttle for 0.6 s, terminal voltage, current, state of charge."""
    em, sim, name, n = ctx["em"], ctx["sim"](), v["name"], int(0.6 / ctx["dt"])
    escs = [x["name"] for x in ctx["devs"] if x["type"] == "esc"]
    base, rows, vs = sim.save(), [], []
    for u in (0.0, 0.25, 0.5, 0.75, 1.0):
        _chk(ctx)
        sim.restore(base)
        for e in escs:
            sim.set(f"/devices/{e}", "throttle", u)
        sim.step(n)
        vs.append((_q(sim, name, "voltage"), _q(sim, name, "current")))
        soc = _q(sim, name, "soc") if v["type"] == "battery" else float("nan")
        rows.append([f"{u:.2f}", f"{vs[-1][0]:.2f}", f"{vs[-1][1]:.1f}", f"{soc:.4f}"])
    em(
        f"[bench] {name}: {len(escs)} ESC(s) at the same throttle, 0.6 s per level"
        if escs
        else f"[bench] {name}: no ESC in the design, idle readout only"
    )
    em(table(["throttle", "terminal V", "current A", "soc"], rows))
    sag = vs[0][0] - vs[-1][0]
    em(
        f"[bench] idle {vs[0][0]:.2f} V -> {vs[-1][0]:.2f} V at full throttle ({vs[-1][1]:.1f} A): sag {sag:.2f} V"
    )
    return {
        f"{name}.v_idle": vs[0][0],
        f"{name}.v_full": vs[-1][0],
        f"{name}.i_full_A": vs[-1][1],
        f"{name}.sag_V": sag,
    }


# sensor type -> (config list in the devices JSON, attr)
SENSORS = dict(
    imu=("imus", ["gyro", "accel"]),
    baro=("baros", ["pressure"]),
    mag=("mags", ["field"]),
    gnss=("gnss_receivers", ["alt", "lat", "lon"]),
    rangefinder=("rangefinders", ["frames"]),
)


def _sensor(ctx, v):
    """Sensor readout at rest: configured rate / latency, then ~100 samples: mean and noise (std of successive differences / sqrt 2,
    which ignores the slow drift of the free-falling body)."""
    em, sim, name, dt = ctx["em"], ctx["sim"](), v["name"], ctx["dt"]
    key, attrs = SENSORS[v["type"]]
    cfg = next(c for c in ctx["dev"][key] if c["name"] == name)
    rate = float(cfg.get("rate_hz", 100.0))
    k = max(int(round(1 / (rate * dt))), 1)
    spec = {
        a: cfg[a]
        for a in (
            "rate_hz",
            "latency_s",
            "noise_std",
            "horiz_std",
            "vert_std",
            "sigma1",
        )
        if a in cfg
    }
    em(
        f"[bench] {name} ({v['type']}): configured {spec}"
        + (
            f"; imu noise density gyro {cfg['gyro']['noise_density']:.3g} rad/s/rtHz, accel {cfg['accel']['noise_density']:.3g} m/s2/rtHz"
            if v["type"] == "imu"
            else ""
        )
    )
    sim.step(int(1.0 / dt))  # the first samples of a device read zero
    if v["type"] == "rangefinder":
        n0 = _q(sim, name, "frames")
        sim.step(int(1.0 / dt))
        fps = _q(sim, name, "frames") - n0
        em(f"[bench] {fps:.0f} frames in 1 s (configured {rate:g} Hz)")
        return {f"{name}.frames_per_s": fps}
    N, rows, M = min(100, int(10 / (k * dt))), [], {}
    data = {a: [] for a in attrs}
    for i in range(N):
        if i % 20 == 0:
            _chk(ctx)
        sim.step(k)
        for a in attrs:
            data[a].append(sim.query(f"/devices/{name}", a)[0].copy())
    for a in attrs:
        x = np.array(data[a])
        x = x.reshape(len(x), -1)
        noise = np.diff(x, axis=0).std(axis=0) / math.sqrt(2)
        for j in range(x.shape[1]):
            lab = a if x.shape[1] == 1 else f"{a}[{'xyz'[j]}]"
            rows.append([lab, f"{x[:, j].mean():.6g}", f"{noise[j]:.3g}"])
            M[f"{name}.{lab}.noise"] = float(noise[j])
    em(f"[bench] {N} samples at {rate:g} Hz, at rest in the design's initial pose")
    em(table(["channel", "mean", "noise (1 sigma)"], rows))
    return M
