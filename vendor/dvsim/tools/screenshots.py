"""Screenshot set of the dvsim simulation interface and the models in action (docs/showcase/screens/, see its README.md).

Everything is a real capture of the real software: the viewer window under Xvfb (UI shots) or the offscreen Vulkan renderer
(`Sim.render`) of a state reached by a real run (model shots). Run on a host with a Vulkan device, xvfb and (for the PX4 shots)
the PX4 stack, e.g. on hfbox (GPU jobs hold the shared GPU lock):

    tools/remote.sh run --gpu '.venv/bin/python tools/screenshots.py all --runs /tmp/dvsim-shots-runs --out build/screens'
    rsync -e 'ssh -o BatchMode=yes' -a hfbox:/workspace/wt/dvsim/src/build/screens/ docs/showcase/screens/

Subcommands (NAME... = a subset of the shot names, default all; `list` prints them):
    scenarios [--runs DIR] [NAME...]   run the scenarios the model shots replay (`dvsim program`; PX4 ones ~1 min each, CPU only)
    frames    [--runs DIR] [--out DIR] [NAME...]   replay a run's log to the shot's sim time and render it (GPU)
    quad      [--out DIR]               the scratch-built quad (tests/python/test_design_scratch.py) under the built-in hover (GPU)
    ui        [--out DIR] [NAME...]     viewer-window shots: `xvfb-run` one Xvfb per shot (GPU)
    all       = scenarios + frames + quad + ui
    optimize  DIR                       re-encode PNGs over 400 KB with Pillow (256-colour palette); run where Pillow exists
Writes <out>/<name>.png and <out>/manifest.json (sim time, tick, state hash and the command of every model shot).
Frames are deterministic per GPU / driver only (the sim state is deterministic: the replay's final hash is checked against the run's).
"""

import argparse
import csv
import json
import os
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
W, H = 1280, 720
LIMIT = 400_000

# ---- UI shots: (viewer script, design requests to wait for, extra settle seconds). Backend = the X500 design editor ----------
ESC = "holybro-x500v2-esc-blheli-s-20a"
UI = {
    # scene tree + inspector with a body selected (units in the inspector), the sim running
    "ui-main-x500": ("2 inst mot1; 4 run 1.0", 0, 4.0),
    # remove the PDB, snap one back: the Library tab lists the ranked placements (alternatives) of the snap
    "ui-design-library": (
        "2 dremove pdb; 4 dsuggest bottom.pdb pcb; 6 dsnap holybro-pdb-60a bottom.pdb pdb; 8 dcycle 2; 12 dfinding 4",
        4,
        2.5,
    ),
    # take two ESCs out and snap them back, then autowire all: motor_out / signal ports have several equally good targets
    "ui-design-wiring": (
        f"2 dremove esc1; 4 dremove esc2; 6 dsnap {ESC} tube1.esc esc1; 8 dsnap {ESC} tube2.esc esc2; 12 dautowire_all",
        5,
        2.5,
    ),
    # DRC window after removing the PDB: new findings carry a * ; one finding clicked (its instance selected)
    "ui-drc": ("2 dremove pdb; 8 dfinding 4", 1, 2.5),
    # Simulate tab after the built-in hover run: the numbers Design.simulate returns
    "ui-simulate-hover": ("2 dsim hover", 1, 2.5),
    # the mode bar (BUILD / LIBRARY / WIRING / SIMULATE / PROGRAM). Each is also a backend op (assemblies/README.md "UI operations")
    "ui-build-select": ("4 uisel mot1; 8 uicam iso; 10 uiframe", 0, 3.0),
    "ui-library": ("3 mode library; 6 search motor", 0, 3.0),
    "ui-wiring": ("3 mode wiring", 0, 2.5),
    "ui-simulate": ("3 mode simulate", 0, 2.5),
    "ui-program": ("3 mode program", 0, 2.5),
    # remove the PDB, then the agent API drags it back: the shot is taken while the ghost is still flying to the port
    "ui-drag-ghost": (
        "2 dremove pdb; 4 uicam iso",
        1,
        1.1,  # the window polls ui events once a second, then the ghost flies for 1.2 s: this lands mid-flight
        [
            {
                "tool": "library.drag",
                "args": {
                    "part": "holybro-pdb-60a",
                    "to": {"instance": "bottom", "port": "pdb"},
                    "animate": True,
                },
            }
        ],
    ),
    # the agent API recolours a part: the window opens the colour picker on it and the part turns the new colour (persisted)
    "ui-colour-picker": (
        "4 uisel mot1; 8 uiframe",
        0,
        0.8,
        [
            {
                "tool": "set_color",
                "args": {"instance": "mot1", "rgba": [0.9, 0.25, 0.1, 1.0]},
            }
        ],
    ),
}


def xvfb_png(out, script, expect, settle, fbdir, steps=()):
    """In the Xvfb: open the viewer on the X500 design with SCRIPT, wait for EXPECT design requests, send the backend
    requests in STEPS (agent-API calls the window shows), grab the screen."""
    import design_shot
    from dvsim import Sim, _dvsim
    from dvsim.design import Design
    from dvsim.design.viewer import Backend

    d = Design.open(REPO / "assemblies" / "x500v2.toml")
    b = Backend(lambda: d, tempfile.mkdtemp(prefix="dvsim-shot-"))
    b.auto_thumbs = True  # the library shows real thumbnails (rendered in the background on the first run, then cached)
    sim = Sim.load(b.state(d)["xml"], host=True)
    shot = []

    def grab():
        try:
            t0 = time.time()
            while (
                time.time() - t0 < 60
                and sum(design_shot.is_edit(q["tool"]) for q, _ in b.log) < expect
            ):
                time.sleep(0.2)
            time.sleep(
                1.5
            )  # the window is attached (polling ui_poll) and has loaded the rebuilt model
            for req in steps:
                print("step", b(json.dumps(req))[:160])
            time.sleep(settle)  # model reload + a few frames
            shot.append(design_shot.grab(fbdir))
        finally:
            sim.close_viewer()  # also when the grab raised: otherwise show() never returns

    threading.Thread(target=grab, daemon=True).start()
    design_shot.watchdog(sim, 120)
    sim.show(script=script, width=W, height=H, design=b)
    if not shot:
        raise SystemExit(f"{out}: no screenshot was taken")
    Path(out).write_bytes(_dvsim.encode_png(shot[0]))
    print(
        f"wrote {out}: {shot[0].shape[1]}x{shot[0].shape[0]}; requests {[q['tool'] for q, _ in b.log if design_shot.is_edit(q['tool'])]}"
    )


def cmd_ui(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in a.names or UI:
        with tempfile.TemporaryDirectory(prefix="dvsim-xvfb-") as fb:
            argv = [
                "xvfb-run",
                "-a",
                "-s",
                f"-screen 0 {W}x{H}x24 -fbdir {fb}",
                sys.executable,
                __file__,
                "_ui",
                name,
                str(out / f"{name}.png"),
                fb,
            ]
            subprocess.run(
                argv, check=True, timeout=300
            )  # + the watchdog inside: no run outlives this


# ---- scenarios the model shots replay ------------------------------------------------------------------------------------------
SC = REPO / "examples"
SCENARIOS = {
    "x500-hover": SC / "x500" / "hover.design.scenario.toml",
    "x500-motor-out": SC / "faults" / "x500_motor_out.scenario.toml",
    "so101-pick": SC / "so101" / "pick.scenario.toml",
    "koch-pick": SC / "koch" / "pick.scenario.toml",
    "showcase": SC / "showcase" / "arm_loads_drone.scenario.toml",
    "so101-scratch-pick": None,  # examples/so101/pick.scenario.toml against the arm built from an empty design (see scratch_pick)
    "rover-square": SC / "rover" / "square.scenario.toml",
}
ENV = {"x500-hover": {"HOVER_HOLD_S": "8"}}


def scratch_pick(tmp):
    """The shipped SO-101 pick scenario with [design] = the SO-101 built from nothing (as tests/python/test_design_scratch_arm.py)."""
    sys.path.insert(0, str(REPO / "tests" / "python"))
    import test_design_scratch_arm as t

    saved = t.build_arm().save(Path(tmp) / "scratch-so101.toml")
    return t.pick_on(Path(tmp), saved)


def cmd_scenarios(a):
    from dvsim import program

    for name in a.names or SCENARIOS:
        if name == "rover-square" and not (REPO / "assemblies" / "rover.toml").exists():
            continue
        path = (
            scratch_pick(tempfile.mkdtemp(prefix="dvsim-scratch-"))
            if name == "so101-scratch-pick"
            else SCENARIOS[name]
        )
        os.environ.update(ENV.get(name, {}))
        try:
            res, d = program.run(path, Path(a.runs) / name)
        except (
            Exception
        ) as e:  # a model that does not build / run is reported, not fatal
            print(f"{name}: NOT RUN ({type(e).__name__}: {e})")
            continue
        print(
            f"{name}: {'PASS' if res['pass'] else 'FAIL ' + str(res['fail_reasons'])} tick {res['sim']['tick']}, replay {res['replay']}, {d}"
        )


# ---- replay a run's log to a sim time ------------------------------------------------------------------------------------------
def _flat(buf, tab, slot):
    """Position of a table field (slot) or None; FlatBuffers vtable walk."""
    vt = tab - struct.unpack_from("<i", buf, tab)[0]
    n = struct.unpack_from("<H", buf, vt)[0]
    off = struct.unpack_from("<H", buf, vt + 4 + 2 * slot)[0] if 4 + 2 * slot < n else 0
    return tab + off if off else None


def _ref(buf, pos):
    return pos + struct.unpack_from("<I", buf, pos)[0]


def _text(buf, pos):
    o = _ref(buf, pos)
    return buf[o + 4 : o + 4 + struct.unpack_from("<I", buf, o)[0]].decode()


def setattrs(log):
    """Replay-log bytes (format: src/core/replay.h) -> [(applied_tick, path, attr, values)] of its SetAttr records. Step / Run / Pause
    records are skipped: replay steps to each record's tick, which covers them (as `dvsim-replay` and examples/showcase/render.py)."""
    assert log[:8] == b"DVSIMLOG\0", "not a dvsim replay log"
    p = 8 + 4 + 8
    p += 4 + struct.unpack_from("<I", log, p)[0]
    for _ in range(2):  # model bytes, initial state
        p += 8 + struct.unpack_from("<Q", log, p)[0]
    out = []
    while p < len(log) - 24:
        tick, _err, n = struct.unpack_from("<QBI", log, p)
        env = bytes(log[p + 13 : p + 13 + n])
        p += 13 + n
        tab = _ref(env, 0)
        kp = _flat(env, tab, 2)
        if kp is None or env[kp] != 2:  # dvsim.Command SetAttr = 2
            continue
        sa = _ref(env, _flat(env, tab, 3))
        vp = _flat(env, sa, 3)
        vals = []
        if vp:
            vo = _ref(env, vp)
            vals = list(
                struct.unpack_from(
                    f"<{struct.unpack_from('<I', env, vo)[0]}d", env, vo + 4
                )
            )
        out.append(
            (tick, _text(env, _flat(env, sa, 0)), _text(env, _flat(env, sa, 1)), vals)
        )
    return out


class Replay:
    """A run dir's world, advanced to any increasing sim time by replaying its log's SetAttr commands."""

    def __init__(self, run_dir, model=None):
        from dvsim import Sim

        self.run = Path(
            run_dir
        ).resolve()  # the scene's mesh paths are relative to the run dir
        self.res = json.loads((self.run / "result.json").read_text())
        self.recs = setattrs((self.run / "replay.dvsimog").read_bytes())
        xmls = sorted(
            (self.run / "model").glob("*.xml"),
            key=lambda p: p.name.startswith("scene_"),
        )  # showcase.xml = the merged one
        model = Path(model) if model else xmls[0]
        with open(self.run / "trace.csv") as f:
            self.tr = list(csv.DictReader(f))
        self.t = np.array([float(r["t"]) for r in self.tr])
        self.sim, self.i = Sim.load(model), 0
        self.dt = (
            self.t[-1] / self.res["sim"]["tick"]
        )  # the trace's last row is the run's last tick

    def col(self, k):
        return np.array([float(r[k]) for r in self.tr])

    def to(self, t):
        tick = round(t / self.dt)
        while self.i < len(self.recs) and self.recs[self.i][0] <= tick:
            self._adv(self.recs[self.i][0])
            self.sim.set(*self.recs[self.i][1:])
            self.i += 1
        self._adv(tick)
        return tick

    def _adv(self, tick):
        if tick > self.sim.tick:
            self.sim.step(tick - self.sim.tick)

    def finish(self):
        """Replay the rest; -> (hash, matches the run's)."""
        for tick, *r in self.recs[self.i :]:
            self._adv(tick)
            self.sim.set(*r)
        self._adv(self.res["sim"]["tick"])
        h = f"{self.sim.state_hash():016x}"
        return h, h == self.res["sim"]["state_hash"]


# ---- model shots: (scenario, sim time, camera, what) -------------------------------------------------------------------------
def t_zmax(r):
    return r.t[r.col("z").argmax()]


def cam_follow(dist, az, el, dz=0.0):
    """Orbit camera on the ground-truth position of the trace row at the shot time (x, y, z columns)."""

    def f(r, t):
        j = min(np.searchsorted(r.t, t), len(r.t) - 1)
        return dict(
            camera="orbit",
            az=az,
            el=el,
            dist=dist,
            target=[r.col("x")[j], r.col("y")[j], r.col("z")[j] + dz],
        )

    return f


def cam_fixed(az, el, dist, target):
    return lambda r, t: dict(camera="orbit", az=az, el=el, dist=dist, target=target)


def t_cube_max(r):
    return r.t[r.col("cube_was_lifted").argmax()]


def t_fault(r, after):
    return r.res["faults"][0]["start_t"] + after


def t_handoff(r):
    """Showcase: the first time the payload is lifted off the shelf and over the drone's plate (scans the replay in 0.1 s steps)."""
    for t in np.arange(0.1, r.t[-1], 0.1):
        r.to(t)
        p = r.sim.query("/world/payload", "pos")[0]
        if p[1] > -0.06 and p[2] > 0.28:
            return t + 0.2
    raise RuntimeError("the payload never reached the plate")


def cam_prim(path, az, el, dist):
    """Orbit camera on a prim's position at the shot time."""
    return lambda r, t: dict(
        camera="orbit",
        az=az,
        el=el,
        dist=dist,
        target=[float(x) for x in r.sim.query(path, "pos")[0]],
    )


FRAMES = {
    "x500-hover-px4": (
        "x500-hover",
        t_zmax,
        cam_follow(1.1, 35, 8),
        "X500 V2 (parts-built design) hovering at 5 m under PX4 SITL",
    ),
    "x500-motor-out": (
        "x500-motor-out",
        lambda r: t_fault(r, 0.5),
        cam_follow(1.2, 35, 14),
        "motor-out fault: esc0 open circuit 0.5 s ago, the quad tumbling",
    ),
    "so101-pick": (
        "so101-pick",
        t_cube_max,
        cam_prim("/world/cube", 30, 20, 0.55),
        "SO-101 mid-pick: the cube lifted in the gripper",
    ),
    "koch-pick": (
        "koch-pick",
        t_cube_max,
        cam_prim("/world/cube", 160, 20, 0.5),
        "Koch arm mid-move with the cube in the gripper",
    ),
    "showcase-handoff": (
        "showcase",
        t_handoff,
        cam_prim("/world/payload", 20, 25, 0.55),
        "arm-loads-drone: the SO-101 puts the cube on the drone's top plate",
    ),
    "showcase-flight": (
        "showcase",
        t_zmax,
        cam_follow(1.5, 30, 10),
        "arm-loads-drone: the loaded drone in flight at its 4 m top",
    ),
    "rover-square": (
        "rover-square",
        lambda r: r.t[
            np.argmax(r.col("corner_1_1") < 0.5)
        ],  # first time half a metre from the third corner: mid-leg
        cam_prim("/world/plate", 40, 25, 0.5),
        "rover driving its 1 m square, mid-leg",
    ),
    "so101-scratch-pick": (
        "so101-scratch-pick",
        t_cube_max,
        cam_prim("/world/cube", 30, 20, 0.55),
        "scratch-built SO-101 mid-pick",
    ),
}


def render_to(sim, cam, out, aa=2):
    from dvsim import _dvsim, showcase

    Path(out).write_bytes(_dvsim.encode_png(showcase.shot(sim, W, H, aa, **cam)))


def cmd_frames(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man_p = out / "manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {}
    for name in a.names or FRAMES:
        scen, tfun, cam, what = FRAMES[name]
        runs = sorted((Path(a.runs) / scen).glob("*/result.json"))
        if not runs:
            print(f"{name}: no run of {scen} under {a.runs} (run `scenarios` first)")
            continue
        run = runs[-1].parent
        model = (
            REPO / "tests/data/px4/x500_fixture.xml"
            if scen == "x500-motor-out"
            else None
        )
        r = Replay(run, model)
        t = float(tfun(r))
        tick = r.to(t)
        h = f"{r.sim.state_hash():016x}"
        render_to(r.sim, cam(r, t), out / f"{name}.png")
        fh, ok = r.finish()
        man[name] = dict(
            what=what,
            scenario=scen,
            sim_s=round(tick * r.dt, 3),
            tick=tick,
            state_hash=h,
            run_final_hash=fh,
            replay_matches_run=ok,
        )
        print(
            f"{name}: t={tick * r.dt:.3f} s tick={tick} hash={h}; replay final hash {fh} {'MATCH' if ok else 'DIFFERENT'}"
        )
    man_p.write_text(json.dumps(man, indent=1))


# ---- the scratch-built quad under the built-in hover ------------------------------------------------------------------------
def cmd_quad(a):
    """Design.simulate('hover') on the quad built from an empty design; Sim.step is wrapped to render the hover Sim at ALT / HOLD time."""
    import dvsim
    from dvsim import Sim

    sys.path.insert(0, str(REPO / "tests" / "python"))
    import test_design_scratch as t

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d, _ = t.build_scratch()
    main, step, done, info = [], Sim.step, [], {}
    load = Sim.load.__func__

    def load2(cls, *args, **kw):
        s = load(cls, *args, **kw)
        main.append(
            s
        )  # the first Sim the hover loads is the one it flies (its calibration fork is made from it)
        return s

    def step2(self, n=1):
        k = step(self, n)
        if (
            main and self is main[0] and not done and self.tick * 0.002 >= a.at
        ):  # hover timestep: the model's (0.002 s default)
            done.append(1)
            pos = self.query(
                next(k for k in self.body_positions() if k.count("/") == 2), "pos"
            )[0]
            tgt = pos
            render_to(
                self,
                dict(camera="orbit", az=35, el=14, dist=1.5, target=list(tgt)),
                out / "scratch-quad-hover.png",
            )
            info.update(
                sim_s=self.tick * 0.002,
                tick=self.tick,
                state_hash=f"{self.state_hash():016x}",
                pos=[round(float(x), 3) for x in pos],
            )
        return k

    Sim.load, Sim.step = classmethod(load2), step2
    try:
        res = d.simulate(
            "hover",
            out=tempfile.mkdtemp(prefix="dvsim-quad-"),
            altitude=2.0,
            climb_s=3.0,
            hold_s=4.0,
        )
    finally:
        Sim.load, Sim.step = classmethod(load), step
    print(
        f"scratch-quad-hover: {info}; hover pass={res['pass']} throttle={res.get('hover_throttle')}"
    )
    man_p = out / "manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {}
    man["scratch-quad-hover"] = dict(
        what="the quad built from an empty design, built-in hover",
        **info,
        hover_result={
            k: res.get(k)
            for k in ("pass", "altitude_mean_m", "hover_throttle", "hover_current_A")
        },
    )
    man_p.write_text(json.dumps(man, indent=1))
    assert dvsim._dvsim is not None


# ---- PNG size ---------------------------------------------------------------------------------------------------------------
def cmd_optimize(a):
    from PIL import Image

    for p in sorted(Path(a.dir).glob("*.png")):
        if p.stat().st_size > LIMIT:
            before = p.stat().st_size
            Image.open(p).convert("RGB").quantize(
                256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
            ).save(p, optimize=True)
            print(f"{p.name}: {before} -> {p.stat().st_size}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("scenarios", "frames", "quad", "ui", "all"):
        s = sub.add_parser(n)
        s.add_argument(
            "--runs",
            default="/tmp/dvsim-shots-runs",
            help="where the scenario run dirs go / are read from",
        )
        s.add_argument("--out", default="build/screens")
        s.add_argument(
            "--at",
            type=float,
            default=5.0,
            help="quad: sim s of the hover frame (climb 3 s, then hold)",
        )
        s.add_argument("names", nargs="*")
    sub.add_parser("list")
    sub.add_parser("optimize").add_argument("dir")
    u = sub.add_parser("_ui")
    u.add_argument("name"), u.add_argument("out"), u.add_argument("fbdir")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        print(
            "ui:",
            *UI,
            "\nscenarios:",
            *SCENARIOS,
            "\nframes:",
            *FRAMES,
            "\nquad: scratch-quad-hover",
        )
    elif a.cmd == "_ui":
        xvfb_png(a.out, *UI[a.name][:3], a.fbdir, *UI[a.name][3:])
    elif a.cmd == "all":
        a.names = []
        for f in (cmd_scenarios, cmd_frames, cmd_quad, cmd_ui):
            f(a)
    else:
        globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
