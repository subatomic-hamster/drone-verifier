"""dvsim's real viewer window as a live video feed: ViewerUI (the Session's GUI seam), an fMP4 stream, a recording in the run dir.

The viewer renders off-screen on the GPU (DVSIM_OFFSCREEN=1: no X server; every frame is read back, `_dvsim.viewer_frame`). A grab thread hands the newest frame to
one gst child (H.264 by VA, fragmented MP4 at 200 ms; a busy encoder drops frames, it never queues). Its output is cut into fragments, appended to
<run_dir>/stream.mp4 (capped, playable even if the process is killed) and offered to every client of a unix socket, /tmp/dv-gui-<port>.sock:
`python -m dv.gui_stream --port PORT` copies that to stdout for the laptop's ssh exec. A client that falls behind skips to the next key frame.

fly_live replays a flight the verifier already flew and judged (its replay.dvsimog, tick-exact, on the window's World, at real time; Session uses this). It can also fly a spec live in
the window: same dv.flight.run_flight, its Sim bound to the World, paced to real time (results equal the headless flight's).

Wires are the build's straight chords, not the router's paths: routing the hopper takes 224 s on hfbox per geometry, the window must follow an edit within a second.
ponytail: chords only; route in a niced child (DVSIM_ROUTE_CACHE is its disk cache) if a stable design should get its real paths on screen.

Wiring (dv.server --gui):
    ui = ViewerUI(run_dir, port=port)     # give Session this as its ui
    threading.Thread(target=serve_http, daemon=True).start()
    ui.run()                              # main thread; blocks until ui.stop()

Knobs in the environment: DV_STREAM_FPS, DV_STREAM_KBPS. Check: python -m dv.gui --help.
"""

import collections
import gzip
import json
import math
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

W, H = 1280, 720  # the window
FPS = int(os.environ.get("DV_STREAM_FPS", 15))
KBPS = int(os.environ.get("DV_STREAM_KBPS", 1800))
REC_MAX_BYTES = 400_000_000  # about 30 min at 1800 kbit/s
MAX_BEHIND = (
    8  # fragments of 200 ms a client may lag before it skips to the next key frame
)
TYPE_S = 5.0  # show_code never takes longer than this
REVEAL_S = 1.5  # between parts when one design_set adds several
FLY_TIMEOUT_S = 240.0  # a tool never waits longer than this for the window's flight
BLANK = '<mujoco model="blank"><worldbody><geom name="floor" type="plane" size="3 3 0.1"/></worldbody></mujoco>'
PAGES = {"design": "build", "program": "program", "run": "simulate"}
GREEN, RED, WHITE, DIM = (
    (110, 235, 140),
    (255, 90, 80),
    (255, 255, 255),
    (170, 175, 185),
)


def sock_path(port):
    return f"/tmp/dv-gui-{port}.sock"


def _log(*a):
    print("gui:", *a, file=sys.stderr, flush=True)  # ttsr-ignore: py/no-print


def is_key(mdat):
    """Does this fragment's media data (4-byte length-prefixed H.264 NAL units, without the box header) start on an IDR frame?"""
    p = 0
    while p + 5 <= len(mdat):
        t = mdat[p + 4] & 31
        if t == 5:
            return True
        if t == 1:
            return False
        p += 4 + int.from_bytes(mdat[p : p + 4], "big")
    return False


class _Client(threading.Thread):
    """One socket reader: gets the init segment, then fragments from a key frame on. It never holds more than MAX_BEHIND fragments: past that the backlog goes and it
    waits for the next key frame."""

    def __init__(self, conn, init):
        super().__init__(daemon=True)
        self.conn, self.init = conn, init
        self.q, self.cv, self.need_key, self.gone = (
            collections.deque(),
            threading.Condition(),
            True,
            False,
        )
        self.dropped = 0

    def offer(self, frag, key):
        with self.cv:
            if len(self.q) >= MAX_BEHIND:
                self.dropped += len(self.q)
                self.q.clear()
                self.need_key = True
            if self.need_key and not key:
                return
            self.need_key = False
            self.q.append(frag)
            self.cv.notify()

    def run(self):
        try:
            self.conn.sendall(self.init)
            while True:
                with self.cv:
                    while not self.q:
                        self.cv.wait()
                    frag = self.q.popleft()
                self.conn.sendall(frag)
        except OSError:
            pass
        self.gone = True
        self.conn.close()


class Stream:
    """Frames in (offer, newest only), fMP4 out: a bounded recording and a unix socket."""

    def __init__(self, sock, rec_path=None):
        self.frame, self.ev, self.stopped = None, threading.Event(), False
        self.offered = self.encoded = self.bytes = self.frags = self.keys = 0
        self.init, self.clients, self.t0 = b"", [], time.time()
        desc = (
            f"fdsrc fd=0 ! rawvideoparse format=rgba width={W} height={H} framerate={FPS}/1 ! queue leaky=downstream max-size-buffers=2 ! "
            f"videoconvert ! video/x-raw,format=NV12 ! vah264enc bitrate={KBPS} key-int-max={FPS} b-frames=0 rate-control=cbr ! h264parse ! "
            "mp4mux fragment-duration=200 streamable=true ! fdsink fd=1 sync=false"
        )
        self.p = subprocess.Popen(
            ["gst-launch-1.0", "-q", *desc.split()],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        self.rec = open(rec_path, "wb") if rec_path else None
        self.srv = socket.socket(socket.AF_UNIX)
        Path(sock).unlink(missing_ok=True)
        self.srv.bind(sock)
        self.srv.listen()
        self.sock = sock
        for fn in (self._pump, self._read, self._accept):
            threading.Thread(target=fn, name="gui-" + fn.__name__, daemon=True).start()

    def offer(self, frame):
        self.frame = frame
        self.offered += 1
        self.ev.set()

    def _pump(self):
        try:
            while not self.stopped:
                self.ev.wait()
                self.ev.clear()
                if self.frame is not None:
                    self.p.stdin.write(self.frame)
                    self.p.stdin.flush()
                    self.encoded += 1
        except (OSError, ValueError) as e:
            _log("encoder gone:", e)

    def _read(self):
        """The muxer's boxes -> init segment (everything before the first moof) and moof + mdat fragments."""
        out, moof, started = self.p.stdout, b"", False
        while True:
            h = out.read(8)
            if len(h) < 8:
                _log("encoder output ended")
                return
            body = out.read(int.from_bytes(h[:4], "big") - 8)
            typ = h[4:8]
            if typ == b"moof":
                moof, started = h + body, True
            elif not started:
                self.init += h + body
            elif typ == b"mdat":
                self._publish(moof + h + body, is_key(body))

    def _publish(self, frag, key):
        self.frags += 1
        self.keys += key
        self.bytes += len(frag)
        if self.rec:
            if self.frags == 1:
                self.rec.write(self.init)
                # the replay page lines the recording up with events.jsonl by this
                Path(self.rec.name).with_suffix(".json").write_text(
                    json.dumps(dict(t0=time.time() - 0.2))
                )
            self.rec.write(frag)
            self.rec.flush()
            if self.rec.tell() > REC_MAX_BYTES:
                _log(f"recording stopped at its cap of {REC_MAX_BYTES} bytes")
                self.rec.close()
                self.rec = None
        self.clients = [c for c in self.clients if not c.gone]
        for c in self.clients:
            c.offer(frag, key)

    def _accept(self):
        while True:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            while not self.init and not self.stopped:
                time.sleep(0.05)
            c = _Client(conn, self.init)
            self.clients.append(c)
            c.start()

    def stats(self):
        dt = max(time.time() - self.t0, 1e-3)
        return dict(
            offered_fps=round(self.offered / dt, 1),
            encoded_fps=round(self.encoded / dt, 1),
            kbytes_s=round(self.bytes / dt / 1000, 1),
            frags=self.frags,
            keys=self.keys,
            clients=len(self.clients),
        )

    def close(self):
        self.stopped = True
        self.ev.set()
        self.srv.close()
        Path(self.sock).unlink(missing_ok=True)
        try:
            self.p.stdin.close()
            self.p.wait(5)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            self.p.kill()
        if self.rec:
            self.rec.close()


class _Chords:
    """A Design whose build skips the wire router (the tubes are chords; cosmetic, physics identical). The rest is the Design."""

    def __init__(self, d):
        self._d = d

    def __getattr__(self, k):
        return getattr(self._d, k)

    def build(self, out):
        from dvsim.design import compute

        d = self._d
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        # dv/refly.py routes for real: its wires were routed into DVSIM_ROUTE_CACHE before the window opened
        route = bool(os.environ.get("DV_ROUTE_WIRES"))
        xml, report = compute(d.asm, d.lib.dir, out, d.lib, route=route)
        xp = out / f"{report['assembly']}.xml"
        xp.write_text(xml, encoding="utf-8", newline="\n")
        return {"xml": xp, "report": report}


class _WindowSim:
    """The window's Sim as run_flight sees it: a Host's replay log runs from its start, there is nothing to start."""

    def __init__(self, sim):
        self._s = sim

    def __getattr__(self, k):
        return getattr(self._s, k)

    def start_log(self):
        pass


class ViewerUI:
    """The Session's GUI seam: design_changed, show_code, fly_live, page, stop. Every call only queues and returns (the window's work runs on one worker thread, in
    order); nothing raises into a tool. `run()` owns the calling thread until stop()."""

    def __init__(self, run_dir, port=0, design=None, lock=None):
        from dvsim.design.viewer import Backend

        class _B(Backend):
            frozen = None  # while a flight replay owns the World the window does not hear of design edits (it would load the design over the flight)
            _shim = (None, None)

            def _dispatch(self, req):
                if req.get("tool") == "rev" and self.frozen:
                    return self.frozen
                if req.get("tool") == "prog.hover":
                    return {
                        "ok": False,
                        "error": "no pointer on stage",
                    }  # the unmoved mouse hovers a word: the popup covers the code
                return super()._dispatch(req)

            def state(self, d):
                if isinstance(d, _Chords):
                    return super().state(d)
                if self._shim[0] is not d:
                    self._shim = (d, _Chords(d))
                return super().state(self._shim[1])

        self.run_dir = Path(run_dir)
        self.prog_dir = (
            self.run_dir / "gui_prog"
        )  # the PROGRAM page's root: only what we type, never the session's own files
        self.prog_dir.mkdir(parents=True, exist_ok=True)
        self.port = port
        self.design = getattr(design, "design", design)
        # Not the session's lock, which a tool holds for all of a verify: the window would stop following for that long.
        # Nothing else touches a Design once apply() has built it, and a new design is a new object.
        self.lock = threading.RLock()
        self.b = _B(
            lambda: self.design,
            self.run_dir / "gui_build",
            lock=self.lock,
            program_root=self.prog_dir,
        )
        self.q = queue.Queue()
        self.stopped = threading.Event()
        self.stream = self.sim = self.hud = None
        self.mode = "build"
        self.seq = 0
        self.opened = None
        self.grabbed = 0

    # ---- the seam ----------------------------------------------------------------------------------------------------------------------
    def design_changed(self, design):
        self.q.put(("design", getattr(design, "design", design)))

    def reveal(self, designs):
        """Intermediate designs of one design_set, one part further each: for the audience only, the tool has already answered."""
        self.q.put(("reveal", designs))

    def show_code(self, text, path="controller.py"):
        self.q.put(("code", text, str(path)))

    def fly_live(self, scenario, spec, emit=None):
        """spec = dict(run_dir=<replay.dvsimog[.gz], native.json, scene.xml>, passed=, reason=): queues the replay of a flight already flown and judged. -> None.
        spec = a flight spec as for `python -m dv.flight`: the flight itself in the window, dv.flight.run_flight on the window's World paced to real time; emit gets its flight_tick
        events. The one seam call that blocks: the tool waits for its flight. -> the result, or None if the window could not fly it (fly headless then)."""
        if spec.get("run_dir"):
            self.q.put(("replay", scenario, spec))
            return None
        done = queue.Queue(1)
        self.q.put(("fly", scenario, spec, emit, done))
        try:
            return done.get(timeout=FLY_TIMEOUT_S)
        except queue.Empty:
            return None

    def page(self, name):
        self.q.put(("page", name))

    def stop(self):
        self.stopped.set()
        self.q.put(("stop",))
        if self.sim is not None:
            self.sim.close_viewer()

    # ---- the window ----------------------------------------------------------------------------------------------------------------------
    def run(self):
        """Opens the window and blocks. Call on the main thread."""
        os.environ["DVSIM_OFFSCREEN"] = "1"
        from dvsim import Sim
        from dvsim import hud as hudmod

        xml = None
        if self.design is not None:
            with self.lock:
                xml = self.b.state(self.design)["xml"]
        if not xml:
            xml = self.run_dir / "gui_blank.xml"
            xml.write_text(BLANK)
        self.sim = self.b.sim = Sim.load(
            xml, root="/", host=True
        )  # root: flight replays load models from their run dirs
        self.hud = hudmod.Hud()
        self.stream = Stream(sock_path(self.port), self.run_dir / "stream.mp4")
        threading.Thread(target=self._grab, name="gui-grab", daemon=True).start()
        threading.Thread(target=self._work, name="gui-work", daemon=True).start()
        try:
            self.sim.show(width=W, height=H, design=self.b, hud=self.hud)
        finally:
            self.stopped.set()
            self.stream.close()
            self.b.prog.close()

    def _grab(self):
        from dvsim import _dvsim

        nxt, last = time.monotonic(), 0
        while not self.stopped.is_set():
            f = _dvsim.viewer_frame()
            if f is not None and f[1].shape[:2] == (H, W):
                self.stream.offer(
                    f[1].data
                )  # also an unchanged frame: the video's clock is the frame count
                self.grabbed += f[0] != last
                last = f[0]
            nxt = max(nxt + 1 / FPS, time.monotonic())
            time.sleep(max(0.0, nxt - time.monotonic()))

    def _ui(self, tool, **args):
        r = json.loads(self.b(json.dumps({"tool": tool, "args": args})))
        if not r["ok"]:
            _log(tool, r.get("error"))
        return r

    def _mode(self, name):
        if self.mode != name:
            self.mode = name
            self._ui("ui.mode", name=name)

    def _work(self):
        idle = 0.0
        self._ui(
            "ui.view", tooltips="off"
        )  # no pointer on stage: a stuck hover box otherwise
        while not self.stopped.is_set():
            try:
                items = [self.q.get(timeout=0.5)]
            except queue.Empty:
                if (
                    time.monotonic() - idle > 9.5
                    and self.mode in ("build", "simulate")
                    and self.design is not None
                ):
                    idle = time.monotonic()  # the camera never sits still
                    self._ui(
                        "camera.orbit",
                        yaw_deg=36,
                        seconds=10.0,
                        ease="linear",
                        relative=True,
                    )
                continue
            while not self.q.empty():
                items.append(self.q.get())
            for k, it in enumerate(items):
                if it[0] == "stop":
                    return
                if it[0] in ("design", "code") and any(
                    j[0] == it[0] for j in items[k + 1 :]
                ):
                    continue  # superseded by a newer one in the same batch
                try:
                    getattr(self, "_do_" + it[0])(*it[1:])
                except Exception as e:  # the window is decoration: a failure here must never reach a tool
                    _log(it[0], "failed:", type(e).__name__, e)
            idle = 0.0

    def _do_design(self, design):
        self.design = design
        self._leave_flight()
        was = self.mode
        self._mode("build")
        if was != "build" or not self.seq:
            self.seq += 1
            self._ui("ui.camera", preset="iso")  # pitch above the floor
        self._ui("ui.frame", instances=[])

    def _do_reveal(self, designs):
        for d in designs:
            self._do_design(d)
            if self.stopped.wait(REVEAL_S):
                return

    def _do_page(self, name):
        name = PAGES.get(name, name)
        if name != "simulate":
            self._leave_flight()
        self._mode(name)

    def _do_code(self, text, path):
        """A fresh file and buffer per version; the text arrives in chunks over TYPE_S (the window drains its events every 100 ms)."""
        self._leave_flight()
        self._mode("program")
        name = Path(path).name or "controller.py"
        if self.opened:
            self._ui("prog.ui", cmd="cmd", name="close_tab")
            time.sleep(0.2)
        (self.prog_dir / name).write_text("")
        del self.b.log[:-50]  # the Backend logs every request, for good
        seen = len(self.b.log)
        self._ui("prog.ui", cmd="open", path=name)
        self.opened = name
        # the buffer loads through the window's queue; text typed before its prog.read comes back is lost
        limit = time.monotonic() + 5.0
        while time.monotonic() < limit and not any(
            r[0].get("tool") == "prog.read" for r in self.b.log[seen:]
        ):
            time.sleep(0.05)
        time.sleep(0.25)  # the reply reaches the window at its next 100 ms drain
        self._ui("prog.ui", cmd="dock", name="output")  # not the shell's tab
        pause = 0.12
        lines = text.splitlines(keepends=True)
        per = max(1, math.ceil(len(lines) / (TYPE_S / pause)))
        for i in range(0, len(lines), per):
            self._ui("prog.ui", cmd="type", text="".join(lines[i : i + per]))
            if i + per < len(
                lines
            ):  # the view follows the cursor (whole lines: it sits at column 0)
                self._ui("prog.ui", cmd="goto", line=i + per + 1)
            time.sleep(pause)
        self._ui("prog.ui", cmd="cmd", name="save")

    # ---- the flight: a replay of the log the verifier judged -------------------------------------------------------------------------------
    def _doc(self, items, tick=None):
        from dvsim import hud as h

        self.seq += 1
        self.hud.set(h.doc(items, seq=self.seq, tick=tick))

    def _leave_flight(self):
        if self.b.frozen:
            self.b.frozen = None
            self._doc([])
            self.b.runs.held = True
            self.b.runs.restore()  # the design's model back in the World
            self._ui("ui.camera", preset="iso")
            self._ui("ui.frame", instances=[])

    def _load_flight(self, path):
        """The flight's own model into the window's World; design edits wait until the window leaves it."""
        self.b.frozen = {"ok": True, "result": self._ui("rev")["result"]}
        self._mode("simulate")
        self.sim.call({"cmd_type": "Pause", "cmd": {}})
        r = self.sim.call({"cmd_type": "LoadModel", "cmd": {"path": str(path)}})
        if not r["ok"]:
            raise RuntimeError(f"LoadModel {path}: {r['message']}")

    def _follower(self, scenario):
        """-> follow(t), called with increasing sim time: holds the caller to real time, moves the camera with the vehicle, slowly orbiting it, keeps the HUD current."""
        from dvsim import hud as h

        sim = self.sim
        names = list(sim.body_positions())
        path = next(
            (k for k in names if k.rsplit("/", 1)[-1] in ("hopper_frame", "frame")),
            next((k for k in names if "frame" in k), names[-1]),
        )
        pos = lambda: [float(x) for x in sim.query(path, "pos")[0]]
        st = dict(az=2.2, cam=0.0, t0=None)
        z0 = pos()[2]

        def look(p, seconds):
            eye = [
                p[0] + 1.6 * math.cos(st["az"]),
                p[1] + 1.6 * math.sin(st["az"]),
                max(p[2] + 0.7, 0.5),  # above the floor, looking down at the vehicle
            ]
            self._ui("camera.fly", eye=eye, look=p, seconds=seconds)

        look(pos(), 0.8)
        time.sleep(0.9)

        def follow(t):
            now = time.monotonic()
            if st["t0"] is None:
                st["t0"] = now - t
            if st["t0"] + t > now:
                time.sleep(st["t0"] + t - now)
            if now - st["cam"] > 0.25:
                st["cam"] = now
                st["az"] += 0.02
                p = pos()
                look(p, 0.3)
                rows = [
                    ("SCENARIO", scenario),
                    ("T", f"{t:5.1f} s"),
                    ("HEIGHT", f"{p[2] - z0:5.2f} m"),
                ]
                self._doc([h.nudge(h.kv(rows, at=(0, 0), size=20, gap=30), 16, 16)])

        return follow

    def _verdict(self, scenario, ok, reason):
        from dvsim import hud as h

        self._doc(
            [
                h.text(
                    "PASS" if ok else "FAIL",
                    at=(0.5, 0.12),
                    a="t",
                    size=64,
                    col=GREEN if ok else RED,
                    bg=(0, 0, 0, 228),
                    pad=18,
                    f=True,
                ),
                h.text(
                    str(reason or "")[:100],
                    at=(0.5, 0.32),
                    a="t",
                    size=20,
                    col=WHITE,
                    bg=(0, 0, 0, 200),
                    pad=10,
                    f=True,
                ),
                h.text(
                    f"the verifier's judgement of this flight, {scenario}",
                    at=(0.5, 0.97),
                    a="b",
                    size=14,
                    col=DIM,
                    f=True,
                ),
            ]
        )

    def _do_replay(self, scenario, spec):
        sys.path.insert(0, str(Path(os.environ.get("DVSIM_ROOT", ".")) / "tools"))
        from screenshots import setattrs  # parses a replay.dvsimog's SetAttr records

        run = Path(spec["run_dir"])
        log = run / "replay.dvsimog"
        recs = setattrs(
            log.read_bytes()
            if log.exists()
            else gzip.decompress((run / "replay.dvsimog.gz").read_bytes())
        )
        native = json.loads((run / "native.json").read_text())
        end, dt = int(native["tick"]), float(native["dt"])
        self._load_flight(spec.get("model") or run / "scene.xml")
        sim = self.sim
        base = sim.tick  # (a Host's tick may carry on across loads)
        follow = self._follower(scenario)
        i, tick, stride = 0, 0, 20
        while not self.stopped.is_set():
            follow(tick * dt)
            tick = min(tick + stride, end)
            while i < len(recs) and recs[i][0] <= tick:
                if recs[i][0] + base > sim.tick:
                    sim.step(recs[i][0] + base - sim.tick)
                sim.set(*recs[i][1:])
                i += 1
            if tick + base > sim.tick:
                sim.step(tick + base - sim.tick)
            if tick >= end:
                break
        self.replayed = f"{sim.state_hash():016x}"
        self._verdict(scenario, spec.get("passed"), spec.get("reason"))

    def _do_fly(self, scenario, spec, emit, done):
        from dv.flight import run_flight

        res = None
        st = {}

        def load(path):
            self._load_flight(path)
            return _WindowSim(self.sim)

        def on_tick(sim, t):
            if "follow" not in st:
                st["follow"] = self._follower(scenario)
            st["follow"](t)

        try:
            res = run_flight(spec, emit, load_sim=load, on_tick=on_tick)
            self._verdict(scenario, res["pass"], res["reason"])
        finally:
            done.put(res)


def main():
    """Check + stub: the window on an assembly, a scripted build / code / flight, PNG shots, assertions at the end (stream parses, typing is whole and under the cap)."""
    import argparse

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--port", type=int, default=8890)
    ap.add_argument("--no-flight", action="store_true")
    ap.add_argument(
        "--live",
        action="store_true",
        help="also fly it live in the window and compare with the headless flight",
    )
    ap.add_argument("--out", default="/tmp/dv-gui-check")
    ap.add_argument("--shots", help="directory for a PNG at each step")
    ap.add_argument(
        "--hold", type=float, default=0, help="stay up this long after the script"
    )
    a = ap.parse_args()

    from dvsim import _dvsim

    from dv import design as D
    from dv import flight as F

    ref = json.loads(Path("reference/design.json").read_text())
    code_text = Path("reference/controller.py").read_text()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    say = lambda *x: print(*x, flush=True)  # ttsr-ignore: py/no-print
    t = time.time()
    b = D.apply(ref)
    ui = ViewerUI(out, a.port, b)
    fails = []
    say("design applied", round(time.time() - t, 1), "s, cost", round(b.cost, 1))

    def flight_spec(built, log=None):
        xml = out / "fly.xml"
        xml.write_text(built.xml)
        return dict(
            xml=str(xml),
            scenario="nominal",
            seed=1,
            scatter=None,
            scenario_params=D.TASK["scenarios"]["nominal"],
            code=code_text,
            cfg=D.controller_cfg(built),
            mission=D.TASK["mission"],
            sandbox=D.TASK["sandbox"],
            soc_min_pct=20.0,
            log=log,
            **D.flight_inputs(built),
        )

    def shot(name):
        f = _dvsim.viewer_frame()
        if a.shots and f:
            Path(a.shots).mkdir(parents=True, exist_ok=True)
            (Path(a.shots) / f"{name}.png").write_bytes(_dvsim.encode_png(f[1]))
        say(
            name,
            ui.stream.stats(),
            "viewer frames",
            f and f[0],
            "grab distinct",
            ui.grabbed,
        )

    def script():
        while ui.stream is None or ui.stream.frags < 5:
            time.sleep(0.2)
        say("first frags after", round(time.time() - t, 1), "s")
        c = socket.socket(socket.AF_UNIX)
        c.connect(sock_path(a.port))
        got = c.recv(65536)
        if got[4:8] != b"ftyp":
            fails.append(f"stream starts with {got[:8]!r}")
        c.close()
        ui.design_changed(b)
        time.sleep(4)
        shot("1_build")
        slot, opts = next((k, v["options"]) for k, v in D.SLOTS.items() if k == "esc")
        other = next(o["part"] for o in opts if o["part"] != ref.get(slot))
        t1 = time.time()
        b2 = D.apply(dict(ref, **{slot: other}))
        t_apply = time.time() - t1
        t1 = time.time()
        ui.design_changed(b2)
        st = ui.b.state(
            b2.design
        )  # what the window's next poll asks for; includes the chord build
        say(
            "edit: core apply",
            round(t_apply, 2),
            "s, window state",
            round(time.time() - t1, 2),
            "s, build error",
            st["build_error"],
        )
        time.sleep(1.5)
        shot("2_edited")
        code = code_text
        t2 = time.monotonic()
        ui.show_code("superseded", "controller.py")
        ui.show_code(code, "controller.py")
        time.sleep(3)
        shot("3_typing")
        while (
            out / "gui_prog/controller.py"
        ).read_text() != code and time.monotonic() - t2 < TYPE_S + 5:
            time.sleep(0.1)
        typed = time.monotonic() - t2
        say("typed + saved in", round(typed, 1), "s")
        if (out / "gui_prog/controller.py").read_text() != code:
            fails.append("the saved file differs from the text")
        if typed > TYPE_S + 3:
            fails.append(f"typing took {typed:.1f} s")
        time.sleep(0.5)
        shot("4_typed")
        if not a.no_flight:
            ui.design_changed(b)
            time.sleep(1)
            keys = ("t_end_s", "z_max_m", "tilt_max_deg", "land_dist_m")
            # the replay (what Session does): headless flight with a log, then the window plays it; must end on the flight's hash, at about real time
            fd = out / "flight_last"
            fd.mkdir(exist_ok=True)
            spec = flight_spec(b, log=str(fd / "replay.dvsimog"))
            t3 = time.time()
            head = F.run_flight(spec)
            say("headless flight wall", round(time.time() - t3, 1), "s", {k: head["metrics"][k] for k in keys}, head["pass"], head["state_hash"])  # fmt: skip
            t3 = time.time()
            threading.Timer(9, shot, ["5_replay"]).start()
            if ui.fly_live("nominal", dict(run_dir=str(fd), passed=head["pass"], reason=head["reason"])) is not None:  # fmt: skip
                fails.append("a replay returned a result")
            if time.time() - t3 > 0.2:
                fails.append("a replay blocked its caller")
            while getattr(ui, "replayed", None) is None and time.time() - t3 < head["metrics"]["t_end_s"] * 2 + 10:  # fmt: skip
                time.sleep(0.1)
            secs = time.time() - t3
            say("replay wall", round(secs, 1), "s of", head["metrics"]["t_end_s"], "hash", ui.replayed, "flight", head["state_hash"])  # fmt: skip
            if ui.replayed != head["state_hash"]:
                fails.append("the replay's hash differs from the flight's")
            if secs > 1.3 * head["metrics"]["t_end_s"] + 3:
                fails.append(f"replay not near real time: {secs:.1f} s")
            time.sleep(1.5)
            shot("6_verdict")
            ui.page("design")
            time.sleep(3)
            shot("7_back")
            if ui.b.frozen:
                fails.append("still frozen after leaving the flight")
            if a.live:  # the same flight flown in the window
                t3 = time.time()
                res = ui.fly_live("nominal", flight_spec(b), lambda *x: None)
                secs = time.time() - t3
                if res is None:
                    fails.append("the window did not fly it")
                else:
                    say("live flight wall", round(secs, 1), "s", {k: res["metrics"][k] for k in keys}, res["metrics"]["late_steps"])  # fmt: skip
                    if any(abs(head["metrics"][k] - res["metrics"][k]) > 1e-6 for k in keys):  # fmt: skip
                        fails.append("the live flight differs from the headless one")
                ui.page("design")
                time.sleep(1)
        time.sleep(a.hold)
        st = ui.stream.stats()
        if st["encoded_fps"] < FPS * 0.7:
            fails.append(f"encoded fps {st['encoded_fps']} < {FPS * 0.7}")
        if not st["keys"] or (out / "stream.mp4").stat().st_size == 0:
            fails.append("no key frames, or no recording")
        ui.stop()

    threading.Thread(target=script, daemon=True).start()
    ui.run()
    say("stats", ui.stream.stats(), "rec bytes", (out / "stream.mp4").stat().st_size)
    say("FAIL " + "; ".join(fails) if fails else "OK")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
