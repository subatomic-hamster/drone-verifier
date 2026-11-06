"""PX4 SITL + dvsim-px4-bridge as one stack, for `dvsim program` (scenario `[interfaces.px4]`).

    st = Stack(bridge_cfg, run_dir)          # st.mavlink = "udp://:14540": the MAVSDK connection string of a user program
    st.start()                               # bridge (own World, lockstep speed 1.0) + PX4 SITL, returns once PX4 is connected
    ...                                      # user program flies; bridge writes run_dir/trace.csv and run_dir/replay.dvsimog
    final = st.stop()                        # {"tick": n, "state_hash": "<hex>"} of the bridge's World; kills both processes

tests/python/test_px4_flight.py launches through this same Stack (after stop(): st.rc = the bridge's exit code).
Env: DVSIM_PX4_DIR (PX4 checkout, default the hfbox path), DVSIM_PX4_INSTANCE (default 0), DVSIM_BRIDGE_EXE, DVSIM_REPLAY_EXE,
DVSIM_PX4_SPEED (default 1 = real time; the bridge paces sim time at this factor, PX4 follows the HIL clock in lockstep. Measured on
12 Skydio seeds x goal/follow at 1/2/4/8: success within the Wilson intervals, but whole-MC wall only 1084 -> 977 -> 906 -> 824 s,
because PX4 boot, the planner and the replay are wall-bound; measured on a loaded host, so the default stays 1).
Everything binds 127.0.0.1-reachable ports 4560+N (tcp) and 14540+N / 14550+N / 18570+N (udp); never 5177-5194, 8000, 8010, 8093.
"""

import json
import os
import queue
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PX4_DIR = Path(os.environ.get("DVSIM_PX4_DIR", "/workspace/wt/dvsim/px4/src"))
BUILD = PX4_DIR / "build" / "px4_sitl_default"
PX4_BIN = BUILD / "bin" / "px4"
BRIDGE = Path(os.environ.get("DVSIM_BRIDGE_EXE", REPO / "build" / "dvsim-px4-bridge"))
REPLAY = Path(os.environ.get("DVSIM_REPLAY_EXE", REPO / "build" / "dvsim-replay"))
PARAMS = REPO / "px4" / "params" / "dvsim_sitl.params"
FORBIDDEN = set(range(5177, 5195)) | {8000, 8010, 8093}


def skip_reason():
    """None if PX4, the bridge exe, dvsim-replay and mavsdk are all present, else why not."""
    try:
        import mavsdk  # noqa: F401
    except ImportError:
        return "mavsdk not installed (pip install -e '.[px4]')"
    for what, p in (
        ("PX4 binary", PX4_BIN),
        ("bridge exe", BRIDGE),
        ("dvsim-replay", REPLAY),
    ):
        if not p.exists():
            return f"{what} missing: {p}"
    return None


def _busy(proto, ports):
    """Local ports in use, from /proc/net. TCP TIME_WAIT is not a conflict (SO_REUSEADDR)."""
    busy = set()
    for name in (proto, proto + "6"):
        for line in Path(f"/proc/net/{name}").read_text().splitlines()[1:]:
            f = line.split()
            port = int(f[1].rsplit(":", 1)[1], 16)
            if port in ports and not (proto == "tcp" and f[3] == "06"):
                busy.add(port)
    return busy


class _Proc:
    """Subprocess in its own process group; stdout+stderr pumped into a log file and a line queue."""

    def __init__(self, argv, log, cwd=None, env=None, stdin=subprocess.DEVNULL):
        self.q, self.lines, self.name = queue.Queue(), [], Path(argv[0]).name
        self.log = open(log, "w")
        self.p = subprocess.Popen(
            argv, cwd=cwd, env=env, stdin=stdin, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, start_new_session=True,
        )  # fmt: skip
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.p.stdout:
            self.log.write(line)
            self.log.flush()
            self.lines.append(line.rstrip())
            self.q.put(line.rstrip())
        self.q.put(None)

    def wait_line(self, prefix, timeout):
        end = time.monotonic() + timeout
        while (left := end - time.monotonic()) > 0:
            try:
                line = self.q.get(timeout=min(left, 1.0))
            except queue.Empty:
                continue
            if line is None:
                raise RuntimeError(
                    f"{self.name} exited (rc={self.p.poll()}) before '{prefix}'"
                )
            if line.startswith(prefix):
                return line
        raise RuntimeError(
            f"timeout {timeout}s waiting for '{prefix}' from {self.name}"
        )

    def kill(self):
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(self.p.pid, sig)
            except ProcessLookupError:
                break
            try:
                self.p.wait(5)
                break
            except subprocess.TimeoutExpired:
                pass
        self.log.close()


class Stack:
    def __init__(
        self,
        cfg,
        run_dir,
        model=None,
        instance=None,
        speed=None,
        connect_timeout=120,
        params=None,
        ctl=False,
    ):  # params: extra PX4 parameters (name -> value) applied after dvsim_sitl.params; ctl: the bridge also serves st.ctl
        self.params = {
            **(params or {}),
            **json.loads(os.environ.get("DVSIM_PX4_PARAMS", "{}")),
        }  # DVSIM_PX4_PARAMS='{"MC_ROLL_P": 10}': per-run override of the scenario's parameters (tuning sweeps, examples/skydio/px4_montecarlo.py --param)
        n = (
            int(os.environ.get("DVSIM_PX4_INSTANCE", 0))
            if instance is None
            else instance
        )
        self.cfg, self.dir, self.model, self.speed, self.ct = (
            Path(cfg),
            Path(run_dir),
            model,
            float(os.environ.get("DVSIM_PX4_SPEED", 1.0))
            if speed is None
            else speed,  # DVSIM_PX4_SPEED: bridge pacing, sim s per wall s (lockstep, PX4 follows HIL time); 1 = real time, 0 = as fast as possible
            connect_timeout,
        )
        off = (
            14540 + n if n <= 9 else 14549
        )  # px4-rc.mavlink: instances above 9 all send their offboard stream to 14549
        self.tcp, self.udp = [4560 + n], [off, 14550 + n, 14580 + n, 18570 + n]
        self.n, self.bridge, self.px4, self.rc = n, None, None, None
        self.mavlink = f"udp://:{off}"  # a MAVSDK program listens here (PX4 sends its offboard stream to 14540 + n)
        self.trace, self.log = self.dir / "trace.csv", self.dir / "replay.dvsimog"
        # ctl: unix socket onto the bridge's World (dvsim.ctl.RemoteSim); short path (sun_path is 107 bytes), removed by the bridge
        self.ctl = (
            Path(tempfile.mkdtemp(prefix="dvsim-ctl-")) / "ctl.sock" if ctl else None
        )

    def start(self):
        if (why := skip_reason()) is not None:
            raise RuntimeError(why)
        assert not (set(self.tcp + self.udp) & FORBIDDEN), (
            "instance maps onto a forbidden port"
        )
        if _busy("tcp", self.tcp) or _busy("udp", self.udp):
            raise RuntimeError(
                f"ports busy: tcp {_busy('tcp', self.tcp)} udp {_busy('udp', self.udp)}"
            )
        n, run = self.n, self.dir / "px4run"
        run.mkdir(parents=True, exist_ok=True)
        # --sysid: PX4 instance n runs with MAV_SYS_ID n + 1 (rcS) and drops BATTERY_STATUS from any other system id
        argv = [str(BRIDGE), "--config", str(self.cfg), "--port", str(4560 + n), "--px4-udp", str(18570 + n),
                "--udp-bind", str(14550 + n), "--speed", str(self.speed), "--trace", str(self.trace),
                "--log", str(self.log), "--connect-timeout", str(self.ct),
                "--sysid", str(n + 1)]  # fmt: skip
        if self.model:
            argv += ["--model", str(self.model)]
        if self.ctl:
            argv += ["--ctl", str(self.ctl)]
        try:
            self.bridge = _Proc(argv, self.dir / "bridge.log", stdin=subprocess.PIPE)
            self.bridge.wait_line("listening", 30)
            env = {k: v for k, v in os.environ.items() if not k.startswith("PX4_")}
            env["PX4_SYS_AUTOSTART"] = "4902"
            for line in PARAMS.read_text().splitlines():
                if line.strip() and not line.lstrip().startswith("#"):
                    name, val = line.split(None, 1)
                    env[f"PX4_PARAM_{name}"] = val.strip()
            env.update({f"PX4_PARAM_{k}": str(v) for k, v in self.params.items()})
            self.px4 = _Proc([str(PX4_BIN), "-d", "-i", str(n), "-s", "etc/init.d-posix/rcS", str(BUILD / "etc")],
                             self.dir / "px4.log", cwd=run, env=env)  # fmt: skip
            self.bridge.wait_line("connected", 90)
        except BaseException:
            self.stop(final=False)
            raise

    def stop(self, final=True):
        """Quit the bridge (it writes the replay log), kill both. -> {"tick", "state_hash"} or None (final=False / no line)."""
        out = None
        b = self.bridge
        if b and b.p.poll() is None:
            try:
                b.p.stdin.write("quit\n")
                b.p.stdin.flush()
                b.p.stdin.close()
                if final:
                    m = re.fullmatch(
                        r"final tick=(\d+) state_hash=([0-9a-f]+)",
                        b.wait_line("final", 30),
                    )
                    out = {"tick": int(m[1]), "state_hash": m[2]}
                    self.rc = b.p.wait(
                        30
                    )  # the bridge's exit code (0 = clean shutdown)
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired):
                pass  # best effort: kill() below is the guarantee
        for p in (b, self.px4):
            if p:
                p.kill()
        self.bridge = self.px4 = None
        if self.ctl:
            shutil.rmtree(self.ctl.parent, ignore_errors=True)
        return out
