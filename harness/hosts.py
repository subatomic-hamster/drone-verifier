"""Where the sessions run: DV_HOST (default `hfbox`) names an ssh host; DV_HOST=local runs the same commands as local subprocesses.
Paths: DV_DIR (default /workspace/wt/dv on a host, <repo>/.dv locally); DVSIM_ROOT (default: the hfbox build on a host, <repo>/vendor/dvsim locally).
scripts/hf.sh uses the same three variables. All of harness/ goes through argv()/ENV/RUNS/SRC, nothing else knows the host."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOST = os.environ.get("DV_HOST", "hfbox")
LOCAL = HOST == "local"
DIR = os.environ.get("DV_DIR") or (str(ROOT / ".dv") if LOCAL else "/workspace/wt/dv")
SRC = str(ROOT) if LOCAL else f"{DIR}/src"
# DV_RUNS and DVSIM_ROOT only count locally: on a host they would be the laptop's paths. Same rule as scripts/hf.sh.
RUNS = (os.environ.get("DV_RUNS") if LOCAL else None) or f"{DIR}/runs"
DVSIM = (os.environ.get("DVSIM_ROOT") if LOCAL else None) or (
    str(ROOT / "vendor" / "dvsim") if LOCAL else "/workspace/wt/dv-dvsim/src"
)
# environment of every session / flight command; same string hf.sh builds
ENV = f"export OMP_NUM_THREADS=1 PYTHONPATH={SRC}:{DVSIM}/examples/hopper DV_RUNS={RUNS} DVSIM_ROOT={DVSIM} PATH={DVSIM}/.venv/bin:$PATH; cd {SRC}; "


def argv(cmd, tty=False):
    """argv that runs shell command `cmd` on the session host."""
    if LOCAL:
        return ["bash", "-c", cmd]
    return [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=3",
        HOST,
        cmd,
    ]


def run(cmd, input=None, timeout=300):
    return subprocess.run(
        argv(cmd), input=input, capture_output=True, text=True, timeout=timeout
    )


def bridge(ports):
    """Popen of harness/bridge.py for these ports (sshd on hfbox forbids -L). None when local: the sessions already listen on 127.0.0.1."""
    if LOCAL:
        return None
    return subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "harness" / "bridge.py"),
            "--port",
            ",".join(map(str, ports)),
        ]
    )


def cat(path):
    r = run(f"cat {path}")
    return r.stdout if r.returncode == 0 else None
