"""Library thumbnails on demand: the viewer's LIBRARY / sidebar show build/thumbs/<part>.png (tools/thumbs.py, needs a Vulkan device).
When some are missing the Backend starts that script once, in the background (`--missing`: it renders only what the cache lacks,
the parts of the open design first), and counts the finished PNGs; `rev` changes as they land, so the window re-asks the library
and the placeholders turn into pictures. The cache (build/thumbs) survives restarts; thumbs.py bumps its version stamp when the
look changes. DVSIM_THUMBS=0 disables the job (no GPU)."""

import json
import os
import subprocess
import sys
import threading
import time

from .uilib import thumbs_dir

VERSION = 2  # keep equal to tools/thumbs.py VERSION


class ThumbJob:
    def __init__(self, get_lib, get_priority=list):
        self.get_lib, self.get_priority = get_lib, get_priority
        self.proc, self.out = None, None
        self.settled = False  # start() has looked at a library (and started the job or found nothing missing)
        self._lock = threading.Lock()

    @staticmethod
    def missing(lib) -> list:
        """Part ids the cache has no entry for (a part with nothing to draw has an index entry with thumb null)."""
        d = thumbs_dir(lib)
        try:
            idx = json.loads((d / "index.json").read_text())
        except (OSError, ValueError):
            idx = {}
        if idx.get("_v") != VERSION:
            idx = {}
        return [p for p in lib.parts if p not in idx and not (d / f"{p}.png").is_file()]

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> bool:
        """Start the background render if thumbnails are missing and nothing runs yet. True when a job is (now) running."""
        with self._lock:
            if self.running():
                return True
            if os.environ.get("DVSIM_THUMBS") == "0":
                return False
            try:
                lib = self.get_lib()
            except Exception:  # no design open yet (the MCP window before design_open)
                return False
            script = lib.root / "tools" / "thumbs.py"
            if not script.is_file() or not self.missing(lib):
                self.settled = True
                return False
            out = thumbs_dir(lib)
            out.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(script),
                "--missing",
                "--quiet",
                "--out",
                str(out),
                "--parts",
                str(lib.dir),
            ]
            pri = [p for p in self.get_priority() if p in lib.parts]
            if pri:
                cmd += ["--priority", ",".join(pri)]
            self.settled = True
            log = open(out / "job.log", "w")
            self.proc = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                cwd=lib.root,
            )
            self.out = out
            return True

    @property
    def rev(self) -> int:
        """Changes while thumbnails land (PNG count) and once more when the job ends (the index is final); 0 before any job."""
        if self.proc is None:
            return 0
        return sum(1 for _ in self.out.glob("*.png")) + (0 if self.running() else 1)

    def wait(self, timeout: float = 900.0) -> bool:
        """Block until the job (if any) has finished; True when nothing is missing afterwards."""
        t0 = time.time()
        while self.running() and time.time() - t0 < timeout:
            time.sleep(0.5)
        return not self.missing(self.get_lib())
