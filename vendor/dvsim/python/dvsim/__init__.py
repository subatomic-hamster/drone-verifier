"""dvsim: thin Python front end over the C++ kernel (`_dvsim`). No simulation logic lives here."""

import json
import os
from os import fspath

import numpy as np

try:
    from . import _dvsim
except ImportError as _e:  # no built kernel: `dvsim.design` / `dvsim design` still work; Sim.load re-raises this
    _dvsim, _dvsim_error = None, _e

__all__ = ["Sim", "render_style"]


def render_style(words: str = "") -> str:
    """The process-wide look of Sim.render and the viewer: comma-separated words added to the current style --
    "outdoor" | "studio" (sky), "edges" | "noedges" (CAD outlines), "shadows" | "noshadows", "wires" | "nowires" (cables as
    tubes), "boldwires" | "plainwires" (cables at least 4 px wide on screen; the data keeps their diameter), "rain=<mm/h>" | "overcast=<0..1>" (weather look
    override; by default the scene's rain device sets it, and rain=0 / no device is the clear look). DVSIM_RENDER_STYLE in the environment is applied on top. Returns the style now in force."""
    return _dvsim.render_style(words)


class Sim:
    """One World. Not thread-safe per instance; separate Sims run in parallel (the GIL is released).
    A Sim loaded with `host=True` keeps its World on a C++ thread (the Host), so it can be used from several Python
    threads at once and shown in the viewer while other threads (e.g. an MCP server) keep calling it."""

    def __init__(self, core: "_dvsim.Sim | _dvsim.Host"):
        self._c = core

    @classmethod
    def load(
        cls,
        path,
        root=None,
        max_sweep_runs: int = 100000,
        max_wall_s: float = 0.0,
        caps=("read", "control", "destructive"),
        max_steps_per_call: int = 2**31 - 1,
        host: bool = False,
    ) -> "Sim":
        """Load MJCF/URDF. `root` confines later LoadModel commands (default: the model's directory).
        Session limits: `max_sweep_runs` (larger sweeps -> limit_exceeded), `max_wall_s` (0 = none; a RunTest or
        sweep running longer fails with limit_exceeded and changes nothing). `caps` is the session's capability
        set, enforced by the C++ bus (a command whose `cap` is missing gets permission_denied); forks inherit it.
        `max_steps_per_call`: a Step with larger n, or a RunTest/Sweep longer than this, gets limit_exceeded.
        `host=True`: the World lives on a Host thread (starts paused, logs from tick 0): Run / Pause commands then
        work (via call()), and show() can run the viewer beside it."""
        if _dvsim is None:
            raise _dvsim_error
        if os.environ.get("DVSIM_ATTACH") and os.path.realpath(path) == os.environ.get(
            "DVSIM_ATTACH_MODEL"
        ):
            from .attach import (
                Remote,
            )  # live run: the scenario's model is already loaded in the viewer's Sim (dvsim.attach)

            return cls(Remote(os.environ["DVSIM_ATTACH"]))
        return cls(
            (_dvsim.Host if host else _dvsim.Sim)(
                fspath(path),
                "" if root is None else fspath(root),
                max_sweep_runs,
                max_wall_s,
                list(caps),
                max_steps_per_call,
            )
        )

    def show(
        self,
        frames: int = 0,
        script: str = "",
        width: int = 1280,
        height: int = 720,
        design=None,
        hud=None,
        clean: bool = False,
    ) -> None:
        """Open the Vulkan viewer on THIS thread and block until the window is closed (or `frames` > 0 frames were
        drawn). The Sim's World runs on a Host thread meanwhile: the UI's pause / run / step / rate and SetAttr
        edits are bus commands (source "viewer"), logged and replayable. Plain Sim: its World is handed to a Host
        for the duration and returned afterwards (state, tick and log carry on). Linux: any thread may call this.
        macOS: GLFW needs the process's MAIN thread (not built there yet). `script` = "FRAME ACTION ARGS; ..." issues
        UI-equivalent commands at given frames (see dvsim-view --script). Raises RuntimeError with the reason if there
        is no display, no Vulkan device, or this build has no viewer (`_dvsim.has_viewer`).
        `design` = a `dvsim.design.viewer.Backend` (JSON request -> JSON reply): turns the Design window into an editor
        whose edits are the Design API's (see `dvsim.design.viewer.view` for the one-call form).
        `hud` = a `dvsim.hud.Hud` whose latest document is drawn over the 3D view (and may drive the camera); `clean=True`
        hides the panes, top and status bars so the view fills the window (see dvsim.hud)."""
        (self._c.view if isinstance(self._c, _dvsim.Host) else self._c.show)(
            int(frames), script, int(width), int(height), design, hud, bool(clean)
        )

    def close_viewer(self) -> None:
        """Ask a show() running on another thread to return (host=True Sims only)."""
        self._c.request_close()

    def step(self, n: int = 1) -> int:
        """Advance n steps through the command bus (logged); returns the tick."""
        return self._c.step(n)

    @property
    def tick(self) -> int:
        return self._c.tick

    @property
    def diverged(self) -> bool:
        return self._c.diverged

    @property
    def caps(self) -> list:
        return self._c.caps

    def call(self, cmd, elevate: bool = False) -> dict:
        """Run one dvsim.Envelope given as dict or JSON text; returns the dvsim.Reply as a dict (never raises
        for command-level failures: check reply["ok"]). `elevate=True` runs this one call with all capabilities
        (the caller has obtained approval); the session itself is unchanged."""
        return json.loads(
            self._c.call(cmd if isinstance(cmd, str) else json.dumps(cmd), elevate)
        )

    def query(self, path: str, attr: str, frame: str = ""):
        """-> (float64 ndarray, unit string). Raises KeyError / ValueError for unknown / invalid requests."""
        return self._c.get(path, attr, frame)

    def set(self, path: str, attr: str, values, frame: str = "") -> dict:
        """SetAttr through the bus (logged). Raises ValueError if the reply is not ok."""
        vals = np.atleast_1d(np.asarray(values, dtype=np.float64)).tolist()
        r = self.call(
            {
                "cmd_type": "SetAttr",
                "cmd": {"path": path, "attr": attr, "frame": frame, "values": vals},
            }
        )
        if not r["ok"]:
            raise ValueError(r["message"])
        return r

    def set_many(self, paths, attrs, values) -> None:
        """Batch of single-value SetAttrs through the bus (logged and replayed exactly like set()); raises on the first failure."""
        self._c.set_many(list(paths), list(attrs), [float(v) for v in values])

    def body_positions(self) -> dict:
        """{prim path: world position (3,)} of every body prim (those that answer a `pos` query)."""
        out = {}
        for p in self.call({"cmd_type": "Query", "cmd": {"path": ""}}).get("prims", []):
            try:
                out[p] = self.query(p, "pos")[0]
            except (KeyError, ValueError):
                pass
        return out

    def orbit_camera(self, preset="orbit", az=45.0, el=25.0, dist=None, target=None):
        """-> (pos, lookat) for an orbit camera around `target` (body prim path, 3-vector, or None = middle of the
        bodies' bounding box). az / el in degrees: az 0 looks from +x towards the target, counter-clockwise seen from above; el up.
        `preset` "iso" | "front" (from -y) | "back" | "left" | "right" | "top" overrides az / el; "orbit" uses them.
        dist None = fit the bodies' bounding sphere (+ 0.08 m for geom extent) in a 45 degree view."""
        presets = {
            "iso": (45, 28),
            "front": (-90, 8),
            "back": (90, 8),
            "left": (180, 8),
            "right": (0, 8),
            "top": (-90, 89),
        }
        if preset != "orbit":
            if preset not in presets:
                raise ValueError(
                    f"unknown camera preset {preset!r} (orbit, {', '.join(presets)})"
                )
            az, el = presets[preset]
        bodies = None
        if target is None or dist is None:
            bodies = np.array(list(self.body_positions().values()) or [[0, 0, 0]])
        if isinstance(target, str):
            target = self.query(target, "pos")[0]
        t = np.asarray(
            (bodies.min(0) + bodies.max(0)) / 2 if target is None else target,
            dtype=float,
        )
        if dist is None:
            r = float(np.linalg.norm(bodies - t, axis=1).max()) + 0.08
            dist = r / np.sin(np.radians(22.5))
        a, e = np.radians(az), np.radians(el)
        return t + dist * np.array(
            [np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]
        ), t

    def render(
        self,
        pos=None,
        lookat=None,
        fovy: float = 45.0,
        width: int = 320,
        height: int = 240,
        *,
        camera=None,
        az: float = 45.0,
        el: float = 25.0,
        dist=None,
        target=None,
    ):
        """Offscreen camera image (Render command, read cap; needs a Vulkan 1.3 device). World frame, +z up; fovy is the
        vertical field of view in degrees. -> (rgba uint8 (height, width, 4), depth float32 (height, width)): depth is
        linear metres along the optical axis, +inf where nothing was hit. Lossless (binary reply, not the JSON edge).
        Raises ValueError for invalid arguments and RuntimeError otherwise (e.g. "solver_error: no Vulkan device ...").
        Images are deterministic per GPU/driver only. Either `pos` + `lookat`, or `camera="orbit"` (with az, el, dist,
        target) / a preset name ("iso", "front", "top", ...): see orbit_camera."""
        if camera is not None:
            pos, lookat = self.orbit_camera(camera, az, el, dist, target)
        elif pos is None or lookat is None:
            raise ValueError("render needs pos and lookat, or camera=")
        return self._c.render(
            np.asarray(pos, dtype=np.float64).tolist(),
            np.asarray(lookat, dtype=np.float64).tolist(),
            float(fovy),
            int(width),
            int(height),
        )

    def raycast(self, origin, directions) -> np.ndarray:
        """Lidar: float64 distances (metres, -1 = miss) from `origin` (3,) along each row of `directions` (n, 3; any
        nonzero length) via the solver's ray casts. CPU, fully deterministic, full double precision."""
        d = np.atleast_2d(np.asarray(directions, dtype=np.float64))
        return self._c.raycast(
            np.asarray(origin, dtype=np.float64).tolist(), d.ravel().tolist()
        )

    def tubes(self) -> list:
        """The cables drawn as tubes (a design's wires: spatial tendons through sites; visual only): a list of
        (radius_m, length_m, rgba, points (n, 3) world positions), following the bodies they are anchored on."""
        return self._c.tubes()

    def run_test(self, spec: dict) -> dict:
        """RunTest (a dvsim.TestSpec as a dict) through the bus; returns the dvsim.TestResult dict with
        `fail_snapshot` as bytes (feed to restore()). Steps this Sim. Raises ValueError if the reply is not ok."""
        r = self.call({"cmd_type": "RunTest", "cmd": {"spec": spec}})
        if not r["ok"]:
            raise ValueError(r["message"])
        res = r["result"]
        res["fail_snapshot"] = bytes(res.get("fail_snapshot", []))
        return res

    def sweep(
        self,
        spec: dict,
        params: list,
        mode: str = "grid",
        n: int = 0,
        seed: int = 0,
        threads: int = 1,
        base: int = 0,
    ) -> list:
        """Sweep command: run `spec` (a TestSpec dict) once per parameter point on clones of the current state (`base` =
        a bus snapshot id instead, 0 = now). params: [{"path","attr","values":[...]}] (mode "grid": cartesian product,
        first param slowest) or [{"path","attr","lo","hi"}] (mode "random": `n` runs, run i seeded by seed+i). Returns one
        dict per run, ordered by index: index, params (dict "path.attr" -> value), pass, diverged, error, monitors.
        This Sim is not modified. The GIL is released while running, so it can be called from a background thread
        (do not use this same Sim from another thread meanwhile; fork() it first). `threads` is capped at the host's core count.
        Raises ValueError (message starts with the error kind, e.g. "limit_exceeded") if the sweep is refused."""
        r = self.call(
            {
                "cmd_type": "Sweep",
                "cmd": {
                    "base": base,
                    "spec": spec,
                    "params": params,
                    "mode": mode,
                    "n": n,
                    "seed": seed,
                    "threads": threads,
                },
            }
        )
        if not r["ok"]:
            raise ValueError(f"{r['error']}: {r['message']}")
        names = r["sweep"].get("params", [])
        return [
            {
                "index": row.get("index", 0),
                "params": dict(zip(names, row.get("params", []))),
                "pass": row.get("pass", False),
                "diverged": row.get("diverged", False),
                "error": row.get("error", ""),
                "monitors": row.get("monitors", []),
            }
            for row in r["sweep"].get("rows", [])
        ]

    def fork(self) -> "Sim":
        """Independent copy of the current state (World::clone); no log, no snapshots."""
        return Sim(self._c.clone())

    def save(self) -> bytes:
        return self._c.save()

    def restore(self, state: bytes) -> None:
        self._c.restore(state)

    def state_hash(self) -> int:
        return self._c.state_hash()

    def start_log(self) -> None:
        """Start the replay log; call on a fresh Sim, before any snapshot."""
        self._c.start_log()

    def log(self):
        """Complete replay-log bytes, or None if logging is off."""
        return self._c.log()
