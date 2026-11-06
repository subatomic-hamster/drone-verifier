"""Re-fly a finished run's saved design and controller in a recorded GUI session, with no agent. Started by harness/demo.py refly.

  python -m dv.refly --warm --design D                   routes the wires into DVSIM_ROUTE_CACHE, exits (224 s per geometry, then a lookup)
  python -m dv.refly --session R --port P --model M --trial T --feedback F --design D --code C --scenario wind

The flight is judged as ever on the chord-wire scene the verifier flies (tubes are visual-only tendons); the window replays it on the same scene
with routed tubes. Waits for `refly.go` in its run dir, so two sessions start together; leaves `refly.done` beside it after the verdict. Never calls
finish: no results.json, no score.
"""

import argparse
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path

DEVICES = re.compile(r"<text name=\"dvsim.devices\" data='.*?'\s*/>", re.DOTALL)


def routed(design, visual=False, out=None):
    from dvsim.design import compute

    lib = design.lib
    return compute(design.asm, lib.dir, out or lib.root, lib, visual=visual, route=True)[0]  # fmt: skip


def warm(path):
    """The two routes a session asks for: the window's build (cosmetic meshes count as obstacles) and the flight scene's."""
    from dv import design as D

    d = D.apply(json.loads(Path(path).read_text())).design
    routed(d, visual=True, out=tempfile.mkdtemp(prefix="dv-route-"))
    routed(d)


def routed_scene(design, flown):
    """The flown scene as the window should draw it: built with routed tubes, this flight's devices (seeds, wind, rain) put back."""
    xml = re.sub(r'file="(?!/)', f'file="{design.lib.root}/', routed(design))
    return DEVICES.sub(lambda _: DEVICES.search(flown)[0], xml, count=1)


def script(sess, ui, choices, code, scenario):
    from dv import verify

    d = sess.run_dir
    sess.ev.emit(
        "config", replays_all=True
    )  # the page holds each verdict for its picture
    while not (d / "refly.go").exists():
        time.sleep(0.1)

    def fly(scenario):
        # the same refusals as Session.fly: a design stopped at L0 or L1 has no flight to show
        if sess.built.missing:
            return dict(ok=False, error=f"choose parts for: {', '.join(sess.built.missing)}")  # fmt: skip
        bad = verify.layers(sess.built)
        if bad:
            return dict(ok=False, error=f"{bad['gate']} fails, not flown: {bad['reason']}")  # fmt: skip
        if not sess.code:
            return dict(ok=False, error="no controller")
        sess._ui("page", "run")
        seed = verify.seeds_for(verify.TASK["public"]["seed"])[scenario]
        log = d / "flight_last" / "replay.dvsimog"
        log.parent.mkdir(parents=True, exist_ok=True)
        spec = verify.make_spec(sess.built, sess.code, scenario, seed, log=str(log))
        r = verify.fly(spec, sess.built, sess.ev.emit)
        verify.compress_log(spec)
        scene = log.parent / "scene_routed.xml"
        scene.write_text(
            routed_scene(sess.built.design, (log.parent / "scene.xml").read_text())
        )
        ui.replayed = None
        sess._ui(
            "fly_live",
            scenario,
            dict(
                run_dir=str(log.parent),
                model=str(scene),
                passed=r["pass"],
                reason=r["reason"],
            ),
        )
        end = time.time() + r["metrics"].get("t_end_s", 60) * 2 + 30
        while ui.replayed is None and time.time() < end:
            time.sleep(0.2)
        native = json.loads((log.parent / "native.json").read_text())
        print("replay hash", ui.replayed, "flight", native["state_hash"], flush=True)
        return dict(
            ok=r["pass"],
            summary=f"{scenario} seed {seed}: {'pass' if r['pass'] else 'fail'}: {r['reason']}",
        )

    try:
        sess.call("design_set", dict(choices=choices), sess.design_set)
        time.sleep(5)
        sess.call("write_controller", dict(code=code), sess.write_controller)
        time.sleep(8)  # typed in about 5 s
        sess.call("fly", dict(scenario=scenario), fly)
        time.sleep(8)  # the verdict holds, in the recording too
    finally:
        (d / "refly.done").write_text("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", action="store_true")
    ap.add_argument("--design", required=True)
    ap.add_argument("--code")
    ap.add_argument("--session")
    ap.add_argument("--port", type=int)
    ap.add_argument("--model")
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--feedback", default="whatif")
    ap.add_argument("--scenario", default="wind")
    a = ap.parse_args()
    if a.warm:
        t = time.time()
        warm(a.design)
        print(f"routes ready in {time.time() - t:.0f} s", flush=True)
        return

    import uvicorn

    from dv.gui import ViewerUI
    from dv.server import build_app
    from dv.session import Session

    os.environ["DV_ROUTE_WIRES"] = "1"
    sess = Session(a.session, a.model, a.trial, feedback=a.feedback)
    ui = ViewerUI(sess.run_dir, port=a.port, design=sess.built.design)
    sess.ui = ui
    uv = uvicorn.Server(
        uvicorn.Config(
            build_app(sess), host="127.0.0.1", port=a.port, log_level="warning"
        )
    )
    threading.Thread(target=uv.run, daemon=True).start()
    threading.Thread(
        target=script,
        args=(
            sess,
            ui,
            json.loads(Path(a.design).read_text()),
            Path(a.code).read_text(),
            a.scenario,
        ),
        daemon=True,
    ).start()
    ui.run()  # main thread: the window; the session stays up until demo.py closes it
    uv.should_exit = True


if __name__ == "__main__":
    main()
