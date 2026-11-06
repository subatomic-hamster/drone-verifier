"""Screenshot of the viewer's Design window after a scripted editing session on the X500 (docs/showcase/design_window.png).

    xvfb-run -a -s "-screen 0 1280x720x24 -fbdir DIR" python tools/design_shot.py OUT.png DIR

Off-screen by default (no Xvfb needed, renders on the GPU; grab() below). With DVSIM_OFFSCREEN=0, Xvfb's -fbdir keeps the
screen as an XWD file and this reads it while the window is up.
"""

import os
import struct
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from dvsim import Sim, _dvsim
from dvsim.design import Design
from dvsim.design.viewer import Backend

REPO = Path(__file__).resolve().parents[1]
# The viewer renders off-screen on the best Vulkan device (the GPU) and every capture tool reads its frames with grab();
# DVSIM_OFFSCREEN=0 goes back to the Xvfb window + XWD read (llvmpipe), which is also what a build without viewer_frame gets.
if hasattr(_dvsim, "viewer_frame"):
    os.environ.setdefault("DVSIM_OFFSCREEN", "1")
else:
    os.environ["DVSIM_OFFSCREEN"] = "0"
# remove the PDB (new DRC findings), browse parts for its seat, snap one back (4 placements), preview the third
SCRIPT = "2 dremove pdb; 4 dsuggest bottom.pdb pcb; 6 dsnap holybro-pdb-60a bottom.pdb pdb; 8 dcycle 2; 12 dfinding 4"
EXPECT = (
    4  # design requests the script makes (the picture is taken after the last answer)
)


def xwd_rgba(path):
    b = Path(path).read_bytes()
    h = struct.unpack(">25I", b[:100])
    hdr, depth, w, hgt, bpl, ncol = h[0], h[3], h[4], h[5], h[12], h[19]
    assert h[11] == 32 and h[7] == 0, "expects 32 bpp little-endian Xvfb output"
    px = np.frombuffer(b, np.uint8, offset=hdr + ncol * 12, count=hgt * bpl).reshape(
        hgt, bpl
    )
    bgrx = px[:, : w * 4].reshape(hgt, w, 4)
    return np.ascontiguousarray(
        np.dstack(
            [
                bgrx[:, :, 2],
                bgrx[:, :, 1],
                bgrx[:, :, 0],
                np.full((hgt, w), 255, np.uint8),
            ]
        )
    )


def grab(fbdir):
    """The viewer's picture right now, (h, w, 4) uint8 RGBA. Off-screen (default): the newest frame the viewer rendered, no
    Xvfb involved; raises while there is none yet, like an unreadable XWD does. DVSIM_OFFSCREEN=0: the Xvfb screen kept in
    FBDIR (Xvfb's -fbdir)."""
    if os.environ.get("DVSIM_OFFSCREEN") != "0":
        f = _dvsim.viewer_frame()
        if f is None:
            raise RuntimeError("the off-screen viewer has not drawn a frame yet")
        return f[1]
    return xwd_rgba(Path(fbdir) / "Xvfb_screen0")


def watchdog(sim, seconds=240):
    """Every headless run must end: a helper thread that raised (or waited on a reply that never came) used to leave
    sim.show() blocking forever (the 2 h hang). After SECONDS the viewer is asked to close; 15 s later the process dies."""

    def run():
        time.sleep(seconds)
        print(f"watchdog: {seconds}s budget spent, closing the viewer", flush=True)
        sim.close_viewer()
        time.sleep(15)
        print("watchdog: viewer did not close, exiting", flush=True)
        os._exit(3)

    threading.Thread(target=run, daemon=True).start()


def is_edit(tool):
    """A design request the script made (not the window's own housekeeping: rev / state polls, ui_poll, library browsing)."""
    return tool not in (
        "rev",
        "state",
        "ui_poll",
        "select",
        "design_text",
    ) and not tool.startswith("library.")


def main(out, fbdir):
    d = Design.open(REPO / "assemblies" / "x500v2.toml")
    b = Backend(lambda: d, tempfile.mkdtemp(prefix="dvsim-shot-"))
    b.auto_thumbs = (
        True  # real library thumbnails (background job, cached in build/thumbs)
    )
    sim = Sim.load(b.state(d)["xml"], host=True)
    shot = []

    def shoot():
        try:
            t0 = time.time()
            while (
                time.time() - t0 < 120
                and sum(is_edit(q["tool"]) for q, _ in b.log) < EXPECT
            ):
                time.sleep(0.2)
            time.sleep(2.5)  # reload + a few frames
            shot.append(grab(fbdir))
        finally:
            sim.close_viewer()  # also when the grab raised: otherwise show() never returns

    threading.Thread(target=shoot, daemon=True).start()
    watchdog(sim)
    sim.show(script=SCRIPT, width=1280, height=720, design=b)
    if not shot:
        raise SystemExit("no screenshot was taken")
    img = shot[0]
    Path(out).write_bytes(_dvsim.encode_png(img))
    print(
        f"wrote {out}: {img.shape[1]}x{img.shape[0]}, {Path(out).stat().st_size} bytes; requests {[q['tool'] for q, _ in b.log if is_edit(q['tool'])]}"
    )


if __name__ == "__main__":
    main(*sys.argv[1:3])
