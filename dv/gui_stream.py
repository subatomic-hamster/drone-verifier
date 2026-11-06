"""python -m dv.gui_stream --port PORT: the session's fMP4 video (init segment, then fragments from a key frame) on stdout, for the laptop bridge's ssh exec. Holds no encoder:
the session owns that. Waits up to --wait s for the session to come up. Exits when stdout closes (ssh gone: EPIPE), or when the session ends."""

import argparse
import signal
import socket
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--wait", type=float, default=300)
    a = ap.parse_args()
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # stdout closed: die at once
    path = f"/tmp/dv-gui-{a.port}.sock"  # dv.gui.sock_path; dv.gui imports dvsim, this stays light
    t0 = time.monotonic()
    while True:
        c = socket.socket(socket.AF_UNIX)
        try:
            c.connect(path)
            break
        except OSError:
            c.close()
            if time.monotonic() - t0 > a.wait:
                sys.exit(f"no GUI session behind {path} after {a.wait:.0f} s")
            time.sleep(1)
    out = sys.stdout.buffer
    while b := c.recv(1 << 16):
        out.write(b)
        out.flush()


if __name__ == "__main__":
    main()
