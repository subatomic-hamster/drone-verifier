"""Live attach: a program process drives a Sim that lives in ANOTHER process (the viewer's, or any host=True Sim) over a unix socket.

`dvsim program` normally gives the program its own World (it calls `Sim.load(DVSIM_MODEL)`). With `serve(sim, path, ...)` running in the
parent and DVSIM_ATTACH=<socket> / DVSIM_ATTACH_MODEL=<model path> in the program's environment, `Sim.load` of that model returns a Sim
whose core is a `Remote` proxy instead: every call (step, query, set, call, log, ...) runs on the parent's Sim, so the viewer showing that
World draws the program's flight as it happens, and the commands land in the parent's replay log like any other client's.

Trust: the socket is mode 0600 in the run dir and the peer is the user's own program (which already runs with the user's permissions);
messages are pickles, so this is not a boundary against that program, only a local transport.

Pacing: `step` returns once the sim clock is not ahead of the wall clock x rate (rate 0 = as fast as possible), so a live run plays
in real time. A program that computes slower than real time just runs slower; no tick is skipped.
"""

import os
import pickle
import socket
import struct
import threading
import time


def _send(s, obj):
    b = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    s.sendall(struct.pack("<I", len(b)) + b)


def _recv(s):
    h = b""
    while len(h) < 4:
        c = s.recv(4 - len(h))
        if not c:
            raise EOFError
        h += c
    n, buf = struct.unpack("<I", h)[0], bytearray()
    while len(buf) < n:
        c = s.recv(min(n - len(buf), 1 << 20))
        if not c:
            raise EOFError
        buf += c
    return pickle.loads(bytes(buf))


class Server:
    """Serves `sim` (a host=True dvsim.Sim) at `path`. `dt`: seconds per tick (pacing), `rate`: sim seconds per wall second (0 = unpaced)."""

    def __init__(self, sim, path, dt, rate):
        self.sim, self.path, self.dt, self.rate = sim, str(path), dt, float(rate)
        self._t0 = None  # (wall, tick) at the first step
        self.sock = socket.socket(socket.AF_UNIX)
        old = os.umask(0o177)
        try:
            self.sock.bind(self.path)
        finally:
            os.umask(old)
        self.sock.listen(4)
        self.conns, self.closed = [], False
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while not self.closed:
            try:
                c, _ = self.sock.accept()
            except OSError:
                return
            self.conns.append(c)
            threading.Thread(target=self._serve, args=(c,), daemon=True).start()

    def _serve(self, c):
        core = self.sim._c
        try:
            while True:
                op, name, a, kw = _recv(c)
                try:
                    if name == "start_log":  # the served Host logs from tick 0 already
                        r = ("method", None) if op == "attr" else ("ok", None)
                    elif op == "attr":
                        v = getattr(core, name)
                        r = ("method", None) if callable(v) else ("val", v)
                    else:
                        r = ("ok", getattr(core, name)(*a, **kw))
                        if name == "step":
                            self._pace()
                except (
                    Exception
                ) as e:  # the program sees it as an exception from the call
                    r = (
                        "err",
                        e,
                    )  # raised again in the program with its own type (KeyError, ValueError, ...)
                try:
                    _send(c, r)
                except (
                    Exception
                ) as e:  # unpicklable result / exception: nothing was sent yet
                    _send(
                        c,
                        (
                            "err",
                            RuntimeError(f"{name}(): cannot send {r[0]} result: {e}"),
                        ),
                    )
        except (EOFError, OSError):
            pass
        finally:
            c.close()

    def _pace(self):
        if self.rate <= 0:
            return
        tick = self.sim.tick
        if self._t0 is None:
            self._t0 = (time.monotonic(), tick)
        w0, k0 = self._t0
        late = w0 + (tick - k0) * self.dt / self.rate - time.monotonic()
        if late > 0:
            time.sleep(late)

    def close(self):
        self.closed = True
        self.sock.close()
        for c in self.conns:
            try:
                c.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        try:
            os.unlink(self.path)
        except OSError:
            pass


class Remote:
    """The `_c` of a Sim whose World is on the other end of the socket: any attribute is looked up on the served Sim's core."""

    def __init__(self, path):
        self._s = socket.socket(socket.AF_UNIX)
        self._s.connect(path)
        self._lock = threading.Lock()
        self._kind = {}

    def _rpc(self, op, name, a=(), kw=None):
        with self._lock:
            _send(self._s, (op, name, a, kw or {}))
            k, v = _recv(self._s)
        if k == "err":
            raise v
        return k, v

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        if name not in self._kind:
            k, v = self._rpc("attr", name)
            if k == "val":
                return v  # a property: read every time, not cached
            self._kind[name] = k
        return lambda *a, **kw: self._rpc("call", name, a, kw)[1]
