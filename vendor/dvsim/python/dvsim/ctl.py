"""RemoteSim: the slice of `dvsim.Sim` that FeetechPort and the program-runner monitors use (call SetAttr, query, tick), served over
the dvsim-px4-bridge `--ctl` unix socket, i.e. onto the World the bridge steps in lockstep with PX4.

    rs = RemoteSim("/tmp/dvsim-ctl-x/ctl.sock")
    rs.set("/devices/arm_servo1", "torque_enable", [1])            # a SetAttr command: logged in the bridge's replay log, replayed
    rs.query("/world/payload", "pos", frame="/world/bottom")        # -> (ndarray, "")  (frame = a body prim path, as Query)
    rs.query_many("/devices/arm_servo1", ["counts", "alarm"])       # -> ndarray, one round trip (FeetechPort uses it if present)
    rs.tick                                                         # the World's tick

Thread-safe (one request in flight at a time). The wire protocol is documented in src/px4/main.cpp (Bridge::on_ctl).
"""

import socket
import threading
import time

import numpy as np


class RemoteSim:
    def __init__(self, path, timeout=60.0):
        self._lock = threading.Lock()
        self._s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        end = time.monotonic() + timeout
        while True:  # the bridge creates the socket before it prints "listening"; retry only for a late caller
            try:
                self._s.connect(str(path))
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() > end:
                    raise
                time.sleep(0.05)
        self._buf = b""

    def _rpc(self, line):
        with self._lock:
            self._s.sendall(line.encode() + b"\n")
            while b"\n" not in self._buf:
                d = self._s.recv(65536)
                if not d:
                    raise ConnectionError("dvsim-px4-bridge closed the ctl socket")
                self._buf += d
            reply, _, self._buf = self._buf.partition(b"\n")
        t = reply.decode().split()
        if t[:1] != ["ok"]:
            raise ValueError(" ".join(t[1:]) or "ctl error")
        return t[1:]

    def close(self):
        self._s.close()

    @property
    def tick(self) -> int:
        return int(self._rpc("tick")[0])

    def call(self, cmd) -> dict:
        """SetAttr envelopes only ({"source", "cmd_type": "SetAttr", "cmd": {path, attr, values}}); -> {"ok", "message"}."""
        if cmd.get("cmd_type") != "SetAttr":
            raise ValueError(f"ctl serves SetAttr only, not {cmd.get('cmd_type')}")
        c = cmd["cmd"]
        vals = " ".join(repr(float(v)) for v in c["values"])
        try:
            self._rpc(
                f"set {cmd.get('source') or 'ctl'} {c['path']} {c['attr']} {vals}"
            )
        except ValueError as e:
            return {"ok": False, "message": str(e)}
        return {"ok": True, "message": ""}

    def set(self, path, attr, values, frame="") -> dict:
        r = self.call(
            {
                "cmd_type": "SetAttr",
                "cmd": {
                    "path": path,
                    "attr": attr,
                    "values": list(np.atleast_1d(values)),
                },
            }
        )
        if not r["ok"]:
            raise ValueError(r["message"])
        return r

    def get(self, path, attr, frame=""):
        """-> (tick the value belongs to, float64 ndarray)."""
        t = self._rpc(f"get {path} {attr} {frame}".rstrip())
        return int(t[0]), np.array([float(x) for x in t[1:]])

    def query(self, path, attr, frame=""):
        return self.get(path, attr, frame)[1], ""

    def imu(self, since=0):
        """The IMU samples the bridge sent to PX4 after world tick `since` -> (current tick, ndarray (n, 7): counter, gyro xyz, accel xyz, device
        FLU frame). The bridge keeps 16 s; a companion computer's estimator calls this each cycle with the last counter it has."""
        t = self._rpc(f"imu {int(since)}")
        return int(t[0]), np.array([float(x) for x in t[2:]]).reshape(int(t[1]), 7)

    def query_many(self, path, attrs):
        return np.array(
            [float(x) for x in self._rpc(f"mget {path} {' '.join(attrs)}")[1:]]
        )
