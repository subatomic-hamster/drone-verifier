"""Laptop bridge to hfbox sessions without port forwarding (sshd forbids it).

  bridge.py --port 8801[,8802,...] [--session A]

Serves 127.0.0.1:PORT with exactly these routes: /mcp (GET POST DELETE), /state, /events (SSE), /events.jsonl, /agent-event (POST), /stream,
/bridge-state. Requests travel over one long-lived `ssh hfbox python3 dv/pipe.py PORT` (stdin/stdout, framed JSON, multiplexed by id), so a
slow fly/verify never blocks /state. /stream is a separate ssh exec of the GUI's `dv.gui_stream` (fragmented MP4), one upstream, fan-out.
"""

import argparse
import base64
import json
import os
import queue
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = os.environ.get("DV_HOST", "hfbox")
SSH = [
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
]
ENV = (
    "export OMP_NUM_THREADS=1 PYTHONPATH=/workspace/wt/dv/src:/workspace/wt/dv-dvsim/src/examples/hopper DV_RUNS=/workspace/wt/dv/runs "
    "DVSIM_ROOT=/workspace/wt/dv-dvsim/src PATH=/workspace/wt/dv-dvsim/src/.venv/bin:$PATH; cd /workspace/wt/dv/src; "
)
PIPE_CMD = "python3 /workspace/wt/dv/src/dv/pipe.py {port}"
STREAM_CMD = (
    ENV + "python -m dv.gui_stream --port {port}"
)  # GUI provides: fragmented MP4 (H.264) on stdout
ROUTES = {
    "/mcp": {"GET", "POST", "DELETE"},
    "/state": {"GET"},
    "/events": {"GET"},
    "/events.jsonl": {"GET"},
    "/agent-event": {"POST"},
}
FWD_H = (
    "content-type",
    "accept",
    "mcp-session-id",
    "mcp-protocol-version",
    "last-event-id",
)
CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "*",
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
    "Access-Control-Expose-Headers": "mcp-session-id",
}


class Link:
    """One ssh exec running pipe.py for a port; reconnects with backoff. request() returns a queue of frames."""

    def __init__(self, port):
        self.port, self.up, self.err, self.reconnects, self.since = (
            port,
            False,
            "starting",
            0,
            None,
        )
        self.proc, self.pending, self.n, self.wlock, self.plock = (
            None,
            {},
            0,
            threading.Lock(),
            threading.Lock(),
        )
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        back = 1
        while True:
            t0 = time.time()
            try:
                self.proc = p = subprocess.Popen(
                    SSH + [PIPE_CMD.format(port=self.port)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                threading.Thread(
                    target=lambda: setattr(
                        self, "err", p.stderr.read().decode()[-200:] or self.err
                    ),
                    daemon=True,
                ).start()
                for line in p.stdout:
                    f = json.loads(line)
                    if f.get("hello"):
                        self.up, self.since = True, time.time()
                    elif f["id"] in self.pending:
                        self.pending[f["id"]].put(f)
            except Exception as e:
                self.err = str(e)
            self.up = False
            self.reconnects += 1
            with self.plock:
                for q in self.pending.values():
                    q.put({"error": "link down"})
            if time.time() - t0 > 30:
                back = 1
            time.sleep(back)
            back = min(back * 2, 15)

    def request(self, method, path, headers, body):
        if not self.up:
            return None, None
        with self.plock:
            self.n += 1
            i = self.n
            q = self.pending[i] = queue.Queue()
        self.send(
            {
                "id": i,
                "method": method,
                "path": path,
                "headers": headers,
                "body": base64.b64encode(body).decode(),
            }
        )
        return i, q

    def send(self, f):
        try:
            with self.wlock:
                self.proc.stdin.write(json.dumps(f).encode() + b"\n")
                self.proc.stdin.flush()
        except (OSError, ValueError, AttributeError):
            pass

    def done(self, i):
        self.send({"id": i, "cancel": True})
        self.pending.pop(i, None)


class Stream:
    """Fragmented-MP4 fan-out: one ssh upstream while viewers exist; new viewers get the init segment (boxes before the first moof) then live fragments."""

    def __init__(self, port):
        self.port, self.viewers, self.init, self.lock, self.running = (
            port,
            set(),
            b"",
            threading.Lock(),
            False,
        )

    def subscribe(self):
        q = queue.Queue(
            maxsize=20
        )  # a viewer that cannot keep up is dropped, not buffered
        with self.lock:
            self.viewers.add(q)
            if not self.running:
                self.running = True
                threading.Thread(target=self.pump, daemon=True).start()
        return q

    def unsubscribe(self, q):
        with self.lock:
            self.viewers.discard(q)

    def pump(self):
        while self.viewers:
            p = subprocess.Popen(
                SSH + [STREAM_CMD.format(port=self.port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            last = [time.time()]

            def watchdog(p=p, last=last):  # stall -> kill, the loop restarts it
                while p.poll() is None:
                    if time.time() - last[0] > 15:
                        p.kill()
                    time.sleep(2)

            threading.Thread(target=watchdog, daemon=True).start()
            self.init, frag, header = b"", b"", True
            while True:
                h = p.stdout.read(8)
                if len(h) < 8:
                    break
                size, typ = int.from_bytes(h[:4], "big"), h[4:8]
                body = p.stdout.read(size - 8) if size > 8 else b""
                last[0] = time.time()
                box = h + body
                if typ == b"moof":
                    header = False
                if header:
                    self.init += box
                    continue
                if typ == b"moof" and frag:
                    self.emit(frag)
                    frag = b""
                frag += box
            p.kill()
            time.sleep(2)
        self.running = False

    def emit(self, frag):
        for q in list(self.viewers):
            try:
                q.put_nowait(frag)
            except queue.Full:
                self.viewers.discard(
                    q
                )  # its handler times out on the empty queue and closes


def make_handler(links, session):
    streams = {p: Stream(p) for p in links}
    port_of = lambda h: h.server.server_address[1]

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"  # close-delimited bodies: simplest for SSE and streamed MCP replies

        def log_message(self, *a):
            pass

        def reply(self, code, body=b"", ctype="application/json"):
            self.send_response(code)
            for k, v in CORS.items():
                self.send_header(k, v)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.reply(204)

        def serve(self, method):
            path = self.path
            bare = path.split("?")[0]
            port = port_of(self)
            if bare == "/bridge-state" and method == "GET":
                return self.reply(
                    200,
                    json.dumps(
                        {
                            "session": session,
                            "port": port,
                            "up": links[port].up,
                            "since": links[port].since,
                            "reconnects": links[port].reconnects,
                            "last_error": links[port].err,
                        }
                    ).encode(),
                )
            if bare == "/stream" and method == "GET":
                return self.stream(streams[port])
            if method not in ROUTES.get(bare, ()):
                return self.reply(404, b'{"error":"route not allowed"}')
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            i, q = links[port].request(
                method,
                path,
                {k: v for k, v in self.headers.items() if k.lower() in FWD_H},
                body,
            )
            if q is None:
                return self.reply(503, b'{"error":"link down"}')
            try:
                f = q.get(timeout=3600)
                if "status" not in f:
                    return self.reply(
                        502, json.dumps({"error": f.get("error")}).encode()
                    )
                self.send_response(f["status"])
                for k, v in CORS.items():
                    self.send_header(k, v)
                for k, v in f["headers"].items():
                    self.send_header(k, v)
                self.end_headers()
                while True:
                    f = q.get(timeout=3600)
                    if "chunk" in f:
                        self.wfile.write(base64.b64decode(f["chunk"]))
                        self.wfile.flush()
                    else:
                        break
            except (BrokenPipeError, ConnectionResetError, queue.Empty):
                pass
            finally:
                links[port].done(i)

        def stream(self, st):
            q = st.subscribe()
            try:
                self.send_response(200)
                for k, v in CORS.items():
                    self.send_header(k, v)
                self.send_header("Content-Type", "video/mp4")
                self.end_headers()
                while not st.init and st.running:
                    time.sleep(0.2)
                self.wfile.write(st.init)
                while True:
                    self.wfile.write(q.get(timeout=20))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, queue.Empty, OSError):
                pass
            finally:
                st.unsubscribe(q)

        do_GET = lambda self: self.serve("GET")
        do_POST = lambda self: self.serve("POST")
        do_DELETE = lambda self: self.serve("DELETE")

    return H


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--port", required=True)
    a.add_argument("--session", default="")
    a = a.parse_args()
    links = {}
    servers = []
    for p in map(int, a.port.split(",")):
        links[p] = Link(p)
    for p in links:
        s = ThreadingHTTPServer(("127.0.0.1", p), make_handler(links, a.session))
        s.daemon_threads = True
        threading.Thread(target=s.serve_forever, daemon=True).start()
        servers.append(s)
    print(f"bridge up on {list(links)}", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
