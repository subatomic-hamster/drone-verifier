"""Stand-in for dv.server for harness tests (stdlib only). Same HTTP surface: /mcp (JSON-RPC), /events, /events.jsonl, /agent-event, /state.
Run: python fake_server.py PORT [RUNS_DIR]. Tool `finish` writes results.json under RUNS_DIR/<session>-<model>-<trial>."""

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOOLS = [
    "task_info",
    "parts_search",
    "design_set",
    "design_report",
    "write_controller",
    "fly",
    "verify",
    "finish",
]
EV, LOCK, STATE = (
    [],
    threading.Lock(),
    {"session": "A", "model": "fake", "finished": False},
)


def emit(kind, **kw):
    with LOCK:
        EV.append(
            {
                "seq": len(EV),
                "t": time.time(),
                "session": STATE["session"],
                "kind": kind,
                **kw,
            }
        )


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/state":
            self._send(
                200,
                {
                    **STATE,
                    "events": len(EV),
                    "score": 0,
                    "budget": {"verify_left": 12, "wall_left_s": 1500},
                },
            )
        elif self.path == "/events.jsonl":
            self._send(
                200,
                "".join(json.dumps(e) + "\n" for e in EV).encode(),
                "application/x-ndjson",
            )
        elif self.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            i = 0
            try:
                while True:
                    while i < len(EV):
                        self.wfile.write(f"data: {json.dumps(EV[i])}\n\n".encode())
                        i += 1
                    self.wfile.flush()
                    time.sleep(0.2)
            except OSError:
                pass
        else:
            self._send(404, {"error": "no"})

    def do_POST(self):
        body = json.loads(
            self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}"
        )
        if self.path == "/agent-event":
            kind = "agent_usage" if body.get("kind") == "agent_usage" else "agent_text"
            body.pop("kind", None)
            emit(kind, **body)
            return self._send(200, {"ok": True})
        if self.path != "/mcp":
            return self._send(404, {})
        m, i = body.get("method"), body.get("id")
        if i is None:
            return self._send(202, b"", "text/plain")
        if m == "initialize":
            r = {
                "protocolVersion": body["params"].get("protocolVersion", "2025-03-26"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "dv", "version": "fake"},
            }
        elif m == "tools/list":
            r = {
                "tools": [
                    {
                        "name": t,
                        "description": f"fake {t}",
                        "inputSchema": {"type": "object", "properties": {}},
                    }
                    for t in TOOLS
                ]
            }
        elif m == "tools/call":
            t = body["params"]["name"]
            emit("tool_call", tool=t, args=body["params"].get("arguments"))
            if t == "finish":
                STATE["finished"] = True
                d = os.path.join(RUNS, f"{STATE['session']}-{STATE['model']}-0")
                os.makedirs(d, exist_ok=True)
                json.dump(
                    {
                        "model": STATE["model"],
                        "trial": 0,
                        "score": 42,
                        "gates": [],
                        "verify_calls": 1,
                        "flights": 1,
                    },
                    open(d + "/results.json", "w"),
                )
            r = {"content": [{"type": "text", "text": f"fake {t} ok"}]}
        else:
            r = {}
        self._send(200, {"jsonrpc": "2.0", "id": i, "result": r})


RUNS = sys.argv[2] if len(sys.argv) > 2 else "/tmp/dv-fake-runs"
if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
