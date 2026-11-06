"""Remote half of harness/bridge.py (runs on hfbox, stdlib only). sshd forbids port forwarding, so the laptop runs
`ssh hfbox python3 .../dv/pipe.py PORT` and speaks newline-delimited JSON over its stdin/stdout, multiplexed by request id.
It makes HTTP requests to 127.0.0.1:PORT (PORT in 8801-8804) for the allowlisted routes below, and nothing else.

laptop -> here: {"id":n,"method":"POST","path":"/mcp","headers":{..},"body":b64}   or {"id":n,"cancel":true}
here -> laptop: {"hello":true} once, then per request {"id":n,"status":200,"headers":{..}}, {"id":n,"chunk":b64}*, {"id":n,"end":true}
"""

import base64
import http.client
import json
import os
import sys
import threading

ALLOW = {
    "/mcp": {"GET", "POST", "DELETE"},
    "/state": {"GET"},
    "/events": {"GET"},
    "/events.jsonl": {"GET"},
    "/agent-event": {"POST"},
}
REQ_H = {
    "content-type",
    "accept",
    "mcp-session-id",
    "mcp-protocol-version",
    "last-event-id",
}
RESP_H = {"content-type", "mcp-session-id", "content-length", "cache-control"}
out, lock, live = sys.stdout.buffer, threading.Lock(), {}


def send(obj):
    line = json.dumps(obj).encode() + b"\n"
    with lock:
        out.write(line)
        out.flush()


def handle(port, f):
    i = f["id"]
    try:
        path, method = f["path"], f["method"]
        if method not in ALLOW.get(path.split("?")[0], ()):
            return (
                send(
                    {
                        "id": i,
                        "status": 404,
                        "headers": {"content-type": "application/json"},
                        "end": False,
                    }
                )
                or send(
                    {
                        "id": i,
                        "chunk": base64.b64encode(
                            b'{"error":"route not allowed"}'
                        ).decode(),
                    }
                )
                or send({"id": i, "end": True})
            )
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=None)
        live[i] = c
        c.request(
            method,
            path,
            base64.b64decode(f.get("body") or "") or None,
            {k: v for k, v in f.get("headers", {}).items() if k.lower() in REQ_H},
        )
        r = c.getresponse()
        send(
            {
                "id": i,
                "status": r.status,
                "headers": {
                    k.lower(): v for k, v in r.getheaders() if k.lower() in RESP_H
                },
            }
        )
        while True:
            b = r.read1(16384)
            if not b:
                break
            send({"id": i, "chunk": base64.b64encode(b).decode()})
        send({"id": i, "end": True})
    except Exception as e:  # upstream closed (cancel) or session down
        try:
            send({"id": i, "error": str(e)[:200]})
        except Exception:
            pass
    finally:
        c = live.pop(i, None)
        if c:
            c.close()


def main():
    port = int(sys.argv[1])
    if port not in range(8801, 8811):
        sys.exit("port not allowed")
    send({"hello": True})
    for line in sys.stdin.buffer:
        f = json.loads(line)
        if f.get("cancel"):
            c = live.get(f["id"])
            if c and c.sock:
                try:
                    c.sock.shutdown(2)
                except OSError:
                    pass
        else:
            threading.Thread(target=handle, args=(port, f), daemon=True).start()
    os._exit(0)  # laptop side hung up


if __name__ == "__main__":
    main()
