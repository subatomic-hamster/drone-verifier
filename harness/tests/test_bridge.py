"""Needs a session answering on hfbox 8801 (real dv.server or tests/fake_server.py via hf.sh bg). Run: python harness/tests/test_bridge.py [PORT]
Checks /state latency, MCP initialize + tools/list, a slow SSE not blocking /state, route allowlist, and reconnect after killing the ssh."""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8801
B = Path(__file__).resolve().parent.parent / "bridge.py"
U = f"http://127.0.0.1:{PORT}"


def get(path, **kw):
    return urllib.request.urlopen(
        urllib.request.Request(U + path, **kw), timeout=30
    ).read()


def rpc(i, method, params=None):
    r = urllib.request.urlopen(
        urllib.request.Request(
            U + "/mcp",
            json.dumps(
                {"jsonrpc": "2.0", "id": i, "method": method, "params": params or {}}
            ).encode(),
            {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
        ),
        timeout=60,
    )
    return r.read().decode()


def wait_up(secs=40):
    end = time.time() + secs
    while time.time() < end:
        try:
            if json.loads(get("/bridge-state"))["up"]:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise SystemExit("FAIL link never came up")


p = subprocess.Popen([sys.executable, str(B), "--port", str(PORT)])
try:
    wait_up()
    t = []
    for _ in range(5):
        t0 = time.time()
        json.loads(get("/state"))
        t.append(time.time() - t0)
    print(f"/state round trip s: {[round(x, 3) for x in t]}")
    out = rpc(
        1,
        "initialize",
        {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "t", "version": "1"},
        },
    )
    t0 = time.time()
    out = rpc(2, "tools/list")
    print(f"tools/list {time.time() - t0:.3f}s")
    assert "verify" in out and "finish" in out, out[:200]
    sse = urllib.request.urlopen(
        U + "/events", timeout=30
    )  # open stream stays open ...
    t0 = time.time()
    get("/state")
    assert time.time() - t0 < 5  # ... without blocking others
    sse.close()
    try:
        get("/etc/passwd")
        raise SystemExit("FAIL allowlist")
    except urllib.error.HTTPError as e:
        assert e.code == 404
    # reconnect: kill the ssh child of the bridge
    subprocess.run(["pkill", "-f", f"ssh .*pipe.py {PORT}"])
    time.sleep(1)
    t0 = time.time()
    wait_up(60)
    json.loads(get("/state"))
    print(
        f"reconnected after {time.time() - t0:.1f}s, reconnects={json.loads(get('/bridge-state'))['reconnects']}"
    )
    print("bridge ok")
finally:
    p.kill()
