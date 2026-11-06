"""The Start panel's guards, offline: python harness/tests/test_panel.py. Starts only a sleeping child; no model, no hfbox."""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

H = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(H))
import demo

SLEEP = [sys.executable, "-c", "import subprocess,sys;subprocess.Popen([sys.executable,'-c','import time;time.sleep(300)'],start_new_session=True).wait()"]  # fmt: skip


def call(port, path, body=None, headers=None, raw=None):
    h = {
        "Origin": f"http://127.0.0.1:{port}",
        "Content-Type": "application/json",
        **(headers or {}),
    }
    h = {k: v for k, v in h.items() if v is not None}
    data = (
        raw if raw is not None else None if body is None else json.dumps(body).encode()
    )
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data, h)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        body = e.read()
        return e.code, json.loads(body) if body[:1] == b"{" else None


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def test_validate():
    ok = dict(mode="benchmark", models=["fake"], feedback="diag", trials=2)
    known = {**demo.MODELS, "fake": demo.FAKE}
    assert demo.validate(ok, known) == ("benchmark", ["fake"], "diag", 2)
    for bad in (
        {**ok, "models": ["haiku; rm -rf /"]},
        {**ok, "models": ["fake", "fake"]},
        {**ok, "models": "fake"},
        {**ok, "models": []},
        {**ok, "feedback": "--sweep"},
        {**ok, "mode": "shell"},
        {**ok, "trials": True},
        {**ok, "trials": "2"},
        {**ok, "trials": 99},
        {**ok, "cmd": "x"},
        {**ok, "mode": "design", "models": ["haiku", "opus"]},
        {**ok, "mode": "versus", "models": ["haiku"]},
    ):
        try:
            demo.validate(bad, known)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    a = demo.command("benchmark", ["haiku", "nemotron"], "diag", 2)
    assert a[1:] == [str(H / "bench.py"), "--models", "haiku,nemotron", "--trials", "2", "--feedback", "diag"]  # fmt: skip
    assert demo.command("versus", ["haiku", "opus"], "none", 1)[2:] == ["versus", "haiku", "opus", "--feedback", "none"]  # fmt: skip
    assert demo.command("design", ["fake"], "whatif", 1)[2:] == ["single", "fake", "--feedback", "whatif"]  # fmt: skip


def test_http():
    panel = demo.Panel(fake=True, command=lambda *a: SLEEP)
    p = demo.serve(demo.panel_handler(panel))
    start = dict(mode="design", models=["fake"], feedback="whatif", trials=1)
    # who may ask
    assert call(p, "/api/start", start, {"Origin": None})[0] == 403
    assert call(p, "/api/start", start, {"Origin": "https://evil.example"})[0] == 403
    assert call(p, "/api/start", start, {"Host": "evil.example"})[0] == 403
    assert call(p, "/api/status", headers={"Host": "evil.example"})[0] == 403
    assert call(p, "/api/start", start, {"Content-Type": "text/plain"})[0] == 415
    assert call(p, "/api/start", raw=b"x" * 5000)[0] == 400
    assert not panel.runs, "a refused request started something"
    # what may be read
    for path in ("/.git/config", "/harness/demo.py", "/web/../harness/demo.py", "/web/%2e%2e/harness/demo.py"):  # fmt: skip
        assert call(p, path)[0] == 404, path
    # money needs the confirm step, and starts nothing before it
    code, r = call(p, "/api/start", {**start, "models": ["haiku"]})
    assert code == 409 and r["need_confirm"] == ["haiku"] and not panel.runs
    code, r = call(p, "/api/start", {**start, "models": ["deepseek-pro"]})
    assert code == 409 and r["need_confirm"] == ["deepseek-pro"]
    # start, one at a time; stop kills what it started and nothing else
    other = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    code, r = call(p, "/api/start", start)
    assert code == 200, r
    rid, pid = r["id"], r["pid"]
    time.sleep(0.5)
    kids = demo.descendants(pid)
    assert kids and alive(kids[0])
    assert call(p, "/api/start", start)[0] == 409
    assert call(p, "/api/stop", dict(id=999))[0] == 404
    assert call(p, "/api/stop", dict(id=str(other.pid)))[0] == 404
    st = call(p, "/api/status")[1]
    assert [x["name"] for x in st["models"]][-1] == "fake"
    assert st["runs"][0]["alive"]
    assert call(p, "/api/stop", dict(id=rid))[0] == 200
    end = time.time() + 5
    while time.time() < end and (alive(pid) or alive(kids[0])):
        time.sleep(0.1)
    assert not alive(pid) and not alive(kids[0]), "stop left the run alive"
    assert alive(other.pid), "stop touched a process it did not start"
    other.kill()
    panel.runs[rid].log.unlink()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    signal.alarm(60)
    test_validate()
    test_http()
    print("panel ok")
