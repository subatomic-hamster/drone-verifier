"""Feedback filter over real MCP, on hfbox: `python tests/check_feedback.py PORT`. For each level: the tool list, what the agent gets from design_set / design_report / fly / verify, a malformed
input still errors, a design failing L1 is refused and uses no verify call, and the event log keeps the full result."""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

PORT = int(sys.argv[1])
ROOT = Path(__file__).resolve().parent.parent
BASE = f"http://127.0.0.1:{PORT}"
REF = json.loads((ROOT / "reference" / "design.json").read_text())
BAD = dict(REF, battery="tattu-1300mah-6s-100c-funfly-xt60")  # fails L1: flight time


def wait_up(p):
    for _ in range(100):
        try:
            urllib.request.urlopen(BASE + "/state", timeout=1)
            return
        except OSError:
            time.sleep(0.2)
    p.kill()
    sys.exit("server did not start")


async def level(name):
    async with streamable_http_client(BASE + "/mcp") as streams:
        async with ClientSession(streams[0], streams[1]) as s:
            await s.initialize()
            tools = sorted(t.name for t in (await s.list_tools()).tools)
            assert ("what_if" in tools) == (name == "whatif"), tools

            async def call(tool, **a):
                r = await s.call_tool(tool, a)
                return json.loads(r.content[0].text)

            bad = await call("design_set", choices={"esc": "no-such-esc"})
            assert bad.get("ok") is False and "Options" in bad["error"], bad
            d1 = await call("design_set", choices=BAD)
            await call("write_controller", code=(ROOT / "reference" / "controller.py").read_text())  # fmt: skip
            refused = await call("fly", scenario="nominal")
            v = await call("verify")
            info = await call("task_info")
            assert info["budget"]["verify_left"] == 12, info[
                "budget"
            ]  # no sim, no charge
            d2 = await call("design_set", choices=REF)
            rep = await call("design_report")
            f = await call("fly", scenario="nominal")
            wi = await call("what_if", slot="esc", part="tmotor-alpha-60a-6s") if name == "whatif" else None  # fmt: skip
            state_after = await call("task_info")
            assert (
                state_after["design_chosen"]["esc"] == REF["esc"]
            )  # what_if changes nothing
            print(name, "tools", len(tools))
            for label, out in (("set bad", d1), ("refused", refused), ("verify L1 fail", v), ("set good", d2), ("report", rep), ("fly", f), ("what_if", wi)):  # fmt: skip
                print(f"   {label}: {json.dumps(out)[:170] if out else None}")
            if name == "none":
                for out in (d1, refused, v, d2, rep, f):
                    assert out == {"recorded": True}, out
            if name == "passfail":
                for out in (d1, refused, v, d2, rep, f):
                    assert list(out) == ["gates"], out
                    assert all(set(g) == {"id", "pass"} for g in out["gates"]), out
                assert refused["gates"][-1] == {"id": "G2", "pass": False}, refused
                assert f["gates"] == [{"id": "G3", "pass": True}], f
            if name in ("diag", "whatif"):
                assert "flight time" in refused["error"] and refused["gate"] == "G2"
                assert f["pass"] and "saturation_pct" in f["metrics"], f
            if name == "whatif":
                assert wi["would_break"] and wi["change"]["thrust_to_weight"] < 0, wi


def main():
    for name in ("none", "passfail", "diag", "whatif"):
        with tempfile.TemporaryDirectory(dir=os.environ.get("DV_RUNS")) as runs:
            p = subprocess.Popen([sys.executable, "-m", "dv.server", "--session", "fb", "--port", str(PORT), "--model", "check", "--feedback", name], cwd=ROOT, env=dict(os.environ, DV_RUNS=runs), stdout=subprocess.DEVNULL)  # fmt: skip
            try:
                wait_up(p)
                asyncio.run(level(name))
                evs = [json.loads(x) for x in (Path(runs) / "fb-check-1" / "events.jsonl").read_text().splitlines()]  # fmt: skip
                kinds = {e["kind"] for e in evs}
                assert {"config", "meters", "flight_end", "verify"} <= kinds, kinds
                assert next(e for e in evs if e["kind"] == "config")["feedback"] == name
                assert any(e["kind"] == "flight_end" and "metrics" in e for e in evs)
                assert not next(e for e in evs if e["kind"] == "verify")["charged"]
            finally:
                p.terminate()
                p.wait(timeout=10)
    print("OK")


main()
