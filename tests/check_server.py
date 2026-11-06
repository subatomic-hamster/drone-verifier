"""End to end over real MCP: starts dv.server, lists tools with an SDK client, then plays a short agent (design, controller, fly, verify, finish) and checks the HTTP routes, on hfbox.
`python tests/check_server.py PORT`"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from dv.tools import TOOLS
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

PORT = int(sys.argv[1])
ROOT = Path(__file__).resolve().parent.parent
BASE = f"http://127.0.0.1:{PORT}"


def get(path):
    t = time.perf_counter()
    with urllib.request.urlopen(BASE + path, timeout=5) as r:
        return r.read().decode(), time.perf_counter() - t


def post(path, obj):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(obj).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status


def rpc(method, params, rid=1):
    """Plain JSON-RPC POST, as the harness's fallback finish does."""
    req = urllib.request.Request(
        BASE + "/mcp",
        data=json.dumps(
            dict(jsonrpc="2.0", id=rid, method=method, params=params)
        ).encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, r.read().decode()[:200]


async def main():
    async with streamable_http_client(BASE + "/mcp") as streams:
        async with ClientSession(streams[0], streams[1]) as s:
            await s.initialize()
            names = sorted(t.name for t in (await s.list_tools()).tools)
            print("MCP tool list:", names)
            assert names == sorted(TOOLS)

            async def call(name, **args):
                t = time.perf_counter()
                r = await s.call_tool(name, args)
                d = json.loads(r.content[0].text)
                print(
                    f"{name} {time.perf_counter() - t:.1f}s -> {d.get('summary') or d.get('error') or list(d)[:4]}",
                    flush=True,
                )
                return d

            cat = await call("parts_search", slot="esc")
            assert "reference" not in json.dumps(cat).lower()
            bad = await call("design_set", choices={"esc": "no-such-esc"})
            assert bad["ok"] is False and "Options" in bad["error"]
            d = await call(
                "design_set",
                choices=json.loads((ROOT / "reference" / "design.json").read_text()),
            )
            assert d["ok"], d
            await call("design_report")
            w = await call("write_controller", code="import os\nclass Controller: pass")
            assert not w["ok"]
            await call(
                "write_controller",
                code=(ROOT / "reference" / "controller.py").read_text(),
            )
            f = await call("fly", scenario="nominal")
            assert f["pass"], f
            # /state answers while a flight is running
            task = asyncio.create_task(call("verify"))
            await asyncio.sleep(2.0)
            _, dt = await asyncio.to_thread(get, "/state")
            print(f"/state during verify: {dt * 1000:.0f} ms")
            assert dt < 1.0
            print(
                "agent-event",
                await asyncio.to_thread(
                    post,
                    "/agent-event",
                    dict(
                        kind="agent_usage",
                        input_tokens=5,
                        output_tokens=7,
                        cost_usd=None,
                    ),
                ),
            )
            print(
                "agent-text",
                await asyncio.to_thread(
                    post, "/agent-event", dict(kind="whatever", text="hello")
                ),
            )
            v = await task
            assert v["score"] == 100, v
            fin = await call("finish", note="check")
            print("finish score", fin.get("score"))
            assert fin["score"] == 100


if __name__ == "__main__":
    out = tempfile.TemporaryDirectory(prefix="dv-check-")
    env = dict(os.environ, DV_RUNS=out.name)
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"
    srv = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "dv.server",
            "--session",
            "T",
            "--port",
            str(PORT),
            "--model",
            "check",
            "--trial",
            "1",
        ],
        cwd=ROOT,
        env=env,
    )
    try:
        for _ in range(100):
            try:
                print("state before", get("/state")[0][:300])
                break
            except OSError:
                time.sleep(0.2)
        print(
            "plain JSON-RPC",
            rpc(
                "initialize",
                dict(
                    protocolVersion="2025-06-18",
                    capabilities={},
                    clientInfo=dict(name="x", version="1"),
                ),
            ),
        )
        asyncio.run(main())
        ev, _ = get("/events.jsonl")
        kinds = {}
        for line in ev.splitlines():
            k = json.loads(line)["kind"]
            kinds[k] = kinds.get(k, 0) + 1
        print("events", kinds)
        rd = Path(out.name) / "T-check-1"
        print(
            "run dir",
            sorted(p.name for p in rd.iterdir()),
            sorted(p.name for p in (rd / "flight_last").iterdir()),
        )
        print(
            "replay kB",
            (rd / "flight_last" / "replay.dvsimog.gz").stat().st_size // 1024,
        )
        res = json.loads((rd / "results.json").read_text())
        print(
            "results",
            {
                k: res[k]
                for k in (
                    "model",
                    "trial",
                    "score",
                    "verify_calls",
                    "flights",
                    "wall_s",
                    "tokens",
                    "cost_usd",
                    "first_pass_flight_s",
                    "failure_reasons",
                    "cost_usd_bom",
                    "mass_kg",
                    "cost_by_kind",
                    "feedback",
                    "pass_within_budget",
                    "first_viable",
                    "robustness_gap",
                    "prediction_error",
                    "thrust_l1_vs_static_sim_pct",
                    "viable_designs",
                )
            },
        )
        print("energy_invariant", res["energy_invariant"]["ok"], res["energy_invariant"]["flights"][0], len(res["energy_invariant"]["flights"]))  # fmt: skip
        assert res["feedback"] == "whatif" and res["pass_within_budget"] is True
        assert res["first_viable"] and res["viable_designs"] and res["energy_invariant"]["ok"]  # fmt: skip
        print("OK")
    finally:
        srv.terminate()
        srv.wait(timeout=5)
