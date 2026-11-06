"""One session's server: `python -m dv.server --session A --port 8801 [--gui] [--model NAME] [--trial N]`. MCP (streamable HTTP, JSON answers, stateless) at /mcp and the session's HTTP routes on one port.
Tools run in worker threads, so a long verify never blocks /state, /events or /agent-event."""

import argparse
import asyncio
import json
import threading

import anyio
import uvicorn
from mcp.server import MCPServer
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, PlainTextResponse, StreamingResponse

from dv.session import Session
from dv.tools import TOOLS


def build_app(sess, host="127.0.0.1"):
    srv = MCPServer(
        "dv",
        instructions="Design and program a VTVL hopper. Nothing you claim counts: verify flies it.",
    )

    async def run(tool, args, fn):
        out = await anyio.to_thread.run_sync(lambda: sess.call(tool, args, fn))
        return json.dumps(out, default=str)

    @srv.tool(description=TOOLS["task_info"])
    async def task_info() -> str:
        return await run("task_info", {}, sess.task_info)

    @srv.tool(description=TOOLS["parts_search"])
    async def parts_search(slot: str | None = None, query: str | None = None) -> str:
        return await run(
            "parts_search", dict(slot=slot, query=query), sess.parts_search
        )

    @srv.tool(description=TOOLS["design_set"])
    async def design_set(choices: dict) -> str:
        return await run("design_set", dict(choices=choices), sess.design_set)

    @srv.tool(description=TOOLS["design_report"])
    async def design_report() -> str:
        return await run("design_report", {}, sess.design_report)

    @srv.tool(description=TOOLS["write_controller"])
    async def write_controller(code: str) -> str:
        return await run("write_controller", dict(code=code), sess.write_controller)

    @srv.tool(description=TOOLS["fly"])
    async def fly(scenario: str) -> str:
        return await run("fly", dict(scenario=scenario), sess.fly)

    @srv.tool(description=TOOLS["verify"])
    async def verify() -> str:
        return await run("verify", {}, sess.verify)

    @srv.tool(description=TOOLS["finish"])
    async def finish(note: str = "") -> str:
        return await run("finish", dict(note=note), sess.finish)

    @srv.custom_route("/state", methods=["GET"])
    async def state(_):
        return JSONResponse(sess.state())

    @srv.custom_route("/events.jsonl", methods=["GET"])
    async def events_jsonl(_):
        return PlainTextResponse(
            "".join(json.dumps(e, default=str) + "\n" for e in list(sess.ev.items)),
            media_type="application/x-ndjson",
        )

    @srv.custom_route("/events", methods=["GET"])
    async def events(_):
        async def gen():
            seq = 0
            while True:
                new = await asyncio.to_thread(sess.ev.wait_after, seq, 10.0)
                if not new:
                    yield ": keep-alive\n\n"
                    continue
                for e in new:
                    yield f"data: {json.dumps(e, default=str)}\n\n"
                seq += len(new)

        return StreamingResponse(
            gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    @srv.custom_route("/agent-event", methods=["POST"])
    async def agent_event(req):
        try:
            body = await req.json()
        except ValueError:
            return JSONResponse(
                dict(ok=False, error="body is not JSON"), status_code=400
            )
        if not isinstance(body, dict):
            return JSONResponse(
                dict(ok=False, error="body is not an object"), status_code=400
            )
        kind = "agent_usage" if body.get("kind") == "agent_usage" else "agent_text"
        fields = {
            k: v for k, v in body.items() if k not in ("kind", "seq", "t", "session")
        }
        sess.ev.emit(kind, **fields)
        return JSONResponse(dict(ok=True))

    app = srv.streamable_http_app(json_response=True, stateless_http=True, host=host)
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--model", default="unknown")
    ap.add_argument("--trial", type=int, default=1)
    a = ap.parse_args()
    sess = Session(a.session, a.model, a.trial)
    uv = uvicorn.Server(
        uvicorn.Config(
            build_app(sess, a.host), host=a.host, port=a.port, log_level="warning"
        )
    )
    if a.gui:
        from dv.gui import ViewerUI  # lazy: needs the viewer build

        ui = ViewerUI(
            sess.run_dir, port=a.port, design=sess.built.design, lock=sess.lock
        )
        sess.ui = ui
        threading.Thread(target=uv.run, daemon=True).start()
        ui.run()  # main thread: the window; returns on ui.stop()
        uv.should_exit = True
    else:
        uv.run()


if __name__ == "__main__":
    main()
