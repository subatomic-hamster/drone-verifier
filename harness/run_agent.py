"""Run one benchmarked agent: Claude Code headless, isolated, one MCP server (the session on 127.0.0.1:PORT).

  python harness/run_agent.py --model haiku --port 8801 --run-dir results/haiku-0 --prompt-file prompt.txt

Isolation (proved in the smoke run, see README of the report): --tools "" (no built-in tools), --strict-mcp-config, --setting-sources ""
(no user/project settings, hooks, plugins from settings), --disable-slash-commands, CLAUDE_CODE_DISABLE_AUTO_MEMORY=1, empty temp cwd,
no session persistence. --bare is NOT used: it refuses the OAuth login and demands ANTHROPIC_API_KEY.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import tomllib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "haiku": "claude-haiku-5-5",
    "sonnet": "claude-sonnet-5-5",
    "opus": "claude-opus-5-5",
    "fable": "claude-fable-5-1",
}
# alias -> omp model selector (omp CLI instead of claude). nemotron goes through NVIDIA NIM (free tier, cost 0). The deepseek-* ones use the
# direct deepseek provider and cost real money: never run without the owner's go-ahead.
OMP_MODELS = {
    "nemotron": "nvidia/nvidia/nemotron-3.5-super-vl-120b-a12b",
    "deepseek-flash": "deepseek/deepseek-flash",
    "deepseek-pro": "deepseek/deepseek-v4-pro",
    "gemini-flash-lite": "google/gemini-3.5-flash-lite",  # needs GEMINI_API_KEY in the environment
    "grok": "xai/grok-4.7",  # needs XAI_API_KEY in the environment; limited quota, 1-2 runs at most
}


def task_limits():
    """(wall_s, budget_usd) from dv/task.toml; searches any table for the keys, falls back to the DESIGN.md numbers."""
    wall_min, usd = 25, 15.0
    try:
        t = tomllib.loads((ROOT / "dv" / "task.toml").read_text())
        flat = {}

        def walk(d):
            for k, v in d.items():
                walk(v) if isinstance(v, dict) else flat.setdefault(k, v)

        walk(t)
        wall_min = flat.get("wall_min", wall_min)
        usd = flat.get("agent_budget_usd", flat.get("max_budget_usd", usd))
    except FileNotFoundError:
        pass
    return wall_min * 60, usd


def post(port, path, body, timeout=5):
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            json.dumps(body).encode(),
            # the MCP endpoint answers 406 without both Accept types
            {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
        )
        return urllib.request.urlopen(req, timeout=timeout).read()
    except Exception as e:
        post.last_error = repr(e)
        return None  # a dead feed must never kill the agent


def mcp_call(port, tool, args, timeout=900):
    """Plain JSON-RPC over the streamable-HTTP endpoint (used for the fallback finish)."""

    def rpc(i, method, params):
        raw = post(
            port,
            "/mcp",
            {
                "jsonrpc": "2.0",
                **({"id": i} if i else {}),
                "method": method,
                "params": params,
            },
            timeout,
        )
        return raw

    rpc(
        1,
        "initialize",
        {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "harness", "version": "1"},
        },
    )
    rpc(0, "notifications/initialized", {})
    return rpc(2, "tools/call", {"name": tool, "arguments": args})


class Stream:
    """Turns stream-json lines into /agent-event posts. Pure: feed() returns the events, the caller posts them."""

    def __init__(self):
        self.inp = self.out = 0
        self.cost = None
        self.finish_called = False
        self.errors = []

    def feed(self, d):
        ev = []
        if d.get("type") == "assistant":
            m = d["message"]
            for b in m.get("content", []):
                if b.get("type") == "text" and b["text"].strip():
                    ev.append({"kind": "agent_text", "text": b["text"]})
                elif b.get("type") == "tool_use" and b.get("name", "").endswith(
                    "__finish"
                ):
                    self.finish_called = True
            u = m.get("usage") or {}
            self.inp += (
                u.get("input_tokens", 0)
                + u.get("cache_creation_input_tokens", 0)
                + u.get("cache_read_input_tokens", 0)
            )
            self.out += u.get("output_tokens", 0)
            ev.append(
                {
                    "kind": "agent_usage",
                    "input_tokens": self.inp,
                    "output_tokens": self.out,
                    "cost_usd": self.cost,
                }
            )
        elif d.get("type") == "result":
            self.cost = d.get("total_cost_usd")
            if d.get("is_error"):
                self.errors.append(str(d.get("result"))[:300])
            ev.append(
                {
                    "kind": "agent_usage",
                    "input_tokens": self.inp,
                    "output_tokens": self.out,
                    "cost_usd": self.cost,
                }
            )
        return ev


class OmpStream(Stream):
    """Same contract as Stream for `omp --mode json`: one message_end per assistant message carries text blocks, toolCall blocks and usage."""

    def feed(self, d):
        m = d.get("message") or {}
        if d.get("type") != "message_end" or m.get("role") != "assistant":
            return []
        ev = []
        for b in m.get("content", []):
            if b.get("type") == "text" and b["text"].strip():
                ev.append({"kind": "agent_text", "text": b["text"]})
            elif b.get("type") == "toolCall" and b.get("name", "").endswith("__finish"):
                self.finish_called = True
        u = m.get("usage") or {}
        self.inp += u.get("input", 0) + u.get("cacheRead", 0) + u.get("cacheWrite", 0)
        self.out += u.get("output", 0)
        self.cost = (self.cost or 0) + (u.get("cost") or {}).get("total", 0)
        if m.get("stopReason") == "error":
            self.errors.append(str(m.get("errorMessage"))[:300])
        ev.append(
            {
                "kind": "agent_usage",
                "input_tokens": self.inp,
                "output_tokens": self.out,
                "cost_usd": self.cost,
            }
        )
        return ev


def omp_cmd(model, port, prompt, cwd, wall_s):
    """(cmd, env, extra temp dir) for omp. Isolation proved by negative test (see README): --no-tools leaves only the MCP tools, the project
    .omp/mcp.json in the fresh cwd supplies ours, and a throwaway HOME + agent dir hide the user's mcp.json, skills, rules, memory and
    AGENTS.md. Only agent.db (the provider logins) is symlinked in: omp --profile would lose them. Nothing is copied or printed."""
    home = tempfile.mkdtemp(prefix="dv-omp-home-")
    agent = Path(home) / "agent"
    agent.mkdir()
    real = Path(os.environ.get("PI_CODING_AGENT_DIR") or Path.home() / ".omp" / "agent")
    (agent / "agent.db").symlink_to(real / "agent.db")
    (Path(cwd) / ".omp").mkdir()
    (Path(cwd) / ".omp" / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "dv": {
                        "type": "http",
                        "url": f"http://127.0.0.1:{port}/mcp",
                    }
                }
            }
        )
    )
    cmd = [
        "omp",
        "--model",
        OMP_MODELS[model],
        "--mode",
        "json",
        "--no-tools",
        "--no-extensions",
        "--no-skills",
        "--no-rules",
        "--no-lsp",
        "--no-session",
        "--no-title",
        "--max-time",
        str(int(wall_s)),
        "-p",
        prompt,
    ]
    return cmd, {**os.environ, "HOME": home, "PI_CODING_AGENT_DIR": str(agent)}, home


def run_agent(model, port, run_dir, prompt, wall_s=None, budget_usd=None, quiet=False):
    lim_wall, lim_usd = task_limits()
    wall_s, budget_usd = wall_s or lim_wall, budget_usd or lim_usd
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    mcp = run_dir / "mcp.json"
    mcp.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "dv": {"type": "http", "url": f"http://127.0.0.1:{port}/mcp"}
                }
            }
        )
    )
    cwd = tempfile.mkdtemp(prefix="dv-agent-")  # fresh and empty
    cmd = [
        "claude",
        "-p",
        prompt,
        "--model",
        MODELS.get(model, model),
        "--output-format",
        "stream-json",
        "--verbose",
        "--tools",
        "",
        "--strict-mcp-config",
        "--mcp-config",
        str(mcp),
        "--allowedTools",
        "mcp__dv",
        "--setting-sources",
        "",
        "--no-session-persistence",
        "--disable-slash-commands",
        "--permission-mode",
        "dontAsk",
        "--max-budget-usd",
        str(budget_usd),
    ]
    env = {**os.environ, "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}
    omp = model in OMP_MODELS
    if omp:
        cmd, env, omp_home = omp_cmd(model, port, prompt, cwd, wall_s)
    t0 = time.time()
    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,  # omp waits for EOF on a piped stdin
        stdout=subprocess.PIPE,
        stderr=open(run_dir / "stderr.log", "w"),
        text=True,
        start_new_session=True,
    )
    killed = []

    def kill():
        killed.append(1)
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    timer = threading.Timer(wall_s + 60, kill)
    timer.start()
    st = OmpStream() if omp else Stream()
    with open(run_dir / "stream.jsonl", "w") as raw:
        for line in p.stdout:
            raw.write(line)
            raw.flush()
            try:
                d = json.loads(line)
            except ValueError:
                continue
            for e in st.feed(d):
                post(port, "/agent-event", e)
            if (
                omp and (st.cost or 0) > budget_usd
            ):  # claude enforces its own --max-budget-usd
                kill()
    p.wait()
    timer.cancel()
    if omp:
        shutil.rmtree(omp_home, ignore_errors=True)

    def is_finished():
        try:
            return bool(
                json.load(
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/state", timeout=10)
                ).get("finished")
            )
        except Exception:
            return False

    harness_finish = (
        not is_finished()
    )  # the server's word, not the stream's: the agent's own finish may have errored
    deadline = (
        time.time() + 240
    )  # a dead link (network drop) is retried for 4 minutes, then the run is reported lost, never hung
    while time.time() < deadline and not is_finished():
        # agent ran out of budget / time / died: score what it left behind (held-out rerun runs inside finish)
        mcp_call(
            port,
            "finish",
            {
                "note": "harness: agent ended without calling finish"
                + (" (wall limit)" if killed else "")
            },
            timeout=min(600, max(30, deadline - time.time() + 300)),
        )
        if not is_finished():
            time.sleep(15)
    meta = {
        "model": model,
        "port": port,
        "wall_s": round(time.time() - t0, 1),
        "killed_at_wall_limit": bool(killed),
        "rc": p.returncode,
        "finish_by_harness": harness_finish,
        "claude_cost_usd": st.cost,
        "tokens": {"input": st.inp, "output": st.out},
        "errors": st.errors,
    }
    (run_dir / "agent.json").write_text(json.dumps(meta, indent=1))
    if not quiet:
        print(json.dumps(meta))
    return meta


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--model", required=True)
    a.add_argument("--port", type=int, required=True)
    a.add_argument("--run-dir", required=True)
    a.add_argument("--prompt-file", required=True)
    a.add_argument("--wall-s", type=float)
    a.add_argument("--budget-usd", type=float)
    a = a.parse_args()
    run_agent(
        a.model,
        a.port,
        a.run_dir,
        Path(a.prompt_file).read_text(),
        a.wall_s,
        a.budget_usd,
    )
