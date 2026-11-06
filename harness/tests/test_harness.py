"""Run: python harness/tests/test_harness.py   (stdlib only; spawns fake_server on 8898 and one real haiku call unless --offline)"""

import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

H = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(H))
import leaderboard
import run_agent


def test_stream():
    s = run_agent.Stream()
    ev = s.feed(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "tool_use", "name": "mcp__dv__finish"},
                ],
                "usage": {
                    "input_tokens": 2,
                    "cache_creation_input_tokens": 10,
                    "output_tokens": 5,
                },
            },
        }
    )
    assert (
        ev[0] == {"kind": "agent_text", "text": "hi"}
        and s.finish_called
        and ev[1]["input_tokens"] == 12
        and ev[1]["output_tokens"] == 5
    )
    ev = s.feed({"type": "result", "total_cost_usd": 0.5})
    assert ev[0]["cost_usd"] == 0.5 and s.feed({"type": "system"}) == []


def test_leaderboard():
    d = Path(tempfile.mkdtemp())
    for name, model, score, ok in [
        ("a-0", "a", 40, True),
        ("b-0", "b", 90, True),
        ("b-1", "b", 70, False),
    ]:
        (d / name).mkdir()
        (d / name / "results.json").write_text(
            json.dumps(
                {
                    "model": model,
                    "score": score,
                    "verify_calls": 3,
                    "flights": 4,
                    "wall_s": 10,
                    "tokens": {"input": 5, "output": 5},
                    "cost_usd": 1,
                    "failure_reasons": [] if ok else ["x"],
                    "gates": [
                        {"id": "G1", "pass": True, "points": 10, "max_points": 10},
                        {"id": "G2", "pass": score > 50, "points": 0, "max_points": 10},
                    ],
                }
            )
        )
    e = leaderboard.build(d)["entries"]
    assert (
        [x["model"] for x in e] == ["b", "a"]
        and e[0]["score"] == 80
        and e[0]["trials"] == 2
        and e[0]["best_gate"] == "G2"
    )
    assert (
        e[1]["best_gate"] == "G1"
        and e[0]["tokens"] == 10
        and e[0]["failure_reasons"] == ["x"]
    )


def test_agent_with_fake():
    """Real haiku call (cheap) against the fake server: events arrive, finish is called by the harness, results.json appears."""
    srv = subprocess.Popen(
        [
            sys.executable,
            str(H / "tests" / "fake_server.py"),
            "8898",
            tempfile.mkdtemp(),
        ]
    )
    try:
        time.sleep(1)
        meta = run_agent.run_agent(
            "haiku",
            8898,
            tempfile.mkdtemp(),
            "Call the task_info tool once, then reply DONE.",
            wall_s=120,
            budget_usd=0.5,
            quiet=True,
        )
        ev = [
            json.loads(l)
            for l in urllib.request.urlopen("http://127.0.0.1:8898/events.jsonl")
            .read()
            .decode()
            .splitlines()
        ]
        kinds = [e["kind"] for e in ev]
        assert (
            "agent_text" in kinds and "agent_usage" in kinds and "tool_call" in kinds
        ), kinds
        assert meta["finish_by_harness"] and kinds.count("tool_call") >= 2, (
            meta,
            kinds,
        )  # task_info + fallback finish
    finally:
        srv.kill()


if __name__ == "__main__":
    test_stream()
    test_leaderboard()
    print("offline ok")
    if "--offline" not in sys.argv:
        test_agent_with_fake()
        print("fake-server run ok")
