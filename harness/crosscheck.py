"""Second sim: re-fly each run's final design + controller at half timestep on the held-out seeds and add `crosscheck` to its results.json.
Not run during agent runs. Uses CORE's dv.heldout.crosscheck(run_dir). Only whatif runs scoring >= MIN_SCORE (50).

  python harness/crosscheck.py            laptop: ships results/*/design.json + controller.py to hfbox, flies there, merges the answer
  python harness/crosscheck.py --remote DIR...   on hfbox: DIR holds design.json, controller.py, results.json; prints one JSON line per DIR
crosscheck = {score_full, score_half, score_change, gates_half:[{id,pass}], dt_scale: 0.5}
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
import hosts

HF = str(ROOT / "scripts" / "hf.sh")
RR = hosts.RUNS
MIN_SCORE = 50  # whatif runs scoring at least this are re-flown


def fly_half(rd):
    from dv import heldout

    r = heldout.crosscheck(
        rd
    )  # CORE's half-timestep re-fly; does not touch results.json
    return dict(
        score_full=r["score_was"],
        score_half=r["score"],
        score_change=r["change"],
        gates_half=[{"id": g["id"], "pass": g["pass"]} for g in r["gates"]],
        dt_scale=0.5,
    )


def ssh(cmd, data=None):
    return hosts.run(cmd, input=data)


def main():
    if sys.argv[1:2] == ["--remote"]:
        sys.path.insert(0, str(ROOT))
        for d in sys.argv[2:]:
            print(
                json.dumps({"dir": Path(d).name, "crosscheck": fly_half(d)}), flush=True
            )
        return
    todo = [
        d
        for d in sorted((ROOT / "results").glob("*/"))
        if (d / "results.json").exists()
        and (d / "design.json").exists()
        and (d / "controller.py").read_text().strip()
        and (r := json.loads((d / "results.json").read_text()))
        and "crosscheck" not in r
        and r.get("feedback") == "whatif"
        and (r.get("score") or 0) >= MIN_SCORE
    ]
    if not todo:
        sys.exit(
            "nothing to cross-check (needs results/<run>/{results.json,design.json,controller.py} without a crosscheck field)"
        )
    for d in todo:
        ssh(f"mkdir -p {RR}/crosscheck-{d.name}")
        for f in ("results.json", "design.json", "controller.py"):
            ssh(f"cat > {RR}/crosscheck-{d.name}/{f}", (d / f).read_text())
    dirs = " ".join(f"{RR}/crosscheck-{d.name}" for d in todo)
    r = subprocess.run(
        [HF, "run", f"python harness/crosscheck.py --remote {dirs}"],
        capture_output=True,
        text=True,
    )
    got = {}
    for line in r.stdout.splitlines():
        if line.startswith("{"):
            j = json.loads(line)
            got[j["dir"].removeprefix("crosscheck-")] = j["crosscheck"]
    if not got:
        sys.exit(f"ERROR: crosscheck produced nothing: {(r.stdout + r.stderr)[-400:]}")
    for d in todo:
        if d.name in got:
            res = json.loads((d / "results.json").read_text())
            res["crosscheck"] = got[d.name]
            (d / "results.json").write_text(json.dumps(res, indent=1))
            sys.stdout.write(
                f"{d.name}: {got[d.name]['score_full']} -> {got[d.name]['score_half']}\n"
            )
    subprocess.run([sys.executable, str(ROOT / "harness" / "leaderboard.py")])


if __name__ == "__main__":
    main()
