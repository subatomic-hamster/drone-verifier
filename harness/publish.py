"""Copies the scores worth publishing from results/ (gitignored) into benchmark/ (tracked): leaderboard.json, planted.json and every run's
results.json as benchmark/runs/<run>.json. No logs, no events, no media. Rebuilds the leaderboard first.  python harness/publish.py"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R, B = ROOT / "results", ROOT / "benchmark"

subprocess.run(
    [sys.executable, str(ROOT / "harness" / "leaderboard.py")],
    capture_output=True,
    check=True,
)
(B / "runs").mkdir(parents=True, exist_ok=True)
for f in ("leaderboard.json", "planted.json"):
    if (R / f).exists():
        shutil.copy(R / f, B / f)
for d in sorted(R.glob("*/results.json")):
    shutil.copy(d, B / "runs" / f"{d.parent.name}.json")
sys.stdout.write(f"benchmark/: {len(list((B / 'runs').glob('*.json')))} runs\n")
