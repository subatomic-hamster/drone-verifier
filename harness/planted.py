"""Turns the table printed by tests/check_planted.py into results/planted.json.
  python harness/planted.py [LOGFILE]    default: the hfbox tmux log of `hf.sh bg planted "python tests/check_planted.py"`
Rows: {name, flagged, layer, gate, why} for the planted flaws; `clean`: lines of the clean-design table; `summary`: the closing lines."""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROW = re.compile(r"^(\S.*?)\s{2,}(yes|NO)\s+(L\d|-)\s+(G\d|-)\s+(.*)$")


def parse(text):
    flaws, clean, summary, mode = [], [], [], "flaws"
    for line in text.splitlines():
        if line.startswith("clean design"):
            mode = "clean"
            continue
        m = ROW.match(line)
        if mode == "flaws" and m:
            flaws.append(
                dict(name=m[1], flagged=m[2] == "yes", layer=m[3], gate=m[4], why=m[5])
            )
        elif (
            mode == "clean"
            and line.strip()
            and not line.startswith(("depth", "verify", "Connecting"))
        ):
            clean.append(line)
    summary = [
        l
        for l in clean
        if "flagged" in l.lower()
        or "false positive" in l.lower()
        or l.startswith("L0+L1")
    ]
    return dict(
        updated=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        flagged=sum(f["flagged"] for f in flaws),
        total=len(flaws),
        flaws=flaws,
        clean=clean,
        summary=summary,
    )


if __name__ == "__main__":
    if sys.argv[1:]:
        text = Path(sys.argv[1]).read_text()
    else:
        text = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "hfbox",
                "cat /workspace/wt/dv/logs/planted.log",
            ],
            capture_output=True,
            text=True,
        ).stdout
    out = parse(text)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "planted.json").write_text(json.dumps(out, indent=1))
    sys.stdout.write(
        f"planted.json: {out['flagged']}/{out['total']} flagged, {len(out['clean'])} clean lines\n"
    )
