"""`python -m dv.render_prompt` prints harness/prompt.md with every {placeholder} filled from dv/task.toml, dv/tools.py and harness/conventions.md. Only the prompt goes to stdout. Stdlib only."""

import argparse
import re
import tomllib
from pathlib import Path

from dv.tools import ABOUT, LEVELS, tools_for

ROOT = Path(__file__).resolve().parent.parent


def _num(v):
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def render(level="whatif"):
    t = tomllib.loads((ROOT / "dv" / "task.toml").read_text())
    m, lim, bud = t["mission"], t["limits"], t["budget"]
    vals = {
        k: _num(m[k])
        for k in (
            "alt_m",
            "alt_tol_m",
            "hold_s",
            "land_radius_m",
            "touchdown_m_s",
            "tilt_max_deg",
            "mission_s",
            "t_cal_s",
        )
    }
    vals.update(
        cost_max_usd=_num(lim["cost_max_usd"]),
        mass_max_kg=_num(lim["mass_max_kg"]),
        envelope_mm=" x ".join(_num(v) for v in lim["envelope_mm"]),
        tw_min=_num(lim["tw_min"]),
        soc_min_pct=_num(lim["soc_min_pct"]),
        verify_budget=_num(bud["verify"]),
        wall_min=_num(bud["wall_min"]),
        tools_table="| tool | what |\n| ---- | ---- |\n"
        + "\n".join(f"| `{k}` | {v} |" for k, v in tools_for(level).items())
        + "\n\n"
        + ABOUT[level]
        + " G1 and G2 need no simulator: a `verify` that stops there does not use up a call, and `fly` refuses a design failing them.",
        conventions=(ROOT / "harness" / "conventions.md").read_text().strip(),
    )
    text = (ROOT / "harness" / "prompt.md").read_text()
    # the L1 limits, under the hard constraints (harness/prompt.md lists the original ones)
    anchor = "- Thrust to weight at full charge: at least {tw_min}\n"
    assert anchor in text
    text = text.replace(
        anchor,
        anchor
        + f"- Hover at full charge: at most {_num(lim['hover_throttle_max_pct'])} % throttle and {_num(lim['hover_power_max_w'])} W\n"
        + f"- Flight time at hover down to {_num(lim['soc_min_pct'])} % charge: at least {_num(lim['flight_time_min_s'])} s\n"
        + "- Pack and ESC current ratings cover full throttle; regulators cover two stalled servos; 6S and the airframe's XT60 lead\n",
    )
    rows = "\n".join(
        f"| {g['id']} {g['name']} | {g['what']} | {g['points']} |" for g in t["gates"]
    )
    text = re.sub(r"^\| \{gates_table\} \|$", rows, text, flags=re.MULTILINE)
    text = re.sub(
        r"\{(\w+)\}",
        lambda g: vals[g.group(1)] if g.group(1) in vals else g.group(0),
        text,
    )
    left = [p for p in re.findall(r"\{(\w+)\}", text) if p in ("gates_table", *vals)]
    assert not left, left
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feedback", choices=LEVELS, default="whatif")
    print(render(ap.parse_args().feedback))
