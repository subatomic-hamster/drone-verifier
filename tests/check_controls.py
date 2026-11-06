"""Positive and negative controls through the same path agents use (Session, sandbox, gates). Run on hfbox: `python tests/check_controls.py [SEEDS]`.
Exits non-zero if a control does not behave. SEEDS = also fly the reference on that many extra held-out base seeds (margin)."""

import json
import sys
import tempfile
from pathlib import Path

from dv import design as D
from dv import flight as F  # noqa: F401
from dv import heldout, verify

R = Path(__file__).resolve().parent.parent / "reference"
REF = json.loads((R / "design.json").read_text())
HELD = heldout.heldout_seed()


def show(title, r):
    print(title, "score", r["score"], flush=True)
    for g in r["gates"]:
        if not g["reason"].startswith("not run"):
            print(
                f"  {g['id']} {'PASS' if g['pass'] else 'FAIL'} {g['points']}/{g['max_points']}: {g['reason'][:200]}",
                flush=True,
            )


def gates(choices, code, seed=HELD):
    b = D.apply(choices)
    b.static = None
    print(
        f"  cost ${b.cost:.2f} ({b.cost_by_kind}) mass {b.mass:.3f} kg envelope {b.envelope_mm} cg_mm {[round(v * 1000, 1) for v in b.props['cg']]}",
        flush=True,
    )
    return verify.run_gates(b, code, seed), b


def main():
    bad = []
    code = (R / "controller.py").read_text()
    print("controller lines", code.count("\n") + 1)
    extra = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if not extra:
        print("REFERENCE, held out")
        r, b = gates(REF, code)
        show("reference", r)
        if r["score"] != 100:
            bad.append("reference is not 100")
        for name in ("do_nothing", "open_loop"):
            print(name)
            r, _ = gates(REF, (R / f"{name}.py").read_text())
            show(name, r)
            if any(
                g["pass"] for g in r["gates"] if g["id"] in ("G3", "G4", "G5", "G6")
            ):
                bad.append(f"{name} scored flight points")
        print("STOCK hopper")
        stock = dict(REF, gnss="sparkfun-gps-rtk-sma-16481")
        r, b = gates(stock, code)
        show("stock", r)
        if r["gates"][0]["pass"]:
            bad.append("stock hopper passes G1")
        for slot, part in (
            ("battery", "gnb-3000mah-4s-100c-xt60"),
            ("esc", "tmotor-alpha-60a-6s"),
            ("esc", "bluerobotics-basic-esc"),
            ("esc", "hobbywing-seaking-60a-v3-1"),
        ):
            print("DISTRACTOR", slot, part)
            r, _ = gates(dict(REF, **{slot: part}, battery_x_mm=65.0), code)
            show(part, r)
            if r["score"] > 10:
                bad.append(f"{part} passes G2")
    else:
        scores = []
        for i in range(extra):
            b = D.apply(REF)
            b.static = None
            ends = []
            r = verify.run_gates(
                b,
                code,
                7000 + 100 * i,
                emit=lambda k, **f: ends.append(f) if k == "flight_end" else None,
            )
            scores.append(r["score"])
            for f in ends:
                m = f["metrics"]
                print(
                    "   ",
                    f["scenario"],
                    "dist",
                    m.get("land_dist_m"),
                    "td",
                    m.get("touchdown_m_s"),
                    "tilt",
                    m.get("tilt_max_deg"),
                    "zmax",
                    m.get("z_max_m"),
                    "soc",
                    m.get("soc_end_pct"),
                    "wall",
                    m.get("wall_s"),
                    flush=True,
                )
            fails = [
                f"{g['id']}: {g['reason'][:140]}"
                for g in r["gates"]
                if not g["pass"] and not g["reason"].startswith("not run")
            ]
            print(7000 + 100 * i, r["score"], fails, flush=True)
        print("scores", scores)
    print("BAD" if bad else "OK", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    _ = tempfile
    sys.exit(main())
