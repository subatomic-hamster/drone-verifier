"""Planted flaws: each design or controller has one known defect; clean designs must pass. Runs the gates on the held-out seed, through the same code agents use, on hfbox: `python tests/check_planted.py`.
Prints the table: flaw, flagged yes/no, layer, why; then the flagged fraction and the false positives. A flaw the verifier misses is a finding: do not weaken this list to make it pass."""

import json
import re
import sys
import time
from pathlib import Path

from dv import design as D
from dv import heldout
from dv import verify as V

R = Path(__file__).resolve().parent.parent / "reference"
REF = json.loads((R / "design.json").read_text())
CODE = (R / "controller.py").read_text()
SEED = heldout.heldout_seed()
LAYER = dict(G1="L0", G2="L1", G3="L2", G4="L2", G5="L2", G6="L3")


def sub(pattern, repl, code=CODE):
    """The controller with one line changed. Fails loudly if the reference moved."""
    out, n = re.subn(pattern, repl, code)
    assert n, pattern
    return out


HARDWARE = {
    "undersized ESC": dict(esc="tmotor-alpha-60a-6s"),
    "overloaded low-C pack": dict(battery="iflight-fullsend-6s1p-4000mah-liion-40t", esc="iflight-blitz-e80-single", battery_x_mm=75.0),
    "pack too small for the mission": dict(battery="tattu-1300mah-6s-100c-funfly-xt60"),
    "wrong pack voltage (4S on a 6S fan)": dict(battery="gnb-3000mah-4s-100c-xt60", battery_x_mm=65.0),
    "wrong pack voltage (6S on a 3S ESC)": dict(esc="hobbywing-seaking-60a-v3-1"),
    "connector mismatch": dict(battery="gnb-4000mah-6s-70c-tplug", esc="iflight-blitz-e80-single", battery_x_mm=75.0),
    "over budget": dict(gnss="sparkfun-gps-rtk-sma-16481"),
    "overweight": dict(battery="tattu-4500mah-6s-25c-xt60"),
    "under-rated regulator": dict(regulator="pololu-d24v10f5"),
}  # fmt: skip
CONTROLLER = {
    "no yaw control": sub(r"e\[2\] = -math\.atan2\(R\[1, 0\], R\[0, 0\]\)", "e[2] = 0.0", sub(r"tau\[2\] \+= self\.c\.fan\.yaw_torque_per_newton \* T", "tau[2] = 0.0")),
    "no altitude hold": sub(r"az = 2\.5 \* \(self\.zref - z\) \+ 3\.0 \* \(vref - vz\) \+ self\.iz", "az = 0.5"),
    "hard landing": sub(r"vref = -0\.8 if z > 1\.0 else -0\.45", "vref = -3.0"),
    # holds position loosely in still air, nothing against a steady push: low gains, no integrators
    "ignores wind": sub(r"tau\[:2\] \+= self\.itau", "pass", sub(r"axy = np\.clip\(1\.2 \* exy - 2\.0 \* s\.vel\[:2\] \+ self\.ixy, -2\.5, 2\.5\)", "axy = np.clip(0.3 * exy - 0.8 * s.vel[:2], -2.5, 2.5)")),
}  # fmt: skip
CLEAN = {
    "reference": {},
    "M9N GNSS, rocket legs": dict(gnss="matek-m9n-5883", legs="generic-rocket-landing-leg"),
    "R-Line pack": dict(battery="tattu-2200mah-6s-150c-rline5-xt60"),
    "M8N GNSS, VL53L1X range": dict(gnss="holybro-m8n-gps", range="pololu-vl53l1x-carrier"),
}  # fmt: skip


def run(choices, code):
    t = time.perf_counter()
    b = D.apply(choices)
    t_apply = time.perf_counter() - t
    t = time.perf_counter()
    bad = V.layers(b)
    t_layers = t_apply + time.perf_counter() - t
    r = V.run_gates(b, code, SEED)
    r["built"] = b
    fail = next((g for g in r["gates"] if not g["pass"]), None)
    return r, fail, t_layers, bad


def main():
    flags = 0
    n_flaws = len(HARDWARE) + len(CONTROLLER)
    times = []
    print(f"{'flaw':38s} {'flagged':7s} layer  gate  why")
    for name, delta in HARDWARE.items():
        r, fail, t_l, _ = run({**REF, **delta}, CODE)
        times.append(t_l)
        flags += fail is not None
        print(f"{name:38s} {'yes' if fail else 'NO':7s} {LAYER[fail['id']] if fail else '-':5s}  {fail['id'] if fail else '-':4s}  {fail['reason'][:120] if fail else 'score ' + str(r['score'])}", flush=True)  # fmt: skip
    for name, code in CONTROLLER.items():
        r, fail, t_l, _ = run(REF, code)
        flags += fail is not None
        print(f"{name:38s} {'yes' if fail else 'NO':7s} {LAYER[fail['id']] if fail else '-':5s}  {fail['id'] if fail else '-':4s}  {fail['reason'][:120] if fail else 'score ' + str(r['score'])}", flush=True)  # fmt: skip
    fp = 0
    print(f"{'clean design':38s} {'passes':7s} score")
    for name, delta in CLEAN.items():
        r, fail, t_l, _ = run({**REF, **delta}, CODE)
        fp += fail is not None
        b = r["built"]
        energy = [f["metrics"]["energy"] for f in r["flown"]]
        print(f"{name:38s} {'yes' if not fail else 'NO':7s} {r['score']:5g}  ${b.cost:.0f} {b.mass:.3f} kg L1 {b.l1['hover_throttle_pct']} %, {b.l1['flight_time_s']} s; error {heldout.prediction_error(b, r['flown'])}; energy ok {sum(e['ok'] for e in energy)}/{len(energy)}, first {energy[0]}; {(fail['id'] + ': ' + fail['reason'][:120]) if fail else ''}", flush=True)  # fmt: skip
        walls = [f["metrics"]["wall_s"] for f in r["flown"]]
    print(
        f"flagged {flags}/{n_flaws} = {flags / n_flaws:.2f}; false positives {fp}/{len(CLEAN)}"
    )
    print(f"L0+L1 wall (apply + checks), worst of {len(times)}: {max(times):.3f} s, best {min(times):.3f} s; one flight in the sim: {min(walls):.1f} s to {max(walls):.1f} s")  # fmt: skip
    return 0


sys.exit(main())
