"""Monte Carlo of the hopper-land scenario families: every flight draws, from its seed, the hopper's physical scatter (fan thrust +-5 %, mass +-3 %, CG +-4 mm, servo stiffness / backlash, sensor noise,
gyro / accelerometer offsets, battery charge: examples/hopper/hop.draw_hidden, unchanged), the wind (speed 0.65 to 1 x the scenario's, direction +-15 deg), the gust (amplitude 0.7 to 1 x, peak time +-3 s)
and the terrain (new fBm detail / crest or cliff phases, same pad positions).

    python examples/hopper-land/landmc.py --family ridge --train 40 --test 100 --workers 8 --out /tmp/mc-ridge.json

Protocol (the same as examples/hopper/mc.py):
  1. TRAINING seeds (1000...) pick the tunable(s) of the flight software from `--cands` (JSON list of parameter overrides): here `k_dob`, the gain of the wind-push feed-forward. Nothing else is tuned.
  2. TEST seeds (21000...) are then flown ONCE with the pick; success rate (Wilson 95 % interval), touchdown speed / distance quantiles and the failure causes are reported as they came out.
  3. The envelope: the same test seeds (the first --sweep-n) with one environment knob pushed (wind x 0.5 ... 2, gust, rain rate, site altitude, deck motion), pick unchanged.
A flight succeeds when it lands (no abort, no fault) with every monitor of the scenario files: touchdown speed < 0.5 m/s, touchdown tilt < 5 deg (relative to the deck on a deck), within 0.5 m of the
pad centre (the pad is a 2 m radius disc), upright at the end, battery above 18 V and 15 % charge left, roll / pitch vane command at the limit for < 10 % of the powered time, feet at least 1 m above the
terrain whenever away from the pads, full-throttle thrust / weight >= 1.15 at the site's density."""

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "examples" / "hopper"))
sys.path.insert(0, str(HERE))

import landhop
import landing
from mc import wilson  # examples/hopper/mc.py (the original, unchanged)

MAX_DIST, MAX_SPEED, MAX_TILT, MAX_SAT, MIN_V, MIN_SOC, MIN_CLR = (
    0.5,
    0.5,
    5.0,
    0.10,
    18.0,
    0.15,
    1.0,
)
TRAIN_BASE, TEST_BASE = 1000, 21000
SCRATCH = Path(os.environ.get("DVSIM_MC_TMP", "/tmp/dvsim-hopperland-mc"))
SCEN = {
    "mesa_down": "mesa/down.scenario.toml",
    "mesa_up": "mesa/up.scenario.toml",
    "ridge": "ridge/summit.scenario.toml",
    "ledge": "ledge/lee.scenario.toml",
    "rain25": "rain/rain25.scenario.toml",
    "rain100": "rain/rain100.scenario.toml",
    "deck": "deck/scripted.scenario.toml",
    "osv": "osv/underway.scenario.toml",
}
# The envelope sweeps: (label, overrides) per family; `wind_x` / `gust_x` multiply the scenario's own numbers (resolved in `one`).
SWEEPS = {
    "mesa_down": [("wind x 0.5", dict(wind_x=0.5)), ("wind x 1.5", dict(wind_x=1.5)), ("wind x 2", dict(wind_x=2.0)), ("site 4500 m", dict(origin_alt=4500.0)),
                  ("site 5500 m", dict(origin_alt=5500.0)), ("site 6500 m", dict(origin_alt=6500.0)), ("site 7500 m", dict(origin_alt=7500.0))],
    "mesa_up": [("wind x 1.5", dict(wind_x=1.5)), ("wind x 2", dict(wind_x=2.0)), ("site 5500 m", dict(origin_alt=5500.0))],
    "ridge": [("wind x 0.5", dict(wind_x=0.5)), ("wind x 1.5", dict(wind_x=1.5)), ("wind x 2", dict(wind_x=2.0)), ("gust x 1.5", dict(gust_x=1.5)), ("gust x 2", dict(gust_x=2.0)),
              ("no wind-push feed-forward (k_dob 0)", dict(k_dob=0.0))],
    "ledge": [("wind x 0.5", dict(wind_x=0.5)), ("wind x 1.5", dict(wind_x=1.5)), ("wind x 2", dict(wind_x=2.0)), ("gust x 2", dict(gust_x=2.0)), ("no wind-push feed-forward (k_dob 0)", dict(k_dob=0.0))],
    "rain25": [("rain 50 mm/h", dict(rain_mmh=50.0)), ("rain 150 mm/h", dict(rain_mmh=150.0)), ("wind x 2", dict(wind_x=2.0))],
    "rain100": [("rain 150 mm/h", dict(rain_mmh=150.0)), ("wind x 2", dict(wind_x=2.0))],
    "deck": [("heave 0.6 m", dict(deck_heave_m=0.6)), ("heave 1.0 m", dict(deck_heave_m=1.0)), ("roll 4 deg, pitch 2 deg", dict(deck_roll_deg=4.0, deck_pitch_deg=2.0)),
             ("drift 2 m/s", dict(deck_vy_ms=2.0)), ("datalink 0.3 s late", dict(deck_latency_s=0.3)), ("wind 5 m/s", dict(wind_ms=5.0))],
    "osv": [("Hs 0.25 m", dict(hs_m=0.25)), ("Hs 0.5 m", dict(hs_m=0.5)), ("Hs 1.0 m", dict(hs_m=1.0)), ("Hs 1.5 m", dict(hs_m=1.5)), ("beam seas, Hs 0.5 m", dict(wave_dir_deg=90.0, hs_m=0.5)),
            ("vessel stopped, Hs 0.5 m", dict(ship_speed_ms=0.0, hs_m=0.5)), ("vessel stopped, Hs 1.0 m", dict(ship_speed_ms=0.0, hs_m=1.0)), ("vessel 1 m/s, Hs 0.5 m", dict(ship_speed_ms=1.0, hs_m=0.5)),
            ("vessel 3.5 m/s, Hs 0.5 m", dict(ship_speed_ms=3.5, hs_m=0.5)), ("no quiet-window wait, Hs 0.5 m", dict(use_window=0.0, hs_m=0.5)),
            ("no deck-rise hold, Hs 0.5 m", dict(deck_vz_hold=99.0, hs_m=0.5)), ("deck link 0.3 s late, Hs 0.5 m", dict(ship_link_latency_s=0.3, hs_m=0.5)), ("wind x 2, Hs 0.5 m", dict(wind_x=2.0, hs_m=0.5))],
}  # fmt: skip


def one(args):
    """One flight -> dict of the metrics (picklable, for the pool)."""
    seed, over, fam = args
    o = dict(over)
    wx, gx = o.pop("wind_x", 1.0), o.pop("gust_x", 1.0)
    sc = HERE / SCEN[fam]
    base = landing.resolve(sc)
    o["seed"] = int(seed)
    o.setdefault("terrain_seed", int(seed))
    o["wind_ms"] = o.get("wind_ms", base["wind_ms"] * wx)
    o["gust_ms"] = o.get("gust_ms", base["gust_ms"] * gx)
    p = landing.resolve(sc, o)
    out = SCRATCH / f"w{os.getpid()}"
    model, h, info = landhop.prepare(seed, p, out)
    r = landhop.fly(seed, p, model, h, info, None, log=False, telemetry=False)
    res = r["res"]
    wmax = max((row[landhop.COLS.index("wind_ms")] for row in r["rows"]), default=0.0)
    return dict(
        seed=int(seed), landed=int(res["landed"]), aborted=int(bool(res["abort"]) or res["state"] == "FAULT"), td_speed=float(res["td_speed"]), td_tilt=float(res["td_tilt"]),
        td_dist=float(res["td_dist"]), final_tilt=float(res["final_tilt"]), sat=float(res["sat_frac"]), v_min=float(res["v_min"]), soc_min=float(res["soc_min"]),
        clr_min=float(res["clr_min"]), margin=float(res["margin"]), diverged=int(res["diverged"]), wind=float(np.hypot(h["wind"][0], h["wind"][1])), wind_peak=float(wmax),
        state=res["state"], abort=res["abort"], t_end=float(res["t_end"]),
    )  # fmt: skip


def ok(r):
    return bool(
        r["landed"] and not r["aborted"] and r["td_speed"] < MAX_SPEED and r["td_tilt"] < MAX_TILT and r["td_dist"] < MAX_DIST and r["final_tilt"] < MAX_TILT
        and r["sat"] < MAX_SAT and r["v_min"] >= MIN_V and r["soc_min"] >= MIN_SOC and r["clr_min"] >= MIN_CLR and not r["diverged"]
    )  # fmt: skip


def why(r):
    if r["aborted"]:
        return "abort:" + (r["abort"] or r["state"])
    if not r["landed"]:
        return "no landing"
    return (
        "speed" if r["td_speed"] >= MAX_SPEED else "dist" if r["td_dist"] >= MAX_DIST else "tilt" if max(r["td_tilt"], r["final_tilt"]) >= MAX_TILT
        else "vane saturation" if r["sat"] >= MAX_SAT else "battery" if (r["v_min"] < MIN_V or r["soc_min"] < MIN_SOC) else "terrain clearance" if r["clr_min"] < MIN_CLR else "diverged"
    )  # fmt: skip


def summarize(results):
    n = len(results)
    if n == 0:
        raise ValueError("no results")
    sp = np.array([r["td_speed"] if r["landed"] else np.nan for r in results])
    di = np.array([r["td_dist"] if r["landed"] else np.nan for r in results])
    q = lambda a, p: float(np.nanpercentile(a, p)) if np.isfinite(a).any() else None
    fails = {}
    for r in results:
        if not ok(r):
            fails[why(r)] = fails.get(why(r), 0) + 1
    k = sum(ok(r) for r in results)
    lo, hi = wilson(k, n)
    return dict(
        n=n, k=k, success=k / n, wilson=[lo, hi], landed=sum(1 for r in results if r["landed"] and not r["aborted"]) / n, aborted=sum(r["aborted"] for r in results) / n, failures=fails,
        speed_ms=dict(p50=q(sp, 50), p90=q(sp, 90), p99=q(sp, 99)), dist_m=dict(p50=q(di, 50), p90=q(di, 90), p99=q(di, 99)),
        wind_peak_p50=float(np.median([r["wind_peak"] for r in results])),
    )  # fmt: skip


def fmt(tag, s):
    f = lambda v: "  -  " if v is None else f"{v:.2f}"
    sp, di = s["speed_ms"], s["dist_m"]
    return (
        f"{tag:<38} n {s['n']:4d}  success {100 * s['success']:5.1f}% [{100 * s['wilson'][0]:.1f}, {100 * s['wilson'][1]:.1f}]  abort {100 * s['aborted']:4.1f}%  "
        f"speed p50/p90/p99 {f(sp['p50'])}/{f(sp['p90'])}/{f(sp['p99'])} m/s  dist {f(di['p50'])}/{f(di['p90'])}/{f(di['p99'])} m  wind peak p50 {s['wind_peak_p50']:.1f}  fails {s['failures']}"
    )


def run(pool, seeds, over, fam):
    return pool.map(one, [(s, over, fam) for s in seeds], chunksize=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--family", required=True, choices=sorted(SCEN))
    ap.add_argument(
        "--train", type=int, default=40, help="training seeds per candidate"
    )
    ap.add_argument("--test", type=int, default=100, help="report seeds")
    ap.add_argument("--workers", type=int, default=landing.nparams.default_workers())
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--pick",
        default=None,
        help="JSON of parameter overrides: skip the training and use these",
    )
    ap.add_argument(
        "--cands",
        default="[{}]",
        help="JSON list of parameter-override dicts tried on the training seeds",
    )
    ap.add_argument("--test-base", type=int, default=TEST_BASE)
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument(
        "--sweep-n",
        type=int,
        default=40,
        help="report seeds (the first n) per envelope row",
    )
    ap.add_argument(
        "--over",
        default="{}",
        help="JSON of scenario parameter overrides for every flight",
    )
    a = ap.parse_args(argv)
    base = json.loads(a.over)
    base.setdefault("disperse", True)
    base.setdefault("wind_fixed", False)
    t0 = time.time()
    out = {"protocol": __doc__.split("\n\n")[1], "family": a.family, "over": base,
           "criteria": dict(speed=MAX_SPEED, tilt=MAX_TILT, dist=MAX_DIST, clearance=MIN_CLR)}  # fmt: skip
    with mp.get_context("spawn").Pool(a.workers) as pool:
        train, cands = {}, json.loads(a.cands)
        if a.pick is None and len(cands) > 1:
            tr = range(TRAIN_BASE, TRAIN_BASE + a.train)
            for c in cands:
                s = summarize(run(pool, tr, {**base, **c}, a.family))
                train[json.dumps(c, sort_keys=True)] = s
                sys.stdout.write(fmt(f"train {json.dumps(c)}", s) + "\n")
                sys.stdout.flush()
            pick = max(
                cands,
                key=lambda c: (
                    train[json.dumps(c, sort_keys=True)]["success"],
                    -(train[json.dumps(c, sort_keys=True)]["speed_ms"]["p50"] or 99),
                ),
            )
        else:
            pick = (
                json.loads(a.pick)
                if a.pick is not None
                else (cands[0] if cands else {})
            )
        out["train"], out["pick"] = train, pick
        sys.stdout.write(f"chosen {json.dumps(pick)}\n")
        seeds = list(range(a.test_base, a.test_base + a.test))
        res = run(pool, seeds, {**base, **pick}, a.family)
        out["test"], out["test_flights"] = summarize(res), res
        sys.stdout.write(fmt("TEST (report seeds)", out["test"]) + "\n")
        sys.stdout.flush()
        out["sweep"] = {}
        if not a.no_sweep:
            for tag, ov in SWEEPS[a.family]:
                s = summarize(
                    run(pool, seeds[: a.sweep_n], {**base, **pick, **ov}, a.family)
                )
                out["sweep"][tag] = s
                sys.stdout.write(fmt(tag, s) + "\n")
                sys.stdout.flush()
    out["wall_s"] = round(time.time() - t0, 1)
    shutil.rmtree(SCRATCH, ignore_errors=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
    sys.stdout.write(f"wall {out['wall_s']} s\n")
    return out


if __name__ == "__main__":
    main()
