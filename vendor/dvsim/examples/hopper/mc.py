"""Monte Carlo of the hopper missions: fan thrust, mass, centre of mass, servo stiffness and backlash, IMU / barometer / lidar noise, gyro and accelerometer offsets, battery charge, wind
and gusts, all seeded (hop.draw_hidden).

    python examples/hopper/mc.py --scenario divert25 --train 40 --test 300 --workers 6 --out runs/hopper-mc-divert25.json

Protocol (the point of keeping two seed ranges apart):
  1. TRAINING seeds (1000...) pick the tunable(s) of the flight software from a short list of candidates (--cands, JSON list of parameter overrides). Nothing else is tuned on them.
  2. TEST seeds (5000...) are then flown ONCE with the pick, and the success rate and the touchdown speed / position distributions are reported as they came out.
  3. The same test seeds are re-flown with one scatter changed at a time (sensitivity table), pick unchanged.
A flight is a success when it lands (no abort) with all of the monitors of the scenario file: touchdown speed < 0.5 m/s, touchdown tilt < 5 deg, within 0.3 m of the pad, upright at the end,
battery above 18 V and 15 % charge left, vane command at the limit for < 10 % of the powered time (and, for the hover mission, within 0.5 m of the hover height).

Run it under the CPU lock with an outer timeout: flock /workspace/wt/dvsim/.cpu.lock timeout 3600 python examples/hopper/mc.py ..."""

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import hop
from dvsim import params as nparams

HERE = Path(__file__).parent
MAX_DIST, MAX_SPEED, MAX_TILT, MAX_FINAL_TILT, MAX_SAT = 0.3, 0.5, 5.0, 5.0, 0.10
MIN_V, MIN_SOC, MAX_HOVER_ERR = 18.0, 0.15, 0.5
TRAIN_BASE, TEST_BASE = 1000, 5000
SCRATCH = Path(os.environ.get("DVSIM_MC_TMP", "/tmp/dvsim-hopper-mc"))
SCEN = {
    "hop10": "hop10.scenario.toml",
    "divert25": "divert25.scenario.toml",
    "gusts": "gusts.scenario.toml",
}


def one(args):
    """One flight -> dict of the metrics (picklable, for the pool)."""
    seed, over, scen = args
    o = dict(over)
    o["seed"] = int(seed)
    sc = HERE / SCEN[scen]
    p = nparams.resolve(sc, o)
    out = SCRATCH / f"w{os.getpid()}"
    model, h = hop.prepare(seed, p, out)
    r = hop.fly(seed, p, model, h, None, log=False, telemetry=False)
    res, rows = r["res"], r["rows"]
    hover = max((row[13] for row in rows), default=0.0)
    return dict(
        seed=int(seed), landed=int(res["landed"]), aborted=int(bool(res["abort"]) or res["state"] == "FAULT"), td_speed=float(res["td_speed"]), td_tilt=float(res["td_tilt"]),
        td_dist=float(res["td_dist"]), final_tilt=float(res["final_tilt"]), sat=float(res["sat_frac"]), sat_yaw=float(res["sat_yaw_frac"]),v_min=float(res["v_min"]), soc_min=float(res["soc_min"]),
        hover_err=float(hover), diverged=int(res["diverged"]), wind=float(np.hypot(h["wind"][0], h["wind"][1])), state=res["state"], abort=res["abort"],
    )  # fmt: skip


def ok(r, hover=False):
    return bool(
        r["landed"] and not r["aborted"] and r["td_speed"] < MAX_SPEED and r["td_tilt"] < MAX_TILT and r["td_dist"] < MAX_DIST and r["final_tilt"] < MAX_FINAL_TILT
        and r["sat"] < MAX_SAT and r["v_min"] >= MIN_V and r["soc_min"] >= MIN_SOC and (not hover or r["hover_err"] < MAX_HOVER_ERR)
    )  # fmt: skip


def summarize(results, hover=False):
    """-> dict: n, success rate (all criteria), per-criterion rates, quantiles of touchdown speed / distance."""
    n = len(results)
    if n == 0:
        raise ValueError("no results")
    sp = np.array([r["td_speed"] if r["landed"] else np.nan for r in results])
    di = np.array([r["td_dist"] if r["landed"] else np.nan for r in results])
    q = lambda a, p: float(np.nanpercentile(a, p)) if np.isfinite(a).any() else None
    frac = lambda f: sum(1 for r in results if f(r)) / n
    fails = {}
    for r in results:
        if not ok(r, hover):
            why = (
                "abort:" + (r["abort"] or r["state"]) if r["aborted"] else "no landing" if not r["landed"]
                else "speed" if r["td_speed"] >= MAX_SPEED else "dist" if r["td_dist"] >= MAX_DIST else "tilt" if max(r["td_tilt"], r["final_tilt"]) >= MAX_TILT
                else "vane saturation" if r["sat"] >= MAX_SAT else "battery" if (r["v_min"] < MIN_V or r["soc_min"] < MIN_SOC) else "hover error"
            )  # fmt: skip
            fails[why] = fails.get(why, 0) + 1
    return dict(
        n=n, success=frac(lambda r: ok(r, hover)), landed=frac(lambda r: r["landed"] and not r["aborted"]), speed_lt=frac(lambda r: r["landed"] and r["td_speed"] < MAX_SPEED),
        tilt_lt=frac(lambda r: r["landed"] and max(r["td_tilt"], r["final_tilt"]) < MAX_TILT), dist_lt=frac(lambda r: r["landed"] and r["td_dist"] < MAX_DIST),
        sat_lt=frac(lambda r: r["sat"] < MAX_SAT), aborted=frac(lambda r: r["aborted"]), failures=fails,
        speed_ms=dict(p10=q(sp, 10), p50=q(sp, 50), p90=q(sp, 90), p99=q(sp, 99)), dist_m=dict(p10=q(di, 10), p50=q(di, 50), p90=q(di, 90), p99=q(di, 99)),
    )  # fmt: skip


def wilson(k, n, z=1.96):
    """95 % Wilson interval of a success count."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def run(pool, seeds, over, scen):
    return pool.map(one, [(s, over, scen) for s in seeds], chunksize=2)


def fmt(tag, s):
    sp, di = s["speed_ms"], s["dist_m"]
    lo, hi = wilson(round(s["success"] * s["n"]), s["n"])
    f = lambda v: "  -  " if v is None else f"{v:.2f}"
    return (
        f"{tag:<34} n {s['n']:4d}  success {100 * s['success']:5.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]  v<{MAX_SPEED} {100 * s['speed_lt']:5.1f}%  tilt<{MAX_TILT:.0f} {100 * s['tilt_lt']:5.1f}%  "
        f"dist<{MAX_DIST} {100 * s['dist_lt']:5.1f}%  abort {100 * s['aborted']:4.1f}%  speed p50/p90/p99 {f(sp['p50'])}/{f(sp['p90'])}/{f(sp['p99'])}  dist p50/p90/p99 {f(di['p50'])}/{f(di['p90'])}/{f(di['p99'])} m  fails {s['failures']}"
    )


SWEEP = [
    ("thrust_tol", (0.0, 0.10), "fan thrust +-{:.0%}"),
    ("mass_tol", (0.0, 0.06), "mass +-{:.0%}"),
    ("cg_tol_m", (0.0, 0.008), "centre of mass +-{:.3f} m"),
    ("servo_p_drop", (0.0, 0.6), "servo stiffness drop up to {:.0%}"),
    ("backlash_deg", (0.0, 1.5), "servo backlash up to {:.1f} deg"),
    ("noise_extra", (0.0, 1.5), "sensor noise +{:.0%} (upper)"),
    ("wind_ms", (0.0, 3.0, 5.0), "wind up to {:.0f} m/s"),
    ("gnss_rtk", (False,), "GNSS: NEO-M8N instead of RTK"),
    ("fault_servo", (True,), "one servo degraded at t_fault"),
    ("fault_batt_v", (2.0,), "battery open-circuit step -{:.0f} V"),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenario", default="divert25", choices=sorted(SCEN))
    ap.add_argument(
        "--train", type=int, default=40, help="training seeds per candidate"
    )
    ap.add_argument("--test", type=int, default=300, help="report seeds")
    ap.add_argument(
        "--workers",
        type=int,
        default=nparams.default_workers(),
        help="processes (default: every core, bounded by free memory)",
    )
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
    ap.add_argument(
        "--test-base", type=int, default=TEST_BASE, help="first report seed"
    )
    ap.add_argument(
        "--no-sweep", action="store_true", help="skip the sensitivity table"
    )
    ap.add_argument(
        "--sweep-n",
        type=int,
        default=100,
        help="report seeds (the first n) per sensitivity row",
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
    base.setdefault("wind_dir_deg", -1.0)
    if a.workers < 1:
        raise SystemExit("--workers must be >= 1")
    hover = a.scenario == "hop10"
    t0 = time.time()
    out = {
        "protocol": __doc__.split("\n\n")[1],
        "scenario": a.scenario,
        "over": base,
        "criteria": dict(speed=MAX_SPEED, tilt=MAX_TILT, dist=MAX_DIST),
    }
    with mp.get_context("spawn").Pool(a.workers) as pool:
        train = {}
        cands = json.loads(a.cands)
        if a.pick is None:
            tr_seeds = range(TRAIN_BASE, TRAIN_BASE + a.train)
            for c in cands:
                s = summarize(run(pool, tr_seeds, {**base, **c}, a.scenario), hover)
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
            pick = json.loads(a.pick)
        out["train"], out["pick"] = train, pick
        sys.stdout.write(f"chosen {json.dumps(pick)}\n")
        te_seeds = list(range(a.test_base, a.test_base + a.test))
        res = run(pool, te_seeds, {**base, **pick}, a.scenario)
        out["test"] = summarize(res, hover)
        out["test_flights"] = res
        sys.stdout.write(fmt("TEST (report seeds)", out["test"]) + "\n")
        sys.stdout.flush()
        out["sweep"] = {}
        if not a.no_sweep:
            for key, vals, tag in SWEEP:
                for v in vals:
                    s = summarize(
                        run(
                            pool,
                            te_seeds[: a.sweep_n],
                            {**base, **pick, key: v},
                            a.scenario,
                        ),
                        hover,
                    )
                    out["sweep"][f"{key}={v}"] = s
                    sys.stdout.write(
                        fmt(tag.format(v) if "{" in tag else tag, s) + "\n"
                    )
                    sys.stdout.flush()
    out["wall_s"] = round(time.time() - t0, 1)
    shutil.rmtree(SCRATCH, ignore_errors=True)  # only this tool's own scene files
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
    sys.stdout.write(f"wall {out['wall_s']} s\n")
    return out


if __name__ == "__main__":
    main()
