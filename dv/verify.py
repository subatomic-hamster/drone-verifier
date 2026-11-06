"""The gauntlet: G1-G6 in order, stop at the first failed gate. Flights are subprocesses bounded by the flock slots."""

import fcntl
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dv import design as D
from dv import flight as F

TASK = D.TASK
GATES = {g["id"]: g for g in TASK["gates"]}
SLOTS_MAX = 8


def slots_dir():
    return Path(
        os.environ.get("DV_SLOTS")
        or Path(os.environ.get("DV_RUNS", "runs")).parent / "slots"
    )


class slot:
    """One of the machine's flight slots (flock, shared by every session)."""

    def __enter__(self):
        d = slots_dir()
        d.mkdir(parents=True, exist_ok=True)
        while True:
            for i in range(SLOTS_MAX):
                f = open(d / f"slot{i}", "w")
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.f = f
                    return self
                except OSError:
                    f.close()
            time.sleep(0.1)

    def __exit__(self, *a):
        self.f.close()


def seeds_for(base):
    """The seeds of one verification, derived from the base seed."""
    n = TASK["robust"]["seeds"]
    return dict(
        nominal=base + 1,
        wind=base + 2,
        rain=base + 3,
        robust=[base + 10 + i for i in range(n)],
    )


def make_spec(b, code, scenario, seed, scatter=None, log=None):
    spec = dict(
        xml=None,
        scenario=scenario,
        seed=seed,
        scatter=scatter,
        scenario_params=TASK["scenarios"].get(
            scenario,
            {"wind_ms": TASK["robust"]["wind_ms"], "w20_ms": 0.0, "rain_mmh": 0.0},
        ),
        code=code,
        cfg=D.controller_cfg(b, getattr(b, "static", None)),
        mission=TASK["mission"],
        sandbox=TASK["sandbox"],
        soc_min_pct=TASK["limits"]["soc_min_pct"],
        log=log,
        **D.flight_inputs(b),
    )
    return spec


def fly(spec, b, emit=None, ticks=True):
    """One flight in a subprocess inside a slot. emit(kind, **fields) gets flight_start, flight_tick, flight_end. -> result dict"""
    wd = Path(tempfile.mkdtemp(prefix="dv-fly-"))
    (wd / "scene.xml").write_text(b.xml)
    spec = dict(spec, xml=str(wd / "scene.xml"))
    sf = wd / "spec.json"
    sf.write_text(json.dumps(spec))
    os.chmod(wd, 0o700)
    if emit:
        emit("flight_start", scenario=spec["scenario"], seed=spec["seed"], live=False)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    res = None
    err = tempfile.TemporaryFile()
    try:
        with slot():
            p = subprocess.Popen(
                [sys.executable, "-m", "dv.flight", str(sf)],
                stdout=subprocess.PIPE,
                stderr=err,
                text=True,
                env=env,
                cwd=Path(__file__).resolve().parent.parent,
            )
            for line in p.stdout:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if "result" in ev:
                    res = ev["result"]
                elif emit and ticks and ev.get("kind") == "flight_tick":
                    ev.pop("kind")
                    emit("flight_tick", **ev)
            p.wait(timeout=30)
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    if res is None:
        err.seek(0)
        tail = err.read()[-600:].decode(errors="replace")
        res = {
            "scenario": spec["scenario"],
            "seed": spec["seed"],
            "pass": False,
            "reason": f"the flight process failed: {tail}",
            "first_violation": None,
            "metrics": {},
            "telemetry": [],
        }
    if emit:
        emit(
            "flight_end",
            scenario=spec["scenario"],
            seed=spec["seed"],
            **{"pass": res["pass"]},
            reason=res["reason"],
            metrics=res["metrics"],
        )
    return res


def compress_log(spec):
    """Keep only the newest replay log, gzipped; the GUI replays it."""
    log = spec.get("log")
    if log and Path(log).exists():
        raw = Path(log).read_bytes()
        with gzip.open(str(log) + ".gz", "wb", compresslevel=6) as f:
            f.write(raw)
        Path(log).unlink()


def g1(b):
    bad = [c for c in D.static_violations(b) if not c["ok"]]
    if b.build_error and not bad:
        bad = [dict(message=f"the design does not build: {b.build_error}")]
    return dict(
        ok=not bad,
        reason="; ".join(c["message"] for c in bad)
        or "cost, mass, envelope and design rules within limits",
    )


def g2(b):
    st = getattr(b, "static", None) or F.static_thrust(b.xml)
    b.static = st
    tw = st["thrust_n"] / (b.props["mass"] * F.veh.G)
    lim = TASK["limits"]["tw_min"]
    pack = D.library().get(b.choices["battery"]).value("i_max_continuous")
    msgs = []
    if tw < lim:
        msgs.append(
            f"thrust to weight {tw:.2f} below {lim:g}: {st['thrust_n']:.1f} N at full duty for {b.props['mass']:.3f} kg, pack {st['vbat']:.1f} V at {st['i_batt_a']:.0f} A"
        )
    if pack and st["i_batt_a"] > pack:
        msgs.append(f"pack current {st['i_batt_a']:.0f} A over its {pack:g} A rating")
    return dict(
        ok=not msgs,
        reason="; ".join(msgs) or f"thrust to weight {tw:.2f}, {st['i_batt_a']:.0f} A",
        tw=tw,
    )


def run_gates(b, code, base_seed, emit=None, log_dir=None):
    """-> dict(gates, score, flights). Later gates are listed as not run."""
    seeds = seeds_for(base_seed)
    gates, score, flights = [], 0.0, 0
    stop = False

    def done(gid, ok, reason, points=None):
        nonlocal score, stop
        g = GATES[gid]
        pts = (g["points"] if ok else 0) if points is None else points
        score += pts
        gates.append(
            dict(
                id=gid,
                name=g["name"],
                **{"pass": bool(ok)},
                points=pts,
                max_points=g["points"],
                reason=reason,
            )
        )
        stop = not ok

    def flight(gid, scenario):
        nonlocal flights
        log = (
            str(Path(log_dir) / "flight_last" / "replay.dvsimog")
            if log_dir and scenario == "nominal"
            else None
        )
        if log:
            Path(log).parent.mkdir(parents=True, exist_ok=True)
        spec = make_spec(b, code, scenario, seeds[scenario], log=log)
        r = fly(spec, b, emit)
        flights += 1
        compress_log(spec)
        done(
            gid,
            r["pass"],
            r["reason"] if r["pass"] else f"{scenario} flight failed: {r['reason']}",
        )
        return r

    r = g1(b)
    done("G1", r["ok"], r["reason"])
    if not stop:
        r = g2(b)
        done("G2", r["ok"], r["reason"])
    if not stop and not code:
        done("G3", False, "no controller: call write_controller")
    for gid, sc in (("G3", "nominal"), ("G4", "wind"), ("G5", "rain")):
        if not stop:
            flight(gid, sc)
    if not stop:
        specs = [
            make_spec(b, code, "robust", s, scatter=TASK["robust"])
            for s in seeds["robust"]
        ]
        with ThreadPoolExecutor(len(specs)) as ex:
            rs = list(ex.map(lambda s: fly(s, b, emit, ticks=s is specs[0]), specs))
        flights += len(rs)
        good = sum(x["pass"] for x in rs)
        pts = round(GATES["G6"]["points"] * good / len(rs), 1)
        bad = [x for x in rs if not x["pass"]]
        done(
            "G6",
            not bad,
            f"{good} of {len(rs)} seeds pass"
            + (f"; first failure: {bad[0]['reason']}" if bad else ""),
            points=pts,
        )
    seen = {g["id"] for g in gates}
    for g in TASK["gates"]:
        if g["id"] not in seen:
            gates.append(
                dict(
                    id=g["id"],
                    name=g["name"],
                    **{"pass": False},
                    points=0,
                    max_points=g["points"],
                    reason="not run: an earlier gate failed",
                )
            )
    return dict(gates=gates, score=round(score, 1), flights=flights)
