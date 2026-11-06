"""Partial credit, alongside the strict gates (which it never changes): a 0-1 mission score for one flight, from the metrics every flight already has, and a 0-100 score over the gates with the
flights the strict gauntlet would have stopped before. Numbers in task.toml [partial]; formula in docs/LAYERS.md. A passing flight is 1.0, so the reference is exactly 100; a flight that never
lifted off is 0. Called after the fact (harness/first_test.py); nothing here reaches an agent."""

import math
from concurrent.futures import ThreadPoolExecutor

from dv import design as D
from dv import verify as V

P = D.TASK["partial"]
H = D.TASK["hard"]
M = D.TASK["mission"]
G = {g["id"]: g for g in D.TASK["gates"]}


def lifted(m):
    return (m.get("z_max_m") or 0.0) >= P["lift_m"]


def landed(m):
    """Came to the ground and stopped, upright, before the flight ran out. The loop only ends early at rest, on its side (the tilt rules that out) or after a dead controller's fall."""
    return (
        lifted(m)
        and (m.get("touchdown_m_s") or 0.0) > 0.0
        and (m.get("final_tilt_deg") or 0.0) <= M["tilt_max_deg"]
        and (m.get("t_end_s") or 1e9) < M["mission_s"] + 2.0
    )


def parts(m):
    """The five components of a flight, each 0-1."""
    if not lifted(m):
        return dict(liftoff=0.0, altitude=0.0, hold=0.0, landing=0.0, accuracy=0.0)
    z = m["z_max_m"]
    top = M["alt_m"] + M["alt_tol_m"]
    alt = min(1.0, z / (M["alt_m"] - M["alt_tol_m"]))
    if z > top:
        alt *= max(0.0, 1.0 - (z - top) / (P["overshoot_zero_x"] * M["alt_m"]))
    land = acc = 0.0
    if landed(m):
        over = m["touchdown_m_s"] / M["touchdown_m_s"]
        land = min(1.0, max(0.0, (P["touchdown_zero_x"] - over) / (P["touchdown_zero_x"] - 1.0)))  # fmt: skip
        miss = m["land_dist_m"] / M["land_radius_m"]
        acc = min(1.0, max(0.0, (P["miss_zero_x"] - miss) / (P["miss_zero_x"] - 1.0)))
    return dict(
        liftoff=1.0,
        altitude=alt,
        hold=min(1.0, (m.get("hold_s") or 0.0) / M["hold_s"]),
        landing=land,
        accuracy=acc,
    )


def mission_score(m, passed):
    """0-1 for one flight. A passing flight is 1.0; a failed one is capped below it (it can fail on something the components do not see, such as a late controller)."""
    if passed:
        return 1.0
    w = P["weights"]
    s = sum(w[k] * v for k, v in parts(m).items())
    over = ((m.get("tilt_max_deg") or 0.0) - M["tilt_max_deg"]) / M["tilt_max_deg"]
    if over > 0:  # past the limit, down to tilt_floor at twice the limit
        s *= max(P["tilt_floor"], 1.0 - over * (1.0 - P["tilt_floor"]))
    return round(min(s, P["fail_cap"]), 4)


def rule_fractions(b):
    """(G1, G2) fractions. A design with unchosen slots, or that does not build, gets neither: cost and mass rules pass trivially on an empty deck."""
    if b.missing or not b.xml:
        return 0.0, 0.0
    rules = D.static_violations(b)
    checks = (b.l1 or {}).get("checks", [])
    return (
        sum(c["ok"] for c in rules) / len(rules),
        sum(c["ok"] for c in checks) / len(checks) if checks else 0.0,
    )


def _gate(gid, fraction, why, **extra):
    pts = G[gid]["points"]
    return dict(id=gid, name=G[gid]["name"], points=round(pts * fraction, 2), max_points=pts, fraction=round(fraction, 4), reason=why, **extra)  # fmt: skip


def hold_quality(telemetry):
    """(rms error from the target height, standard deviation of height) over the true height, from the first to the last 2 Hz sample in the band. (None, None) if it never was in the band."""
    zs = [row["z"] for row in telemetry]
    idx = [i for i, z in enumerate(zs) if abs(z - M["alt_m"]) <= M["alt_tol_m"]]
    if not idx:
        return None, None
    w = zs[idx[0] : idx[-1] + 1]
    mean = sum(w) / len(w)
    return (
        math.sqrt(sum((z - M["alt_m"]) ** 2 for z in w) / len(w)),
        math.sqrt(sum((z - mean) ** 2 for z in w) / len(w)),
    )


def hard_mission(m, telemetry):
    """0-1 for one flight of the hidden exam: the partial parts, each scaled by a continuous quality that is below 1 for any real flight, so no flight reaches 1.0."""
    p = parts(m)
    s = H["scales"]
    rms, std = hold_quality(telemetry)
    q = dict(
        liftoff=1.0,
        altitude=0.0 if rms is None else math.exp(-rms / s["alt_rms_m"]),
        hold=0.0 if std is None else math.exp(-std / s["hold_std_m"]),
        landing=math.exp(-(m.get("touchdown_m_s") or 0.0) / s["touchdown_m_s"]),
        accuracy=math.exp(-(m.get("land_dist_m") or 0.0) / s["land_dist_m"]),
    )
    w = P["weights"]
    tilt = m.get("tilt_max_deg") or 0.0
    total = sum(w[k] * p[k] * q[k] for k in p) * math.exp(-tilt / s["tilt_deg"])
    over = (tilt - M["tilt_max_deg"]) / M["tilt_max_deg"]
    if over > 0:
        total *= max(P["tilt_floor"], 1.0 - over * (1.0 - P["tilt_floor"]))
    return round(total, 4)


def _flight(b, code, scenario, seed, scatter=None, hard=False):
    spec = V.make_spec(b, code, scenario, seed, scatter=scatter)
    if hard and scenario in H["scenarios"]:
        spec["scenario_params"] = H["scenarios"][scenario]
    r = V.fly(spec, b)
    m = r["metrics"]
    mission = hard_mission(m, r["telemetry"]) if hard else mission_score(m, r["pass"])
    out = dict(seed=seed, scenario=scenario, **{"pass": r["pass"]}, mission=mission, parts={k: round(v, 3) for k, v in parts(m).items()}, reason=r["reason"][:160], lifted=lifted(m))  # fmt: skip
    if hard:
        rms, std = hold_quality(r["telemetry"])
        out.update(alt_rms_m=rms and round(rms, 3), hold_std_m=std and round(std, 3), touchdown_m_s=m.get("touchdown_m_s"), land_dist_m=m.get("land_dist_m"), tilt_max_deg=m.get("tilt_max_deg"))  # fmt: skip
    return out


def perfect():
    """What score() gives a state whose strict gates all passed on the same seed: every rule met and every flight 1.0."""
    gates = [_gate(g, 1.0, "strict gate passed") for g in G]
    return dict(partial_score=100.0, partial_gates=gates)


def score(b, code, base_seed, hard=False):
    """-> {partial_score, partial_gates}, or with hard {hard_score, hard_gates} (harder conditions, continuous quality). L0 and L1 failures still stop before the simulator, so an over-budget design
    earns no flight points. After G3 the other flights fly whether or not G3 passed, if the design lifted off in G3; otherwise they are 0 unflown."""
    f1, f2 = rule_fractions(b)
    gates = [
        _gate("G1", f1, "fraction of the static rules passed"),
        _gate("G2", f2, "fraction of the L1 margins met"),
    ]
    seeds = V.seeds_for(base_seed)
    why = None
    if V.layers(b):
        why = f"not flown: {V.layers(b)['gate']} failed"
    elif not code:
        why = "not flown: no controller"
    nom = None
    if not why:
        nom = _flight(b, code, "nominal", seeds["nominal"], hard=hard)
        gates.append(_gate("G3", nom["mission"], nom["reason"], flight=nom))
        if not nom["lifted"]:
            why = "not flown: did not lift off in G3"
    if why:
        gates += [_gate(g, 0.0, why) for g in G if g not in {x["id"] for x in gates}]
    else:
        robust = H["robust"] if hard else D.TASK["robust"]
        jobs = [("wind", seeds["wind"], None), ("rain", seeds["rain"], None)]
        jobs += [("robust", s, robust) for s in seeds["robust"]]
        with ThreadPoolExecutor(len(jobs)) as ex:
            fl = list(ex.map(lambda j: _flight(b, code, *j, hard=hard), jobs))
        gates.append(_gate("G4", fl[0]["mission"], fl[0]["reason"], flight=fl[0]))
        gates.append(_gate("G5", fl[1]["mission"], fl[1]["reason"], flight=fl[1]))
        rb = fl[2:]
        mean = sum(x["mission"] for x in rb) / len(rb)
        gates.append(_gate("G6", mean, f"mean mission score over {len(rb)} seeds", flights=rb))  # fmt: skip
    name = "hard" if hard else "partial"
    return {f"{name}_score": round(sum(g["points"] for g in gates), 1), f"{name}_gates": gates}  # fmt: skip
