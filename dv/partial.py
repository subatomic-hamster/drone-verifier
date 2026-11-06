"""Partial credit, alongside the strict gates (which it never changes): a 0-1 mission score for one flight, from the metrics every flight already has, and a 0-100 score over the gates with the
flights the strict gauntlet would have stopped before. Numbers in task.toml [partial]; formula in docs/LAYERS.md. A passing flight is 1.0, so the reference is exactly 100; a flight that never
lifted off is 0. Called after the fact (harness/first_test.py); nothing here reaches an agent."""

from concurrent.futures import ThreadPoolExecutor

from dv import design as D
from dv import verify as V

P = D.TASK["partial"]
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


def _flight(b, code, scenario, seed, scatter=None):
    r = V.fly(V.make_spec(b, code, scenario, seed, scatter=scatter), b)
    m = r["metrics"]
    return dict(seed=seed, scenario=scenario, **{"pass": r["pass"]}, mission=mission_score(m, r["pass"]), parts={k: round(v, 3) for k, v in parts(m).items()}, reason=r["reason"][:160], lifted=lifted(m))  # fmt: skip


def score(b, code, base_seed):
    """-> {partial_score, partial_gates}. L0 and L1 failures still stop before the simulator, so an over-budget design earns no flight points. After G3 the other flights fly whether or not G3
    passed, if the design lifted off in G3; otherwise they are 0 unflown."""
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
        nom = _flight(b, code, "nominal", seeds["nominal"])
        gates.append(_gate("G3", nom["mission"], nom["reason"], flight=nom))
        if not nom["lifted"]:
            why = "not flown: did not lift off in G3"
    if why:
        gates += [_gate(g, 0.0, why) for g in G if g not in {x["id"] for x in gates}]
    else:
        robust = D.TASK["robust"]
        jobs = [("wind", seeds["wind"], None), ("rain", seeds["rain"], None)]
        jobs += [("robust", s, robust) for s in seeds["robust"]]
        with ThreadPoolExecutor(len(jobs)) as ex:
            fl = list(ex.map(lambda j: _flight(b, code, *j), jobs))
        gates.append(_gate("G4", fl[0]["mission"], fl[0]["reason"], flight=fl[0]))
        gates.append(_gate("G5", fl[1]["mission"], fl[1]["reason"], flight=fl[1]))
        rb = fl[2:]
        mean = sum(x["mission"] for x in rb) / len(rb)
        gates.append(_gate("G6", mean, f"mean mission score over {len(rb)} seeds", flights=rb))  # fmt: skip
    return dict(partial_score=round(sum(g["points"] for g in gates), 1), partial_gates=gates)  # fmt: skip
