"""Open ports, dependency hints and auto-wiring for `Design` (edit.py): `autowire` / `autowire_all` ops and `needs` hints.

An open power / signal port is one that no `[[wires]]` entry names. Auto-wiring finds the port that needs it, sizes the wire
from the DRC's worst-case current, takes the length from the port positions and pins the mating connectors. Every number below
is a design choice (named, with its reason), not physical data; the part files carry the physical ratings.
"""

import math
from collections import Counter

import numpy as np

from . import assembly as A
from . import pins as PN
from .assembly import Ctx
from .build import compute
from .drc import _n as _num
from .drc import _Net, _wires, _World, check
from .library import DesignError, types_ok

POWER, SIGNAL = "power", "signal"
WIRE_MARGIN = 1.2  # wire ampacity >= need x 1.2: the part files' i_max are rule-of-thumb charts with "no bundling /
#                    temperature-rise condition given", so a harness in a frame gets 20 % headroom (design margin)
MIN_NEED_A = {
    POWER: 2.0,
    SIGNAL: 1.0,
}  # A: floor when the DRC models no load on the wire (avionics, logic): sized for this
#                    much, so a signal lead comes out as a 26 AWG harness wire, not a 30 AWG hair (design assumption)
ROUTING_FACTOR = 1.3  # wire length = straight line between the two port positions x 1.3: a lead follows the frame, leaves
#                    slack for strain relief and service loops (design margin; no routing is solved)
LENGTH_QUANTUM = 0.005  # m: a length with no stock list is rounded up to 5 mm
MIN_LENGTH = 0.03  # m: shortest wire (strip + crimp allowance)
CONDUCTORS = {
    "motor-3phase": 3
}  # per port type; else 2 for power (supply + return, as the X500 file) and 1 for signal


def _split(ref):
    i, _, p = ref.partition(".")
    return i, (p or None)


def wired_ports(doc) -> set:
    """{(instance, port)} named by a wire end."""
    return {_split(w[e]) for w in doc.get("wires", []) for e in ("from", "to")}


def _roles_ok(ra, rb):
    return not (
        ra == rb and ra in ("source", "sink")
    )  # source-source / sink-sink cannot be one net


class _State:
    """Nets of the current document (drc's own `_Net`), so "fed" means what the DRC means."""

    def __init__(self, d, doc):
        self.ctx = Ctx(A.Assembly(doc), d.lib)
        self.net = _Net(self.ctx, _wires(self.ctx, []))
        self.direct = wired_ports(doc)
        self.wired_nodes = {
            self.net.node.find(w[k]) for w in self.net.wires for k in ("ka", "kb")
        }

    def fed(self, iid, port):
        return bool(self.net.sources((iid, port)))

    def loose(self, iid, port):  # drc FLOATING_PORT: no wire on the node
        return self.net.node.find((iid, port)) not in self.wired_nodes


# ---- candidates --------------------------------------------------------------------------------------------------------------
def _supplier(r):
    return r in ("source", "passthrough", None)


def _load(r):
    return r in ("sink", "passthrough", None)


def _candidates(d, st, ia, pa):
    """Open ports that `ia.pa` could be wired to, best tier first -> (candidates, rejected reasons).
    Same kind and same connector family (types_ok; a wire carries its own connectors, so genders need not complement).
    Power: tier 0 = exactly one of the two nets is fed (a source reaches it) and the roles complement (one side can supply,
    the other can load; passthrough-passthrough counts only here, when one side is fed: a PM02 feeding a PDB) = the unfed side
    "needs" the other; tier 1 = neither is fed yet.
    Signal: any port of another part (an ESC does not signal an ESC)."""
    kind, out, why = pa["kind"], [], []
    for iid, inst in st.ctx.insts.items():
        if iid == ia or (kind == SIGNAL and inst.part.id == st.ctx.insts[ia].part.id):
            continue
        for q in inst.part.ports:
            if q.get("kind") != kind or (iid, q["name"]) in st.direct:
                continue
            if not types_ok(pa.get("type", ""), q.get("type", "")):
                continue
            tier = 0
            if kind == POWER:
                ra, rq = pa.get("role"), q.get("role")
                if not _roles_ok(ra, rq):
                    continue
                if not ((_supplier(ra) and _load(rq)) or (_load(ra) and _supplier(rq))):
                    continue
                fa, fq = st.fed(ia, pa["name"]), st.fed(iid, q["name"])
                if fa and fq:
                    continue
                if ra == rq == "passthrough" and fa == fq:
                    continue  # a pass-through pair says nothing about direction unless one side is already fed (PM02 -> PDB)
                tier = 0 if fa != fq else 1
                src, snk = (pa, q) if (ra == "source" or rq == "sink") else (q, pa)
                hi, lo = _num(src.get("v_max")), _num(snk.get("v_max"))
                if hi and lo and lo < hi:
                    why.append(
                        f"{iid}.{q['name']}: v_max {lo:g} V below the source's {hi:g} V"
                    )
                    continue
            out.append((tier, iid, q))
    best = min((t for t, _, _ in out), default=None)
    return [(i, q) for t, i, q in out if t == best], why, best


def _signature(d, st, iid, q):
    p = st.ctx.insts[iid].part
    return (
        p.id,
        q.get("type"),
        q.get("gender"),
        q.get("role"),
        _num(q.get("i_max")),
        _num(q.get("v_max")),
    )


def resolve(d, st, ia, pa, ib=None, fed_only=False):
    """-> ("ok", (iid, port), note) | ("ambiguous", [candidate refs], note) | ("none", None, reason).
    fed_only: only tier-0 targets (a fed net that needs this port's unfed net); a tier-1 pairing is "none"."""
    cands, why, tier = _candidates(d, st, ia, pa)
    if fed_only and tier != 0:
        return "none", None, "no fed net needs it"
    if ib is not None:
        hit = [(i, q) for i, q in cands if (i, q["name"]) == ib]
        return (
            ("ok", ib, "named by the caller")
            if hit
            else (
                "none",
                None,
                f"{ib[0]}.{ib[1]} is not an open compatible {pa['kind']} port",
            )
        )
    if not cands:
        return (
            "none",
            None,
            "no open compatible port that needs it"
            + (f" ({'; '.join(why)})" if why else ""),
        )
    if len(cands) == 1:
        return (
            "ok",
            (cands[0][0], cands[0][1]["name"]),
            f"the only open compatible {pa['kind']} port ({'a fed net needs it' if tier == 0 else 'neither net is fed yet'})",
        )
    one = {c[0] for c in cands}
    sig = {_signature(d, st, i, q) for i, q in cands}
    if (
        pa["kind"] == POWER and len(one) == 1 and len(sig) == 1
    ):  # outlets of one part, electrically identical
        i, q = cands[0]
        others = ", ".join(f"{i}.{x['name']}" for _, x in cands[1:])
        return (
            "ok",
            (i, q["name"]),
            f"first of {len(cands)} identical free outlets of {i} (also free: {others})",
        )
    return (
        "ambiguous",
        [f"{i}.{q['name']}" for i, q in cands],
        f"{len(cands)} equally good open ports",
    )


# ---- connectors --------------------------------------------------------------------------------------------------------------
def pin_connector(d, doc, st, iid, port):
    """The mating connector of a wire end on `iid.port` (builder rule: library parts of category exactly `connector` that have
    a port of the same type and the opposite gender; unpinned, several fit -> CONNECTOR_AMBIGUOUS and the smallest id is used).
    Preference here, in order: (1) the connector pinned on a wire end of this port type + gender on another instance of the same
    part in this design (siblings: the four ESCs of a quad, the outlets of one PDB, use one), (2) the one pinned most often on that port type + gender in this design, (3) the builder's
    default, the smallest id. -> {"part", "rule", "alternatives"} or None (no gender, or no library connector)."""
    g = port.get("gender")
    if g not in ("male", "female"):
        return None
    opp = "female" if g == "male" else "male"
    ids = sorted(
        p.id
        for p, pt in d.lib.compatible(port["type"], g)
        if pt.get("gender") == opp and p.category == "connector"
    )
    if not ids:
        return None
    here, there = Counter(), Counter()
    for w in doc.get("wires", []):
        for end in ("from", "to"):
            pin = w.get("connectors", {}).get(end)
            wi, wp = _split(w[end])
            if pin not in ids or wi not in st.ctx.insts:
                continue
            q = st.ctx.insts[wi].part.port(wp)
            if (q.get("type"), q.get("gender")) == (port["type"], g):
                there[pin] += 1
                if st.ctx.insts[wi].part.id == st.ctx.insts[iid].part.id:
                    here[pin] += 1
    for cnt, rule in (
        (
            here,
            "pinned on a port of that type on the same part elsewhere in this design",
        ),
        (there, f"pinned most often on {port['type']} {g} ports in this design"),
    ):
        if cnt:
            best = max(cnt.values())
            return {
                "part": min(i for i in cnt if cnt[i] == best),
                "rule": rule,
                "alternatives": [i for i in ids if i not in cnt][:3],
            }
    return {
        "part": ids[0],
        "rule": "no precedent in this design: the library default (smallest id, as the builder picks)",
        "alternatives": ids[1:4],
    }


# ---- wire part, length -------------------------------------------------------------------------------------------------------
def _wire_parts(d):
    """Wire-category parts that end in two wire-lead power ports with a rating and a mass per metre -> [(kg/m, A, part)]."""
    out = []
    for p in d.lib.by_category("wire"):
        leads = [q for q in p.ports if q.get("type") == "wire-lead"]
        rated = [_num(q.get("i_max")) for q in leads]
        kgm = p.value("mass_per_m") or p.value("mass_per_m_copper")
        if len(leads) == 2 and all(rated) and kgm:
            out.append((kgm, min(rated), p))
    return sorted(out, key=lambda t: (t[0], t[2].id))


def choose_wire(d, need, kind, forced=None):
    """Lightest library wire whose ampacity >= need x WIRE_MARGIN. -> (part, reason text)."""
    pool = _wire_parts(d)
    if not pool:
        raise DesignError(
            "autowire: the library has no wire part with two wire-lead ports, a rating and a mass per metre"
        )
    eff = max(need or 0.0, MIN_NEED_A[kind])
    req = eff * WIRE_MARGIN
    basis = (
        f"worst-case {need:.1f} A (DRC electrical model)"
        if need
        else f"no modelled load on this wire: floor {MIN_NEED_A[kind]:g} A"
    )
    if forced:
        p = d._part(forced)
        amp = min((_num(q.get("i_max")) or 0.0 for q in p.ports[:2]), default=0.0)
        return (
            p,
            f"part {p.id} given by the caller ({amp:g} A vs {basis} x {WIRE_MARGIN:g} = {req:.1f} A)",
        )
    weak = []
    for kgm, amp, p in pool:
        if amp >= req:
            return p, (
                f"{basis}; x {WIRE_MARGIN:g} margin = {req:.1f} A; lightest wire rated >= that: {p.id} "
                f"({amp:g} A, {kgm * 1000:.1f} g/m)"
                + (f"; lighter but too weak: {', '.join(weak[-3:])}" if weak else "")
            )
        weak.append(f"{p.id} {amp:g} A")
    raise DesignError(
        f"autowire: no library wire is rated for {req:.1f} A ({basis} x {WIRE_MARGIN:g}); the strongest is {pool[-1][2].id} "
        f"at {max(a for _, a, _ in pool):g} A. Split the load over two wires or add a heavier wire part"
    )


def _stock(part):
    e = part.params.get("stock_lengths")
    v = e.get("value") if isinstance(e, dict) else None
    if isinstance(v, list) and v:
        return sorted(float(x) for x in v), "stock_lengths"
    if part.value("length_default") is not None:
        return [part.value("length_default")], "length_default"
    return None, None


def wire_length(part, dist):
    """Straight-line distance (m) -> (length m, derivation text). x ROUTING_FACTOR, then up to a stocked length when the wire
    part lists them (`stock_lengths` or `length_default`), else up to LENGTH_QUANTUM; never below MIN_LENGTH."""
    raw = max(dist * ROUTING_FACTOR, MIN_LENGTH)
    stock, key = _stock(part)
    fit = next((s for s in stock if s >= raw - 1e-12), None) if stock else None
    if fit is not None:
        n, how = fit, f"rounded up to the stocked length {fit:g} m ({part.id} {key})"
    else:
        n = math.ceil(raw / LENGTH_QUANTUM - 1e-9) * LENGTH_QUANTUM
        how = f"rounded up to {LENGTH_QUANTUM * 1000:g} mm" + (
            f" (longest stocked length {stock[-1]:g} m is shorter: cut to size)"
            if stock
            else ""
        )
    return round(n, 6), (
        f"autowire: straight line {dist:.4f} m between the port positions x routing factor {ROUTING_FACTOR:g} = {dist * ROUTING_FACTOR:.4f} m, "
        f"at least {MIN_LENGTH:g} m, {how}"
    )


def _world(d, doc):
    try:
        xml, _ = compute(A.Assembly(doc), d.lib.dir, d.lib.root, d.lib, route=False)
        return _World(xml, d.lib.root)
    except (DesignError, KeyError, ValueError, StopIteration):
        return None


def _port_pos(world, inst, port):
    if world is None or inst.id not in world.idx:
        return None
    T = world.bodies[world.idx[inst.id]]["T"]
    return T[:3, :3] @ np.array(port.get("pos", [0, 0, 0]), dtype=float) + T[:3, 3]


# ---- the ops -----------------------------------------------------------------------------------------------------------------
def _direction(st, ia, pa, ib, pb):
    """(from, to): the fed / supplying end first; a signal wire keeps the caller's order."""
    if pa["kind"] == POWER:
        fa, fb = st.fed(ia, pa["name"]), st.fed(ib, pb["name"])
        if fa != fb:
            return ((ia, pa), (ib, pb)) if fa else ((ib, pb), (ia, pa))
        if pa.get("role") == "sink" or pb.get("role") == "source":
            return (ib, pb), (ia, pa)
    return (ia, pa), (ib, pb)


def _probe(d, doc, st, world, f, t, kind, forced_len):
    """Append a provisional wire (lightest rated part, unrounded length) so the DRC can tell what current it carries."""
    (fi, fp), (ti, tp) = f, t
    pf = _port_pos(world, st.ctx.insts[fi], fp)
    pt = _port_pos(world, st.ctx.insts[ti], tp)
    if forced_len is None and (pf is None or pt is None):
        raise DesignError(
            f"autowire {fi}.{fp['name']} -> {ti}.{tp['name']}: the port positions are unknown (an end is offboard or the design "
            "does not build yet: mate the floating parts first); pass length = a provenance table in metres"
        )
    dist = None if forced_len is not None else float(((pf - pt) ** 2).sum() ** 0.5)
    probe = _wire_parts(d)
    probe = probe[0][2] if probe else None
    if probe is None:
        raise DesignError("autowire: the library has no usable wire part")
    name, n = f"{fi}.{fp['name']}--{ti}.{tp['name']}", 1
    names = {w.get("name") for w in doc.get("wires", [])}
    while name in names:
        n += 1
        name = f"{fi}.{fp['name']}--{ti}.{tp['name']}-{n}"
    spec = {
        "name": name,
        "from": f"{fi}.{fp['name']}",
        "to": f"{ti}.{tp['name']}",
        "part": probe.id,
        "length": forced_len
        or {
            "value": max(dist * ROUTING_FACTOR, MIN_LENGTH),
            "unit": "m",
            "fidelity": "estimated",
            "rationale": "autowire probe",
        },
    }
    n_c = CONDUCTORS.get(fp.get("type"), 2 if kind == POWER else 1)
    if n_c != 1:
        spec["conductors"] = n_c
    return spec, dist


def _lead(d, doc, lead) -> dict:
    """Resolve a port's `lead` = {length?, awg?} (each a param name of the part, or a provenance table) ->
    {inst, what, length (reference string / table or None), length_m, awg, wire (library wire part with that awg, or None)}."""
    iid, port = lead
    part = d._part(d._inst(doc, iid)["part"])
    spec = port["lead"] if isinstance(port["lead"], dict) else {}
    what = f"lead of {iid}.{port['name']}"

    def val(x, what_):
        if isinstance(x, str):
            if part.value(x) is None:
                raise DesignError(
                    f"{part.id} port {port['name']}: lead.{what_} names param '{x}', which the part does not have"
                )
            return part.value(x)
        return x.get("value") if isinstance(x, dict) else None

    ln, awg = spec.get("length"), spec.get("awg")
    n = val(awg, "awg") if awg is not None else None
    wire = next(
        (p for p in d.lib.by_category("wire") if n is not None and p.value("awg") == n),
        None,
    )
    return {
        "inst": iid,
        "what": what,
        "length": f"{iid}.{ln}" if isinstance(ln, str) else ln,
        "length_m": val(ln, "length") if ln is not None else None,
        "awg": n,
        "wire": wire.id if wire else None,
    }


def _finish(d, doc, st, rec, current, forced_part, forced_len):
    """Size the probe wire of `rec` in place: part, length, connectors; -> the reasons dict."""
    w, kind = rec["wire"], rec["kind"]
    pk, sus = current if current else (None, None)
    need = max(pk or 0.0, sus or 0.0) or None
    rated = None
    if (
        need is None and kind == POWER
    ):  # no ESC device yet: a phase lead carries up to the weaker end's rating
        r = [
            _num(p.get("i_max"))
            for _, p in (rec["f"], rec["t"])
            if p.get("type") == "motor-3phase"
        ]
        rated = min((x for x in r if x), default=None)
        need = rated
    lead = _lead(d, doc, rec["lead"]) if rec.get("lead") else None
    lead_part = forced_part is None and lead and lead["wire"]
    part, why_part = choose_wire(d, need, kind, forced_part or lead_part)
    if lead_part:
        why_part = f"{lead['what']} awg {lead['awg']:g}: {part.id}"
    if rated:
        why_part += f"; no ESC device yet, so sized for the motor-3phase port rating {rated:g} A (re-run autowire after auto_devices to size from the DRC model)"
    elif current is None and kind == POWER:
        why_part += "; the wire is in a net with no source yet, so no current is modelled: re-run autowire / swap the wire part after the source is wired"
    reasons = {"part": why_part}
    if lead:
        reasons["included_in"] = (
            f"{lead['what']}: the wire is a lead inside {lead['inst']}'s datasheet mass"
        )
    w["part"] = part.id
    if forced_len is None and lead and lead["length"] is not None:
        w["length"] = lead["length"]
        reasons["length"] = f"the length of {lead['what']}"
        if (
            lead["length_m"]
            and rec["dist"]
            and rec["dist"] * ROUTING_FACTOR > lead["length_m"]
        ):
            reasons["warning"] = (
                f"the lead ({lead['length_m']:g} m) is shorter than the routed run ({rec['dist'] * ROUTING_FACTOR:.3f} m): "
                "the rest of the run is not modelled"
            )
    elif forced_len is None:
        n, deriv = wire_length(part, rec["dist"])
        w["length"] = {
            "value": n,
            "unit": "m",
            "fidelity": "derived",
            "derivation": deriv,
        }
        reasons["length"] = deriv
    else:
        reasons["length"] = "given by the caller"
    if part.params.get("mass_per_connector") is None:
        pins = {}
        for end, (iid, p) in (("from", rec["f"]), ("to", rec["t"])):
            c = pin_connector(d, doc, st, iid, p)
            if c:
                pins[end] = c["part"]
                reasons[f"connector_{end}"] = f"{c['part']}: {c['rule']}" + (
                    f"; other fits: {', '.join(c['alternatives'])}"
                    if c["alternatives"]
                    else ""
                )
        if pins:
            w["connectors"] = pins
    lim = [
        (_num(p.get("i_max")), f"{i}.{p['name']}")
        for i, p in (rec["f"], rec["t"])
        if _num(p.get("i_max")) and "i_peak" not in p
    ]
    if lim and need and min(lim)[0] < need:
        reasons["warning"] = (
            f"port {min(lim)[1]} is rated {min(lim)[0]:g} A, below the {need:.1f} A this wire carries: the wire is not the limit, the DRC will flag it"
        )
    if pk is not None:
        reasons["current_A"] = {
            "peak": round(pk, 3),
            "sustained": None if sus is None else round(sus, 3),
        }
    return reasons


def autowire(d, doc, s) -> dict:
    """Op `autowire` {a, b?, kind?, part?, length?, included_in?}: wire the open port `a` to the port that needs it (or to `b`)."""
    return _run(
        d,
        doc,
        [(s["a"], s.get("b"))],
        s.get("kind"),
        s.get("part"),
        s.get("length"),
        many=False,
        included_in=s.get("included_in"),
    )


def pinwire(d, doc, s) -> dict:
    """Op `pinwire` {a, b, part?, length?, included_in?, name?}: one conductor between two pins ('inst.CONN.PIN'). Several of these may
    share a port (autowire joins whole ports and refuses a second wire on one). The wire part is the lightest one rated for the pins'
    i_max (else the autowire floor), the length the autowire rule (port positions x routing factor), both overridable."""
    ends = []
    for k in ("a", "b"):
        iid, _, ref = str(s.get(k, "")).partition(".")
        part = d._part(d._inst(doc, iid)["part"])
        pin = PN.find_pin(part, ref)
        pname = PN.port_of_pin(part, ref) if pin is not None else None
        if pname is None:
            raise DesignError(
                f"pinwire '{s.get(k)}': name a pin as 'inst.CONN.PIN' (connectors of {part.id}: {sorted(part.connectors)})"
            )
        ends.append((iid, ref, pin, part.port(pname)))
    (ia, ra, qa, pa), (ib, rb, qb, pb) = ends
    if pa["kind"] != pb["kind"]:
        raise DesignError(
            f"pinwire {s['a']} -> {s['b']}: a {pa['kind']} port and a {pb['kind']} port cannot be joined"
        )
    st, world = _State(d, doc), _world(d, doc)
    spec, dist = _probe(
        d, doc, st, world, (ia, pa), (ib, pb), pa["kind"], s.get("length")
    )
    need = max((_num(q.get("i_max")) or 0.0 for q in (qa, qb)), default=0.0) or None
    part, _ = choose_wire(d, need, pa["kind"], s.get("part"))
    spec.update(
        name=s.get("name") or f"{s['a']}--{s['b']}",
        part=part.id,
        from_pins=[ra],
        to_pins=[rb],
        conductors=1,
    )
    if s.get("length") is None:
        n, deriv = wire_length(part, dist)
        spec["length"] = {
            "value": n,
            "unit": "m",
            "fidelity": "derived",
            "derivation": deriv,
        }
    if s.get("included_in"):
        d._inst(doc, s["included_in"])
        spec["included_in"] = s["included_in"]
    ch = d._op_wire(doc, spec)
    return {"op": "pinwire", "wire": ch["wire"]}


def autowire_all(d, doc, s) -> dict:
    """Op `autowire_all`: every open power / signal port that has exactly one sensible target; ambiguous ones are reported."""
    return _run(
        d,
        doc,
        None,
        s.get("kind"),
        None,
        None,
        many=True,
        included_in=s.get("included_in"),
    )


def _decisions(ambiguous):
    """One entry per decision: reports that share a port or a candidate are the same choice seen from both sides (a 2 x 2
    ESC-signal / PWM-channel ambiguity is one decision, not four) -> [{"ports": [...], "candidates": [...], "why"}]."""
    adj = {}  # a wire joins a port and a candidate: the two ends of a decision are the two sides of this graph
    for e in ambiguous:
        for c in e["candidates"]:
            adj.setdefault(e["port"], set()).add(c)
            adj.setdefault(c, set()).add(e["port"])
    side, out = {}, []
    for root in sorted(adj):
        if root in side:
            continue
        side[root], todo, comp = 0, [root], [root]
        while todo:
            x = todo.pop()
            for y in adj[x]:
                if y not in side:
                    side[y] = 1 - side[x]
                    todo.append(y)
                    comp.append(y)
        p = sorted(x for x in comp if side[x] == 0)
        c = sorted(x for x in comp if side[x] == 1)
        out.append(
            {
                "ports": p,
                "candidates": c,
                "why": f"{len(p)} open port(s) x {len(c)} equally good target(s): which goes to which is the caller's call "
                "(autowire a, b)",
            }
        )
    return out


def _run(d, doc, pairs, kind, forced_part, forced_len, many, included_in=None):
    world = _world(d, doc)
    st = _State(d, doc)
    done, ambiguous, unmatched, recs = set(), [], [], []
    if many:  # initiators: ports the DRC would call floating, every signal port, any port the part marks required
        pairs = []
        for iid, inst in st.ctx.insts.items():
            for p in inst.part.ports:
                if (
                    p.get("kind") not in ((kind,) if kind else (POWER, SIGNAL))
                    or (iid, p["name"]) in st.direct
                ):
                    continue
                if (
                    p.get("kind") == SIGNAL
                    or p.get("required")
                    or (p.get("role") != "source" and st.loose(iid, p["name"]))
                ):
                    pairs.append((f"{iid}.{p['name']}", None))
    work = [(a, b, False) for a, b in pairs] + [(None, None, False)]
    mark = -1
    for (
        a,
        b,
        fed_only,
    ) in work:  # grows: the sentinel (a is None) appends the next sweep
        if a is None:
            # end of a sweep: a power port still unfed, with no wire of its own, that a fed net can now supply (the PM02 that
            # the battery just reached feeds the PDB) is a further sweep, repeated while it wires something
            if many and len(recs) != mark:
                mark = len(recs)
                more = [
                    (f"{i}.{p['name']}", None, True)
                    for i, inst in st.ctx.insts.items()
                    for p in inst.part.ports
                    if p.get("kind") == POWER
                    and kind in (None, POWER)
                    and p.get("role") != "source"
                    and (i, p["name"]) not in st.direct
                    and (i, p["name"]) not in done
                    and not st.fed(i, p["name"])
                ]
                if more:
                    work += more + [(None, None, False)]
            continue
        ia, pn = _split(a)
        if pn is None:
            raise DesignError(f"autowire '{a}': name the port as 'inst.port'")
        if (ia, pn) in done:
            continue
        pa = d._part(d._inst(doc, ia)["part"]).port(pn)
        if (
            many
            and not fed_only
            and pa["kind"] == POWER
            and not pa.get("required")
            and not st.loose(ia, pn)
        ):
            continue  # an earlier wire of this batch already powers its node (a spare outlet of a wired PDB)
        if pa.get("kind") not in (POWER, SIGNAL):
            raise DesignError(
                f"autowire '{a}': port kind {pa.get('kind')!r}, wires join power or signal ports"
            )
        if kind and pa.get("kind") != kind:
            raise DesignError(f"autowire '{a}': a {pa['kind']} port, not {kind}")
        if (ia, pn) in st.direct:
            raise DesignError(f"autowire '{a}': already has a wire")
        ib = _split(b) if b else None
        if ib and ib[1] is None:
            raise DesignError(f"autowire '{b}': name the port as 'inst.port'")
        if ib:
            d._inst(doc, ib[0])
            d._part(d._inst(doc, ib[0])["part"]).port(ib[1])
        status, tgt, note = resolve(d, st, ia, pa, ib, fed_only)
        if status == "ambiguous":
            if many:
                ambiguous.append({"port": a, "candidates": tgt, "why": note})
                continue
            raise DesignError(
                f"autowire '{a}': {note}: {', '.join(tgt)}; name one as b"
            )
        if status == "none":
            if many:
                if not fed_only and (pa["kind"] == POWER or pa.get("required")):
                    unmatched.append({"port": a, "why": note})
                continue
            raise DesignError(f"autowire '{a}': {note}")
        ti, tn = tgt
        pb = st.ctx.insts[ti].part.port(tn)
        f, t = _direction(st, ia, pa, ti, pb)
        spec, dist = _probe(d, doc, st, world, f, t, pa["kind"], forced_len)
        inc = (  # one instance for `autowire`; {"inst.port": instance} for `autowire_all`, matched on either end of the wire
            included_in
            if isinstance(included_in, str)
            else next(
                (
                    v
                    for k, v in (included_in or {}).items()
                    if k in (spec["from"], spec["to"])
                ),
                None,
            )
        )
        # a port that declares `lead` (parts/README.md) carries its own wire: that wire is already inside the part's mass
        lead = next(((i, p) for i, p in (f, t) if p.get("lead") is not None), None)
        if lead and not inc:
            inc = lead[0]
        if inc:
            d._inst(doc, inc)
            spec["included_in"] = inc
        ch = d._op_wire(doc, spec)
        recs.append(
            {
                "wire": ch["wire"],
                "kind": pa["kind"],
                "f": f,
                "t": t,
                "dist": dist,
                "note": note,
                "a": a,
                "lead": lead,
            }
        )
        done |= {(ia, pn), (ti, tn)}
        st = _State(d, doc)
    diag = {}
    if recs:
        check(A.Assembly(doc), d.lib, diag=diag)
    out = []
    for r in recs:
        reasons = _finish(
            d,
            doc,
            st,
            r,
            diag.get("currents", {}).get(r["wire"]["name"]),
            forced_part,
            forced_len,
        )
        reasons["target"] = r["note"]
        out.append({"wire": r["wire"], "reasons": reasons})
    if not many:
        return {"op": "autowire", **out[0]}
    # a port reported ambiguous / unmatched that a later pair wired anyway is no longer open; one entry per decision
    open_ = lambda r: _split(r) not in done
    ambiguous = [
        {**x, "candidates": [c for c in x["candidates"] if open_(c)]}
        for x in ambiguous
        if open_(x["port"])
    ]
    return {
        "op": "autowire_all",
        "wired": out,
        "ambiguous": _decisions([x for x in ambiguous if x["candidates"]]),
        "unmatched": [x for x in unmatched if open_(x["port"])],
    }


# ---- dependency hints ----------------------------------------------------------------------------------------------------------
def needs(d, doc, part, skip=(), limit=4) -> list:
    """What a part would need next, from the kinds of its other ports (a hint, not a planner): a male / female mech port wants
    its counterpart (library parts that mate it); a power / signal port wants an open compatible port in this design to be
    wired to, else says which library parts have one."""
    direct = wired_ports(doc)
    by_id = {x["id"]: d._part(x["part"]) for x in doc["instances"]}
    out, seen = [], set()
    for q in part.ports:
        key = (q.get("kind"), q.get("type"), q.get("gender"), q.get("role"))
        if q["name"] in skip or key in seen:
            continue
        seen.add(key)
        kind, typ = q.get("kind"), q.get("type", "")
        if kind == "mech":
            if q.get("gender") in ("male", "female"):
                fits = [
                    p.id
                    for p, pt in d.lib.compatible(typ, q["gender"])
                    if pt.get("kind") == "mech" and pt.get("gender") != "neutral"
                ]
                if fits:
                    out.append(
                        f"{q['name']} ({typ}): mate a part that takes it, e.g. {', '.join(fits[:3])}"
                        + (f" (+{len(fits) - 3})" if len(fits) > 3 else "")
                    )
            continue
        if kind not in (POWER, SIGNAL):
            continue
        free = [
            f"{i}.{p['name']}"
            for i, pt in by_id.items()
            for p in pt.ports
            if p.get("kind") == kind
            and (i, p["name"]) not in direct
            and types_ok(typ, p.get("type", ""))
            and _roles_ok(q.get("role"), p.get("role"))
        ]
        if free:
            out.append(
                f"{q['name']} ({kind} {typ}): wire it (autowire) to an open port here: {', '.join(free[:3])}"
                + (f" (+{len(free) - 3})" if len(free) > 3 else "")
            )
        else:
            lib = [
                p.id
                for p in d.lib.parts.values()
                if p.id != part.id
                and any(
                    x.get("kind") == kind
                    and types_ok(typ, x.get("type", ""))
                    and _roles_ok(q.get("role"), x.get("role"))
                    for x in p.ports
                )
            ]
            out.append(
                f"{q['name']} ({kind} {typ}): no open {typ} port here; add a part that has one, e.g. {', '.join(sorted(lib)[:3]) or 'none in the library'}"
            )
    return out[:limit]
