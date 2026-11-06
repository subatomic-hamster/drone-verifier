"""Pin-level design rules (parts/README.md "Connectors and pins"): run on every wire that resolves to concrete pins.

    PIN_ALREADY_USED      error    one pin on two conductors (ground pins and pins of a `joinable` connector may be shared)
    PIN_KIND_MISMATCH     error    a conductor joins unlike pins (power-ground short, signal-power, ...); nc pins are a warning
    PIN_VOLTAGE_MISMATCH  error    both pins state a voltage and they differ (power); signal levels differ: warning (level shifter)
    PIN_TWO_OUTPUTS       error    two signal outputs drive one net
    PIN_GND_MISSING       warning  a multi-conductor wire carries power / signal but no ground return
    PIN_ALT_CONFLICT      error    one peripheral function claimed by two pins, or a wire's TX / RX on different peripherals
    PIN_FUNCTION_MISSING  warning  a multiplexed pin is wired to a device pin whose role (TX, SDA, ...) it cannot provide

Same entry as the other rule groups: `pin_rules(ctx, F)` appends Findings; `pin_nets(ctx)` is the net structure the 2D canvas reuses.
"""

import re

from . import pins as PN

_ROLE = re.compile(r"(?:^|[ _\-.])(TX|RX|SDA|SCL|CTS|RTS)D?$")
_OPP = {"TX": "RX", "RX": "TX", "CTS": "RTS", "RTS": "CTS", "SDA": "SDA", "SCL": "SCL"}


def voltage_mismatch(a: dict, b: dict):
    """Why two joined pins disagree on voltage (None when they agree or one side states nothing). The driving side (`dir` out) is
    checked against the other side's range `v_min` / `v_max` when it has one, else against its nominal `v` (+-10 %, at least 0.5 V);
    with no driving side two nominal voltages must agree."""
    src, snk = (
        (a, b)
        if a.get("dir") == "out"
        else (b, a)
        if b.get("dir") == "out"
        else (None, None)
    )
    if src is None:
        src, snk = a, b
    v = src.get("v")
    if v is None:
        return None
    lo, hi = snk.get("v_min"), snk.get("v_max")
    if lo is not None or hi is not None:
        rng = f"{snk['name']} range {'-' if lo is None else format(lo, 'g')}..{'-' if hi is None else format(hi, 'g')} V"
        if hi is not None and v > hi + 1e-9:
            return "error", f"{src['name']} drives {v:g} V, above the {rng}"
        if lo is not None and v < lo - 1e-9:
            return "warning", f"{src['name']} drives {v:g} V, below the {rng}"
        return None
    w = snk.get("v")
    if w is not None and abs(v - w) > max(0.5, 0.1 * max(v, w)):
        return None, f"{src['name']} is {v:g} V but {snk['name']} is {w:g} V"
    return None


def alt_function(alt: str) -> str:
    """'a3:I2C1_SDA' -> 'I2C1_SDA' (the mux prefix is not part of the function name)."""
    return alt.split(":", 1)[1] if re.match(r"^a\d+:", alt) else alt


def role_of(pin: dict):
    """TX / RX / SDA / SCL / CTS / RTS from the pin's own name or its unprefixed alt names; None when it has no single role."""
    for s in [pin.get("name", "")] + [
        a for a in pin.get("alt", []) if not re.match(r"^a\d+:", a)
    ]:
        m = _ROLE.search(str(s).upper())
        if m:
            return m.group(1)
    return None


def functions_with_role(pin: dict, role: str) -> list:
    """The multiplexed functions of `pin` that end in `role` ('UART0_TX' for TX)."""
    out = []
    for a in pin.get("alt", []):
        f = alt_function(a)
        m = _ROLE.search(f.upper())
        if m and m.group(1) == role:
            out.append(f)
    return out


def peripheral(fn: str) -> str:
    return _ROLE.sub("", fn.upper())


def conductors(ctx):
    """[(wire dict, [(ref_a, pin_a, ref_b, pin_b)])] with `ref` = 'inst.CONN.PIN'; only wires that resolve to concrete pins."""
    out = []
    for w in ctx.doc.get("wires", []):
        try:
            ia = ctx.insts[w["from"].partition(".")[0]]
            ib = ctx.insts[w["to"].partition(".")[0]]
        except (KeyError, AttributeError):
            continue
        rows = []
        for ra, rb in PN.wire_pairs(ctx, w):
            pa, pb = PN.find_pin(ia.part, ra), PN.find_pin(ib.part, rb)
            if pa is not None and pb is not None:
                rows.append((f"{ia.id}.{ra}", pa, f"{ib.id}.{rb}", pb))
        if rows:
            out.append((w, rows))
    return out


def pin_nets(ctx) -> dict:
    """{pin ref: net id} over all conductors (union-find), nets numbered by first member in sorted order."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, rows in conductors(ctx):
        for ra, _, rb, _ in rows:
            parent[find(ra)] = find(rb)
    roots = {}
    for k in sorted(parent):
        roots.setdefault(find(k), len(roots))
    return {k: roots[find(k)] for k in parent}


def pin_rules(ctx, F, add):
    """Append the pin findings to F with `add(F, code, severity, message, where, fix)` (drc._add)."""
    cond = conductors(ctx)
    used = {}  # pin ref -> wire name
    nets = {}  # net id -> [(ref, pin)]
    net_of = pin_nets(ctx)
    fn_owner = {}  # (instance, function) -> pin ref
    for w, rows in cond:
        name = w.get("name", "?")
        wh = [name, w.get("from", ""), w.get("to", "")]
        kinds = set()
        periph = {}
        for ra, pa, rb, pb in rows:
            for r, p_ in ((ra, pa), (rb, pb)):
                if p_["kind"] == "ground" or p_.get("joinable"):
                    continue  # a common return, or a bare lead / solder pad / screw terminal: several conductors may be joined on one pin
                if r in used and used[r] != name:
                    add(
                        F,
                        "PIN_ALREADY_USED",
                        "error",
                        f"pin {r} is on wire '{used[r]}' and on wire '{name}'",
                        [r, used[r], name],
                        "move one wire to a free pin of the connector",
                    )
                elif r in used:
                    add(
                        F,
                        "PIN_ALREADY_USED",
                        "error",
                        f"pin {r} appears twice in wire '{name}'",
                        [r, name],
                        "list each pin once",
                    )
                used.setdefault(r, name)
            ka, kb = pa["kind"], pb["kind"]
            kinds.add(ka)
            kinds.add(kb)
            if "nc" in (ka, kb):
                add(
                    F,
                    "PIN_KIND_MISMATCH",
                    "warning",
                    f"wire '{name}' lands on a no-connect pin ({ra if ka == 'nc' else rb})",
                    wh + [ra, rb],
                    "use a connected pin",
                )
            elif ka != kb:
                add(
                    F,
                    "PIN_KIND_MISMATCH",
                    "error",
                    f"wire '{name}': {ra} ({pa['name']}, {ka}) is joined to {rb} ({pb['name']}, {kb})",
                    wh + [ra, rb],
                    "join power to power, ground to ground and signal to signal; check the pin order of the wire",
                )
            else:
                vm = voltage_mismatch(pa, pb) if ka != "ground" else None
                if vm:
                    why = vm[1]
                    sev = vm[0] or ("error" if ka == "power" else "warning")
                    add(
                        F,
                        "PIN_VOLTAGE_MISMATCH",
                        sev,
                        f"wire '{name}': {ra} / {rb}: {why}",
                        wh + [ra, rb],
                        "use a regulator or level shifter between the two"
                        if ka == "signal"
                        else "power the pin from a matching supply",
                    )
            for r, p, o_r, o in ((ra, pa, rb, pb), (rb, pb, ra, pa)):
                if p["kind"] != "signal" or not p.get("alt"):
                    continue
                role = role_of(o)
                if role is None:
                    continue
                need = _OPP[role] if role in ("TX", "RX", "CTS", "RTS") else role
                cands = functions_with_role(p, need)
                if not cands:
                    add(
                        F,
                        "PIN_FUNCTION_MISSING",
                        "warning",
                        f"wire '{name}': {r} ({p['name']}) has no {need} function for {o_r} ({o['name']})",
                        wh + [r, o_r],
                        f"pick a pin of that connector that offers a {need} function",
                    )
                    continue
                inst = r.split(".")[0]
                fam = "I2C" if need in ("SDA", "SCL") else "UART"
                periph.setdefault((inst, fam), []).append(
                    {peripheral(c) for c in cands}
                )
                for c in cands:
                    key = (inst, c)
                    if fn_owner.setdefault(key, r) != r:
                        add(
                            F,
                            "PIN_ALT_CONFLICT",
                            "error",
                            f"{inst}: function {c} is claimed by {fn_owner[key]} and {r}",
                            wh + [r, fn_owner[key]],
                            "wire each peripheral function to one pin",
                        )
        for (inst, _fam), sets in periph.items():
            if len(sets) > 1 and not set.intersection(*sets):
                add(
                    F,
                    "PIN_ALT_CONFLICT",
                    "error",
                    f"wire '{name}': the pins of {inst} belong to different peripherals ({', '.join(sorted(set.union(*sets)))})",
                    wh,
                    "move the conductors to the pins of one UART / I2C peripheral",
                )
        dc = any(
            p["kind"] == "signal" or (p["kind"] == "power" and p.get("v") is not None)
            for _, pa_, _, pb_ in rows
            for p in (pa_, pb_)
        )  # three motor phases carry AC and need no return; a logic level or a DC rail does
        if len(rows) > 1 and "ground" not in kinds and dc:
            add(
                F,
                "PIN_GND_MISSING",
                "warning",
                f"wire '{name}' has {len(rows)} conductors and no ground return",
                wh,
                "add a ground conductor to the wire (shared reference for the signal)",
            )
    for ref, nid in net_of.items():
        ia = ref.split(".")[0]
        pin = PN.find_pin(ctx.insts[ia].part, ref.partition(".")[2])
        if pin is not None:
            nets.setdefault(nid, []).append((ref, pin))
    for members in nets.values():
        outs = [
            r for r, p in members if p["kind"] == "signal" and p.get("dir") == "out"
        ]
        if len(outs) > 1:
            add(
                F,
                "PIN_TWO_OUTPUTS",
                "error",
                f"outputs {', '.join(sorted(outs))} drive the same net",
                sorted(outs),
                "cross the TX / RX pair, or put a buffer between the outputs",
            )
    return F
