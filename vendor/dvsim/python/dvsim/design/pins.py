"""Pin-level connectors (parts/README.md "Connectors and pins"): `[[connectors]]` of a part file (one physical connector each,
`[[connectors.pins]]` = its pins), the `pins = ["J8.8", ...]` of a logical port, and the `from_pins` / `to_pins` of an assembly
wire. A wire end `inst.CONN.PIN` (shorthand) is folded into the port + pin lists by `normalize_wires`. Stdlib only here;
numpy is imported by the geometry functions (the schema check in tools/check_parts.py runs without it).

    validate(doc) -> [str]                     schema problems of one part document (also used by Library / check_parts)
    connectors(part) -> {id: Connector dict}   normalized: pins keyed by id, each with `loc` (connector frame) and `pos` (part frame)
    port_pins(part, port) -> ["J8.8", ...]     the pins a logical port stands for
    wire_pairs(ctx, w) -> [(ref_a, ref_b)]     the pin pair of every conductor of a wire ("CONN.PIN" or None); explicit lists win,
                                               else the two ports' pin lists when both have the same length
"""

FIDELITY = {"datasheet", "literature", "measured", "derived", "estimated"}
SOURCED = {"datasheet", "literature", "measured"}
PIN_KINDS = ("power", "ground", "signal", "nc")
DIRS = (
    "in",
    "out",
    "io",
)  # seen from the part: a power "out" pin supplies, a signal "out" pin drives
LAYOUTS = ("dual", "line", "seq")
CONN_KEYS = {
    "id", "name", "type", "gender", "pos", "axis", "ref", "rows", "cols", "pitch", "layout", "src", "fidelity", "derivation",
    "rationale", "pins", "note", "pose_rationale", "housing", "port_type", "joinable",
}  # fmt: skip
PIN_KEYS = {
    "number", "name", "alt", "kind", "dir", "v", "v_min", "v_max", "i_max", "at", "offset", "src", "fidelity", "derivation", "rationale", "note",
}  # fmt: skip


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _vec3(v):
    return isinstance(v, list) and len(v) == 3 and all(_num(a) for a in v)


def split_ref(ref: str):
    """'J8.8' -> ('J8', '8'). Pin ids never contain a dot."""
    c, _, p = str(ref).partition(".")
    return c, p


def pin_id(p: dict) -> str:
    return str(p.get("number"))


def _prov(d, label, sources, bad, own=True):
    """Provenance of a connector / pin table (src + fidelity [+ derivation | rationale])."""
    fid = d.get("fidelity")
    if fid not in FIDELITY:
        return bad(f"{label}: fidelity {fid!r} not in {sorted(FIDELITY)}")
    if fid in SOURCED and d.get("src") not in sources:
        bad(f"{label}: src {d.get('src')!r} not in [[sources]]")
    if "src" in d and d["src"] not in sources:
        bad(f"{label}: src {d['src']!r} not in [[sources]]")
    if own and fid == "derived" and not d.get("derivation"):
        bad(f"{label}: derived without 'derivation'")
    if own and fid == "estimated" and not d.get("rationale"):
        bad(f"{label}: estimated without 'rationale'")


def validate(doc: dict) -> list:
    """Problems of the connectors / pins / port pin lists of one part document ([] = fine)."""
    out = []

    def bad(m):
        out.append(m)

    sources = {s.get("id") for s in doc.get("sources", []) if isinstance(s, dict)}
    conns = doc.get("connectors", [])
    if not isinstance(conns, list) or not all(isinstance(c, dict) for c in conns):
        return ["connectors: must be an array of tables"]
    ids, pins_of = set(), {}
    for i, c in enumerate(conns):
        cid = c.get("id")
        lab = f"connector {cid if cid else i}"
        if not isinstance(cid, str) or not cid or "." in cid:
            bad(f"{lab}: 'id' must be a non-empty string without '.'")
        elif cid in ids:
            bad(f"{lab}: duplicate id")
        ids.add(cid)
        for k in set(c) - CONN_KEYS:
            bad(f"{lab}: unknown key '{k}'")
        if not c.get("type"):
            bad(f"{lab}: missing type")
        if c.get("gender", "neutral") not in ("male", "female", "neutral"):
            bad(f"{lab}: gender {c.get('gender')!r}")
        if not isinstance(c.get("joinable", False), bool):
            bad(f"{lab}: 'joinable' must be true or false")
        for k in ("pos", "axis", "ref"):
            if not _vec3(c.get(k)):
                bad(f"{lab}: '{k}' must be 3 numbers")
        ax, rf = c.get("axis"), c.get("ref")
        if _vec3(ax) and abs(sum(a * a for a in ax) - 1) > 1e-6:
            bad(f"{lab}: axis not unit length")
        if _vec3(rf) and abs(sum(a * a for a in rf) - 1) > 1e-6:
            bad(f"{lab}: ref not unit length")
        if _vec3(ax) and _vec3(rf) and abs(sum(a * b for a, b in zip(ax, rf))) > 1e-6:
            bad(f"{lab}: ref not perpendicular to axis")
        rows, cols = c.get("rows", 1), c.get("cols")
        if not isinstance(rows, int) or isinstance(rows, bool) or rows < 1:
            bad(f"{lab}: 'rows' must be an integer >= 1")
        if cols is not None and (
            not isinstance(cols, int) or isinstance(cols, bool) or cols < 1
        ):
            bad(f"{lab}: 'cols' must be an integer >= 1")
        if c.get("layout", "dual" if rows == 2 else "line") not in LAYOUTS:
            bad(f"{lab}: layout {c.get('layout')!r} not in {LAYOUTS}")
        if "pitch" in c and not (_num(c["pitch"]) and c["pitch"] > 0):
            bad(f"{lab}: 'pitch' must be a positive number (m)")
        _prov(c, lab, sources, bad)
        pins = c.get("pins", [])
        if (
            not isinstance(pins, list)
            or not pins
            or not all(isinstance(p, dict) for p in pins)
        ):
            bad(f"{lab}: needs [[connectors.pins]]")
            continue
        seen = set()
        for j, p in enumerate(pins):
            plab = f"{lab} pin {p.get('number', j)}"
            n = p.get("number")
            if (
                isinstance(n, bool)
                or not isinstance(n, (int, str))
                or str(n) == ""
                or "." in str(n)
            ):
                bad(f"{plab}: 'number' must be an int or a string without '.'")
            elif str(n) in seen:
                bad(f"{plab}: duplicate number")
            seen.add(str(n))
            for k in set(p) - PIN_KEYS:
                bad(f"{plab}: unknown key '{k}'")
            if not isinstance(p.get("name"), str) or not p.get("name"):
                bad(f"{plab}: missing name")
            if p.get("kind") not in PIN_KINDS:
                bad(f"{plab}: kind {p.get('kind')!r} not in {PIN_KINDS}")
            if "dir" in p and p["dir"] not in DIRS:
                bad(f"{plab}: dir {p['dir']!r} not in {DIRS}")
            alt = p.get("alt", [])
            if not isinstance(alt, list) or not all(isinstance(a, str) for a in alt):
                bad(f"{plab}: 'alt' must be a list of strings")
            for k in ("v", "v_min", "v_max", "i_max"):
                if k in p and not (
                    _num(p[k]) or (isinstance(p[k], dict) and _num(p[k].get("value")))
                ):
                    bad(
                        f"{plab}: '{k}' must be a number ({'A' if k == 'i_max' else 'V'}) or a provenance table"
                    )
            if "at" in p and not (
                isinstance(p["at"], list)
                and len(p["at"]) == 2
                and all(isinstance(a, int) for a in p["at"])
            ):
                bad(f"{plab}: 'at' must be [col, row]")
            if "offset" in p and not (
                isinstance(p["offset"], list)
                and len(p["offset"]) == 2
                and all(_num(a) for a in p["offset"])
            ):
                bad(f"{plab}: 'offset' must be [x, y] metres in the connector frame")
            if any(k in p for k in ("src", "fidelity")):
                _prov(
                    {
                        "src": p.get("src", c.get("src")),
                        "fidelity": p.get("fidelity", c.get("fidelity")),
                        **p,
                    },
                    plab,
                    sources,
                    bad,
                    own=False,
                )
        pins_of[cid] = seen
        if (
            len(c.get("pins", [])) > 1
            and "pitch" not in c
            and not all("offset" in p for p in c["pins"])
        ):
            bad(f"{lab}: several pins need a 'pitch' (or an 'offset' per pin)")
        if _num(c.get("pitch")) and isinstance(rows, int) and rows >= 1:
            try:
                for p in c["pins"]:
                    col, row = _cell(c, p, c["pins"].index(p))
                    if (
                        row >= rows
                        or (cols is not None and col >= cols)
                        or col < 0
                        or row < 0
                    ):
                        bad(
                            f"{lab} pin {p.get('number')}: grid cell ({col}, {row}) outside {cols} x {rows}"
                        )
            except (TypeError, ValueError):
                pass
    for p in doc.get("ports", []):
        if not isinstance(p, dict) or "pins" not in p:
            continue
        lab = f"port {p.get('name')}"
        if p.get("kind") == "mech":
            bad(f"{lab}: mech ports have no pins")
        if not isinstance(p["pins"], list) or not p["pins"]:
            bad(f"{lab}: 'pins' must be a non-empty list of 'CONN.PIN'")
            continue
        for r in p["pins"]:
            c, n = split_ref(r)
            if c not in pins_of or n not in pins_of[c]:
                bad(f"{lab}: pin '{r}' is not a pin of this part's connectors")
    return out


def _cell(c, p, idx):
    """(col, row) of pin number idx (0-based position in the connector's pin list)."""
    if "at" in p:
        return p["at"][0], p["at"][1]
    rows = c.get("rows", 1)
    lay = c.get("layout", "dual" if rows == 2 else "line")
    if lay == "dual":
        return idx // 2, idx % 2
    if lay == "seq" and c.get("cols"):
        return idx % c["cols"], idx // c["cols"]
    return idx, 0


# ---- geometry ---------------------------------------------------------------------------------------------------------------
def connectors(part) -> dict:
    """Normalized connectors of a Part ({} when it has none). Cached on the part. Each pin gets `loc` (connector frame, m),
    `pos` (part frame, m), `cell` (col, row) and its provenance (own or the connector's)."""
    cached = part.__dict__.get("_connectors")
    if cached is not None:
        return cached
    import numpy as np

    out = {}
    for c in part.raw.get("connectors", []):
        if not isinstance(c.get("id"), str):
            continue
        ax = np.array(c.get("axis", [0, 0, 1]), dtype=float)
        rf = np.array(c.get("ref", [1, 0, 0]), dtype=float)
        ax /= np.linalg.norm(ax) or 1.0
        rf /= np.linalg.norm(rf) or 1.0
        rw = np.cross(ax, rf)
        R = np.stack([rf, rw, ax], axis=1)
        pos = np.array(c.get("pos", [0, 0, 0]), dtype=float)
        pitch = float(c.get("pitch", 0.0))
        raw_pins = c.get("pins", [])
        cells = [_cell(c, p, k) for k, p in enumerate(raw_pins)]
        ncol = c.get("cols") or (max((a for a, _ in cells), default=0) + 1)
        nrow = max(c.get("rows", 1), max((b for _, b in cells), default=0) + 1)
        pins, order = {}, []
        for p, (col, row) in zip(raw_pins, cells):
            if "offset" in p:
                xy = np.array(p["offset"], dtype=float)
            else:
                xy = np.array(
                    [(col - (ncol - 1) / 2) * pitch, (row - (nrow - 1) / 2) * pitch]
                )
            loc = np.array([xy[0], xy[1], 0.0])
            pid = pin_id(p)
            q = {k: v for k, v in p.items() if k not in ("number",)}
            q.update(
                id=pid,
                connector=c["id"],
                ref=f"{c['id']}.{pid}",
                alt=list(p.get("alt", [])),
                loc=loc,
                pos=pos + R @ loc,
                cell=(col, row),
                src=p.get("src", c.get("src")),
                fidelity=p.get("fidelity", c.get("fidelity")),
                joinable=bool(c.get("joinable", False)),
            )
            for k in ("v", "v_min", "v_max", "i_max"):
                if isinstance(q.get(k), dict):
                    q[k] = q[k].get("value")
            pins[pid] = q
            order.append(pid)
        out[c["id"]] = {
            "id": c["id"],
            "name": c.get("name", c["id"]),
            "type": c.get("type", ""),
            "port_type": c.get("port_type", ""),
            "gender": c.get("gender", "neutral"),
            "pos": pos,
            "axis": ax,
            "ref": rf,
            "row_axis": rw,
            "R": R,
            "rows": nrow,
            "cols": ncol,
            "pitch": pitch,
            "pins": pins,
            "order": order,
        }
    part.__dict__["_connectors"] = out
    return out


def find_pin(part, ref: str):
    """The normalized pin dict for 'CONN.PIN' (None when it does not exist)."""
    c, n = split_ref(ref)
    return connectors(part).get(c, {}).get("pins", {}).get(n)


def port_pins(part, port_name: str) -> list:
    for p in part.ports:
        if p.get("name") == port_name:
            return list(p.get("pins", []))
    return []


def synthetic_ports(part) -> list:
    """One port per connector for the pins no authored port claims (name = the connector id in lower case, `synthetic = true`):
    a pin-level wire onto any pin of a connector therefore always has a port to hang its checks on. Not part of `Part.ports`
    (autowire, snapping and the floating-port rule never see them); `Part.port(name)` finds them."""
    cached = part.__dict__.get("_synthetic")
    if cached is not None:
        return cached
    claimed = {r for p in part.ports for r in p.get("pins", [])}
    taken = {p.get("name") for p in part.ports}
    out = []
    for cid, c in connectors(part).items():
        free = [
            c["pins"][i]["ref"]
            for i in c["order"]
            if c["pins"][i]["ref"] not in claimed
        ]
        if not free:
            continue
        name = cid.lower()
        while name in taken:
            name += "_pins"
        taken.add(name)
        power = all(
            c["pins"][split_ref(r)[1]]["kind"] in ("power", "ground", "nc")
            for r in free
        )
        out.append(
            {
                "name": name,
                "kind": "power" if power else "signal",
                "type": c["port_type"] or c["type"],
                "gender": c["gender"],
                "pins": free,
                "pos": [float(x) for x in c["pos"]],
                "axis": [float(x) for x in c["axis"]],
                "connector": cid,
                "synthetic": True,
            }
        )
    part.__dict__["_synthetic"] = out
    return out


def port_of_pin(part, ref: str):
    """The logical port that holds 'CONN.PIN': an authored port first, else the connector's synthetic one (None: no such pin)."""
    for p in list(part.ports) + synthetic_ports(part):
        if ref in p.get("pins", []):
            return p["name"]
    return None


# ---- wires ------------------------------------------------------------------------------------------------------------------
def normalize_wires(doc: dict, lib) -> None:
    """Fold the shorthand `from = "inst.CONN.PIN"` (also `to`) into `from = "inst.port"` + `from_pins = ["CONN.PIN"]`, in place.
    A pin no port claims is left alone (the DRC reports it as WIRE_INVALID)."""
    insts = {i["id"]: i for i in doc.get("instances", [])}
    for w in doc.get("wires", []):
        for end in ("from", "to"):
            ref = w.get(end)
            if not isinstance(ref, str) or ref.count(".") < 2:
                continue
            iid, _, rest = ref.partition(".")
            inst = insts.get(iid)
            if inst is None:
                continue
            try:
                part = lib.get(inst["part"])
            except Exception:
                continue
            port = port_of_pin(part, rest)
            if port is None:
                continue
            w[end] = f"{iid}.{port}"
            lst = w.setdefault(end + "_pins", [])
            if rest not in lst:
                lst.append(rest)


def wire_pairs(ctx, w: dict) -> list:
    """[(pin_ref_a, pin_ref_b)] per conductor: the wire's own `from_pins` / `to_pins` (equal length), else the two ports' pin lists
    when they have the same length, else [] (port-level wire on a part without pinout, or an ambiguous pairing)."""
    ends = []
    for end in ("from", "to"):
        iid, _, pn = str(w.get(end, "")).partition(".")
        inst = ctx.insts.get(iid)
        explicit = w.get(end + "_pins")
        if explicit:
            ends.append(list(explicit))
        elif inst is not None:
            ends.append(port_pins(inst.part, pn))
        else:
            ends.append([])
    a, b = ends
    return list(zip(a, b)) if a and len(a) == len(b) else []
