"""The viewer's 2D "flatten" wiring mode, as design_api ops (the viewer's `src/viewer/wire2d.cpp` draws what these return; a
scripted agent drives the same ops through `Backend`, so a recording shows the UI reacting).

    flatten.begin   {instances: [id], ports?: {id: [port]} | ["id.port"]}   select components + the terminals to show; opens the canvas
    flatten.layout  {}                                                      auto-layout: signal flow left to right (see `layout`)
    flatten.move_card {id, x, y}                                            move a card (canvas units)
    flatten.wire    {from, to, awg?, length?}                      drag pin -> pin. `from` / `to` are "inst.port" (the `autowire` op: kind,
                                                                            connector type, gender, voltage rules) or "inst.CONN.PIN" (the
                                                                            `pinwire` op: one conductor between two real connector pins);
                                                                            a refusal comes back as {"ok": false, "error": reason}.
                                                                            With `part` (+ length, conductors?, included_in?, name?) it is an
                                                                            explicit `wire` op instead: a lead the user chose, bridging unlike connectors
    flatten.set_wire {name, awg?, length?}                                  change a wire's gauge / length (unwire + wire in one edit)
    flatten.unwire  {name}
    flatten.preview {instances, ports?}                                     the cards `begin` would open, as {"cards": [...]} in the result;
                                                                            changes nothing (the picker's preview of the ticked components)
    flatten.state   {}                                                      the canvas state (cards, pins, wires)
    flatten.end     {}                                                      back to 3D: the wires are already in the design (tubes)

Every reply is {"ok": true, "result": {...}, "flatten": <state>, "state": <design state when the design changed>}.

Canvas state: {"active", "rev",
  "cards": [{id, hidden, part, group, x, y, w, h, band, layer,
             pins: [{name, label, kind, type, gender, role, side, row, wired, conn, num, dir, v, tip, flag, flags,
                    port, connects: [{component, port, pin}]}],  (connects = the far ends of the pin's wires; `pin_links` gives the same for any ref)
             connectors: [{id, name, type, rows, cols, side: "left" | "right" | "both", row, nrows}]}],
  "wires": [{name, from, to, part, awg, length, kind, conductors, flag, flags, hidden,
             route: [[x, y], ...], strands: [{from, to, route, flag}], label: {text, x, y, w, h} | null}],
  "hidden_wires": n}.  `hidden` = the part (either end of the wire) is switched off with the eye in the window: it stays in the list so an
agent can still read it, the window does not draw it.

A card terminal (`pins`) is either a port of the part (`conn` = "", name = the port name, as before) or one real connector pin
(`conn` = the connector id, name = "CONN.PIN", `num` = the pin id as printed on the part, `label` = "num name", `dir` / `v` from the
part file, `tip` = the hover text: functions, alternates, ratings). `connectors` are the boxes drawn around pin groups: a one-row
connector is a column of pins on one side, a two-row header (the Pi's 2 x 20) puts row 0 on the left and row 1 on the right of the
box, as a schematic symbol, so every pin has a free edge. A connector is shown when a requested port has pins on it or a wire
lands on it; a port without pins is a single terminal as before.
A wire is drawn one `strand` per conductor, from the real pins of each end (`from_pins` / `to_pins`, else the two ports' pin lists
when they match, else one strand port to port); `route` is the first strand's. `flag` = "" | "warning" | "error" from the pin DRC
rules (drc_pins.py); `flags` = [{code, severity, message}]. Every route is an orthogonal polyline that stays out of every card
(wire2d_route.py); `label` = where the gauge text goes (centre), null when no free spot is left.

Layout (`layout`): a layered signal-flow drawing. Power band on top: sources (batteries, supplies) in the left layer, then power
modules / PDBs / BECs, then ESCs / controllers / compute, then loads (motors, servos) on the right; every wire pushes its target at
least one layer right of its source. Cards that carry no power terminal (receivers, GPS, sensors) form a second band below with
their own layers, so signal buses stay out of the power flow. Inside a layer the order minimises crossings (barycentre sweeps).
A one-row connector (or port) sits on the card side facing its peers (wire targets to the right -> right side), else by its role
(source right, sink left). Card width is sized to the text of its pins so labels never overlap.
"""

from . import assembly as A
from . import pins as PN
from .drc import _add
from .drc_pins import pin_rules
from .library import DesignError
from .wire2d_route import place_labels, route_wires

HEAD_H, ROW_H, PAD, COL_GAP, ROW_GAP, BAND_GAP = 30, 22, 10, 170, 30, 90
MIN_W = 190
# text width estimates in canvas units (Roboto 15 px: head 15 px, pin name 13.8 px, pin type 10.8 px, part id 11.7 px), +8 % margin
CW_HEAD, CW_PART, CW_NAME, CW_TYPE, PIN_PAD, MID_GAP = 8.4, 6.6, 7.6, 6.0, 30, 26
WIRE_KINDS = ("power", "signal")
# category prefix -> layer rank (0 = sources on the left ... 3 = loads on the right); longest prefix wins
RANKS = {
    "battery": 0,
    "psu": 0,
    "psu.dcdc": 1,
    "pcb.pm": 1,
    "pcb.pdb": 1,
    "pcb.bec": 1,
    "connector": 1,
    "esc": 2,
    "driver": 2,
    "computer": 2,
    "pcb.fc": 2,
    "pcb.motor_controller": 2,
    "pcb.servo_driver": 2,
    "motor": 3,
    "actuator": 3,
    "effector": 3,
    "wheel": 3,
    "sensor": 0,
    "rc": 0,
    "radio": 0,
    "pcb.gps": 0,
}
_SEV = {"error": 2, "warning": 1, "info": 0}


def _rank(cat, roles):
    best = None
    for pre, r in RANKS.items():
        if (cat == pre or cat.startswith(pre + ".")) and (
            best is None or len(pre) > len(best[0])
        ):
            best = (pre, r)
    if best:
        return best[1]
    if roles and set(roles) == {"source"}:
        return 0
    if roles and set(roles) == {"sink"}:
        return 3
    return 1


def _pin_role(port, pin):
    r = port.get("role") if port else None
    if r:
        return r
    d = pin.get("dir")
    return "source" if d == "out" else "sink" if d == "in" else ""


def _tip(conn, pin):
    """Hover text of a pin: id + name, kind / direction / voltage / current, alternate functions, source."""
    bits = [pin["kind"]]
    if pin.get("dir"):
        bits.append(pin["dir"])
    if pin.get("v") is not None:
        bits.append(f"{pin['v']:g} V")
    if pin.get("i_max") is not None:
        i = pin["i_max"]
        bits.append(f"max {i:g} A" if i >= 1 else f"max {i * 1000:g} mA")
    lines = [f"{conn['id']} pin {pin['id']}: {pin['name']}", ", ".join(bits)]
    alts = list(pin.get("alt", []))
    for k in range(0, len(alts), 3):
        lines.append(("alt: " if k == 0 else "     ") + "  ".join(alts[k : k + 3]))
    if pin.get("note"):
        lines.append(str(pin["note"]))
    if pin.get("fidelity"):
        lines.append(f"{pin.get('src') or ''} {pin['fidelity']}".strip())
    return "\n".join(lines)


def wire_strands(d, want=lambda ia, ib: True):
    """One entry per conductor of every wire of the design whose end instances pass `want(ia, ib)`: {wire, wn, k, ia, ta, ib, tb}
    (terminal names: 'CONN.PIN' of a real pin, else the port name; a wire whose pin lists do not resolve is one port-to-port
    strand)."""
    try:
        ctx = A.Ctx(A.Assembly(d.doc), d.lib)
    except Exception:
        ctx = None
    out = []
    for w in d.doc.get("wires", []):
        ia, _, pa = str(w.get("from", "")).partition(".")
        ib, _, pb = str(w.get("to", "")).partition(".")
        if not want(ia, ib):
            continue
        try:
            pairs = PN.wire_pairs(ctx, w) if ctx else []
        except Exception:
            pairs = []
        if not pairs:
            pairs = [(None, None)]
        for k, (ra, rb) in enumerate(pairs):
            out.append(
                {
                    "wire": w,
                    "wn": w.get("name") or f"{w.get('from')}--{w.get('to')}",
                    "k": k,
                    "ia": ia,
                    "ta": ra or pa,
                    "ib": ib,
                    "tb": rb or pb,
                }
            )
    return out


def terminal_ref(d, inst, term) -> dict:
    """{component, port, pin} of one wire end: `term` is a port name, or 'CONN.PIN' (port = the port that pin belongs to)."""
    port, pin = term, ""
    if "." in term:
        row = next((i for i in d.doc["instances"] if i["id"] == inst), None)
        try:
            port = PN.port_of_pin(d._part(row["part"]), term) or ""
        except Exception:
            port = ""
        pin = term
    return {"component": inst, "port": port, "pin": pin}


def pin_links(d, ref, strands=None) -> list:
    """What a pin / port / component connects to, for tooltips (the 2D canvas and the code editor). `ref` = "inst",
    "inst.port" or "inst.CONN.PIN". Returns [{component, port, pin, connects: [{component, port, pin}]}], one entry per own
    terminal that has a wire (the entry of a bare port ref lists the pins of that port one by one); `connects` = the far ends."""
    inst, _, term = str(ref).partition(".")
    out = {}
    for s in (
        strands
        if strands is not None
        else wire_strands(d, lambda ia, ib: inst in (ia, ib))
    ):
        for i, t, j, u in (
            (s["ia"], s["ta"], s["ib"], s["tb"]),
            (s["ib"], s["tb"], s["ia"], s["ta"]),
        ):
            if i != inst:
                continue
            me = terminal_ref(d, i, t)
            if term and term not in (t, me["port"]):
                continue
            e = out.setdefault((t,), {**me, "connects": []})
            far = terminal_ref(d, j, u)
            if far not in e["connects"]:
                e["connects"].append(far)
    return list(out.values())


class Flatten:
    def __init__(self):
        self.active = False
        self.touched = (
            False  # a card was moved by hand: wiring then leaves the placement alone
        )
        self.rev = 0
        self.cards = {}  # id -> {"id", "x", "y", "ports": [selected port names], "terms", "blocks", "w", "h", "band", "layer"}
        self.msg = ""

    # ---- entry ---------------------------------------------------------------------------------------------------------------
    def handle(self, b, d, tool, a) -> dict:
        self.prune(d)
        op = tool.partition(".")[2]
        fn = getattr(self, "_" + op, None)
        if fn is None or op.startswith("_"):
            raise DesignError(
                f"unknown flatten op {tool!r} (begin preview layout move_card wire set_wire unwire state end)"
            )
        res, changed = fn(b, d, a or {})
        if changed and self.active:
            self._layout_cards(b, d, place=not self.touched)
        self.rev += 0 if op in ("state", "preview") else 1
        out = {"ok": True, "result": res, "flatten": self.state(b, d)}
        if changed:
            out["state"] = b.state(d)
        return out

    def prune(self, d) -> bool:
        """Drops the cards of instances the design no longer has (removed, or swapped to another part: their terminals are
        stale). Any edit path can do that (the BUILD delete, undo, the MCP tools), so the Backend calls this on every rev poll
        and every flatten op. Wires need no pruning: they live in the document and are drawn only between shown cards. No
        card left ends the mode. Returns True (and bumps `rev`, so the canvas refetches) when something was dropped."""
        part = {i["id"]: i["part"] for i in d.doc["instances"]}
        part.update(  # a sub-assembly block stays while its entry (and file) is in the design
            {s["id"]: s["file"] for s in d.asm.raw.get("subassemblies", [])}
        )
        gone = [k for k, c in self.cards.items() if part.get(k) != c["part"]]
        for k in gone:
            del self.cards[k]
        if gone:
            self.rev += 1
            self.active = self.active and bool(self.cards)
        return bool(gone)

    # ---- ops -----------------------------------------------------------------------------------------------------------------
    def _begin(self, b, d, a):
        ids = list(a.get("instances") or [])
        if not ids:
            raise DesignError(
                "flatten.begin: pass instances = [ids] (select components first)"
            )
        by_id = {i["id"]: i for i in d.doc["instances"]}
        for i in ids:
            if i not in by_id:
                raise DesignError(f"flatten.begin: unknown instance '{i}'")
        want = a.get("ports")
        if isinstance(want, list):  # ["inst.port", ...]
            m = {}
            for r in want:
                i, _, p = r.partition(".")
                m.setdefault(i, []).append(p)
            want = m
        cards = {}
        for i in ids:
            org = d.asm.origins.get(i)
            if org and a.get("collapse", True):
                # an item of a sub-assembly: the whole sub-assembly is ONE block exposing its exported connectors
                cards.setdefault(
                    org["sub"],
                    {
                        "id": org["sub"],
                        "part": org["file"],
                        "sub": True,
                        "x": 0.0,
                        "y": 0.0,
                        "ports": [],
                    },
                )
                continue
            part = d._part(by_id[i]["part"])
            names = [p["name"] for p in part.ports if p.get("kind") in WIRE_KINDS]
            if want and i in want:
                allowed = names + [p["name"] for p in PN.synthetic_ports(part)]
                asked = [self._port_name(part, p) for p in want[i]]
                bad = [p for p, n in zip(want[i], asked) if n not in allowed]
                if bad:
                    raise DesignError(
                        f"flatten.begin: {i} has no wire port(s) {bad}; wire ports: {names}"
                    )
                names = [n for n in allowed if n in asked]
            cards[i] = {
                "id": i,
                "part": by_id[i]["part"],
                "x": 0.0,
                "y": 0.0,
                "ports": names,
            }
        self.cards, self.active, self.touched = cards, True, False
        self._layout_cards(b, d)
        return {"cards": len(cards)}, False

    def _preview(self, b, d, a):
        tmp = (
            Flatten()
        )  # the same validation and layout as `begin`, on a scratch canvas
        tmp._begin(b, d, a)
        return {"cards": tmp.state(b, d)["cards"]}, False

    @staticmethod
    def _port_name(part, name):
        """A port name, or a connector id ('J8') for the connector's synthetic port."""
        if any(p["name"] == name for p in part.ports):
            return name
        return name.lower() if name in part.connectors else name

    def _layout(self, b, d, a):
        self._need()
        self.touched = False
        self._layout_cards(b, d)
        return {"cards": len(self.cards)}, False

    def _move_card(self, b, d, a):
        self._need()
        c = self.cards.get(a.get("id"))
        if c is None:
            raise DesignError(f"flatten.move_card: no card '{a.get('id')}'")
        c["x"], c["y"] = float(a["x"]), float(a["y"])
        self.touched = True
        return {"id": c["id"], "x": c["x"], "y": c["y"]}, False

    def _wire(self, b, d, a):
        self._need()
        for e in ("from", "to"):
            i, _, p = str(a.get(e, "")).partition(".")
            if self.cards.get(i, {}).get("sub"):
                raise DesignError(
                    f"flatten.wire: '{i}' is a sub-assembly block; wire its exported connector in the design file ([[wires]] to '{i}:<export>') "
                    "or open the part"
                )
            if i not in self.cards or (
                p not in self.cards[i].get("terms", {})
                and p
                not in self.cards[i]["ports"]  # a port whose connector pins are drawn
            ):
                raise DesignError(
                    f"flatten.wire: {e} '{a.get(e)}' is not a pin on the canvas (flatten.begin chose the terminals)"
                )
        before = {w.get("name") for w in d.doc.get("wires", [])}
        is_pin = [str(a[e]).count(".") >= 2 for e in ("from", "to")]
        if any(is_pin) and not all(is_pin):
            raise DesignError(
                "flatten.wire: both ends must be connector pins (inst.CONN.PIN) or both ports (inst.port)"
            )
        op = (
            {"op": "pinwire", "a": a["from"], "b": a["to"]}
            if all(is_pin)
            else {"op": "autowire", "a": a["from"], "b": a["to"]}
        )
        if a.get(
            "part"
        ):  # an explicit lead (the user picked the wire part): bridges unlike connector families, which autowire refuses
            op = {
                "op": "wire",
                "from": a["from"],
                "to": a["to"],
                "part": a["part"],
                "length": a["length"],
            }
            if all(is_pin):
                op["from_pins"] = [a["from"].partition(".")[2]]
                op["to_pins"] = [a["to"].partition(".")[2]]
                op["conductors"] = 1
            op.update(
                {
                    k: a[k]
                    for k in (
                        "name",
                        "conductors",
                        "included_in",
                        "from_pins",
                        "to_pins",
                        "connectors",
                    )
                    if a.get(k) is not None
                }
            )
        elif a.get("awg") is not None:
            op["part"] = self._wire_part(d, a["awg"]).id
        if a.get("length") is not None and not a.get("part"):
            op["length"] = self._length(a["length"])
        r = d.edit([op])
        new = [w["name"] for w in d.doc.get("wires", []) if w.get("name") not in before]
        return {**r.as_dict(), "wires": new, "undo_depth": len(d._undo)}, True

    def _set_wire(self, b, d, a):
        self._need()
        w = next(
            (x for x in d.doc.get("wires", []) if x.get("name") == a.get("name")), None
        )
        if w is None:
            raise DesignError(f"flatten.set_wire: no wire named '{a.get('name')}'")
        new = dict(w)
        if a.get("awg") is not None:
            new["part"] = self._wire_part(d, a["awg"]).id
        if a.get("length") is not None:
            new["length"] = self._length(a["length"])
        r = d.edit([{"op": "unwire", "name": w["name"]}, {"op": "wire", **new}])
        return {**r.as_dict(), "wires": [w["name"]]}, True

    def _unwire(self, b, d, a):
        self._need()
        r = d.edit([{"op": "unwire", "name": a.get("name")}])
        return r.as_dict(), True

    def _state(self, b, d, a):
        return {}, False

    def _end(self, b, d, a):
        self.active, self.cards = False, {}
        return {}, False

    # ---- helpers -------------------------------------------------------------------------------------------------------------
    def _need(self):
        if not self.active:
            raise DesignError("flatten: not active (flatten.begin first)")

    @staticmethod
    def _wire_part(d, awg):
        parts = {
            int(p.value("awg")): p
            for p in d.lib.by_category("wire")
            if p.value("awg") is not None
        }
        if int(awg) not in parts:
            raise DesignError(f"no wire part with AWG {awg}; gauges: {sorted(parts)}")
        return parts[int(awg)]

    @staticmethod
    def _length(x):
        if isinstance(x, dict):
            return x
        return {
            "value": float(x),
            "unit": "m",
            "fidelity": "estimated",
            "rationale": "length set in the 2D wiring view",
        }

    # ---- terminals -----------------------------------------------------------------------------------------------------------
    def _strands(self, d):
        """One entry per conductor of every wire with an end on the canvas: {wire, k, ia, ta, ib, tb} (terminal names: 'CONN.PIN' of a
        real pin, else the port name). Wires whose pin lists do not resolve get one port-to-port strand."""
        subs = {k for k, c in self.cards.items() if c.get("sub")}
        if not subs:
            return wire_strands(d, lambda ia, ib: ia in self.cards or ib in self.cards)

        def card_of(i):  # the block an instance's terminals sit on
            o = d.asm.origins.get(i)
            return o["sub"] if o and o["sub"] in subs else i

        exp = self._sub_exports(d)
        out = []
        for s in wire_strands(
            d, lambda ia, ib: card_of(ia) in self.cards or card_of(ib) in self.cards
        ):
            ends = []
            for i, t in ((s["ia"], s["ta"]), (s["ib"], s["tb"])):
                c = card_of(i)
                if c in subs:  # only the exported connectors are terminals of the block
                    name = self._export_of(d, exp.get(c, {}), i, t)
                    if name is None:
                        break
                    ends.append((c, name))
                else:
                    ends.append((i, t))
            if len(ends) < 2 or ends[0][0] == ends[1][0]:
                continue  # inside one sub-assembly (or not exported): not drawn on the parent sheet
            out.append(
                {
                    **s,
                    "ia": ends[0][0],
                    "ta": ends[0][1],
                    "ib": ends[1][0],
                    "tb": ends[1][1],
                }
            )
        return out

    def _sub_exports(self, d):
        """{sub id: {"conns": {export name: (flat instance id, connector id)}}} from the exported connectors of each child file."""
        import tomllib
        from pathlib import Path

        out = {}
        if not d.asm.path:
            return out
        for s in d.asm.raw.get("subassemblies", []):
            try:
                cdoc = tomllib.loads(
                    (Path(d.asm.path).resolve().parent / s["file"]).read_text(
                        encoding="utf-8"
                    )
                )
            except (OSError, tomllib.TOMLDecodeError):
                continue
            conns = {}
            for name, ref in (cdoc.get("exports", {}).get("connectors") or {}).items():
                head, _, cid = str(ref).partition(".")
                if (
                    cid and ":" not in head
                ):  # (an export of a nested sub-assembly is not drawn)
                    conns[name] = (f"{s['id']}:{head}", cid)
            out[s["id"]] = {"conns": conns}
        return out

    @staticmethod
    def _export_of(d, exp, inst, term):
        """The export name whose connector holds terminal `term` ('CONN.PIN' or a port name) of flat instance `inst`, else None."""
        row = next((i for i in d.doc["instances"] if i["id"] == inst), None)
        if row is None:
            return None
        part = d._part(row["part"])
        if (
            "." in term
        ):  # a pin: its connector, and the port it belongs to (an export may name either)
            cids = {PN.split_ref(term)[0], PN.port_of_pin(part, term) or ""}
        else:  # a port: itself, and the connectors its pins sit on
            cids = {term} | {
                PN.split_ref(r)[0]
                for p in part.ports
                if p["name"] == term
                for r in p.get("pins", [])
            }
        return next(
            (
                n
                for n, (i, c) in exp.get("conns", {}).items()
                if i == inst and c in cids
            ),
            None,
        )

    def _sub_terms(self, d, c):
        """Terminals of a sub-assembly block: one per exported connector (kind / type / role from the connector's first pin)."""
        inst = {i["id"]: i for i in d.doc["instances"]}
        terms, blocks = {}, []
        for name, (iid, cid) in (
            self._sub_exports(d).get(c["id"], {}).get("conns", {}).items()
        ):
            row = inst.get(iid)
            if row is None:
                continue
            part = d._part(row["part"])
            cn = part.connectors.get(cid)
            ports = {p["name"]: p for p in part.ports}
            if cn is not None and cn["order"]:  # an exported connector
                pin0 = cn["pins"][cn["order"][0]]
                port = ports.get(PN.port_of_pin(part, pin0["ref"]) or "", {})
                typ, gender, role = cn["type"], cn["gender"], _pin_role(port, pin0)
            elif cid in ports:  # an exported wire port
                port = ports[cid]
                typ, gender, role = (
                    port.get("type", ""),
                    port.get("gender", ""),
                    port.get("role", ""),
                )
            else:
                continue
            terms[name] = {
                "name": name,
                "label": name,
                "kind": port.get("kind", ""),
                "pkind": port.get("kind", "signal"),
                "type": typ,
                "gender": gender,
                "role": role,
                "conn": "",
                "num": "",
                "dir": "",
                "v": None,
                "tip": f"{iid}.{cid}",
                "cell": (0, 0),
            }
            blocks.append({"conn": "", "terms": [name], "two": False})
        c["terms"], c["blocks"] = terms, blocks

    def _build_terms(self, d, strands):
        """Terminals + blocks of every card: c["terms"] = {name: terminal dict}, c["blocks"] = [{conn, terms: [names], two}]."""
        inst = {i["id"]: i for i in d.doc["instances"]}
        used = {}  # (instance, terminal) -> True
        for s in strands:
            used[(s["ia"], s["ta"])] = used[(s["ib"], s["tb"])] = True
        for iid, c in self.cards.items():
            if c.get("sub"):
                self._sub_terms(d, c)
                continue
            part = d._part(inst[iid]["part"])
            conns = part.connectors
            ports = {p["name"]: p for p in part.ports}
            ports.update({p["name"]: p for p in PN.synthetic_ports(part)})
            show = set()
            for n in c["ports"]:
                for r in ports.get(n, {}).get("pins", []):
                    show.add(PN.split_ref(r)[0])
            for (i2, t), _ in used.items():
                if i2 == iid and "." in t:
                    show.add(PN.split_ref(t)[0])
            terms, blocks = {}, []
            for cid, cn in conns.items():
                if cid not in show:
                    continue
                names = []
                for pid in cn["order"]:
                    pin = cn["pins"][pid]
                    port = ports.get(PN.port_of_pin(part, pin["ref"]) or "", {})
                    t = {
                        "name": pin["ref"],
                        "label": f"{pin['id']} {pin['name']}",
                        "kind": pin["kind"],
                        "pkind": port.get("kind", "signal"),
                        "type": cn["type"],
                        "gender": cn["gender"],
                        "role": _pin_role(port, pin),
                        "conn": cid,
                        "num": pin["id"],
                        "dir": pin.get("dir", ""),
                        "v": pin.get("v"),
                        "tip": _tip(cn, pin),
                        "cell": pin["cell"],
                    }
                    terms[pin["ref"]] = t
                    names.append(pin["ref"])
                blocks.append(
                    {
                        "conn": cid,
                        "name": cn["name"],
                        "type": cn["type"],
                        "rows": cn["rows"],
                        "cols": cn["cols"],
                        "terms": names,
                        "two": cn["rows"] == 2,
                    }
                )
            covered = {cn["port_type"] or cn["type"] for cn in conns.values()}
            for n in c["ports"]:
                p = ports.get(n)
                if p is None or p.get("pins") or p.get("synthetic"):
                    continue
                if p.get("type") in covered and (iid, n) not in used:
                    continue  # the detailed connector says it better
                terms[n] = {
                    "name": n,
                    "label": n,
                    "kind": p.get("kind", ""),
                    "pkind": p.get("kind", ""),
                    "type": p.get("type", ""),
                    "gender": p.get("gender", ""),
                    "role": p.get("role", ""),
                    "conn": "",
                    "num": "",
                    "dir": "",
                    "v": None,
                    "tip": "",
                    "cell": (0, 0),
                }
                blocks.append({"conn": "", "terms": [n], "two": False})
            c["terms"], c["blocks"] = terms, blocks

    # ---- layout --------------------------------------------------------------------------------------------------------------
    def _edges(self, strands):
        """[(from card, to card, from term, to term)] of the strands between shown terminals, oriented source -> sink by the roles."""
        out = []
        for s in strands:
            fi, ti, fp, tp = s["ia"], s["ib"], s["ta"], s["tb"]
            if fi == ti or fi not in self.cards or ti not in self.cards:
                continue
            if fp not in self.cards[fi]["terms"] or tp not in self.cards[ti]["terms"]:
                continue
            rf = self.cards[fi]["terms"][fp]["role"]
            rt = self.cards[ti]["terms"][tp]["role"]
            if (
                rf == "sink"
                and rt in ("source", "passthrough")
                or rf == "passthrough"
                and rt == "source"
            ):
                fi, ti, fp, tp = ti, fi, tp, fp
            out.append((fi, ti, fp, tp))
        return out

    def _layout_cards(self, b, d, place=True):
        strands = self._strands(d)
        self._build_terms(d, strands)
        ids = sorted(self.cards)
        inst = {i["id"]: i for i in d.doc["instances"]}
        cat = {
            i: "" if self.cards[i].get("sub") else d._part(inst[i]["part"]).category
            for i in ids
        }
        terms = {i: self.cards[i]["terms"] for i in ids}
        roles = {i: [t["role"] for t in terms[i].values()] for i in ids}
        band = {
            i: 0 if any(t["pkind"] == "power" for t in terms[i].values()) else 1
            for i in ids
        }
        edges = [e for e in self._edges(strands) if band[e[0]] == band[e[1]]]
        layer = {i: _rank(cat[i], roles[i]) for i in ids}
        for _ in range(
            len(ids)
        ):  # relaxation: a wire's target sits right of its source; capped, so a cycle just stops moving
            moved = False
            for f, t, _, _ in edges:
                if layer[t] <= layer[f] and layer[f] + 1 <= 2 * len(ids):
                    layer[t], moved = layer[f] + 1, True
            if not moved:
                break
        cols = {}  # (band, layer) -> [ids]
        for bd in (0, 1):  # compress empty layers per band
            used = sorted({layer[i] for i in ids if band[i] == bd})
            for i in ids:
                if band[i] == bd:
                    layer[i] = used.index(layer[i])
        for i in ids:
            cols.setdefault((band[i], layer[i]), []).append(i)
        nbrs = {i: [] for i in ids}
        for f, t, _, _ in edges:
            nbrs[f].append(t)
            nbrs[t].append(f)
        order = {i: k for col in cols.values() for k, i in enumerate(col)}
        for sweep in range(4):  # barycentre ordering, down then up
            keys = sorted(cols, key=lambda k: (k[0], k[1] if sweep % 2 == 0 else -k[1]))
            for key in keys:
                col = cols[key]

                def bary(i, key=key):
                    ps = [order[n] for n in nbrs[i] if (band[n], layer[n]) != key]
                    return sum(ps) / len(ps) if ps else order[i]

                col.sort(key=lambda i: (bary(i), i))
                for k, i in enumerate(col):
                    order[i] = k
        # horizontal key for "which side do the peers sit on": the layer when the placement is automatic, else the card centres
        if place:
            hx = {i: float(layer[i]) for i in ids}
            same = lambda a_, b_: band[a_] == band[b_]
        else:
            hx = {i: self.cards[i]["x"] + self.cards[i].get("w", 0.0) / 2 for i in ids}
            same = lambda a_, b_: True
        peers = {i: {} for i in ids}  # card -> terminal -> [peer cards]
        for s in strands:
            for ci, ct, oi in (
                (s["ia"], s["ta"], s["ib"]),
                (s["ib"], s["tb"], s["ia"]),
            ):
                if ci in self.cards and oi in self.cards and ci != oi:
                    peers[ci].setdefault(ct, []).append(oi)
        for i in ids:
            c = self.cards[i]
            cnt = {"left": 0, "right": 0}
            for blk in c["blocks"]:
                names = blk["terms"]
                dx = sum(
                    (hx[o] - hx[i]) if same(o, i) else 0
                    for n in names
                    for o in peers[i].get(n, [])
                )
                rs = [terms[i][n]["role"] for n in names]
                if dx > 0:
                    side = "right"
                elif dx < 0:
                    side = "left"
                elif "source" in rs and "sink" not in rs:
                    side = "right"
                elif "sink" in rs and "source" not in rs:
                    side = "left"
                else:
                    side = "left" if cnt["left"] <= cnt["right"] else "right"
                if blk["conn"] and blk["two"]:
                    start = max(cnt["left"], cnt["right"])
                    blk["row"], blk["side"] = start, "both"
                    ncol = max(terms[i][n]["cell"][0] for n in names) + 1
                    for n in names:
                        t = terms[i][n]
                        t["side"] = "left" if t["cell"][1] % 2 == 0 else "right"
                        t["row"] = start + 1 + t["cell"][0]
                    blk["nrows"] = ncol + 1
                    cnt["left"] = cnt["right"] = start + 1 + ncol
                elif blk["conn"]:
                    blk["row"], blk["side"] = cnt[side], side
                    for k, n in enumerate(names):
                        terms[i][n]["side"], terms[i][n]["row"] = (
                            side,
                            cnt[side] + 1 + k,
                        )
                    blk["nrows"] = len(names) + 1
                    cnt[side] += 1 + len(names)
                else:
                    t = terms[i][names[0]]
                    t["side"], t["row"] = side, cnt[side]
                    cnt[side] += 1
            nr = max(1, cnt["left"], cnt["right"])
            lw = [t for t in terms[i].values() if t["side"] == "left"]
            rw = [t for t in terms[i].values() if t["side"] == "right"]
            nm = lambda t: (
                len(t["label"]) * CW_NAME
                + (len(t["type"]) * CW_TYPE if not t["conn"] else 0)
                + PIN_PAD
            )
            lmax, rmax = (
                max((nm(t) for t in lw), default=0),
                max((nm(t) for t in rw), default=0),
            )
            # a pin label stays on its half of the card (the viewer elides what does not fit), so each half holds the longer side
            pin_w = max(lmax + rmax + MID_GAP, 2 * max(lmax, rmax) + MID_GAP / 2)
            # a connector title sits in its box: half the card for a one-sided connector, the whole card for a two-row one
            title_w = max(
                (
                    (len(f"{k['conn']}  {k['name']}") * CW_NAME * 0.8 + 20)
                    * (1 if k["two"] else 2)
                    for k in c["blocks"]
                    if k["conn"]
                ),
                default=0,
            )
            ptxt = c["part"] if c.get("sub") else inst[i]["part"]
            head_w = max(len(i) * CW_HEAD, len(ptxt) * CW_PART) + 24
            c["w"], c["h"] = (
                float(max(MIN_W, pin_w, head_w, title_w)),
                float(HEAD_H + PAD + nr * ROW_H),
            )
            c["band"], c["layer"] = band[i], layer[i]
        if not place:
            return
        # positions: columns left to right, cards stacked and centred on a common mid line per band
        top = 30.0
        for bd in (0, 1):
            keys = sorted(k for k in cols if k[0] == bd)
            if not keys:
                continue
            heights = {
                k: sum(self.cards[i]["h"] for i in cols[k])
                + ROW_GAP * (len(cols[k]) - 1)
                for k in keys
            }
            tall = max(heights.values())
            x = 40.0
            for k in keys:
                cw = max(self.cards[i]["w"] for i in cols[k])
                y = top + (tall - heights[k]) / 2
                for i in cols[k]:
                    c = self.cards[i]
                    c["x"], c["y"] = x + (cw - c["w"]) / 2, y
                    y += c["h"] + ROW_GAP
                x += cw + COL_GAP
            top += tall + BAND_GAP

    # ---- state ---------------------------------------------------------------------------------------------------------------
    def _pin_findings(self, d):
        """{terminal 'inst.name' or wire name: [{code, severity, message}]} from the pin DRC rules (cheap: no build)."""
        out = {}
        try:
            ctx = A.Ctx(A.Assembly(d.doc), d.lib)
            F = []
            pin_rules(ctx, F, _add)
        except Exception:
            return out
        for f in F:
            for w in f.where:
                out.setdefault(w, []).append(
                    {"code": f.code, "severity": f.severity, "message": f.message}
                )
        return out

    @staticmethod
    def _worst(flags):
        return max((f["severity"] for f in flags), key=lambda s: _SEV[s], default="")

    def state(self, b, d) -> dict:
        strands = self._strands(d)
        hid = getattr(
            b, "hidden", set()
        )  # eye-hidden parts: still listed, flagged `hidden` (as the window greys / skips them)
        flags = self._pin_findings(d)
        wired, links = (
            set(),
            {},
        )  # links: (instance, terminal) -> the far ends of its wires (the pin tooltip)
        for s in strands:
            wired.add((s["ia"], s["ta"]))
            wired.add((s["ib"], s["tb"]))
            for i, t, j, u in (
                (s["ia"], s["ta"], s["ib"], s["tb"]),
                (s["ib"], s["tb"], s["ia"], s["ta"]),
            ):
                far = terminal_ref(d, j, u)
                if far not in links.setdefault((i, t), []):
                    links[(i, t)].append(far)
        cards = []
        for c in self.cards.values():
            if c.get("sub"):  # a sub-assembly block: its file stands for the part
                part, grp = c["part"], ""
            else:
                part = next(i["part"] for i in d.doc["instances"] if i["id"] == c["id"])
                grp = next(
                    i.get("group", "") for i in d.doc["instances"] if i["id"] == c["id"]
                )
            pins = []
            for t in sorted(c["terms"].values(), key=lambda t: (t["side"], t["row"])):
                fl = flags.get(f"{c['id']}.{t['name']}", [])
                pins.append(
                    {
                        **{
                            k: t[k]
                            for k in (
                                "name",
                                "label",
                                "kind",
                                "type",
                                "gender",
                                "role",
                                "conn",
                                "num",
                                "dir",
                                "v",
                                "tip",
                                "side",
                                "row",
                            )
                        },
                        "wired": (c["id"], t["name"]) in wired,
                        "port": terminal_ref(d, c["id"], t["name"])["port"],
                        "connects": links.get((c["id"], t["name"]), []),
                        "flag": self._worst(fl),
                        "flags": fl,
                    }
                )
            cards.append(
                {
                    "id": c["id"],
                    "sub": bool(c.get("sub")),
                    "hidden": c["id"] in hid,
                    "part": part,
                    "group": grp,
                    "x": c["x"],
                    "y": c["y"],
                    "w": c["w"],
                    "h": c["h"],
                    "band": c["band"],
                    "layer": c["layer"],
                    "pins": pins,
                    "connectors": [
                        {
                            k: blk[k]
                            for k in (
                                "name",
                                "type",
                                "rows",
                                "cols",
                                "side",
                                "row",
                                "nrows",
                            )
                        }
                        | {"id": blk["conn"]}
                        for blk in c["blocks"]
                        if blk["conn"]
                    ],
                }
            )
        rep = []
        try:
            rep = b.state(d)["report"].get("wires", [])
        except Exception:  # a design that does not build has no lengths
            pass
        length = {r["name"]: r.get("length") for r in rep}
        shown = {
            (c["id"], t["name"])
            for c in self.cards.values()
            for t in c["terms"].values()
        }
        by_wire, hidden = {}, set()
        for s in strands:
            if (s["ia"], s["ta"]) in shown and (s["ib"], s["tb"]) in shown:
                by_wire.setdefault(s["wn"], []).append(s)
            else:
                hidden.add(s["wn"])
        flat = []  # the strands to route: {"name", "from", "to"}
        for name, ss in by_wire.items():
            for s in ss:
                flat.append(
                    {
                        "name": f"{name}#{s['k']}",
                        "from": f"{s['ia']}.{s['ta']}",
                        "to": f"{s['ib']}.{s['tb']}",
                    }
                )
        routes = route_wires(cards, flat)
        wires = []
        for w in d.doc.get("wires", []):
            wn = w.get("name") or f"{w.get('from')}--{w.get('to')}"
            ss = by_wire.get(wn)
            if not ss:
                continue
            part = d._part(w["part"])
            kind = self.cards[ss[0]["ia"]]["terms"][ss[0]["ta"]]["pkind"]
            fl = flags.get(wn, [])
            wires.append(
                {
                    "name": wn,
                    "from": w["from"],
                    "to": w["to"],
                    "part": w["part"],
                    "awg": part.value("awg"),
                    "length": length.get(wn),
                    "kind": kind,
                    "conductors": w.get("conductors", 1),
                    "flag": self._worst(fl),
                    "flags": fl,
                    "hidden": any(s["ia"] in hid or s["ib"] in hid for s in ss),
                    "strands": [
                        {
                            "from": f"{s['ia']}.{s['ta']}",
                            "to": f"{s['ib']}.{s['tb']}",
                            "route": routes[f"{wn}#{s['k']}"],
                            "flag": self._worst(
                                flags.get(f"{s['ia']}.{s['ta']}", [])
                                + flags.get(f"{s['ib']}.{s['tb']}", [])
                            ),
                        }
                        for s in ss
                    ],
                }
            )
        first = {w["name"]: w["strands"][0]["route"] for w in wires}
        labels = place_labels(cards, wires, first)
        for w in wires:
            w["route"], w["label"] = first[w["name"]], labels.get(w["name"])
        return {
            "active": self.active,
            "rev": self.rev,
            "cards": cards,
            "wires": wires,
            "hidden_wires": len(hidden - set(by_wire)),
        }
