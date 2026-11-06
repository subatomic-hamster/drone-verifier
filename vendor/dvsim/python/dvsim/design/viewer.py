"""The viewer's Design window talks to a Design through this module: JSON request in, JSON reply out. No design logic lives
here (edit.py is the single source of truth, the same `Design` the MCP `design_*` tools drive): each request maps onto one
Design method, the way the matching MCP tool does, and every reply carries the resulting `state` so the window can show
the new build, the DRC findings and the instance / port lists.

Requests: {"tool": "design_edit" | "design_suggest" | "design_undo" | "design_check" | "design_simulate" | "design_save" |
"design_text" | "state" | "rev" | the ui.* / library.* / gizmo.* ops of uiops.py, "args": {...}, "replace_last": bool}. `replace_last` (design_edit only) = undo the last edit, then apply
`ops` (cycling through the placement alternatives of a snap keeps one undo step). Replies: {"ok": true, "result": ...,
"state": ...} or {"ok": false, "error": "..."} (a failed edit changes nothing).
"""

import hashlib
import json
import tempfile
from pathlib import Path

from .library import DesignError
from .progops import Prog
from .runops import Runs
from .thumbjob import ThumbJob
from .uiops import UI_TOOLS, UiOps
from .wire2d import Flatten

SUB_OPS = {  # sub.* tool -> (Design edit op, accepted args)
    "sub.export": ("sub_export", ("kind", "ref", "name")),
    "sub.unexport": ("sub_unexport", ("kind", "name")),
    "sub.insert": (
        "sub_insert",
        ("file", "id", "mate", "pos", "rpy", "mirror", "group", "color"),
    ),
    "sub.mirror": ("sub_mirror", ("id", "plane", "toggle", "to", "new_id")),
    "sub.override": ("sub_override", ("id", "name", "value")),
}


def _involved(doc, where) -> list:
    """Instance ids a finding's `where` entries (instance, 'inst.port', mate / wire name) refer to."""
    ids = {i["id"] for i in doc["instances"]}
    mates = {m.get("name"): (m["a"], m["b"]) for m in doc.get("mates", [])}
    wires = {w.get("name"): (w["from"], w["to"]) for w in doc.get("wires", [])}
    out = []
    for w in where:
        ends = mates.get(w) or wires.get(w) or (w,)
        for e in ends:
            i = e.partition(".")[0]
            if i in ids and i not in out:
                out.append(i)
    return out


class Backend(UiOps):
    """`Backend(lambda: design, out_dir)(request_json) -> reply_json`. `lock`: shared with whoever else touches the Design
    (the MCP server); `save_to(path)`: how a save is written (default `Design.save`); `max_wall_s`: hover wall-clock cap;
    `program_root`: the directory the PROGRAM page edits (default: see progops.Prog.root)."""

    def __init__(
        self,
        get_design,
        out_dir,
        lock=None,
        save_to=None,
        max_wall_s=600.0,
        program_root=None,
    ):
        import threading

        self.get_design, self.out = get_design, Path(out_dir)
        self.prog = Prog(
            self, program_root
        )  # prog.* / term.*: the PROGRAM page's IDE ops (in-process only, never MCP tools)
        self.lock = lock or threading.RLock()
        self.save_to, self.max_wall_s = save_to, max_wall_s
        self.log = []  # [(request dict, reply dict)] of every request, for tests and replay of a UI session
        self._cache = None  # (design, rev, state)
        self.flat = Flatten()  # the 2D wiring mode (wire2d.py): the flatten.* tools
        self.docs = []  # sub-assembly document stack above the base design (sub.new / sub.open); the top one is edited
        self._doc_seq = 0
        self.hidden = set()  # instances switched off with the eye (ui.visibility); flatten.state flags their cards and wires
        self.runs = Runs(self)  # run.* (runops.py)
        self.sim = None  # the host=True Sim the window shows: set by whoever calls Sim.show(design=this); `run.start full` runs in it
        # UI state + the events the window should show (mode switch, typing, ghost flight, guide flash ...): every ui.* /
        # library.* / gizmo.* op appends one, the window drains them with `ui_poll`. With no window attached they just pile up
        # (capped), the ops themselves never depend on a window.
        self.ui = {
            "mode": "build",
            "selected": [],
            "search": "",
            "filters": {},
            "category": "",
            "guides": True,
            "gizmo": "move",
            "grid_m": 0.005,
            "angle_deg": 15.0,
            "snap_edges": True,
            "snap_ports": True,
            "tooltips": "auto",
            "emphasise_wires": True,
            "keys": False,  # the controls popover of the top bar is open
            "layout": {},  # what the window last reported of itself (ui.report): page, popover rows, status text, chip counts, rects
        }
        self.thumbs = ThumbJob(lambda: self._design().lib, self._thumb_priority)
        self.auto_thumbs = False  # True (dvsim design view, the MCP window, the screenshot / record tools): the first ui_poll starts it
        self.events = []
        self._queued = {}  # part id -> extra `add` fields of an animated library.drag (applied to the window's library.drop)
        self.attached = False  # set by the first ui_poll: a window is watching, animated ops queue instead of committing

    # ---- entry point -------------------------------------------------------------------------------------------------
    def __call__(self, text: str) -> str:
        req = {}
        try:
            req = json.loads(text)
            with self.lock:
                reply = self._dispatch(req)
        except DesignError as e:
            reply = {"ok": False, "error": str(e)}
            if str(req.get("tool", "")).startswith("flatten.") and self.attached:
                # a refusal of a scripted flatten op changes no state, so the window would never hear of it: toast it
                msg = f"{req['tool'].partition('.')[2]}: {e}"
                self.events.append({"cmd": "toast", "text": msg, "bad": True})
        except Exception as e:  # never let an exception cross into the viewer thread
            reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        self.log.append((req, reply))
        return json.dumps(reply, default=str)

    def _thumb_priority(self):
        """Parts to render first: those of the open design (the BUILD view), then the library's default listing order."""
        d = self.get_design()
        return sorted({i["part"] for i in d.doc["instances"]}) if d is not None else []

    def _design(self):
        if self.docs:
            return self.docs[-1]["design"]
        d = self.get_design()
        if d is None:
            raise DesignError("no design open")
        return d

    # ---- sub-assembly workspace (subasm.py): the document stack + sub.* ops ------------------------------------------------
    def _docs_state(self) -> list:
        """[{doc_id, path, crumb, dirty}], base design first; crumb = the breadcrumb to that document (list of labels)."""
        base = self.get_design()
        rows = [] if base is None else [(0, base, [base.asm.name])]
        rows += [(e["doc_id"], e["design"], e["crumb"]) for e in self.docs]
        return [
            {
                "doc_id": i,
                "path": str(d.path or ""),
                "crumb": c,
                "title": " > ".join(c),
                "dirty": d.dirty,
            }
            for i, d, c in rows
        ]

    def _push(self, design, label) -> dict:
        self._doc_seq += 1
        base = self.get_design()
        parent = self.docs[-1]["crumb"] if self.docs else [base.asm.name]
        e = {"doc_id": self._doc_seq, "design": design, "crumb": parent + [label]}
        self.docs.append(e)
        return e

    def _reload_open(self, skip=None) -> list:
        """Re-flatten every open design that uses sub-assembly files (after one was saved). -> doc_ids whose flat document changed."""
        base = self.get_design()
        rows = ([(0, base)] if base is not None else []) + [
            (e["doc_id"], e["design"]) for e in self.docs
        ]
        return [i for i, d in rows if d is not skip and d.reload_children()]

    def _sub_tool(self, tool, a) -> dict:
        from . import assembly as A
        from .edit import Design

        d = self._design()
        if tool == "sub.new":
            name = a.get("name") or "untitled"
            asm = A.Assembly(
                {
                    "assembly": {"name": name, "base": a.get("base", "free")},
                    "instances": [],
                }
            )
            e = self._push(Design(asm, d.lib), name)
            return self._doc_reply(e)
        if tool == "sub.open":
            s = d._sub(d.asm.raw, a["id"])
            if d.asm.path is None:
                raise DesignError("sub.open: save the design first")
            p = Path(d.asm.path).resolve().parent / s["file"]
            e = self._push(Design(A.load(p), d.lib, p), f"{s['id']} ({s['file']})")
            return self._doc_reply(e)
        if tool == "sub.close":
            if not self.docs:
                raise DesignError("sub.close: no sub-assembly document is open")
            e, saved = self.docs[-1], None
            if d.dirty and a.get("save", True):
                p = a.get("path") or d.path
                base_path = self.get_design().path
                if p and not Path(p).is_absolute() and base_path:
                    p = (
                        Path(base_path).resolve().parent / p
                    )  # (relative to the base design's folder)
                if not p:
                    raise DesignError(
                        "sub.close: this new part has no file yet: pass path, or save=false to discard it"
                    )
                saved = str(d.save(p))
            self.docs.pop()
            changed = self._reload_open(skip=d)
            return {
                "ok": True,
                "result": {"closed": e["doc_id"], "saved": saved, "reloaded": changed},
                "state": self.state(self._design()),
            }
        if (
            tool == "sub.reload"
        ):  # a child file changed on disk: re-read it (additive op, not in the SP1 plan)
            return {
                "ok": True,
                "result": {"changed": d.reload_children()},
                "state": self.state(d),
            }
        op = SUB_OPS.get(tool)
        if op is None:
            raise DesignError(f"unknown tool {tool!r}")
        return self._edit(d, [{"op": op[0], **{k: a[k] for k in op[1] if k in a}}])

    def _doc_reply(self, e) -> dict:
        return {
            "ok": True,
            "result": {"doc_id": e["doc_id"], "crumb": e["crumb"]},
            "state": self.state(self._design()),
        }

    def _sub_state(self, d) -> dict:
        """Sub-assembly view of one design for the window: its [[subassemblies]], [exports] and the origin of every flattened item."""
        subs = [dict(s) for s in d.asm.raw.get("subassemblies", [])]
        base = Path(d.asm.path).resolve().parent if d.asm.path else None
        return {
            "subs": subs,
            "exports": d.asm.raw.get("exports", {}),
            "origins": d.asm.origins,
            "docs": self._docs_state(),
            # a child file that is gone (deleted / moved after the parent was opened): the tree shows that node in error
            "sub_errors": {
                s["id"]: f"{s['file']}: file not found"
                for s in subs
                if base is not None and not (base / s["file"]).is_file()
            },
        }

    @staticmethod
    def rev_of(d) -> str:
        txt = json.dumps(d.doc, sort_keys=True, default=str)
        if (
            d.asm.raw is not d.doc
        ):  # sub-assemblies: exports / overrides live only in the document as written
            txt += json.dumps(d.asm.raw, sort_keys=True, default=str)
        return hashlib.sha1(txt.encode()).hexdigest()[:12]

    def _rev(self, d) -> str:
        """Window revision: the document's, plus the open document's id while a sub-assembly document is on top."""
        return self.rev_of(d) + (f".{self.docs[-1]['doc_id']}" if self.docs else "")

    def _dispatch(self, req) -> dict:
        tool, a = req.get("tool"), req.get("args") or {}
        if tool == "rev":
            d = self.docs[-1]["design"] if self.docs else self.get_design()
            if d is not None:
                self.flat.prune(
                    d
                )  # a part removed / swapped by any edit path leaves the open canvas
            return {
                "ok": True,
                "result": {
                    "rev": self._rev(d) if d is not None else None,
                    "flat_rev": self.flat.rev,
                    "thumb_rev": self.thumbs.rev,
                },
            }
        if tool == "ui_poll":
            if self.auto_thumbs and not self.thumbs.settled:
                self.thumbs.start()  # a window polls: render the library thumbnails that are missing, in the background
            self.attached = True
            ev, self.events = self.events, []
            return {"ok": True, "result": {"events": ev}}
        if tool in UI_TOOLS:
            return self._ui_tool(tool, a)
        if (tool or "").startswith(
            "run."
        ):  # the Test / Run button (runs user code: not an MCP tool, see runops.py)
            return self.runs.handle(tool, a)
        if (tool or "").startswith(("prog.", "term.")):
            return self.prog.handle(tool, a)
        if (tool or "").startswith("sub."):
            return self._sub_tool(tool, a)
        d = self._design()
        if (tool or "").startswith("flatten."):
            return self.flat.handle(self, d, tool, a)
        if tool == "state":
            return {"ok": True, "result": {}, "state": self.state(d)}
        if tool == "design_edit":
            if req.get("replace_last"):
                d.undo()
                try:
                    r = d.edit(a["ops"])
                except DesignError:
                    d.redo()
                    raise
            else:
                r = d.edit(a["ops"])
            return {
                "ok": True,
                "result": {**r.as_dict(), "undo_depth": len(d._undo)},
                "state": self.state(d),
            }
        if tool == "design_undo":
            r = d.redo() if a.get("redo") else d.undo()
            return {"ok": True, "result": r.as_dict(), "state": self.state(d)}
        if tool == "design_suggest":
            rows = d.suggest(a["target"], a.get("limit", 10), a.get("category") or None)
            return {"ok": True, "result": {"target": a["target"], "suggestions": rows}}
        if tool == "design_text":  # the PROGRAM tab: the assembly as TOML, read-only
            from . import assembly as A

            return {"ok": True, "result": {"text": A.dumps(d.asm)}}
        if tool == "design_check":
            return {
                "ok": True,
                "result": {"findings": [f.as_dict() for f in d.check()]},
                "state": self.state(d),
            }
        if tool == "design_simulate":
            kw = {"wall_s": self.max_wall_s} if a["kind"] == "hover" else {}
            return {"ok": True, "result": d.simulate(a["kind"], out=self.out, **kw)}
        if tool == "design_save":
            path = a.get("path") or str(d.path or self.out / f"{d.asm.name}.toml")
            use_hook = (
                self.save_to and not self.docs
            )  # a sub-assembly document always saves to its own file
            saved = self.save_to(path) if use_hook else d.save(path)
            if self.docs:
                self._reload_open(
                    skip=d
                )  # every open parent that uses this file re-flattens
            # save_to (the MCP tool's) returns {"saved": relative path}; Design.save a Path
            return {
                "ok": True,
                "result": {
                    "saved": saved["saved"] if isinstance(saved, dict) else str(saved)
                },
            }
        raise DesignError(f"unknown tool {tool!r}")

    # ---- the state the Design window shows ------------------------------------------------------------------------------
    def state(self, d) -> dict:
        """Builds the design (cached per document revision) and returns what the window needs. A design that does not
        build (a lost mate leaves a part floating) still has DRC findings and an instance list: `xml` is then null and
        `build_error` says why."""
        rev = self.rev_of(d)
        if self._cache and self._cache[0] is d and self._cache[1] == rev:
            st = self._cache[2]
        else:
            try:
                r = d.build(self.out)
                rep, xml, err = dict(r["report"]), str(r["xml"]), ""
            except DesignError as e:
                rep, xml, err = None, None, str(e)
            fs = d.check()  # the findings the edit results' DRC delta is computed on (not the build report's copy)
            if rep is None:
                rep = {
                    "assembly": d.asm.name,
                    "mass": {"total": 0, "by_group": {}},
                    "instances": [
                        {
                            "id": i["id"],
                            "part": i["part"],
                            "group": i.get("group", ""),
                            "body": i["id"],
                            "mass": 0,
                        }
                        for i in d.doc["instances"]
                    ],
                    "targets": [],
                }
            rep["findings"] = [
                {**f.as_dict(), "instances": _involved(d.doc, f.where)} for f in fs
            ]
            used = d._ports_used(d.doc)
            geo, root_T = self._geo(d, xml) if xml else ({}, None)
            ports = {
                i["id"]: [
                    {
                        "name": p["name"],
                        "kind": p.get("kind", ""),
                        "type": p.get("type", ""),
                        "used": (i["id"], p["name"]) in used,
                        **geo.get(i["id"], {}).get("port_world", {}).get(p["name"], {}),
                    }
                    for p in d._part(i["part"]).ports
                ]
                for i in d.doc["instances"]
            }
            for g in geo.values():
                g.pop("port_world", None)
            st = {
                "rev": rev,
                "xml": xml,
                "build_error": err,
                "report": rep,
                "ports": ports,
                "path": str(d.path or ""),
                "geo": geo,
                "root_T": root_T,
                "root": next(
                    (i["id"] for i in d.doc["instances"] if i.get("root")), ""
                ),
                "free": [
                    i["id"] for i in d.doc["instances"] if "pos" in i or "rpy" in i
                ],
                # what the window's assembly browser lists besides the instances
                "mates": [
                    {"name": m.get("name", ""), "a": m["a"], "b": m["b"]}
                    for m in d.doc.get("mates", [])
                ],
                "wires": [
                    {
                        "name": w.get("name", ""),
                        "from": w["from"],
                        "to": w["to"],
                        "awg": w.get("awg", 0),
                    }
                    for w in d.doc.get("wires", [])
                ],
                "devices": [
                    {"type": v["type"], "name": v["name"]}
                    for v in d.doc.get("devices", [])
                ],
            }
            self._cache = (d, rev, st)
        return {
            **st,
            **self._sub_state(d),
            "rev": self._rev(d),
            "undo": len(d._undo),
            "redo": len(d._redo),
            "history": {  # the HISTORY tab: the last undo steps (oldest first), the redo steps (next first)
                "undo": d._undo_notes[-14:],
                "redo": d._redo_notes[::-1][:6],
            },
        }


def view(design, out=None, **show_kw):
    """Open `design` (a dvsim.design.Design) in the viewer for interactive editing; blocks until the window closes.
    Builds into `out` (default: a temp dir), loads that build, shows it. Returns (Sim, Backend): the Sim holds the model as
    last built in the window, the Backend its request log. `show_kw` = Sim.show's frames / script / width / height."""
    from .. import Sim

    out = Path(out) if out else Path(tempfile.mkdtemp(prefix="dvsim-view-"))
    backend = Backend(lambda: design, out)
    backend.auto_thumbs = True
    st = backend.state(design)
    if st["xml"] is None:
        raise DesignError(f"the design does not build: {st['build_error']}")
    # a Host Sim: the window shows it while `run.start full` drives it from another thread. Its LoadModel root is the build dir
    # `out`, which is where run.start puts the run dirs (out/runs), so a live run's scenario model loads inside the root.
    sim = Sim.load(st["xml"], root=out, host=True)
    backend.sim = sim
    try:
        sim.show(design=backend, **show_kw)
    finally:
        backend.prog.close()  # the window's shells die with it
    return sim, backend
