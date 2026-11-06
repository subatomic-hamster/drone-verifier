"""The scriptable UI: every action of the viewer's build UI as a named Backend op (mixed into `viewer.Backend`).

An op does its real work on the Design immediately (so an agent without any window gets the same document), and appends an
event for the window to show: the mode tab switching, search text typed character by character, a ghost flying from the library
card to the drop point, snap guides flashing. `animate=true` on library.drag / gizmo.* defers the commit to the window (queued as
an event the window turns back into the real op when the animation ends) -- only honoured while a window polls (`attached`).

Ops (args -> result):
  ui.mode {name: build|library|wiring|simulate|program}     ui.camera {preset: top|front|side|iso | yaw, pitch, dist}
  ui.frame {instances?}   ui.gizmo {mode: move|rotate|none}   ui.guides {on}   ui.snap {grid_m?, angle_deg?, edges?, ports?}
  ui.view {tooltips?: auto|on|off, emphasise_wires?: bool}   (auto = hover tooltips only while a person moves the mouse: recordings stay
  clean; on = always; off = never. emphasise_wires = cables at least 4 px wide on screen in BUILD / LIBRARY / WIRING, data unchanged)
  ui.hover {instance, seconds?}   show that instance's compact tooltip (about 6 s) even when tooltips are auto / off
  ui.program {file?, scroll?}   PROGRAM page shows that source file (path inside the repo; none = the design TOML), scrolled to a 0..1 fraction
  camera.orbit {target?: [x,y,z] m, yaw_deg?, pitch_deg?, dist?: m, seconds?: 3, relative?: true, ease?: smooth|linear}
      the camera glides (wall-clock `seconds`) to orbit `target` (default: the current target). relative=true: yaw_deg / pitch_deg are
      SWEEPS added to the current angles (360 = one full turn, + = counter-clockwise seen from above; pitch is clamped to +-86 deg);
      relative=false: absolute end angles (yaw 0 looks from +x). dist = end distance in m (a dolly). Works in every 3D page
      (BUILD, LIBRARY, SIMULATE; the WIRING page shows the 2D canvas, its 3D tubes are seen from BUILD).
  camera.fly {eye: [x,y,z], look: [x,y,z], seconds?: 3, ease?}   fly the eye and the look-at point along straight lines to these
  ui.visibility {instances?: [..] (default: all), visible?: true}   the browser's eye: show / hide those parts in the 3D view (view only;
      the document is unchanged)
  ui.measure {a: instance, b: instance} | {off: true}   BUILD: the MEASURE readout between two parts (centre distance, dx dy dz, gap)
  ui.wire_view {zoom?: 0.1..4, fit?: bool, center?: [x, y], seconds?: s, panel?: "open"|"fold"}   the WIRING canvas: fit the sheet, or zoom to that factor (1 = 100 %,
      every pin label shown) with the canvas point `center` (default: the current middle) in the middle of the sheet; `seconds` glides
      there instead of jumping (a recorded pan / zoom)
  ui.wire_pick {instances: [..]}   the WIRING picker: tick exactly these components (all their power / signal terminals); the preview follows
  ui.keys {open?: bool}   the controls popover (keys icon, top bar); no argument toggles
  ui.report {layout: {..}}   the window's own report of what it shows (page, popover rows, status text, DRC chip, rects), kept in ui.state.layout
  ui.state {}  (the Backend.ui dict)      select {instances: [..], add?: bool}
  library.search {text, filters, category, limit, offset, type_ms?}   library.categories {}   library.ranges {category?}
  library.info {part}   library.targets {part}  (ghost poses on every open compatible port)
  library.preview {part, to}  library.drop {part, to? | point?, port?, id?, rpy?}  (= add + snap, ONE undo step)
  library.drag {part, to: {instance, port} | {point: [x, y, z]}, animate?, from?}
  gizmo.move {instance, delta | pos, unmate?, snap?}   gizmo.rotate {instance, axis, angle_deg | rpy | rot, unmate?}
  set_color {instance, rgba | null}   duplicate {instance, delta?}   mirror {instances, axis, at?}
  align {instances, axis, to: min|center|max}   distribute {instances, axis}
World-frame arguments (metres, world +z up); the backend converts to the assembly's root frame (free bases are lifted onto the floor).
"""

import math
from pathlib import Path

import numpy as np

from . import uilib
from .library import DesignError
from .snap import port_frame, rpy_matrix

UI_TOOLS = {
    "ui.mode",
    "ui.camera",
    "camera.orbit",
    "camera.fly",
    "ui.wire_view",
    "ui.wire_pick",
    "ui.measure",
    "ui.visibility",
    "ui.frame",
    "ui.gizmo",
    "ui.guides",
    "ui.snap",
    "ui.view",
    "ui.hover",
    "ui.program",
    "ui.state",
    "ui.keys",
    "ui.report",
    "select",
    "library.search",
    "library.categories",
    "library.ranges",
    "library.my_parts",
    "library.info",
    "library.targets",
    "library.preview",
    "library.drop",
    "library.drag",
    "gizmo.move",
    "gizmo.rotate",
    "set_color",
    "duplicate",
    "mirror",
    "align",
    "distribute",
}
DROP_EXTRA = ("id", "group", "offboard", "joint", "mate", "k", "fasteners")
REPO = Path(__file__).resolve().parents[3]
MODES = ("build", "library", "wiring", "simulate", "program")
PRESETS = {  # yaw, pitch (rad): the orbit camera looks from +x at yaw 0
    "front": (-math.pi / 2, 0.0),
    "side": (0.0, 0.0),
    "top": (-math.pi / 2, 1.5),
    "iso": (-0.6, 0.55),
}
SNAP_PX_M = (
    0.004  # scripted-move snap distance (m); the window uses ~8 px at the current zoom
)


def snap_move(geo: dict, iid: str, delta, edges=True, grid=None, thresh=SNAP_PX_M):
    """Snap a world move `delta` of instance `iid`: its bounding box edges / centre planes align to those of the other instances
    (Figma-style smart guides), then the origin to `grid`. Only axes with a non-zero delta move. Returns (delta, guides) with
    guides = [{axis, value, a, b, kind, other}] (a / b = the aligned feature points on the moving and the other box)."""
    d = np.array(delta, float)
    g = geo[iid]
    lo, hi = np.array(g["lo"]) + d, np.array(g["hi"]) + d
    guides = []
    for ax in range(3):
        if d[ax] == 0:
            continue
        if edges:
            best = None
            mf = {"min": lo[ax], "center": (lo[ax] + hi[ax]) / 2, "max": hi[ax]}
            for oid, o in geo.items():
                if oid == iid or "lo" not in o:
                    continue
                of = {
                    "min": o["lo"][ax],
                    "center": (o["lo"][ax] + o["hi"][ax]) / 2,
                    "max": o["hi"][ax],
                }
                for mk, mv in mf.items():
                    for ok, ov in of.items():
                        e = ov - mv
                        if abs(e) <= thresh and (best is None or abs(e) < abs(best[0])):
                            best = (e, mk, ok, oid, ov)
            if best:
                d[ax] += best[0]
                lo[ax] += best[0]
                hi[ax] += best[0]
                guides.append(
                    {
                        "axis": ax,
                        "value": best[4],
                        "kind": f"{best[1]}-{best[2]}",
                        "other": best[3],
                    }
                )
        if grid:
            p = g["pos"][ax] + d[ax]
            d[ax] += round(p / grid) * grid - p
            guides = [q for q in guides if q["axis"] != ax]
    for q in guides:  # the feature points the guide joins (moving box centre / other box centre, on the guide plane)
        a = ((lo + hi) / 2).tolist()
        o = geo[q["other"]]
        b = ((np.array(o["lo"]) + np.array(o["hi"])) / 2).tolist()
        a[q["axis"]] = b[q["axis"]] = q["value"]
        q["a"], q["b"] = a, b
    return d, guides


class UiOps:
    # ---- world geometry of the build (what the window picks, snaps and highlights) ---------------------------------------------
    def _geo(self, d, xml):
        """{iid: {pos, R, lo, hi, boxes: [{c, R, h}], color, port_world}} in world coordinates + the root body's pose."""

        from .drc import _World

        p = Path(xml)
        w = _World(p.read_text(), p.parent)
        by = {i["id"]: i for i in d.doc["instances"]}
        geo = {}
        for b in w.bodies:
            iid = b["name"]
            if iid not in by:
                continue
            T = b["T"]
            boxes, lo, hi = [], np.full(3, np.inf), np.full(3, -np.inf)
            for gi in b["geoms"]:
                g = w.geoms[gi]
                if not g["collide"]:
                    continue
                c, R, h = w.obb(gi)
                boxes.append(
                    {"c": c.tolist(), "R": R.flatten().tolist(), "h": h.tolist()}
                )
                ext = np.abs(R) @ h
                lo, hi = np.minimum(lo, c - ext), np.maximum(hi, c + ext)
            if not boxes:
                lo, hi = T[:3, 3] - 0.005, T[:3, 3] + 0.005
                boxes = [
                    {
                        "c": T[:3, 3].tolist(),
                        "R": np.eye(3).flatten().tolist(),
                        "h": [0.005] * 3,
                    }
                ]
            pw = {}
            for pt in d._part(by[iid]["part"]).ports:
                if pt.get("kind") != "mech":
                    continue
                P = T @ port_frame(pt)
                pw[pt["name"]] = {"pos": P[:3, 3].tolist(), "axis": P[:3, 2].tolist()}
            geo[iid] = {
                "pos": T[:3, 3].tolist(),
                "R": T[:3, :3].flatten().tolist(),
                "lo": lo.tolist(),
                "hi": hi.tolist(),
                "boxes": boxes,
                "color": by[iid].get("color"),
                "port_world": pw,
            }
        root = w.bodies[0]["T"] if w.bodies else np.eye(4)
        return geo, {"pos": root[:3, 3].tolist(), "R": root[:3, :3].flatten().tolist()}

    def _root(self, d):
        """(R, t) of the root body in the world (identity when the design does not build)."""
        r = self.state(d).get("root_T")
        return (
            (np.array(r["R"]).reshape(3, 3), np.array(r["pos"]))
            if r
            else (np.eye(3), np.zeros(3))
        )

    def _geo_of(self, d):
        return self.state(d).get("geo") or {}

    # ---- plumbing ---------------------------------------------------------------------------------------------------------
    def _event(self, **ev):
        if getattr(self, "_silent", False) and ev.get("cmd") != "select":
            return  # the window itself made this call (silent=true): it has already shown the animation
        self.events.append(ev)
        del self.events[:-500]  # nobody is watching: keep the tail

    def _edit(self, d, ops, **extra):
        r = d.edit(ops)
        return {
            "ok": True,
            "result": {**r.as_dict(), "undo_depth": len(d._undo), **extra},
            "state": self.state(d),
        }

    def _ok(self, result=None, **kw):
        return {"ok": True, "result": result if result is not None else {}, **kw}

    # ---- the ops ------------------------------------------------------------------------------------------------------------
    def _ui_tool(self, tool: str, a: dict) -> dict:
        ui = self.ui
        self._silent = bool(a.get("silent"))
        if tool == "ui.mode":
            if a.get("name") not in MODES:
                raise DesignError(f"ui.mode: name is one of {list(MODES)}")
            ui["mode"] = a["name"]
            self._event(cmd="mode", name=a["name"])
            return self._ok({"mode": a["name"]})
        if tool == "ui.state":
            return self._ok(dict(ui))
        if (
            tool == "ui.keys"
        ):  # the top-bar controls popover (open: bool; absent = toggle)
            ui["keys"] = bool(a.get("open", not ui["keys"]))
            self._event(cmd="keys", open=ui["keys"])
            return self._ok({"open": ui["keys"]})
        if (
            tool == "ui.report"
        ):  # the window states what it shows (a test / agent hook: layout rects, popover rows, status text)
            ui["layout"] = a.get("layout", {})
            return self._ok()
        if tool == "ui.view":
            if a.get("tooltips", ui["tooltips"]) not in ("auto", "on", "off"):
                raise DesignError("ui.view: tooltips is auto, on or off")
            for k in ("tooltips", "emphasise_wires"):
                if k in a:
                    ui[k] = a[k]
            ui["emphasise_wires"] = bool(ui["emphasise_wires"])
            self._event(
                cmd="view",
                tooltips=ui["tooltips"],
                emphasise_wires=ui["emphasise_wires"],
            )
            return self._ok({k: ui[k] for k in ("tooltips", "emphasise_wires")})
        if tool == "ui.hover":
            ids = {i["id"] for i in self._design().doc["instances"]}
            if a.get("instance") not in ids:
                raise DesignError(f"ui.hover: unknown instance {a.get('instance')!r}")
            self._event(
                cmd="hover",
                instance=a["instance"],
                seconds=float(a.get("seconds", 6.0)),
            )
            return self._ok({"instance": a["instance"]})
        if tool == "ui.program":
            name, text = a.get("file", ""), ""
            if name:
                p = (REPO / name).resolve()
                if REPO not in p.parents or not p.is_file():
                    raise DesignError(
                        f"ui.program: {name!r} is not a file inside the repository"
                    )
                text = p.read_text()
            self._event(cmd="program", name=name, text=text, scroll=a.get("scroll"))
            return self._ok(
                {"file": name, "lines": text.count("\n") + 1 if text else 0}
            )
        if tool == "ui.camera":
            if "preset" in a:
                if a["preset"] not in PRESETS:
                    raise DesignError(f"ui.camera: preset is one of {list(PRESETS)}")
                self._event(
                    cmd="camera",
                    preset=a["preset"],
                    yaw=PRESETS[a["preset"]][0],
                    pitch=PRESETS[a["preset"]][1],
                )
            else:
                self._event(
                    cmd="camera",
                    **{k: a[k] for k in ("yaw", "pitch", "dist") if k in a},
                )
            return self._ok()
        if tool in ("camera.orbit", "camera.fly"):
            return self._camera_move(tool, a)
        if tool == "ui.visibility":
            ids = {i["id"] for i in self._design().doc["instances"]}
            want = a.get("instances")
            if want is None:
                want = sorted(ids)
            bad = [i for i in want if i not in ids]
            if bad or not isinstance(want, list):
                raise DesignError(f"ui.visibility: unknown instance(s) {bad}")
            vis = bool(a.get("visible", True))
            self.hidden = (
                self.hidden - set(want) if vis else self.hidden | set(want)
            )  # what flatten.state marks `hidden` (the window keeps its own copy)
            self._event(cmd="visibility", instances=list(want), visible=vis)
            return self._ok({"instances": list(want), "visible": vis})
        if tool == "ui.measure":
            if a.get("off"):
                self._event(cmd="measure", a="", b="")
                return self._ok({"off": True})
            ids = {i["id"] for i in self._design().doc["instances"]}
            for k in ("a", "b"):
                if a.get(k) not in ids:
                    raise DesignError(
                        f"ui.measure: {k} must be an instance id, got {a.get(k)!r}"
                    )
            self._event(cmd="measure", a=a["a"], b=a["b"])
            return self._ok({"a": a["a"], "b": a["b"]})
        if tool == "ui.wire_pick":
            ids = a.get("instances")
            if not isinstance(ids, list):
                raise DesignError("ui.wire_pick: instances is a list of instance ids")
            self._event(cmd="wire_pick", instances=[str(i) for i in ids])
            return self._ok({"instances": ids})
        if tool == "ui.wire_view":
            z = a.get("zoom")
            if z is not None and not (isinstance(z, (int, float)) and 0.1 <= z <= 4):
                raise DesignError(
                    "ui.wire_view: zoom is a number within 0.1..4 (1 = 100 %)"
                )
            pan = {}
            if (
                a.get("center") is not None
            ):  # canvas point that ends in the middle of the sheet
                pan = {"cx": float(a["center"][0]), "cy": float(a["center"][1])}
            if (
                a.get("seconds") is not None
            ):  # glide there (wall clock) instead of jumping
                pan["seconds"] = float(a["seconds"])
            if a.get("panel") is not None:  # the canvas's right panel: "open" | "fold"
                if a["panel"] not in ("open", "fold"):
                    raise DesignError("ui.wire_view: panel is open or fold")
                pan["panel"] = a["panel"]
            self._event(cmd="wire_view", zoom=z, fit=bool(a.get("fit")), **pan)
            return self._ok({"zoom": z, "fit": bool(a.get("fit")), **pan})
        if tool == "ui.gizmo":
            if a.get("mode") not in ("move", "rotate", "none"):
                raise DesignError("ui.gizmo: mode is move, rotate or none")
            ui["gizmo"] = a["mode"]
            self._event(cmd="gizmo", mode=a["mode"])
            return self._ok({"gizmo": a["mode"]})
        if tool == "ui.guides":
            ui["guides"] = bool(a.get("on", True))
            self._event(cmd="guides", on=ui["guides"])
            return self._ok({"guides": ui["guides"]})
        if tool == "ui.snap":
            for key, k2 in (
                ("grid_m", "grid_m"),
                ("angle_deg", "angle_deg"),
                ("edges", "snap_edges"),
                ("ports", "snap_ports"),
            ):
                if key in a:
                    ui[k2] = a[key]
            self._event(
                cmd="snap",
                **{
                    k: ui[k]
                    for k in ("grid_m", "angle_deg", "snap_edges", "snap_ports")
                },
            )
            return self._ok(
                {k: ui[k] for k in ("grid_m", "angle_deg", "snap_edges", "snap_ports")}
            )
        if tool == "library.search":
            text = a.get("text", "")
            ui.update(
                search=text,
                filters=a.get("filters") or {},
                category=a.get("category", ""),
            )
            if (
                a.get("category") == "@my"
            ):  # the MY PARTS category: the workspace's sub-assembly files
                r = self._ui_tool("library.my_parts", {})["result"]
                r["total"] = len(r["parts"])
            else:
                r = uilib.search(
                    self._lib(),
                    text,
                    a.get("filters"),
                    a.get("category", ""),
                    a.get("limit", 60),
                    a.get("offset", 0),
                )
            self._event(
                cmd="search",
                text=text,
                filters=ui["filters"],
                category=ui["category"],
                type_ms=a.get("type_ms", 0),
            )
            return self._ok(r)
        if tool == "library.categories":
            return self._ok({"categories": uilib.categories(self._lib())})
        if tool == "library.ranges":
            return self._ok(
                {"ranges": uilib.param_ranges(self._lib(), a.get("category", ""))}
            )
        if tool == "library.info":
            return self._ok(uilib.info(self._lib(), a["part"]))
        if (
            tool == "library.my_parts"
        ):  # MY PARTS: workspace assemblies with an [exports] table (uilib.my_parts)
            d0 = self.get_design()
            root = (
                Path(a["root"])
                if a.get("root")
                else (
                    Path(d0.path).resolve().parent
                    if d0 is not None and d0.path
                    else REPO
                )
            )
            return self._ok(
                {"parts": uilib.my_parts(root, self._lib() if a.get("mass") else None)}
            )
        if tool == "ui.frame":
            self._event(cmd="frame", instances=a.get("instances") or [])
            return self._ok()
        d = self._design() if tool not in ("library.search",) else None
        if tool == "select":
            ids = a.get("instances") or []
            have = {i["id"] for i in d.doc["instances"]}
            bad = [i for i in ids if i not in have]
            if bad:
                raise DesignError(f"select: unknown instance(s) {bad}")
            ui["selected"] = (ui["selected"] + ids) if a.get("add") else list(ids)
            self._event(cmd="select", instances=list(ids), add=bool(a.get("add")))
            return self._ok({"selected": ui["selected"]})
        if tool == "library.targets":
            part = d._part(a["part"])
            st = self.state(d)
            targets = uilib.ghost_poses(d, st["geo"], d._ports_used(d.doc), part)
            return self._ok(
                {
                    "part": part.id,
                    "targets": targets,
                    "boxes": uilib.part_boxes(d.lib, part),
                }
            )
        if tool == "library.preview":
            part = d._part(a["part"])
            iid = a["to"].partition(".")[0]
            rows = [
                r
                for r in uilib.ghost_poses(
                    d, self.state(d)["geo"], d._ports_used(d.doc), part, only=iid
                )
                if r["to"] == a["to"] or "." not in a["to"]
            ]
            if rows:  # the window flies the ghost to the port and leaves it there; nothing is added
                self._event(
                    cmd="drag",
                    part=part.id,
                    to=rows[0]["to"],
                    then="none",
                    duration=1.2,
                )
            return self._ok(
                {
                    "part": part.id,
                    "target": rows[0] if rows else None,
                    "boxes": uilib.part_boxes(d.lib, part),
                }
            )
        if tool in ("library.drop", "library.drag"):
            return self._drop(d, tool, a)
        if tool == "gizmo.move":
            return self._move(d, a)
        if tool == "gizmo.rotate":
            return self._rotate(d, a)
        if tool == "set_color":
            rgba = a.get("rgba")
            self._event(cmd="color", instance=a["instance"], rgba=rgba)
            return self._edit(d, [{"op": "color", "id": a["instance"], "rgba": rgba}])
        if tool == "duplicate":
            r = self._edit(
                d,
                [
                    {
                        "op": "duplicate",
                        "id": a["instance"],
                        "delta": self._to_root_v(d, a.get("delta", [0.02, 0.02, 0.0])),
                    }
                ],
            )
            self._event(
                cmd="select", instances=r["result"]["changes"][0]["created"], add=False
            )
            return r
        if tool == "mirror":
            R, t = self._root(d)
            ax = "xyz".find(a.get("axis", "y"))
            # world plane coordinate -> root frame (a plane normal to a world axis; exact for the usual axis-aligned root)
            at = float(a.get("at", 0.0))
            n = R[:, ax]
            at_root = at - float(n @ t)
            r = self._edit(
                d,
                [
                    {
                        "op": "mirror",
                        "ids": a["instances"],
                        "axis": a.get("axis", "y"),
                        "at": at_root,
                    }
                ],
            )
            self._event(
                cmd="select", instances=r["result"]["changes"][0]["created"], add=False
            )
            return r
        if tool in ("align", "distribute"):
            return self._align(d, tool, a)
        raise DesignError(f"unknown tool {tool!r}")

    def _camera_move(self, tool: str, a: dict) -> dict:
        """camera.orbit / camera.fly: validate, then one `camera_move` event (the window animates it over `seconds`)."""

        def vec(key, required=False):
            v = a.get(key)
            if v is None:
                if required:
                    raise DesignError(f"{tool}: {key} is required ([x, y, z] in m)")
                return None
            if (
                not isinstance(v, (list, tuple))
                or len(v) != 3
                or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)
            ):
                raise DesignError(f"{tool}: {key} must be [x, y, z] in metres")
            return [float(x) for x in v]

        secs = float(a.get("seconds", 3.0))
        if not 0.0 <= secs <= 600.0:
            raise DesignError(f"{tool}: seconds must be within 0..600")
        ease = a.get("ease", "smooth")
        if ease not in ("smooth", "linear"):
            raise DesignError(f"{tool}: ease is smooth or linear")
        ev = {"cmd": "camera_move", "seconds": secs, "ease": ease}
        if tool == "camera.fly":
            eye, look = vec("eye", True), vec("look", True)
            if math.dist(eye, look) < 1e-4:
                raise DesignError("camera.fly: eye and look must differ")
            ev.update(kind="fly", eye=eye, look=look)
            if a.get(
                "chase"
            ):  # the live-run chase camera: the window drops it while the user holds the camera (FOLLOW off)
                ev["chase"] = True
        else:
            ev.update(
                kind="orbit",
                target=vec("target"),
                relative=bool(a.get("relative", True)),
            )
            for k in ("yaw_deg", "pitch_deg", "dist"):
                if a.get(k) is not None:
                    if not isinstance(a[k], (int, float)) or not math.isfinite(a[k]):
                        raise DesignError(f"{tool}: {k} must be a number")
                    ev[k] = float(a[k])
            if ev.get("dist", 1.0) <= 0:
                raise DesignError("camera.orbit: dist must be > 0")
        self._event(**ev)
        return self._ok({k: v for k, v in ev.items() if k != "cmd"})

    def _lib(self):
        d = self.get_design()
        if d is None:
            raise DesignError(
                "no design open (the library needs the parts directory of a design)"
            )
        return d.lib

    def _to_root_v(self, d, v):
        R, _ = self._root(d)
        return (R.T @ np.array(v, float)).tolist()

    def _to_root_p(self, d, p):
        R, t = self._root(d)
        return (R.T @ (np.array(p, float) - t)).tolist()

    def _drop(self, d, tool, a):
        part = d._part(a["part"])
        if tool == "library.drop" and a.get("silent"):
            a = {**self._queued.pop(part.id, {}), **a}
        to = a.get("to")
        if tool == "library.drag":
            tgt = to or {}
            to = f"{tgt['instance']}.{tgt['port']}" if tgt.get("instance") else None
            a = {**a, "to": to, "point": tgt.get("point")}
            ev = {
                "cmd": "drag",
                "part": part.id,
                "to": to,
                "point": a["point"],
                "port": a.get("port"),
                "from": a.get("from"),
                "id": a.get("id"),
            }
            if (
                a.get("animate") and self.attached
            ):  # the window flies the ghost, then sends library.drop itself
                self._event(**ev, then="drop")
                # the window's drop carries only part / target: id, group ... of this drag are applied to it
                self._queued[part.id] = {
                    k: a[k] for k in DROP_EXTRA if a.get(k) is not None
                }
                return self._ok({"queued": True, "part": part.id})
            self._event(**ev, then="none")
        op = {"op": "add", "part": part.id}
        for k in DROP_EXTRA:  # passed through to the `add` op
            if a.get(k) is not None:
                op[k] = a[k]
        if not any(i.get("root") for i in d.doc["instances"]):
            op["root"] = True
        elif to:
            op["to"] = to
            if a.get("port"):
                op["port"] = a["port"]
        else:
            if a.get("point") is None:
                raise DesignError(
                    "library.drop: give `to` ('inst.port') or `point` [x, y, z] (world)"
                )
            op["pos"] = self._to_root_p(d, a["point"])
            if a.get("rpy"):
                R, _ = self._root(d)
                from .snap import matrix_rpy

                op["rpy"] = matrix_rpy(R.T @ rpy_matrix(a["rpy"]))
        r = self._edit(d, [op])
        self._event(
            cmd="select", instances=[r["result"]["changes"][0]["id"]], add=False
        )
        return r

    def _pose_world(self, d, iid):
        g = self._geo_of(d).get(iid)
        if g is None:
            raise DesignError(f"unknown or offboard instance '{iid}'")
        return g

    @staticmethod
    def _ids(a):
        """`instances` (a group, one undo step) or the single `instance`."""
        ids = a.get("instances") or ([a["instance"]] if a.get("instance") else [])
        if not ids:
            raise DesignError("give `instance` or `instances`")
        return list(ids)

    def _move(self, d, a):
        ids = self._ids(a)
        iid = ids[0]
        geo = self._geo_of(d)
        g = self._pose_world(d, iid)
        for other in ids[1:]:
            self._pose_world(d, other)
        delta = (
            np.array(a["delta"], float)
            if a.get("delta") is not None
            else np.array(a["pos"], float) - np.array(g["pos"])
        )
        guides = []
        if a.get("snap"):
            s = a["snap"] if isinstance(a["snap"], dict) else {}
            grid = (
                s.get("grid_m", self.ui["grid_m"])
                if s.get("grid", False) or "grid_m" in s
                else None
            )
            delta, guides = snap_move(
                geo, iid, delta, edges=s.get("edges", self.ui["snap_edges"]), grid=grid
            )
        dv = self._to_root_v(d, delta)
        r = self._edit(
            d,
            [
                {"op": "place", "id": i, "delta": dv, "unmate": bool(a.get("unmate"))}
                for i in ids
            ],
            guides=guides,
        )
        self._event(cmd="move", instance=iid, delta=delta.tolist(), guides=guides)
        return r

    def _rotate(self, d, a):
        ids = self._ids(a)
        for i in ids:
            self._pose_world(d, i)
        iid = ids[0]
        R, _ = self._root(d)
        op = {"op": "place", "id": iid, "unmate": bool(a.get("unmate"))}
        if a.get("about") is not None or len(ids) > 1:  # a group turns about its centre
            c = (
                np.array(a["about"], float)
                if a.get("about") is not None
                else np.mean([self._geo_of(d)[i]["pos"] for i in ids], axis=0)
            )
            op["about"] = self._to_root_p(d, c)
        if a.get("rot") is not None:
            Rw = np.array(a["rot"], float).reshape(3, 3)
            op["rot"] = (R.T @ Rw @ R).flatten().tolist()
        elif a.get("rpy") is not None:
            from .snap import matrix_rpy

            op["rpy"] = matrix_rpy(R.T @ rpy_matrix(a["rpy"]))
        else:
            ax = np.array(
                a.get("axis", "z")
                if not isinstance(a.get("axis"), str)
                else {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[a["axis"]],
                float,
            )
            ang = float(a["angle_deg"])
            step = a.get("snap_deg", self.ui["angle_deg"] if a.get("snap") else 0)
            if step:
                ang = round(ang / step) * step
            ax = ax / np.linalg.norm(ax)
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            th = math.radians(ang)
            Rw = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)
            op["rot"] = (R.T @ Rw @ R).flatten().tolist()
        if "rpy" in op and len(ids) > 1:
            raise DesignError(
                "gizmo.rotate: rpy sets an absolute pose; use axis + angle_deg for a group"
            )
        r = self._edit(d, [{**op, "id": i} for i in ids])
        self._event(cmd="rotate", instance=iid)
        return r

    def _align(self, d, tool, a):
        geo = self._geo_of(d)
        ids = a["instances"]
        ax = "xyz".find(a.get("axis", "x"))
        if ax < 0 or len(ids) < 2:
            raise DesignError(f"{tool}: axis x|y|z and at least two instances")
        feat = {
            "min": lambda g: g["lo"][ax],
            "center": lambda g: (g["lo"][ax] + g["hi"][ax]) / 2,
            "max": lambda g: g["hi"][ax],
        }
        for i in ids:
            self._pose_world(d, i)
        deltas = {}
        if tool == "align":
            f = feat[a.get("to", "center")]
            ref = (
                a["value"]
                if a.get("value") is not None
                else (
                    min(f(geo[i]) for i in ids)
                    if a.get("to") == "min"
                    else max(f(geo[i]) for i in ids)
                    if a.get("to") == "max"
                    else sum(f(geo[i]) for i in ids) / len(ids)
                )
            )
            deltas = {i: ref - f(geo[i]) for i in ids}
        else:  # equal gaps between the boxes, outermost two stay
            order = sorted(ids, key=lambda i: feat["center"](geo[i]))
            sizes = [geo[i]["hi"][ax] - geo[i]["lo"][ax] for i in order]
            first, last = geo[order[0]], geo[order[-1]]
            gap = (last["lo"][ax] - first["hi"][ax] - sum(sizes[1:-1])) / (
                len(order) - 1
            )
            pos = first["hi"][ax] + gap
            for i, s in zip(order[1:-1], sizes[1:-1]):
                deltas[i] = pos - geo[i]["lo"][ax]
                pos += s + gap
        ops = []
        for i, dv in deltas.items():
            if abs(dv) < 1e-9:
                continue
            v = [0.0, 0.0, 0.0]
            v[ax] = dv
            ops.append(
                {
                    "op": "place",
                    "id": i,
                    "delta": self._to_root_v(d, v),
                    "unmate": bool(a.get("unmate")),
                }
            )
        if not ops:
            return self._ok({"note": "already aligned"}, state=self.state(d))
        r = self._edit(d, ops)
        self._event(cmd="select", instances=list(ids), add=False)
        return r
