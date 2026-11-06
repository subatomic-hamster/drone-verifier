"""One session: design, controller, budgets, events; the tool bodies. Tools run in worker threads and hold the session lock, /state never takes it."""

import json
import os
import threading
import time
from pathlib import Path

from dv import design as D
from dv import heldout, verify
from dv.events import Events
from dv.sandbox import Controller, check_source

TASK = D.TASK
RUNS = lambda: Path(os.environ.get("DV_RUNS", "runs"))
FILTERED = ("design_set", "design_report", "fly", "verify", "finish")
FLY_GATE = dict(nominal="G3", wind="G4", rain="G5")


class NullUI:
    """The GUI seam (dv/gui.py ViewerUI implements it). Headless sessions use this."""

    def design_changed(self, design):
        pass

    def show_code(self, text, path="controller.py"):
        pass

    def fly_live(self, scenario, spec=None):
        return None

    def page(self, name):
        pass

    def stop(self):
        pass


class Session:
    def __init__(
        self, name, model="unknown", trial=1, ui=None, run_dir=None, feedback="whatif"
    ):
        self.name, self.model, self.trial = name, model, trial
        self.feedback = feedback
        self.run_dir = Path(run_dir) if run_dir else RUNS() / f"{name}-{model}-{trial}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(
            self.run_dir, 0o700
        )  # the agent's child runs as nobody; it has no business here
        (self.run_dir / "meta.json").write_text(
            json.dumps(dict(model=model, trial=trial, session=name, feedback=feedback))
        )
        self.ev = Events(name, self.run_dir / "events.jsonl")
        self.ev.emit("config", feedback=feedback, model=model, trial=trial)
        self.ui = ui or NullUI()
        self.lock = threading.RLock()
        self.choices, self.code = {}, ""
        self.built = None
        self.verify_left = TASK["budget"]["verify"]
        self.wall_s = TASK["budget"]["wall_min"] * 60.0
        self.t_first = None
        self.score = None
        self.finished = None
        self.attempt = 0
        self.flights = 0
        self.broken = {}
        self.meters = None
        self.apply({})

    # ---- plumbing
    def _ui(self, fn, *a):
        try:
            return getattr(self.ui, fn)(*a)
        except Exception as e:  # noqa: BLE001  # the GUI must never break a tool
            self.ev.emit("agent_text", text=f"[gui] {fn}: {e}")
            return None

    def wall_left(self):
        return (
            None
            if self.t_first is None
            else max(0.0, self.wall_s - (time.time() - self.t_first))
        )

    def budget(self):
        left = self.wall_left()
        return dict(
            verify_left=self.verify_left,
            wall_left_s=round(self.wall_s if left is None else left, 1),
        )

    def emit_budget(self):
        self.ev.emit("budget", **self.budget())

    def call(self, tool, args, fn, summary=None):
        """Runs one tool: first-call clock, events, the refusal once the budget is gone. Tool errors come back as {ok: false, error}."""
        with self.lock:
            if self.t_first is None:
                self.t_first = time.time()
            shown = {
                k: (f"<{len(v)} chars>" if isinstance(v, str) and len(v) > 200 else v)
                for k, v in args.items()
            }
            self.ev.emit("tool_call", tool=tool, args=shown)
            if self.finished is not None and tool != "finish":
                out = dict(ok=False, error="the session is finished")
            elif (
                tool
                not in (
                    "finish",
                    "task_info",
                    "parts_search",
                    "design_report",
                    "what_if",
                )
                and self.wall_left() == 0.0
            ):
                out = dict(
                    ok=False, error="the wall clock budget is used up: call finish"
                )
            else:
                try:
                    out = fn(**args)
                except (D.DesignError, ValueError, TypeError, KeyError) as e:
                    out = dict(
                        ok=False,
                        error=f"{type(e).__name__}: {e}"
                        if not isinstance(e, D.DesignError)
                        else str(e),
                    )
            self.ev.emit(
                "tool_result",
                tool=tool,
                ok=bool(out.get("ok", True)),
                summary=out.get("summary") or out.get("error") or "",
            )
            self.emit_budget()
            return self.for_agent(tool, out)

    def for_agent(self, tool, out):
        """What the agent receives at this feedback level. Events, the GUI and the run's files always have the full result."""
        if self.feedback in ("diag", "whatif") or tool not in FILTERED:
            return out
        if "error" in out and "gate" not in out:  # malformed input still errors
            return out
        if self.feedback == "none":
            return {"recorded": True}
        if tool in ("verify", "finish") and "gates" in out:
            gates = [{"id": g["id"], "pass": g["pass"]} for g in out["gates"]]
        elif tool == "fly" and "scenario" in out:
            gates = [{"id": FLY_GATE[out["scenario"]], "pass": out["pass"]}]
        else:
            bad = out.get("gate") or (verify.layers(self.built) or {}).get("gate")
            gates = [{"id": g, "pass": g != bad} for g in ("G1", "G2")]
            gates = gates[:1] if bad == "G1" else gates
        return {"gates": gates}

    # ---- design
    def apply(self, choices):
        b = D.apply(choices)
        self.choices, self.built = dict(choices), b
        l1 = b.l1 or {}
        tw = l1.get("tw") if b.xml and not b.missing else None
        (self.run_dir / "design.json").write_text(json.dumps(self.choices, indent=1))
        L = TASK["limits"]
        self.meters = dict(
            cost_usd=round(b.cost, 2),
            mass_kg=round(b.mass, 4),
            envelope_mm=b.envelope_mm,
            tw=tw,
            hover_throttle_pct=l1.get("hover_throttle_pct"),
            hover_power_w=l1.get("hover_power_w"),
            flight_time_s=l1.get("flight_time_s"),
            drc_errors=len(b.errors),
            limits=dict(
                cost_usd=L["cost_max_usd"],
                mass_kg=L["mass_max_kg"],
                envelope_mm=L["envelope_mm"],
                tw=L["tw_min"],
                hover_throttle_pct=L["hover_throttle_max_pct"],
                hover_power_w=L["hover_power_max_w"],
                flight_time_s=L["flight_time_min_s"],
            ),
        )
        self.ev.emit("meters", **self.meters)
        self.ev.emit(
            "drc",
            errors=[dict(rule=e["code"], message=e["message"]) for e in b.errors],
            warnings=len(b.warnings),
        )
        cons = D.static_violations(b) + l1.get("checks", [])
        out = []
        for c in cons:
            if c["name"] == "slots" or (c["name"] == "drc" and b.missing):
                continue
            was_ok = self.broken.get(c["name"], True)
            if c["ok"] != was_ok:
                self.ev.emit("constraint", **c)
            self.broken[c["name"]] = c["ok"]
            if not c["ok"]:
                out.append(c["message"])
        self._ui("design_changed", b.design)
        return b, out

    def design_set(self, choices):
        merged = {**self.choices, **choices}
        b, broken = self.apply(merged)
        p = b.props or {}
        self._ui("page", "design")
        return dict(
            ok=not broken and not b.missing,
            choices=merged,
            unchosen_slots=b.missing,
            cost_usd=round(b.cost, 2),
            mass_kg=round(b.mass, 4),
            envelope_mm=b.envelope_mm,
            thrust_to_weight=self.meters["tw"],
            cg_mm=[round(v * 1000, 1) for v in p["cg"]] if p else None,
            broken_constraints=broken,
            design_errors=[f"{e['code']}: {e['message']}" for e in b.errors][:8],
            margins=[c["message"] for c in (b.l1 or {}).get("checks", [])],
            summary=f"${b.cost:.0f}, {b.mass:.3f} kg, T/W {self.meters['tw']}, {len(broken)} broken"
            + (f", unchosen: {b.missing}" if b.missing else ""),
        )

    def what_if(self, slot, part):
        """L0 and L1 of one swap, not applied. Uses no budget."""
        if slot not in D.SLOTS:
            raise D.DesignError(f"unknown slot '{slot}'. Slots: {', '.join(D.SLOTS)}")
        if not part:
            raise ValueError("part is empty")
        now, then = self.built, D.apply({**self.choices, slot: part})

        def nums(b):
            l1 = b.l1 or {}
            return dict(
                cost_usd=round(b.cost, 2),
                mass_kg=round(b.mass, 4),
                thrust_to_weight=l1.get("tw"),
                **{k: l1.get(k) for k in ("hover_throttle_pct", "hover_power_w", "flight_time_s", "i_pack_peak_a", "i_motor_peak_a")},
            )  # fmt: skip

        a, z = nums(now), nums(then)
        broken = [c["message"] for c in D.static_violations(then) + (then.l1 or {}).get("checks", []) if not c["ok"] and c["name"] != "slots" and not (c["name"] == "drc" and then.missing)]  # fmt: skip
        return dict(
            slot=slot,
            part=part,
            current=now.choices.get(slot),
            before=a,
            after=z,
            change={k: round(z[k] - a[k], 3) for k in a if a[k] is not None and z[k] is not None},
            would_break=broken,
            summary=f"{slot} -> {part}: ${then.cost:.0f}, {then.mass:.3f} kg, {len(broken)} broken",
        )  # fmt: skip

    def design_report(self):
        b = self.built
        complete = bool(b.xml) and not b.missing
        return dict(
            choices=self.choices,
            unchosen_slots=b.missing,
            cost_usd=round(b.cost, 2),
            mass_kg=round(b.mass, 4),
            envelope_mm=b.envelope_mm,
            cost_by_kind=b.cost_by_kind,
            bom=b.bom and b.bom["lines"],
            design_errors=[f"{e['code']}: {e['message']}" for e in b.errors],
            design_warnings=len(b.warnings),
            mass_props=b.props
            and {k: b.props[k] for k in ("mass", "cg", "inertia", "h_stand")},
            analytic=b.l1,
            thrust_to_weight=self.meters["tw"],
            cfg=D.controller_cfg(b) if complete else None,
            summary="design report",
        )

    def parts_search(self, slot=None, query=None):
        return dict(slots=D.catalog(slot, query), summary=f"catalog {slot or 'all'}")

    def task_info(self):
        return dict(
            mission=TASK["mission"],
            limits=TASK["limits"],
            gates=[
                dict(id=g["id"], name=g["name"], what=g["what"], points=g["points"])
                for g in TASK["gates"]
            ],
            budget=self.budget(),
            scenarios=list(TASK["scenarios"]),
            design_chosen=self.choices,
            controller_written=bool(self.code),
            summary="task info",
        )

    # ---- controller
    def write_controller(self, code):
        errs = check_source(code)
        self._ui("page", "program")
        if not errs:
            self.code = code
            (self.run_dir / "controller.py").write_text(code)
            self._ui("show_code", code, str(self.run_dir / "controller.py"))
            b = self.built
            if b.xml and not b.missing:
                c = Controller(code, D.controller_cfg(b), deadline_s=1.0, init_s=5.0)
                if not c.dead:
                    c.step(
                        0.0,
                        dict(
                            pos=[0.0] * 3,
                            vel=[0.0] * 3,
                            att=[1.0, 0.0, 0.0, 0.0],
                            rate=[0.0] * 3,
                            height=0.0,
                            vbat=24.0,
                            armed=True,
                            imu=dict(gyro=[0.0] * 3, accel=[0.0, 0.0, 9.8]),
                            baro=101325.0,
                            gnss=None,
                        ),
                    )
                if c.error:
                    errs = [dict(line=1, message=c.error)]
                c.close()
        ok = not errs
        self.ev.emit("code", text=code, lines=code.count("\n") + 1, ok=ok, errors=errs)
        return dict(
            ok=ok,
            errors=errs,
            lines=code.count("\n") + 1,
            summary=f"controller {code.count(chr(10)) + 1} lines"
            if ok
            else f"rejected: {errs[0]['message']} (line {errs[0]['line']})",
        )

    # ---- flights
    def _replay(self, scenario, passed, reason):
        d = self.run_dir / "flight_last"
        if (d / "replay.dvsimog.gz").exists():
            self._ui(
                "fly_live",
                scenario,
                dict(
                    run_dir=str(d),
                    model=str(d / "scene.xml"),
                    passed=passed,
                    reason=reason,
                ),
            )

    def fly(self, scenario):
        if scenario not in TASK["scenarios"]:
            raise ValueError(f"unknown scenario '{scenario}': nominal, wind or rain")
        if self.built.missing:
            return dict(ok=False, error=f"choose parts for: {', '.join(self.built.missing)}")  # fmt: skip
        bad = verify.layers(self.built)
        if bad:
            return dict(ok=False, error=f"{bad['gate']} fails, not flown: {bad['reason']}", gate=bad["gate"])  # fmt: skip
        if not self.code:
            return dict(ok=False, error="no controller: call write_controller")
        self._ui("page", "run")
        seed = verify.seeds_for(TASK["public"]["seed"])[scenario]
        log = None
        if scenario == "nominal":
            log = str(self.run_dir / "flight_last" / "replay.dvsimog")
            Path(log).parent.mkdir(parents=True, exist_ok=True)
        spec = verify.make_spec(self.built, self.code, scenario, seed, log=log)
        r = verify.fly(spec, self.built, self.ev.emit)
        verify.compress_log(spec)
        self.flights += 1
        if log:
            self._replay(scenario, r["pass"], r["reason"])
        return dict(
            ok=r["pass"],
            **{"pass": r["pass"]},
            scenario=scenario,
            reason=r["reason"],
            first_violation=r["first_violation"],
            metrics=r["metrics"],
            telemetry=r["telemetry"],
            summary=f"{scenario}: {'pass' if r['pass'] else 'fail'}: {r['reason']}",
        )

    def verify(self):
        # only a call that reaches the simulator uses up the budget
        if self.verify_left <= 0 and self.code and not verify.layers(self.built):
            return dict(ok=False, error="no verify calls left: call finish")
        self.attempt += 1
        self._ui("page", "run")
        r = verify.run_gates(
            self.built,
            self.code,
            TASK["public"]["seed"],
            emit=self.ev.emit,
            log_dir=self.run_dir,
        )
        if r["flights"]:
            self.verify_left -= 1
        self.flights += r["flights"]
        self.score = r["score"]
        self.ev.emit("verify", attempt=self.attempt, gates=r["gates"], score=r["score"], charged=bool(r["flights"]))  # fmt: skip
        nom = next(g for g in r["gates"] if g["id"] == "G3")
        if nom["reason"] and not nom["reason"].startswith("not run"):
            self._replay("nominal", nom["pass"], nom["reason"])
        bad = next((g for g in r["gates"] if not g["pass"]), None)
        return dict(
            ok=bad is None,
            attempt=self.attempt,
            score=r["score"],
            gates=r["gates"],
            verify_left=self.verify_left,
            summary=f"score {r['score']}"
            + (f", failed {bad['id']}: {bad['reason']}" if bad else ", all gates pass"),
        )

    def finish(self, note=""):
        """Runs the held-out rerun and writes results.json. Idempotent."""
        if self.finished is not None:
            return self.finished
        t0 = self.t_first or time.time()
        self.ev.emit("agent_text", text=f"[finish] {note}"[:2000])
        res = heldout.run(self.run_dir, self.model, self.trial, emit=self.ev.emit)
        self.ev.emit(
            "final", score=res["score"], gates=res["gates"], verify_calls=self.attempt, flights=self.flights, wall_s=round(time.time() - t0, 1), held_out=True,
        )  # fmt: skip
        self.score = res["score"]
        self.finished = dict(
            ok=True,
            score=res["score"],
            gates=res["gates"],
            summary=f"held-out score {res['score']}",
        )
        return self.finished

    def state(self):
        return dict(
            session=self.name,
            model=self.model,
            trial=self.trial,
            feedback=self.feedback,
            meters=self.meters,
            budget=self.budget(),
            score=self.score,
            finished=self.finished is not None,
        )
