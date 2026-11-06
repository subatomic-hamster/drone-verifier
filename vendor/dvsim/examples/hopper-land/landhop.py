"""Harness of the hopper-land scenarios: the program `dvsim program` runs. examples/hopper/hop.py with a landing pad at another height (or a moving deck), the site's wind field with a gust
front, rain, and the thin air of the site. Owns the Sim, plays the world around the flight computer (sensors, servo gear backlash, seeded scatter, faults, the gust, the rain schedule, the
deck's motion and its datalink), steps landflight.Flight at 250 Hz and writes trace.csv / replay.dvsimog / native.json and telemetry.jsonl.

Everything random is drawn from numpy's PCG64 seeded [seed, 7777] (the vehicle: hop.draw_hidden, unchanged) and [seed, 8888] (landing.draw_hidden: wind direction / speed, gust), or lives inside
the sim (sensor noise, Dryden turbulence, rain: seeded by the model build), so a (seed, parameters) pair replays to the same state hash.

Env: DVSIM_MODEL, DVSIM_RUN_DIR, DVSIM_SCENARIO_DIR (set by `dvsim program`), DVSIM_PARAMS (JSON overrides of the scenario's [[parameters]]), DVSIM_ATTACH (live run in the viewer: the model of the
scenario's [model] script is used as it is). Argument: the scenario file name."""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "examples" / "hopper"))
sys.path.insert(0, str(HERE))

import hop  # examples/hopper/hop.py: draw_hidden, _play, q, tilt_deg, STATE_CODE, LIM, constants (unchanged)
import landdeck
import landing
import landscene
import landship
import dvsim
import vehicle as veh
from landflight import DEFAULTS, Flight, Mission
from landflight import DT as CTRL_DT
from dvsim import params as nparams
from dvsim.marine import write_native

SIM_DT, STRIDE, ROW_EVERY, TLM_EVERY, TOUCH_Z = (
    hop.SIM_DT,
    hop.STRIDE,
    hop.ROW_EVERY,
    hop.TLM_EVERY,
    hop.TOUCH_Z,
)
COLS = hop.COLS + [
    "agl_m",
    "clr_m",
    "rho_kgm3",
    "margin",
    "wind_ms",
    "rain_mmh",
    "z_above_land_m",
    "est_agl_m",
]


def tilt_rel(sim, n):
    """Angle (deg) between the vehicle's axis and the unit vector n (a deck's normal)."""
    w, x, y, z = sim.query("/world/frame", "quat")[0]
    zb = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    return math.degrees(math.acos(max(-1.0, min(1.0, float(zb @ n)))))


def prepare(seed, p, out):
    """Build the model of one flight in `out`. -> (model path, hidden dict, scene Info)"""
    h = landing.draw_hidden(seed, p)
    model, info = landscene.build_model(out, p, seed, h["scatter"], visual=False)
    return model, h, info


def fly(seed, p, model, h, info, run_dir=None, log=True, telemetry=True, max_s=None):
    """One flight. -> dict(res, rows, sim, fl). `p` = landing.resolve(...), `h` = landing.draw_hidden(seed, p), `info` = landscene.Info of the built model."""
    raw = dvsim.Sim.load(str(model))
    ship = bool(info.deck and info.deck.get("ship"))
    sim = (
        landship.PrefixSim(raw) if ship else raw
    )  # a composed ship scene prefixes the vehicle
    if log:
        sim.start_log()
    tw = (
        nparams.TelemetryWriter(Path(run_dir) / "telemetry.jsonl")
        if telemetry and run_dir
        else None
    )
    for s in range(1, 5):
        sim.set(f"/devices/servo{s}", "torque_enable", [1])
        sim.set(f"/devices/servo{s}", "p_gain", [veh.SERVO_P * h["servo_p"]])
        sim.set(f"/devices/servo{s}", "d_gain", [veh.SERVO_D])
        sim.set(f"/devices/servo{s}", "goal_position", [0.0])
    sim.set("/devices/imu", "gyro_bias", h["gyro_bias"])
    sim.set("/devices/imu", "accel_bias", h["accel_bias"])
    sim.set("/devices/air", "wind", h["wind"])
    mis = landing.plan(info, p)
    gust = landing.gust_command(h, p, info.land)
    if gust is not None and info.terrain is not None:
        sim.set("/devices/wind", "gust", gust)
    if ship:
        deck = landship.ShipDeck(raw, info, p, h)
        link = landship.ShipLink(deck, p, seed, info.takeoff)
        link.rover_pos = lambda: np.array(raw.query("/world/hopper_frame", "pos")[0])
    else:
        deck = landdeck.ScriptedDeck(sim, info, p, h) if info.deck else None
        link = (
            landdeck.DeckLink(deck, p, np.random.default_rng([int(seed), 9999]))
            if deck
            else None
        )
    fl = Flight(
        Mission(
            mis["alt"],
            mis["dx"],
            mis["dy"],
            mis["dz"],
            mis["pad_r"],
            p["hold_s"],
            mis.get("deck_half", 1.5),
            mis.get("fixed_pads", True),
        ),
        **{k: p[k] for k in DEFAULTS if k in p},
    )
    fx0, fy0, fz0 = (hop.q(sim, "/world/frame", "pos", i) for i in range(3))
    ax, ay, az = info.takeoff
    lx, ly, lz = info.land
    z_land = lz - az  # landing pad above the take-off pad
    tx, ty = lx, ly
    hs0 = fz0 - az  # standing height of the CoM over its pad
    play = np.zeros(4)
    last_baro, last_gnss = None, None
    rows, res = [], dict(td_speed=99.0, td_tilt=99.0, td_dist=99.0, landed=0)
    z_prev, vz, sat_n, sat_yaw_n, pow_n, touch_tick, z_max = (
        0.0,
        0.0,
        0,
        0,
        0,
        None,
        0.0,
    )
    fault_done = dict(servo=False, batt=False)
    v_min, soc_min, abort_seen = 99.0, 1.0, 0
    t_end_state = None
    rain_last = -1.0
    clr_min, agl_prev, wind_ref = 99.0, 0.0, 0.0
    t_prev_deck_z = None
    sim.step(1)
    for k in range(int((max_s or p["max_s"]) / SIM_DT)):
        if k % STRIDE == 0:
            t = sim.tick * SIM_DT
            if deck is not None:
                deck.drive(t)
            gyro = sim.query("/devices/imu", "gyro")[0]
            accel = sim.query("/devices/imu", "accel")[0]
            pr = hop.q(sim, "/devices/baro", "pressure")
            baro = pr if pr != last_baro and pr > 0 else None
            last_baro = pr
            gn = tuple(float(x) for x in np.concatenate([sim.query("/devices/gnss", "lat")[0], sim.query("/devices/gnss", "lon")[0], sim.query("/devices/gnss", "alt")[0], sim.query("/devices/gnss", "vel_ned")[0]]))  # fmt: skip
            # No fix yet (the receiver acquires): until its first sample has come through the latency the device outputs its ORIGIN (lat0, lon0) -- the world origin, which is not where the pad
            # is. A real receiver reports fix type 0 then; the examples/hopper scene had its pad at the origin and never noticed.
            fix = gn if gn != last_gnss and any(gn) and t >= 0.3 else None
            last_gnss = gn
            rng = hop.q(sim, "/devices/lidar", "range")
            vb = hop.q(sim, "/devices/battery", "voltage")
            ds = (
                (link.sample(t, fl) if ship else link.sample(t))
                if link is not None
                else None
            )
            c = fl.step(t, gyro, accel, baro, fix, rng, vb, deck=ds)
            for i in range(4):
                play[i] = hop._play(
                    play[i],
                    float(np.clip(c.goals[i], -hop.LIM, hop.LIM)),
                    h["backlash"],
                )
                sim.set(f"/devices/servo{i + 1}", "goal_position", [play[i]])
            sim.set("/devices/esc", "throttle", [float(np.clip(c.duty, 0.0, 1.0))])
            if p["rain_mmh"] > 0:
                r = landing.rain_rate_at(t, p)
                if abs(r - rain_last) > 0.05 * p["rain_mmh"] or (
                    r != rain_last and r in (0.0, p["rain_mmh"])
                ):
                    sim.set("/devices/rain", "rate", [r])
                    rain_last = r
            if p["fault_servo"] > 0.5 and not fault_done["servo"] and t >= p["t_fault"]:
                s = h["fault_servo"]
                sim.set(
                    f"/devices/servo{s}", "torque_limit", [p["fault_torque"] * 1000]
                )
                sim.set(
                    f"/devices/servo{s}",
                    "p_gain",
                    [veh.SERVO_P * h["servo_p"] * p["fault_p"]],
                )
                fault_done["servo"] = True
            if (
                p["fault_batt_v"] != 0.0
                and not fault_done["batt"]
                and t >= p["t_fault"] + 3.0
            ):
                sim.set("/devices/battery", "ocv_offset", [-abs(p["fault_batt_v"])])
                fault_done["batt"] = True
            # truth
            fx, fy, fz = (hop.q(sim, "/world/frame", "pos", i) for i in range(3))
            zt = fz - fz0
            vz = (zt - z_prev) / CTRL_DT
            z_prev = zt
            if deck is not None:
                surf_z, vdz = deck.surface(fx, fy, t)
            else:
                surf_z, vdz = lz, 0.0
            hs = (
                fz - surf_z - hs0
            )  # CoM height above its standing height over the landing surface
            vrel = vz - vdz
            thrust = hop.q(sim, "/devices/rotor", "thrust")
            tl = (
                hop.tilt_deg(sim) if deck is None else tilt_rel(sim, deck._pose()[1])
            )  # on a deck: relative to its normal
            # the trace's tilt column: the airframe against the vertical, except on a vessel, where it stands on the deck (against the deck normal); a scripted deck is landed on, not taken off from
            tlw = tl if ship or deck is None else hop.tilt_deg(sim)
            soc = hop.q(sim, "/devices/battery", "soc")
            if thrust > 1.0:
                pow_n += 1
                sat_n += bool(fl.sat)
                sat_yaw_n += bool(fl.sat_yaw)
            if fl.state in (
                "CLIMB",
                "TRANSIT",
                "HOVER",
                "DESCENT",
                "FINAL",
                "ABORT_LAND",
            ):
                v_min, soc_min = min(v_min, vb), min(soc_min, soc)
            abort_seen = max(abort_seen, int(bool(fl.abort_reason)))
            z_max = max(z_max, zt)
            if (
                z_max > 0.5
                and hs <= TOUCH_Z
                and vrel < 0
                and (
                    touch_tick is None
                    or (k - touch_tick < 300 and -vrel > res["td_speed"])
                )  # touchdown speed = the peak approach speed relative to the surface in the 0.3 s after first contact (a deck that heaves under the standing vehicle is not an impact)
            ):
                touch_tick = k if touch_tick is None else touch_tick
                dpx, dpy = deck.center(t) if deck is not None else (tx, ty)
                res.update(
                    td_speed=-vrel,
                    td_tilt=tl,
                    td_dist=math.hypot(fx - dpx, fy - dpy),
                    landed=1,
                )
            if k % ROW_EVERY == 0:
                est = fl.telemetry()
                ex, ey, ea = (
                    est.get("x_m", 0.0),
                    est.get("y_m", 0.0),
                    est.get("alt_m", 0.0),
                )
                he = (
                    abs(zt - mis["alt"])
                    if (fl.state == "HOVER" and t - fl.t_state > 2.0)
                    else 0.0
                )
                if info.terrain is not None:
                    gz = float(info.terrain.height_at(fx, fy))
                    feet = fz - veh.H_STAND - gz
                    far = (
                        min(math.hypot(fx - ax, fy - ay), math.hypot(fx - lx, fy - ly))
                        > info.flat_radius + 3.0
                    )
                    clr = feet if (far and k * SIM_DT > 4.0) else 99.0
                else:
                    feet, clr = fz - veh.H_STAND - surf_z, 99.0
                clr_min = min(clr_min, clr)
                wind_ref = (
                    float(np.linalg.norm(sim.query("/devices/wind", "wind_ref")[0]))
                    if info.terrain is not None
                    else float(np.linalg.norm(h["wind"]))
                )
                rows.append([
                    t, fx - fx0, fy - fy0, zt, vz, tlw, hop.STATE_CODE[fl.state], thrust, ex, ey, ea, math.hypot(ex - (fx - fx0), ey - (fy - fy0)),
                    math.hypot(fx - tx, fy - ty) if deck is None else math.hypot(fx - deck.center(t)[0], fy - deck.center(t)[1]), he,
                    vb, hop.q(sim, "/devices/battery", "current"), soc, hop.q(sim, "/devices/esc", "omega") * 30.0 / math.pi, rng,
                    math.degrees(float(np.max(np.abs(c.goals)))), sat_n / max(pow_n, 1), abort_seen, res["td_speed"], res["td_tilt"], res["td_dist"], res["landed"],
                    math.hypot(fx - fx0, fy - fy0),
                    feet, clr_min if clr_min < 99 else 99.0, hop.q(sim, "/devices/air", "density"), fl.margin, wind_ref,
                    landing.rain_rate_at(t, p), zt - z_land, -1.0 if est.get("agl_m") is None else est["agl_m"],
                ])  # fmt: skip
            if tw and k % TLM_EVERY == 0:
                tw.emit(
                    _record(
                        fl,
                        sim,
                        t,
                        zt,
                        vz,
                        tlw,
                        thrust,
                        fx - fx0,
                        fy - fy0,
                        vb,
                        soc,
                        rng,
                        h,
                        c,
                        p,
                        info,
                        fz,
                    )
                )
            if fl.state in ("LANDED", "FAULT") and t_end_state is None:
                t_end_state = t
            if t_end_state is not None and t - t_end_state > 1.0:
                break
            if touch_tick is not None and k - touch_tick > 6000:
                break
            if sim.diverged:
                break
        sim.step(1)
    if tw:
        tw.close()
    res.update(
        seed=seed, state=fl.state, abort=fl.abort_reason, sat_frac=sat_n / max(pow_n, 1), sat_yaw_frac=sat_yaw_n / max(pow_n, 1), t_end=sim.tick * SIM_DT, diverged=bool(sim.diverged),
        v_min=v_min, soc_min=soc_min, events=fl.events, hidden=h, final_tilt=hop.tilt_deg(sim) if deck is None else tilt_rel(sim, deck._pose()[1]), margin=fl.margin, clr_min=clr_min, mission=mis, rho=fl.est.rho,
    )  # fmt: skip
    return dict(res=res, rows=rows, sim=sim, fl=fl)


def _record(fl, sim, t, zt, vz, tl, thrust, x, y, vb, soc, rng, h, c, p, info, fz):
    """Telemetry record: examples/hopper/hop._record plus the air, the ground under the vehicle and the wind's push."""
    rec = hop._record(fl, sim, t, zt, vz, tl, thrust, x, y, vb, soc, rng, h, c)
    e = fl.telemetry()
    rec["air"] = dict(
        rho=None if e.get("rho") is None else round(e["rho"], 4),
        rho_true=round(hop.q(sim, "/devices/air", "density"), 4),
        margin=None if e.get("margin") is None else round(e["margin"], 3),
    )
    rec["ground"] = dict(
        agl_m=None if e.get("agl_m") is None else round(e["agl_m"], 3),
        push_x=round(e.get("wind_push_x", 0.0), 3),
        push_y=round(e.get("wind_push_y", 0.0), 3),
    )
    if info.terrain is not None:
        rec["truth"]["terrain_m"] = round(
            float(
                info.terrain.height_at(
                    *[hop.q(sim, "/world/frame", "pos", i) for i in range(2)]
                )
            ),
            3,
        )
        rec["wind_ref_ms"] = [
            round(float(v), 2) for v in sim.query("/devices/wind", "wind_ref")[0]
        ]
    rec["rain_mmh"] = round(landing.rain_rate_at(t, p), 1)
    return rec


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else os.environ["HOPPER_LAND_SCENARIO"]
    sc = Path(os.environ.get("DVSIM_SCENARIO_DIR", HERE)) / name
    p = landing.resolve(sc, nparams.env_overrides())
    run_dir = Path(os.environ["DVSIM_RUN_DIR"])
    if os.environ.get(
        "DVSIM_ATTACH"
    ):  # live in the viewer: the scenario's own model is the one the window shows
        h = landing.draw_hidden(p["seed"], p)
        model = Path(os.environ["DVSIM_MODEL"])
        _, info = landscene.build_model(
            run_dir / "model-info", p, p["seed"], h["scatter"], visual=False
        )
    else:
        model, h, info = prepare(p["seed"], p, run_dir / "model")
    r = fly(p["seed"], p, model, h, info, run_dir)
    write_native(r["sim"], SIM_DT, COLS, r["rows"])
    brief = {k: v for k, v in r["res"].items() if k not in ("hidden", "events")}
    (run_dir / "result.json").write_text(json.dumps(brief, default=str, indent=1))
    sys.stdout.write(json.dumps(brief, default=str) + "\n")
    for ev in r["res"]["events"]:
        sys.stdout.write(f"  {ev[0]:7.3f}s {ev[1]}\n")


if __name__ == "__main__":
    main()
