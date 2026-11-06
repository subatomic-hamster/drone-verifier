"""The world around the hopper-land flight computer: parameter resolution (declared scenario knobs + the scatter defaults a scenario file does not list), the seeded physical scatter of one
flight (hop.draw_hidden of examples/hopper, then a terrain wind), the gust and rain schedules, and the mission planner's use of the site survey (cruise height over the terrain between the pads).

Nothing here is seen by the flight computer except what the harness hands it as a sensor reading or as a mission input (landflight.Mission)."""

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "examples" / "hopper"))
sys.path.insert(0, str(HERE))

import hop  # examples/hopper/hop.py (draw_hidden, COLS, ...), the original, unchanged
from dvsim import params as nparams

# Knobs a scenario file need not list: the scatter / tolerances (the values of examples/hopper's scenario files) and the defaults of every environment, mission and deck knob. Scenario files
# declare the ones they set (ranges checked by dvsim.params); Monte Carlo sweeps and tests override any of them through `resolve(over=...)`.
HIDDEN = dict(
    thrust_tol=0.05, mass_tol=0.03, cg_tol_m=0.004, soc_drop=0.3, noise_extra=0.5, backlash_deg=0.5, servo_p_drop=0.3, gyro_bias_dps=0.25, accel_bias_ms2=0.1, gnss_rtk=True,
    fault_servo=False, fault_torque=0.35, fault_p=0.5, fault_batt_v=0.0, t_fault=20.0, w20_ms=-1.0,
    wind_dir_jitter_deg=15.0, wind_spread=0.35, gust_spread=0.3, gust_t_jitter_s=3.0, clear_m=12.0, terrain_seed=0,
    reverse=False, origin_alt=-1.0, wind_field=True, wind_dir_deg=0.0, wind_fixed=False, gust_ms=0.0, gust_len_m=20.0, gust_speed_ms=0.0, gust_t_s=40.0, gust_vert=0.0,
    rain_mmh=0.0, rain_t_on_s=5.0, rain_ramp_s=10.0, hold_s=3.0,
    deck_y_m=6.0, deck_vx_ms=0.0, deck_vy_ms=0.0, deck_heave_m=0.3, deck_heave_s=8.0, deck_roll_deg=2.0, deck_roll_s=9.0, deck_pitch_deg=1.0, deck_pitch_s=7.0,
    deck_rate_hz=10.0, deck_latency_s=0.1, deck_sigma_pos_m=0.03,
    hs_m=0.5, tp_s=0.0, wave_dir_deg=180.0, ship_speed_ms=2.0, ship_t_start_s=2.0, window_s=3.0, window_vz_ms=0.25, window_tilt_deg=3.0, use_window=1.0, window_wait_s=20.0, ship_link_latency_s=0.08, deck_hold_h=0.45, deck_vz_hold=0.15, hold_horizon_s=1.8, hold_lead_s=0.9, hold_max_s=12.0,
)  # fmt: skip


def resolve(scenario, over=None):
    """Declared parameters (dvsim.params.resolve: ranges checked) + HIDDEN defaults; `over` may set either kind. -> dict"""
    over = dict(over or {})
    declared = {p["name"] for p in nparams.declared(scenario)[0]}
    mine = {k: over.pop(k) for k in list(over) if k in HIDDEN and k not in declared}
    p = nparams.resolve(scenario, over)
    for k, v in HIDDEN.items():
        p.setdefault(k, v)
    p.update(mine)
    return p


def draw_hidden(seed, p):
    """hop.draw_hidden (the vehicle scatter, same draw order, so the same seed gives the same vehicle as in examples/hopper) with the wind replaced by the site's: direction `wind_dir_deg` (towards, 0 = +x)
    +- `wind_dir_jitter_deg`, speed `wind_ms` x (1 - `wind_spread` u) with u in [0, 1] (both only when `disperse` and not `wind_fixed`), from a second stream seeded [seed, 8888] so the
    original draws are untouched. Adds the gust (amplitude scatter, timing jitter) and the rain rate."""
    h = hop.draw_hidden(
        seed, {**p, "wind_dir_deg": p["wind_dir_deg"], "wind_ms": p["wind_ms"]}
    )
    r = np.random.default_rng([int(seed), 8888])
    scatter = bool(p["disperse"]) and not p["wind_fixed"]
    dirn = math.radians(
        p["wind_dir_deg"]
        + (r.uniform(-1, 1) * p["wind_dir_jitter_deg"] if scatter else 0.0)
    )
    ws = p["wind_ms"] * (1.0 - (p["wind_spread"] * r.uniform(0, 1) if scatter else 0.0))
    h["wind"] = [ws * math.cos(dirn), ws * math.sin(dirn), 0.0]
    h["wind_dir"] = dirn
    gs = bool(p["disperse"])
    h["gust_amp"] = p["gust_ms"] * (
        1.0 - (p["gust_spread"] * r.uniform(0, 1) if gs else 0.0)
    )
    h["gust_t"] = p["gust_t_s"] + (
        r.uniform(-1, 1) * p["gust_t_jitter_s"] if gs else 0.0
    )
    return h


def gust_command(h, p, land_xy):
    """The `gust` input of the wind field device (docs/models.md): [amplitude, H, dx, dy, dz, delay, speed], or None. The 1 - cos gust has its PEAK at the landing pad at `gust_t` s of flight time:
    a front (speed `gust_speed_ms` > 0, moving along the wind) reaches the world origin at t0 = t_peak - (H + n . x) / c; a uniform gust (speed 0) has s = max(|W|, 2) (t - t0), peak at s = H."""
    if h["gust_amp"] == 0.0:
        return None
    H = float(p["gust_len_m"])
    d = h["wind_dir"]
    n = np.array([math.cos(d), math.sin(d)])
    up = float(
        p["gust_vert"]
    )  # vertical share of the direction vector (negative: downdraft)
    c = float(p["gust_speed_ms"])
    wh = max(math.hypot(h["wind"][0], h["wind"][1]), 2.0)
    t0 = (
        h["gust_t"] - (H + float(n @ np.asarray(land_xy[:2]))) / c
        if c > 0
        else h["gust_t"] - H / wh
    )
    return [h["gust_amp"], H, n[0], n[1], up, max(t0, 0.0), max(c, 0.0)]


def rain_rate_at(t, p):
    """mm/h at flight time t: the rain starts at `rain_t_on_s` and ramps up linearly over `rain_ramp_s`."""
    if p["rain_mmh"] <= 0:
        return 0.0
    return float(p["rain_mmh"]) * float(
        np.clip((t - p["rain_t_on_s"]) / max(p["rain_ramp_s"], 1e-3), 0.0, 1.0)
    )


def plan(info, p):
    """Mission inputs from the site survey (the planner's map): (dx, dy, dz) of the landing pad from the take-off pad, the cruise height above the take-off pad (the highest terrain within 6 m of the straight
    line between the pads + `clear_m`, at least the landing pad's height + `clear_m`), the flat radius of the landing pad. Deck sites: the deck's position at t = 0."""
    ax, ay, az = info.takeoff
    bx, by, bz = info.land
    dz = bz - az
    t = info.terrain
    if t is None:
        return dict(
            dx=bx - ax,
            dy=by - ay,
            dz=dz,
            alt=max(dz, 0.0) + p["clear_m"],
            pad_r=50.0,  # a flat floor: the lidar is a height above the floor everywhere
            deck_half=info.deck.get("half", 1.6) if info.deck else 1.5,
            fixed_pads=not (info.deck and info.deck.get("ship")),
        )
    s = np.linspace(0.0, 1.0, int(math.hypot(bx - ax, by - ay) / 1.0) + 2)
    nx, ny = -(by - ay), bx - ax
    nl = max(math.hypot(nx, ny), 1e-9)
    off = np.array([-6.0, -3.0, 0.0, 3.0, 6.0])
    top = max(
        float(
            np.max(
                t.height_at(
                    ax + s * (bx - ax) + o * nx / nl, ay + s * (by - ay) + o * ny / nl
                )
            )
        )
        for o in off
    )
    alt = max(top - az, dz, 0.0) + float(p["clear_m"])
    return dict(
        dx=bx - ax,
        dy=by - ay,
        dz=dz,
        alt=alt,
        pad_r=info.flat_radius,
    )
