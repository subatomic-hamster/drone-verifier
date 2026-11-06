"""L1: what a design can do from its part specs alone, no simulator, in microseconds. Thrust and current come from the fan and motor constants of examples/hopper/vehicle.py (themselves from
the part files), voltage from the pack's full charge and internal resistance, the ESC from its on-resistance, switching loss and continuous rating. Same equations as the sim's steady state."""

import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.environ["DVSIM_ROOT"]) / "examples" / "hopper"))

import vehicle as veh

G = veh.G
_KT = veh.CT0 * veh.RHO0 * veh.D_FAN**4  # thrust / rev_per_s^2
_KQ = veh.CQ0 * veh.RHO0 * veh.D_FAN**5  # torque / rev_per_s^2


def motor_amps(T):
    """Motor current at static thrust T (N)."""
    return _KQ * (T / _KT) / veh.KE + veh.I0


def motor_volts(T, r_esc):
    """Motor terminal volts at static thrust T (N), the ESC's on-resistance included."""
    return veh.KE * 2 * math.pi * math.sqrt(T / _KT) + motor_amps(T) * (
        veh.R_MOTOR + r_esc
    )


def thrust_at_amps(i):
    return max(i - veh.I0, 0.0) * veh.KE * _KT / _KQ


def _duty_and_bus(T, p):
    """Throttle and pack volts holding thrust T: the pack sags with the current it gives."""
    v, ib = p["v_full"], 0.0
    vm = motor_volts(T, p["r_esc"]) * (1 + p["sw_loss"])
    for _ in range(8):
        d = vm / max(v, 1.0)
        ib = min(d, 1.0) * motor_amps(T) + p["i_quiescent"]
        v = p["v_full"] - p["r_pack"] * ib
    return vm / max(v, 1.0), v, ib


def full_throttle(p):
    """The highest thrust with duty <= 1, before the ESC's current limit. -> (thrust, pack amps, pack volts)"""
    lo, hi = 0.1, 100.0
    for _ in range(40):
        T = (lo + hi) / 2
        if _duty_and_bus(T, p)[0] > 1.0:
            hi = T
        else:
            lo = T
    d, v, ib = _duty_and_bus(lo, p)
    return lo, ib, v


def power_specs(b):
    """The pack and ESC numbers L1 needs, from the part files and the catalog's additions. None until both are chosen."""
    from dv import design as D

    c = b.choices
    if not (c.get("battery") and c.get("esc")):
        return None
    lib = D.library()
    pack, esc = lib.get(c["battery"]), lib.get(c["esc"])
    extra = {k: v[0] for k, v in D.option("esc", c["esc"]).get("set", {}).items()}
    val = lambda k, dflt=None: extra.get(k) or esc.value(k) or dflt
    return dict(
        v_full=pack.value("v_full"),
        v_nominal=pack.value("v_nominal"),
        capacity_ah=pack.value("capacity_ah"),
        r_pack=pack.value("r_internal_pack") or 0.0,
        i_pack=pack.value("i_max_continuous"),
        r_esc=val("r_on_total", 0.002),
        sw_loss=val("switching_loss_fraction", veh.SW_LOSS),
        i_quiescent=val("quiescent_current", 0.05),
        i_esc=val("current_continuous"),
    )


def l1(b):
    """-> {ok, reason, tw, thrust_max_n, hover_throttle_pct, hover_power_w, flight_time_s, i_pack_peak_a, i_motor_peak_a, checks: [{name, value, limit, ok, message}]}, None until the design
    builds with a pack and an ESC. `checks` are the named margins; `reason` the failed ones, or the headline when all pass."""
    from dv import design as D

    p = power_specs(b)
    if p is None or not b.report:
        return None
    L = D.LIMITS
    mass = b.mass
    w = mass * G
    T_full, ib_full, v_full = full_throttle(p)
    i_motor_free = motor_amps(T_full)
    thrust = min(T_full, thrust_at_amps(p["i_esc"]))
    tw = thrust / w
    duty, v_hover, ib_hover = _duty_and_bus(w, p)
    power = v_hover * ib_hover
    t_flight = p["capacity_ah"] * (1 - L["soc_min_pct"] / 100.0) * 3600.0 / ib_hover
    checks = []

    def add(name, value, limit, ok, message):
        checks.append(
            dict(
                name=name,
                value=round(value, 3),
                limit=limit,
                ok=bool(ok),
                message=message,
            )
        )

    add(
        "tw",
        tw,
        L["tw_min"],
        tw >= L["tw_min"],
        f"thrust to weight {tw:.2f}, limit {L['tw_min']:g}: {thrust:.1f} N at full throttle for {mass:.3f} kg, pack {v_full:.1f} V, ESC {p['i_esc']:g} A",
    )
    pct = duty * 100.0
    add(
        "hover_throttle",
        pct,
        L["hover_throttle_max_pct"],
        pct <= L["hover_throttle_max_pct"],
        f"hover at {pct:.0f} % throttle, limit {L['hover_throttle_max_pct']:g} %",
    )
    add(
        "hover_power",
        power,
        L["hover_power_max_w"],
        power <= L["hover_power_max_w"],
        f"hover power {power:.0f} W, limit {L['hover_power_max_w']:g} W",
    )
    add(
        "flight_time",
        t_flight,
        L["flight_time_min_s"],
        t_flight >= L["flight_time_min_s"],
        f"flight time {t_flight:.0f} s at hover down to {L['soc_min_pct']:g} % charge, limit {L['flight_time_min_s']:g} s",
    )
    add(
        "pack_current",
        ib_full,
        p["i_pack"],
        ib_full <= p["i_pack"],
        f"pack gives {ib_full:.0f} A at full throttle, rated {p['i_pack']:g} A",
    )
    add(
        "esc_peak_current",
        i_motor_free,
        p["i_esc"],
        i_motor_free <= p["i_esc"],
        f"motor draws {i_motor_free:.0f} A at full throttle, ESC rated {p['i_esc']:g} A",
    )
    bad = [c["message"] for c in checks if not c["ok"]]
    return dict(
        ok=not bad,
        reason="; ".join(bad)
        or f"T/W {tw:.2f}, hover at {pct:.0f} % throttle and {power:.0f} W, {t_flight:.0f} s flight time, {ib_full:.0f} A peak",
        tw=round(tw, 3),
        thrust_max_n=round(thrust, 2),
        hover_throttle_pct=round(pct, 1),
        hover_power_w=round(power, 1),
        flight_time_s=round(t_flight, 1),
        i_pack_peak_a=round(ib_full, 1),
        i_motor_peak_a=round(i_motor_free, 1),
        checks=checks,
    )
