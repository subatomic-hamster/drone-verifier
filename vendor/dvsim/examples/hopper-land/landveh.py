"""Density-aware propulsion model of the hopper at altitude: the flight computer's firmware tables of examples/hopper/vehicle.py (unchanged, imported) with the air density as an argument
instead of the hard-coded sea-level RHO0, and the thrust ceiling of the fan.

Standard day only (docs/models.md: the kernel has the ISA troposphere and nothing else): the density follows from the barometer's static pressure alone, rho = p / (R T) with the ISA
temperature of that pressure, T = T0 (p / p0)^(R L / g). A real flight computer would add the MS5611's own temperature reading; here the kernel has no ambient temperature to read.

Thrust ceiling: at full throttle the fan is limited by the pack (voltage sag), not by the blade loading, so the thrust falls slower than the density. Measured on the kernel's model
(scratch run of the built assembly, full duty from rest, peak thrust within 1.5 s, SoC 100 %; T/W with 12.66 N of weight):

    origin_alt m   0      1500   2500   3200   3800   4500   5500
    rho kg/m3      1.225  1.058  0.957  0.891  0.837  0.777  0.697
    T_full N       25.3   23.2   21.8   20.8   20.0   19.0   17.7      -> T_full = 25.3 (rho / 1.225)^0.63, within 1 % over the table

The firmware uses that fit (`t_full`) for its go / no-go check and the thrust limit. It is a fit of THIS simulation, not a data sheet: the 12-blade C_T / C_Q rows come from one vendor throttle
table at sea level (examples/hopper/README.md, honesty notes), their density scaling is the textbook C_T rho n^2 D^4."""

import math

import vehicle as veh

P0, T0, LAPSE, R_AIR = (
    101325.0,
    288.15,
    0.0065,
    287.05287,
)  # ISA sea level; the kernel's constants (aero.cpp)
K_ISA = veh.G / (R_AIR * LAPSE)  # 5.2559
T_FULL0, T_FULL_EXP = 25.3, 0.63  # N at rho = RHO0, exponent (see the table above)
MARGIN_MIN = 1.15  # go / no-go: full-throttle thrust at least this many times the weight (control authority, gust margin)


def isa_rho_from_pressure(p):
    """Air density (kg/m3) of the ISA troposphere at static pressure p (Pa)."""
    t = T0 * (p / P0) ** (1.0 / K_ISA)
    return p / (R_AIR * t)


def isa_rho(h):
    """ISA density at altitude h (m MSL), for the tests and the scene builder."""
    t = T0 - LAPSE * h
    return P0 * (t / T0) ** K_ISA / (R_AIR * t)


def duty_for_thrust(thrust, v_bus, rho, ct=veh.CT0, cq=veh.CQ0):
    """veh.duty_for_thrust at air density `rho` (the motor torque Q = C_Q rho n^2 D^5 and the speed n = sqrt(T / (C_T rho D^4)) both follow the density)."""
    if thrust <= 0.0:
        return 0.0
    n = math.sqrt(thrust / (ct * rho * veh.D_FAN**4))
    w = 2 * math.pi * n
    q = cq * rho * n * n * veh.D_FAN**5
    i_m = q / veh.KE + veh.I0
    v_m = veh.KE * w + i_m * (veh.R_MOTOR + veh.R_ESC)
    return min(1.0, v_m * (1.0 + veh.SW_LOSS) / max(v_bus, 1.0))


def t_full(rho, soc_scale=1.0):
    """Thrust (N) at full throttle on the sagging pack at density rho (fit of the kernel model, SoC near full); soc_scale < 1 for a tired pack."""
    return T_FULL0 * (rho / veh.RHO0) ** T_FULL_EXP * soc_scale


def thrust_margin(rho, mass=veh.MASS):
    """Full-throttle thrust over weight."""
    return t_full(rho) / (mass * veh.G)


def hover_ceiling_m(margin=MARGIN_MIN, mass=veh.MASS):
    """Altitude (m MSL, ISA) where thrust_margin falls to `margin`: the go / no-go limit of the firmware."""
    rho = veh.RHO0 * (margin * mass * veh.G / T_FULL0) ** (1.0 / T_FULL_EXP)
    lo, hi = 0.0, 20000.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if isa_rho(mid) > rho:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
