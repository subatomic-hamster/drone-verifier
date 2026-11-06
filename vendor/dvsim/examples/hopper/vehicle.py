"""What the flight computer of the EDF hopper is built with: constants of assemblies/hopper.toml and of its propulsion parts.

Firmware tables, not sensors: the numbers below are known to the flight computer at build time (the design report, the part data sheets). tests/python/test_hopper.py
checks them against `dvsim design` so they cannot drift from the assembly. Frames: body x forward, y left, z up (the deck's frame); the origin of the `*_DECK` tuples is the
centre of the deck's top face on the axis; `*_REL` tuples are relative to the centre of mass."""

import math

import numpy as np

G = 9.80665  # m/s2
RHO0 = 1.225  # kg/m3, ISA sea level

# ---- mass properties (`dvsim design info` + the built MJCF, both batteries / wires included), 6S pack full -----------------------------------------------------------------
MASS = 1.2913  # kg
CG_DECK = np.array([0.0019, 0.0, -0.0071])  # m, centre of mass in the deck frame
H_STAND = 0.2438  # m, centre of mass above the ground when standing on the four feet
I_BODY = np.array(
    [7.976e-3, 9.782e-3, 8.414e-3]
)  # kg m2, about the centre of mass (principal axes taken as the body axes)
J_FAN = 4.0e-5  # kg m2, fan (3.6e-5, solid-disc upper bound) + motor rotor (4.0e-6): angular momentum of the spinning set about the axis, per rad/s

# ---- sensors, relative to the centre of mass (m) -----------------------------------------------------------------------------------------------------------------
IMU_REL = np.array(
    [-0.0319, 0.088, 0.0094]
)  # the Teensy board (the IMU breakout is on its sled): lever arm of rotation into the accelerometer
LIDAR_REL = np.array(
    [0.0931, 0.0, 0.0031]
)  # underside of the deck, looking along -z (body)
LIDAR_MIN, LIDAR_MAX = 0.2, 8.0  # m, TF-Luna range limits (90 % reflectivity)

# ---- propulsion: QX-Motor 80 mm 12-blade EDF on the QF3538 2000 KV (parts/propulsion), Hobbywing Platinum 120 A -------------------------------------------------------
KV = 2000.0  # rpm/V
R_MOTOR, R_ESC, I0 = 0.010, 0.0015, 2.0  # ohm (winding line-to-line), ohm (ESC FETs), A
SW_LOSS = 0.01  # switching loss fraction of the ESC
D_FAN = 0.080  # m
CT0, CQ0 = (
    1.355,
    0.2102,
)  # static thrust and torque coefficients (T = CT rho n^2 D^4, Q = CQ rho n^2 D^5)
KE = 60.0 / (2 * math.pi * KV)  # V s/rad, = Kt N m/A
Q_UNRECOVERED = 0.2  # share of the fan's air torque the stator vanes do not return to the housing (1 - swirl_recovery 0.8 of assemblies/hopper.toml): the yaw feed-forward
T_MAX_STIFF = 39.2  # N, thrust at full throttle on a stiff 24 V supply (the vendor table); the real pack sags to about 23 N

# ---- jet vanes ----------------------------------------------------------------------------------------------------------------------------------------------------
A_EXIT = 0.0034  # m2, nozzle exit area
VANE_AREA = 0.00153  # m2, plate
CN_ALPHA = 1.9  # 1/rad, small-angle normal-force slope
JET_FACTOR = 0.9  # lump on the momentum-theory exit speed
VANE_LIMIT = math.radians(17.0)  # rad, software stop; the mechanical range is +-20 deg
# Site of each vane's force relative to the centre of mass, and the direction of the force of a POSITIVE servo angle (measured on the built model, tests/python/test_hopper.py):
#   vane1 on +x pushes +y, vane2 on +y pushes -x, vane3 on -x pushes -y, vane4 on -y pushes +x.
VANE_SITE_REL = np.array(
    [
        [0.0181, 0.0, -0.0929],
        [-0.0019, 0.02, -0.0929],
        [-0.0219, 0.0, -0.0929],
        [-0.0019, -0.02, -0.0929],
    ]
)
VANE_DIR = np.array(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]
)

# ---- aerodynamics of the body (the airframe-level terms the controller feeds forward) ------------------------------------------------------------------------------------
K_H = 8.47e-5  # kg, inlet momentum drag coefficient (F = -k_h |w| v_perp, rotor.k_h)

# ---- power ------------------------------------------------------------------------------------------------------------------------------------------------------------
CELLS = 6
V_FULL, V_NOM = 25.2, 22.2  # V
CAPACITY_AH = 2.2
R_PACK = (
    0.071  # ohm (estimated in the pack part; the firmware sees the voltage sag instead)
)

# ---- servo (XL330-M288-T) set-up -------------------------------------------------------------------------------------------------------------------------------------
SERVO_P, SERVO_D = (
    2000,
    6000,
)  # runtime gains (as the lander: the default P 400 sags under load)


def thrust_vane_gain(thrust):
    """dF/d(delta) of one vane (N/rad) at rotor thrust `thrust` (N): q A_v cn_alpha with q = 1/2 jet_factor^2 T / A_e (momentum-theory exit speed)."""
    return 0.5 * JET_FACTOR**2 * max(thrust, 0.0) / A_EXIT * VANE_AREA * CN_ALPHA


def duty_for_thrust(thrust, v_bus, ct=CT0, cq=CQ0):
    """ESC duty that gives `thrust` (N) at bus voltage `v_bus` in still air at sea level: the device model solved forwards (motor torque balance, back-EMF + resistive drops,
    switching loss and no-load current included). Static, no inflow; the accelerometer loop of control.py trims the rest."""
    if thrust <= 0.0:
        return 0.0
    n = math.sqrt(thrust / (ct * RHO0 * D_FAN**4))  # rev/s
    w = 2 * math.pi * n
    q = cq * RHO0 * n * n * D_FAN**5
    i_m = q / KE + I0
    v_m = KE * w + i_m * (R_MOTOR + R_ESC)
    return min(1.0, v_m * (1.0 + SW_LOSS) / max(v_bus, 1.0))


def thrust_for_duty(duty, v_bus, ct=CT0, cq=CQ0):
    """Inverse of duty_for_thrust (static): thrust (N) at ESC duty `duty` and bus voltage `v_bus` (bisection on the monotone map)."""
    lo, hi = 0.0, 80.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if duty_for_thrust(mid, v_bus, ct, cq) < duty:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
