# EDF hopper: a throttleable thrust-vectored VTVL vehicle from cheap, open parts

A "Grasshopper-style" vertical take-off and landing hopper: one 80 mm 12-blade ducted fan on 6S, four aluminium jet vanes in the exhaust on four small smart servos, a Teensy 4.0
flight computer and a sensor-only flight stack written in Python. It replaces the solid-motor TVC lander (`docs/lander.md`) as the showcase landing vehicle: a solid motor
cannot be throttled or shut down, so the lander's landing is a fixed-impulse problem whose own Delta-v budget (F15: about 0.4 m/s net at 1.45 kg) caps it at 35 % on held-out
seeds. A fan can be throttled to 0 and back, so the hopper hovers, translates, descends at a chosen rate and lands softly.

Everything here is simulated by the dvsim kernel (MuJoCo + device models). The design is `assemblies/hopper.toml`; the flight software never sees truth (the harness plays the world
around it).

## Parts (assemblies/hopper.toml, `dvsim design info`)

Total 1.291 kg, centre of mass 7 mm below the deck's top face (4 mm of x offset; balanced in y by the battery position), standing CG height 0.244 m. Thrust at full duty 33 N on a
loaded 22.2 V (hover needs 12.7 N, duty about 0.57). Prices are what the vendor page showed on 2026-10-08 (US$, one unit, before shipping/tax), `~` = not re-fetched, from memory
of typical list prices.

| Part | Role | Mass | Price | Source |
|---|---|---|---|---|
| QX-Motor 80 mm 12-blade EDF with QF3538 2000 KV (6S), duct + fan | thrust | 0.335 motor (data sheet) + 0.045 fan + 0.06 duct (estimated, +-50 %) | 145 (sale; the page was sold out, 290 list; the bundle includes a 120 A ESC) | qx-motor.com 80 mm EDF page |
| Hobbywing Platinum 120 A V5 ESC | motor drive | 0.150 | 149.90 | originhobbies.com HW30203402 |
| Tattu 2200 mAh 6S 95C | power | 0.321 (data sheet) | ~60 | tattu part file |
| 4x Robotis XL330-M288-T | vane servos (TTL bus, 1 Mbps) | 4 x 0.018 | 27.49 each (listed sold out, 2 month lead time) | robotis.us |
| 4x 45 x 34 mm 6061-T6 vane, 3D-printed deck frame, 4 skid legs | thrust vectoring, structure | 0.0082 each, frame 0.10 | ~30 total | `parts/rocket/generic-jet-vane-alu-45x34`, `hopper-deck-frame-edf80` |
| Teensy 4.0 | flight computer | 0.0035 | ~24 | pjrc.com |
| ICM-42688-P IMU, MS5611 baro | sensors | on the Teensy sled | ~10 each | datasheets |
| u-blox ZED-F9P (SparkFun GPS-RTK-SMA breakout) | GNSS (RTK) | 0.0068 | 259.95 | sparkfun.com GPS-16481 |
| Benewake TF-Luna | downward lidar, 0.2 to 8 m | 0.005 | ~25 | datasheet |
| 2x Pololu D24V50F5 | 5 V regulators for servos / logic | 0.005 each | ~15 each | pololu.com |

Order of magnitude: about US$ 800 to 900. The pieces that carry real uncertainty are marked `estimated` in the part files with a rationale: fan and duct mass, vane geometry,
`jet_factor` 0.9 (nozzle loss), `swirl_recovery`, the servo gear backlash, and the 12-blade C_T / C_Q rows (derived from the vendor's 6S throttle table, one unit, not a test stand).

Wiring (all in the assembly's `[[wires]]`): battery -> ESC (10 AWG) -> motor (3 phase); battery -> two Pololu regulators (18 AWG) -> servo daisy chains (22 AWG, two servos per
regulator, XL330 X3P cables) and Teensy / lidar; Teensy UART to the servo bus; the IMU, baro, lidar and GNSS chips are on the Teensy side of the harness.

## Kernel additions made for it (docs/models.md)

* `JetVane` device: momentum-theory exit speed from the rotor's thrust, flat-plate normal force with a tanh stall, applied at the plate and torqued about the CoM.
* Rotor `stator_body` / `swirl_recovery`: an EDF's stator row hands a share of the fan's reaction torque back to the housing.
* Both appended last in the saved device state, so every earlier state hash is unchanged (lander `1f5922f507aa5ef2`, rocket-hop `5c5ae209e1b52a46`, cuas
  `59fe2bdd7b2009c8`: all three re-run on the final tree, replay match).

## Flight software (examples/hopper/)

| File | What |
|---|---|
| `vehicle.py` | constants of the assembly and the propulsion parts (checked against `dvsim design` by a test), `duty_for_thrust` |
| `estimator.py` | gyro-only attitude (heading and tilt zeroed by the pad calibration), per-axis position / velocity filters fed by IMU + GNSS (delayed fixes), baro and lidar heights |
| `trajectory.py` | min-jerk reference between waypoints |
| `control.py` | cascaded position -> velocity -> attitude; fan thrust loop (accelerometer trim + duty feed-forward); vane allocation: roll / pitch torque by the four vanes' deflections, yaw by the vane pairs' common mode; gyroscopic and fan-spin-up feed-forward |
| `flight.py` | state machine PAD, READY, SPOOL, CLIMB, TRANSIT, HOVER, DESCENT, FINAL (lidar-governed sink rate), TOUCHDOWN, LANDED; ABORT_LAND and FAULT |
| `hop.py` | the program of the scenarios: owns the Sim, plays sensors / servo gear backlash / seeded scatter / faults, writes `trace.csv`, `replay.dvsimog`, `telemetry.jsonl` |
| `scene.py`, `mc.py` | per-flight scene builder (thrust, mass, CG, noise, SoC scatter), Monte Carlo driver |
| `render_frames.py` | GPU stills / frame sequences of a finished run |

Aborts: tilt > 40 deg, battery < 18 V for 0.5 s, GNSS silent for 2 s, off-plan > 12 m or a timeout; an abort cuts the plan and lands where it is.

## Missions (`dvsim program examples/hopper/<name>.scenario.toml`)

| Scenario | Mission | Monitors |
|---|---|---|
| `hop10` | hop to 10 m, hover 5 s, land | touchdown < 0.5 m/s, tilt < 5 deg, < 0.3 m from the pad, no abort, battery 18 V and 15 % left, holds the height within 0.5 m, vane saturation < 10 % |
| `divert25` | climb to 25 m, divert 10 m (8 east, 6 north), precision landing on a second pad, 2 m/s wind | as above, plus "diverts ten metres" |
| `gusts` | 15 m, 6 m divert, 5 m/s wind with Dryden gusts, one servo degraded at 14 s, battery open-circuit step of 1.5 V | as above |

Nominal results, one run each (default seed): all three `dvsim program` runs pass every monitor and end with replay match (hop10 `93cc314eb1d458f0` at tick 26145, divert25
`8fefff2e164b775e` at tick 48045, gusts `777a7d76b2a70483` at tick 35521). hop10 lands 0.033 m from the pad at 0.26 m/s, divert25 0.054 m at 0.18 m/s, gusts 0.042 m at 0.20 m/s
(vane at its limit for 5.6 % of the powered time).

## Monte Carlo (held-out seeds, flown once)

Protocol (`mc.py`): the flight software's one tunable that was tuned (the servo-lag lead `t_lead`) was picked on TRAINING seeds 1000-1079 (80 divert flights per candidate, with and
without the servo fault); the report seeds are different seeds, flown once with the pick. Per-flight scatter drawn from the seed: fan thrust +-5 %, mass +-3 %, CG +-4 mm, servo
stiffness (up to -30 %) and backlash (up to 0.5 deg), IMU / baro / lidar noise (up to +50 %), gyro / accelerometer offsets, SoC (up to -30 %), wind speed (uniform 0.4 to 1 of the
scenario's) with Dryden gusts and a random direction. Success = every monitor of the mission: touchdown < 0.5 m/s, tilt < 5 deg, < 0.3 m from the pad, no abort, vane saturation
< 10 % of the powered time (battery > 18 V and > 15 % SoC left).

History, because the report seeds were used three times (each time a different set; nothing was re-flown on a set already looked at):

| Run | Change | Seeds | divert25 (wind up to 3 m/s) |
|---|---|---|---|
| v1 | `t_lead` 0.035 s (first value) | 5000-5299 | 96.7 % [94.0, 98.2], 7 tilt aborts |
| v2 | `t_lead` 0.065 s, picked on training seeds (servo fault: 41 % -> 99 %) | 9000-9299 | 98.7 % [96.6, 99.5], 2 "no lift-off" faults |
| **v3 (final)** | spool-up keeps ramping while on the ground (fixes the two faults) | 13000-13299 | **99.3 % [97.6, 99.8]** |

The 5000-5031 seeds were also used, before v2, to look at how `t_lead` behaves (a 4 x 4 sweep): disclosed here, and the v2 / v3 numbers are on other seeds. The Wilson 95 % interval
is in brackets.

Final (v3, report seeds 13000+, flown once):

| Mission | Seeds | Success | Touchdown speed p50 / p90 / p99 | Distance to pad p50 / p90 / p99 | Failures |
|---|---|---|---|---|---|
| hop10 | 150 | 99.3 % [96.3, 99.9] | 0.27 / 0.30 / 0.34 m/s | 0.03 / 0.06 / 0.08 m | 1 battery limit |
| divert25 (b), wind up to 3 m/s | 300 | 99.3 % [97.6, 99.8] | 0.24 / 0.34 / 0.48 m/s | 0.05 / 0.09 / 0.15 m | 2 touchdowns above 0.5 m/s |
| gusts (c), 5 m/s + servo fault + 1.5 V step | 200 | 87.5 % [82.2, 91.4] | 0.24 / 0.29 / 0.34 m/s | 0.11 / 0.33 / 10.4 m | 21 landings > 0.3 m from the pad, 2 off-plan aborts, 2 vane saturation |

Mission (c) is the stress case: a steady 5 m/s wind needs about 15 deg of tilt, the yaw trim uses about 5 deg of the vanes' 17 deg, a servo goes soft in flight, and the pack
steps down: precision (< 0.3 m) suffers, speed and tilt do not (touchdown speed < 0.5 m/s in 100 % of the flights).

Sensitivity, divert25, first 60 report seeds of the v3 set (one scatter pushed past its nominal range, the rest as the Monte Carlo; the v2 table on 100 seeds shows the same
picture):

| Pushed | Success | Failures |
|---|---|---|
| nothing (baseline rows, 60 seeds) | 98 to 100 % | at most 1 |
| fan thrust +-10 % | 100 % | none (v2: 90 %, 9 "no lift-off" faults, fixed after) |
| mass +-6 % | 100 % | none |
| centre of mass +-8 mm | 96.7 % | 2 landings 0.3 to 2.7 m off the pad |
| servo stiffness down up to 60 % (all four) | 75 % [63, 84] | 12 tilt aborts: the servo-lag margin is the weak spot |
| servo backlash up to 1.5 deg | 100 % | none |
| sensor noise +150 % | 100 % | none |
| wind up to 5 m/s | 88 % [78, 94] | 6 off the pad (0.3 to 9 m), 1 off-plan abort |
| NEO-M8N GNSS instead of the ZED-F9P RTK | 0 % | 29 of 60 land > 0.3 m off the pad, 25 above 0.5 m/s: a 2.5 m class GNSS cannot do a 0.3 m precision landing, nor hold the velocity loop |
| one servo degraded in flight at 20 s (half the position gain, torque limit 0.35 N m of its 0.52 stall) | 98 % | 1 off the pad |
| battery open-circuit step -2 V in flight | 100 % | none |

Honest reading: with the RTK receiver and the stated scatter the hopper meets the 95 % target for mission (b) with margin; the margin disappears with a 60 % softer servo, with a
non-RTK GNSS or with 5 m/s wind. The numbers describe this simulation, not a vehicle.

## Run it

```
tools/remote.sh run --nolock '.venv/bin/dvsim program examples/hopper/divert25.scenario.toml'
tools/remote.sh run '.venv/bin/python examples/hopper/mc.py --scenario divert25 --train 100 --test 300 --workers 6 --over "{\"wind_ms\":3}" --out /tmp/mc.json'
python -m pytest tests/python/test_hopper.py -q
# stills of a finished run (GPU box, short; the run dir is printed by `dvsim program`)
tools/remote.sh run --gpu --nolock '.venv/bin/python examples/hopper/render_frames.py <run dir> --out build/frames/follow --cam follow --times 14,41'
# a frame sequence for a video: --fps 30 --from 0 --to 48 --cam follow   (cams: close, side, wide, top, follow)
```

Parameters (`[[parameters]]` of every scenario, override with `DVSIM_PARAMS='{"wind_ms": 3}'`): alt_m, hold_s, dx_m, dy_m, wind_ms, wind_dir_deg, wind_fixed, disperse (0 = nominal
vehicle), the tolerances thrust_tol / mass_tol / cg_tol_m / backlash_deg / noise_extra / soc_drop, gnss_rtk, fault_servo, fault_batt_v, t_fault, max_s.

## Telemetry (`telemetry.jsonl`, one JSON object every 48 ms, the lander's backwards-compatible style)

```
t, state
est      {alt_m, vz_ms, x_m, y_m, vx_ms, vy_ms, roll_deg, pitch_deg, yaw_deg, tilt_deg}      estimator output (x east, y north, from the pad)
setpoint {x_m, y_m, alt_m, vz_ms}
throttle                       ESC duty 0..1
vanes    {a1..a4_deg, g1..g4_deg, f1..f4_N}     a = commanded angle, g = measured servo angle, f = vane force from the kernel's vane device
thrust_N, thrust_cmd_N, thrust_est_N
battery  {v, i_a, soc}      motor {rpm}      lidar_m
torque_cmd_Nm {x, y, z}
flags    {abort, vane_sat, yaw_sat, lidar_ok}
truth    {alt_m, vz_ms, x_m, y_m, tilt_deg}      for the overlay, never fed to the flight software
wind_ms  [x, y, z]
```

`trace.csv` (every 16 ms) carries the columns the monitors read (see `COLS` in `hop.py`); `result.json`, `replay.dvsimog` and `native.json` are the usual `dvsim program` outputs.

## Honesty notes

* Physics, not hardware: nothing here has been on a test stand. The 12-blade fan's C_T / C_Q come from one vendor throttle table, the jet-vane force is a flat-plate model with a
  fitted nozzle loss (0.9), there is no hinge moment on the servos, no vane-to-vane wake interaction and no swirl in the vane inflow. A real hopper would need a thrust-stand
  and a vane-force calibration before any claim here transfers.
* The fan's angular momentum (0.11 N m s) is comparable to the attitude control bandwidth: the gyroscopic feed-forward (with a lead for the servo lag) is what keeps the
  nutation stable under a softened servo (before it was added, flights with a 30 % softer servo saturated the vanes or aborted on tilt).
* The Monte Carlo draws the vehicle's scatter from the seed (thrust +-5 %, mass +-3 %, CG, servo stiffness / backlash, sensor noise, SoC, wind): the flight computer knows none of it.
  Training seeds are 1000+, report seeds 5000+; the report seeds are flown once. The sensitivity rows reuse the first 100 report seeds with one parameter pushed past its nominal
  range; they are a robustness picture, not held-out claims.
* Wind: the divert mission is flown at up to 3 m/s mean wind (uniform 0.4 to 1 of that per flight) plus Dryden turbulence. 5 m/s needs about 15 deg of tilt to hold position and
  is at the edge of the vanes' authority (yaw trim alone uses about 5 deg of the 17 deg limit); see the sensitivity table.
* The monitor "vane_saturation" is my own authority-margin metric, not one of the mission criteria.
* The GPU stills show the vehicle and the sky; the scene has no pad marking, so the pad is only in the numbers.

## Versus the solid-motor lander

| | Solid-motor TVC lander | EDF hopper |
|---|---|---|
| Thrust | fixed impulse, no throttle, no re-start | throttleable 0 to 33 N, restartable |
| Mission | fly up and land in one burn | hop, hover, divert, precision landing |
| Budget limit | F15: net Delta-v about 0.4 m/s at 1.45 kg | battery (about 60 % of a 2.2 Ah pack left after the 48 s divert) |
| Held-out success | 35 % (limit documented) | 99.3 % divert-and-land (300 seeds), 99.3 % hop-hover-land, 87.5 % in the 5 m/s + faults stress case |
| Cost | motors are consumables | rechargeable, about US$ 800 to 900 |
