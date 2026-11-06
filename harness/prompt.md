# Task: design and program a VTVL hopper that passes verification

You are designing a small vertical take-off and landing drone in dvsim, a robot design and simulation tool. You work only
through the `dv` tools. Nothing you claim counts: a simulator decides whether your drone works.

## The vehicle

A fixed airframe: a deck frame, one 80 mm ducted fan pointing down, and four flat jet vanes in the exhaust that steer it.
No propellers, no other control surfaces. You choose the rest of the parts and you write the flight controller.

## The mission

From the pad: climb to {alt_m} m (+-{alt_tol_m} m), hold for {hold_s} s, land within {land_radius_m} m of where you took off at no
more than {touchdown_m_s} m/s, staying under {tilt_max_deg} deg of tilt, ending upright, all inside {mission_s} s.

## Hard constraints

- Cost: at most ${cost_max_usd} (catalog prices, whole bill of materials)
- Mass: at most {mass_max_kg} kg all-up
- Size: at most {envelope_mm} mm
- Design rules: 0 errors
- Thrust to weight at full charge: at least {tw_min}
- Battery at landing: at least {soc_min_pct} % state of charge

## How you are judged

`verify` runs these gates in order, and stops at the first one that fails:

| gate          | what | points |
| ------------- | ---- | ------ |
| {gates_table} |

A later gate only runs once the earlier ones pass. If a gate fails, your drone has failed: read why, change the design or the
controller, try again. You have {verify_budget} `verify` calls and {wall_min} minutes. When you stop, your last design and controller
are flown again on seeds you have not seen, and that is your score. A controller tuned to one flight will lose points there.

## Tools

{tools_table}

## The controller

`write_controller` takes one Python file. Imports allowed: `math`, `numpy`. No files, no network, no other imports; the
file is rejected otherwise. It runs at 250 Hz and each step must finish in time, or the last command is held.

```python
class Controller:
    def __init__(self, cfg):
        # cfg: your design's numbers (mass, inertia, max thrust, vane geometry), from design_report
        ...

    def step(self, t, s):
        # s.pos, s.vel  estimated position and velocity, m and m/s, world frame, z up, origin at the pad
        # s.att         estimated attitude quaternion (w, x, y, z), s.rate body rates rad/s
        # s.height      range sensor height above ground, m, or None when there is no return
        # s.vbat        battery voltage, s.armed, raw sensors in s.imu, s.baro, s.gnss
        return {"duty": 0.0, "vanes": [0.0, 0.0, 0.0, 0.0]}
```

{conventions}

All estimates are noisy and delayed. The vehicle starts on its pad, disarmed, for {t_cal_s} s while the estimator calibrates.

## Advice

Start by reading the catalog and getting a legal design; a flight is wasted on a design that fails the static gate. Work out
hover thrust from your mass before you fly. Use `fly` to debug one scenario and read the telemetry it returns; save `verify`
for when you believe it will pass. Stop when `verify` passes every gate, or when you have used your budget. Say briefly what
you are doing as you go: people are watching.
