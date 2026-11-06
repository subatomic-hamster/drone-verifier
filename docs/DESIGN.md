# drone-verifier: design

A benchmark that watches AI agents design a drone and proves or disproves their work in simulation. Built on dvsim
(proprietary robot design + MuJoCo sim kernel) as a thin layer; this repo holds only new code. Hackathon project, 4 h
build, 6 min demo.

## Demo (what this is judged on)

1. Live: an agent builds a hopper in dvsim's own GUI. Viewers see parts appear in the 3D view, control code typed into
   the PROGRAM page, a live action log, constraint errors (over budget, over mass), then the flight: it crashes, the
   agent reads the failure, fixes it, flies again.
2. Benchmark: leaderboard of Haiku, Sonnet, Opus, Fable. Then the lowest and highest scorer side by side, verification
   running in real time, so the audience sees the verifier fail one and pass the other.

Anti-goals: a verdict an agent can talk its way past; a score that cannot be reproduced from a run's files; agent code
touching anything on the shared build box; a demo that dies with the network link (every run is recorded and replayable).

## Architecture

```
laptop (or any Mac)                                   hfbox (AMD GPU box, build of dvsim main)
  harness/ -- claude -p, one per model ---MCP(http)---> dv/server.py  one process per session
       streams agent text into the event feed --POST->    tools (filtered dvsim) -> dvsim Design, DRC, BOM
  web/ dashboard (static, any browser)                    verifier: gates -> flights (subprocess per flight)
       <- SSE events, leaderboard ------------- ssh exec  sandbox: agent controller in a child, nobody uid
       <- MJPEG of dvsim's GUI --------------------------  xvfb + dvsim viewer window (Design / Program / Run pages)
```

hfbox refuses port forwarding (`ssh -L`), so the laptop reaches sessions through an ssh-exec bridge: `harness/bridge.py` serves
the allowlisted routes on local ports and carries each request as framed JSON over one `ssh hfbox python3 dv/pipe.py PORT`
(stdin/stdout); the GUI stream is a separate ssh exec of `dv.gui_stream`.

Sessions listen on 127.0.0.1:8801 (A), 8802 (B), 8803, 8804, each with its own design, run dir, MCP endpoint `/mcp`,
events `/events` (SSE) and `/events.jsonl`, stream `/stream.mjpg` (GUI sessions only), and `POST /agent-event`. Off limits
on hfbox: 5177-5194, 8000, 8010, 8093. Benchmark sessions run headless; GUI sessions are for the demo.

Everything else in dvsim stays out of the agent's reach: no shell, no filesystem, no dvsim repo, no reference design.

## The vehicle and the agent's freedom

Airframe kit, fixed: `hopper-deck-frame-edf80`, QX 80 mm 12-blade EDF (duct, fan, QF3538 motor), four jet vanes, Teensy
4.0 + ICM-42688-P + MS5611 (dvsim `assemblies/hopper.toml`). The agent chooses from a curated catalog of real parts:

| slot         | decision the agent faces                                                                           |
| ------------ | -------------------------------------------------------------------------------------------------- |
| battery      | voltage (the fan needs 6S to lift), capacity vs mass vs C rate, position along x to balance the CG |
| esc          | current rating vs price and mass                                                                   |
| vane servo   | torque and speed vs price and mass (slow servos cannot stabilise)                                  |
| regulator    | servo current                                                                                      |
| gnss         | RTK at $260 blows the budget; cheaper fixes are noisier (landing accuracy pays)                    |
| range sensor | lidar vs short-range time-of-flight (range limits the final descent)                               |
| legs         | span vs mass vs envelope (narrow tips over on landing)                                             |

`design_set` applies the choice through dvsim's real `Design` edit ops so the mates, wires and devices are real, the DRC
is dvsim's own, and the GUI updates. Distractors in the catalog must fail for real reasons (4S cannot lift, a 40 A ESC is
under-rated). The catalog excludes nothing that makes the task infeasible: a reference solution must pass every gate.

The agent writes the flight controller: one Python file with `class Controller`, `step(t, s) -> {duty, vanes}` at 250 Hz.
`s` carries a state estimate from dvsim's hopper estimator (position, velocity, attitude, rates; noisy, delayed, never
truth), raw sensors, battery voltage. The agent does not write an estimator. Docs in the prompt give the vehicle's frame
and vane sign conventions and the design's measured numbers; nothing gives gains or a template controller.

## Constraints given up front

Numbers below are provisional: the reference design pins them so it passes with a margin of about 5-10 % and dvsim's stock
hopper (RTK GNSS, 120 A ESC, about $850) fails on cost. Final values live in `dv/task.toml`, the single source for the
prompt, the tools and the gates.

| constraint                      | value                   | checked by |
| ------------------------------- | ----------------------- | ---------- |
| cost (BOM, catalog prices)      | <= $700                 | static     |
| mass, all-up                    | <= 1.35 kg              | static     |
| envelope                        | <= 350 x 350 x 400 mm   | static     |
| design rules                    | 0 errors from dvsim DRC  | static     |
| thrust to weight at full charge | >= 1.8                  | lift       |
| battery left at landing         | >= 20 % state of charge | flight     |

Mission: from the pad, climb to 3.0 m (+-0.3 m), hold 3 s, land within 0.5 m of the takeoff point at <= 1.5 m/s, tilt never
over 30 deg, upright at rest, inside 40 s.

## Verification gauntlet

`verify()` runs the gates in order and returns every result with the first violation and a short trace summary. A failure
means the drone failed: the agent continues until it passes or the budget runs out.

| gate       | what                                                                                            | points |
| ---------- | ----------------------------------------------------------------------------------------------- | ------ |
| G1 static  | cost, mass, envelope, DRC                                                                       | 10     |
| G2 lift    | thrust to weight, battery and ESC current margins in a static thrust run                        | 10     |
| G3 nominal | mission, no disturbance                                                                         | 30     |
| G4 wind    | mission in 3 m/s steady wind + gusts                                                            | 15     |
| G5 rain    | mission in 25 mm/h rain + 2 m/s wind (dvsim rain device: thrust loss, drop drag, lidar dropouts) | 15     |
| G6 robust  | mission on 5 seeds: mass +-5 %, CG +-5 mm, sensor noise, servo backlash; points pro rata        | 20     |

Later gates do not run until earlier ones pass, so a flight is never wasted on an illegal design. `fly(scenario)` flies one
scenario for debugging and returns telemetry; only `verify` scores.

Honesty rules: public seeds during the run; the final score comes from the harness re-running the agent's last design and
controller on held-out seeds after the agent stops. Two controls prove the verifier is not decorative: the reference
solution scores 100 (positive), a do-nothing and an open-loop controller score no flight points (negative). Flights are
recorded (`replay.dvsimog`, hash) so any score can be replayed.

## Scoring and budgets

Score = gate points, held-out seeds, 0 to 100. Ties break on `verify` calls, then wall time, then tokens. Each agent gets the
same prompt, 25 min wall clock, 12 `verify` calls. Reported alongside: flights flown, tokens, time to first passing flight,
gate reached per attempt, the failure reasons in order. With 1-2 trials per model this is a demonstration, not a statistic;
the leaderboard says n.

## Agents

Each model runs as Claude Code headless (`claude -p --model ...`), the user's choice of platform, isolated: empty working
directory, no built-in tools, no settings, memory or plugins from the host, one MCP server. That keeps the runs comparable
and stops an agent from reading dvsim's reference hopper. The initial prompt is `harness/prompt.md`, generated values from
`dv/task.toml`.

## Agent code on a shared box

hfbox runs jobs as root on a shared machine with other people's data, no `unshare`, no `bwrap`. So agent code gets layered
limits, none of them a true sandbox: AST allowlist before anything runs (imports `math`, `numpy` only; no dunder access, no
`eval`, `exec`, `open`, `getattr`, ...), a child process that drops to uid `nobody`, empty environment, empty working dir,
rlimits (CPU, memory, no files, no new processes), a per-step deadline (a late step holds the last command and counts
against the run). Model API keys stay on the laptop. Good enough for benchmark models; not for untrusted submissions. A
test tries the escapes and must fail them.

## What dvsim provides (vendored)

dvsim is vendored at `vendor/dvsim` (a committed subset of dvsim main, commit and date in `vendor/dvsim/VENDORED`): C++ kernel and
Vulkan viewer sources, `python/dvsim` (`dvsim.design`: Design, DRC, BOM, build, viewer Backend), the part library with prices,
`assemblies/hopper.toml`, `examples/hopper` (flight stack, estimator, scene builder) and the rain code of `examples/hopper-land`.
Build it with dvsim's own build (`uv pip install -e .` in `vendor/dvsim`; third-party sources are fetched by cmake at build time).
`DVSIM_ROOT` is the only path knob: it names the vendored tree (or any dvsim tree) and `dv/` reads parts, assembly and
`examples/hopper` from it. Never edit `vendor/dvsim` here; re-vendor from dvsim instead.
