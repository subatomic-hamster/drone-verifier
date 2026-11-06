# Layered verifier, feedback experiment, trust checks

Adds to DESIGN.md. Cheap checks first, stop at the first failing layer. An agent's design only reaches the simulator after the cheap layers pass.

## Layers

| Layer        | Checks                                                                                                                                              | Gate             | Agent gets (at full feedback)                                        |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- | -------------------------------------------------------------------- |
| L0 static    | cost, mass, envelope, DRC, compatibility: pack voltage vs ESC, motor and regulator input; connector match; ESC and regulator current rating vs draw | G1               | The named rule and both numbers                                      |
| L1 analytic  | T/W, hover throttle, hover power, flight time, peak current vs pack and ESC limits, from part specs only                                            | G2               | Margins: `hover at 78 % throttle, limit 70 %`                        |
| L2 sim       | MuJoCo: the mission under the agent's controller, nominal, wind, rain                                                                               | G3, G4, G5       | Telemetry: max tilt, saturation %, recovery time, miss distance      |
| L3 perturbed | Same mission with randomized mass, thrust, wind and held-out seeds the agent never saw                                                              | G6 + final score | Public runs: pass and telemetry. Held-out result only after `finish` |
| L4 real      | Not available: nothing is built or flown                                                                                                            | none             | none                                                                 |

- L0 and L1 run in well under a second and do not use up a verify call. Only calls that reach the simulator count.
- `fly` also refuses a design failing L0 or L1, with the same message.
- The controller is written by the agent, not auto-tuned, because writing it is part of the benchmark.

## Feedback levels

`dv.server --feedback none|passfail|diag|whatif`. Default `whatif`. Results and events record the level. The prompt names the level and lists only the tools it has.

| Level      | Agent gets from `design_report`, `fly`, `verify`                                                                |
| ---------- | --------------------------------------------------------------------------------------------------------------- |
| `none`     | `{"recorded": true}`. Malformed input (unknown part, wrong slot, code rejected) still errors                    |
| `passfail` | Gate id and pass or fail only. No numbers, no reasons                                                           |
| `diag`     | Everything: named rule, margins, telemetry                                                                      |
| `whatif`   | `diag` plus `what_if(slot, part)`: metric changes from L0 and L1 for one swap, without applying it, not charged |

Filter only what the agent receives. Events, the GUI and the action log always show the full result. Final scoring is the same at every level.

Experiment: pass rate against feedback level, per model. This answers "does verification help?".

## Trust

- **Planted flaws** (`tests/check_planted.py`): each design or controller has one known defect. Report the fraction flagged and at which layer. Hardware: undersized ESC, overloaded low-C pack, pack too small for the mission, wrong pack voltage, connector mismatch, over budget, overweight, under-rated regulator. Controller: no yaw control, no altitude hold, hard landing, ignores wind. Clean designs must pass, so report false positives as well: the reference plus at least 2 other good part sets.
- **Prediction error**: L1 hover throttle and flight time vs measured in L2 on the reference and on every agent's final design. L1 thrust vs the vendor thrust table where a part file has one. This is model against sim and against datasheet. Never present it as real-flight accuracy.
- **Anti-exploit**: agents only pick catalog parts, and all physics comes from part specs. No free parameters exist. Invariant on every flight: pack energy used must not exceed the energy available, and must match integrated power.
- **Second sim**: `harness/crosscheck.py` re-flies final designs at half timestep after the benchmark, and reports score change. Not done during agent runs.

## Metrics

Per run in `results.json`, per model on the leaderboard:

- `feedback`
- `pass_within_budget`: passed G3-G6 inside the verify budget
- `first_viable`: `{iteration, verify_calls, cost_usd, mass_kg}` for the first design passing L0, L1 and G3, else null
- `robustness_gap`: public gates score minus held-out score
- `prediction_error`: `{hover_throttle_pct, flight_time_pct}`
- Cost-versus-mass Pareto front: across all viable designs seen, per model

## Baselines (`harness/baseline.py`)

- **Brute force**: enumerate the whole catalog through L0 and L1, report how many pass, then fly the cheapest and lightest non-dominated few with the reference controller.
- **Human**: the reference design and controller.

Same metrics, same leaderboard, no LLM cost.

## Left out

- Genetic algorithm baseline: brute force covers a catalog this small.
- Real builds and published flight times: the catalog is one EDF hopper, so hobby multirotor builds do not map onto it. L4 stays empty.
- Auto-tuned controller: see above.
