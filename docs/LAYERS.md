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

### Partial credit

`dv/partial.py`, numbers in `task.toml [partial]`. Alongside the strict `score`, `gates` and `first_test_score`, which do not change; nothing here reaches an agent.

- **One flight**, 0 to 1. A passing flight is 1.0. A failed flight sums five parts at weight 0.2 each, multiplies by a tilt factor, and is capped at 0.95:
  - lift-off: 1 if peak height reached 0.3 m. Without it every part is 0.
  - altitude: peak / (target - tolerance), at most 1, falling to 0 at one target height above the band.
  - hold: longest time in the band / hold time, at most 1.
  - landing: 1 up to the touchdown limit, falling to 0 at 3 times it. 0 if it never came down and stopped, or ended tipped over.
  - accuracy: 1 inside the land radius, falling to 0 at 5 times it. Needs a landing.
  - tilt factor: 1 up to the tilt limit, falling to 0.5 at twice it.
- **Gates**: G1 (10) = fraction of static rules passed. G2 (10) = fraction of L1 margins met. Both 0 for a design with unchosen slots or that does not build. G3 30, G4 15, G5 15 = mission score x points. G6 20 = mean mission score over its seeds.
- **Flights**: L0 or L1 failure still stops before the simulator, so an over-budget design earns no flight points. After G3, the rest fly whether or not G3 passed, if the design lifted off in G3; otherwise they are 0 unflown.
- All on the held-out seed, for the state at final and at first test. Fields: `partial_score`, `partial_gates`, `first_test.partial_score`, `first_test.partial_gates`. `harness/first_test.py --all` adds them.

### Hidden exam

The gates an agent sees are the practice test; this is the exam. Same mission, limits and held-out seed, for the state at final and at first test; agents never see these conditions. Numbers in `task.toml [hard]`, code in `dv/partial.py`, fields `hard_score`, `hard_gates`, `first_test.hard_score`, `first_test.hard_gates`. On the leaderboard `score` and `first_test_score` carry it; `score_strict`, `first_test_score_strict`, `score_partial`, `first_test_score_partial` stay. Strict scores, gate passes and planted flaws do not change.

- **Conditions**, about double: wind 6 m/s; rain 50 mm/h with 4 m/s wind; robustness mass +-10 %, CG +-10 mm, thrust +-6 %, extra sensor noise 1.0, backlash 2 deg, gyro bias 0.4 deg/s, accel bias 0.1 m/s2. G3 stays still air.
- **One flight**, 0 to 1: `sum(0.2 x part x quality) x exp(-peak tilt / 300 deg)`, then the partial tilt factor past the limit. Parts as above, passing or not. Each quality is below 1 for any real flight, so 100 is out of reach by construction.
  - lift-off: quality 1.
  - altitude: `exp(-rms / 0.5 m)`, rms of true height from target, first to last 2 Hz sample in the band.
  - hold: `exp(-std / 0.5 m)`, standard deviation of height over that window.
  - landing: `exp(-touchdown / 2.5 m/s)`.
  - accuracy: `exp(-distance / 2.5 m)`.
- **Gates** as partial: G1, G2 rule fractions; L0 or L1 failure flies nothing; no lift-off in G3 stops the rest.
- **Calibration was on the reference only, set once before any agent run got an exam score (partial-credit scores had been seen); these numbers are frozen.** The reference passes all 8 harder flights and scores 86.2: G1 10, G2 10, G3 25.43 of 30, G4 11.61 of 15, G5 12.23 of 15, G6 16.88 of 20. Do-nothing 20 (no flight points), open-loop 26.3.

## Baselines (`harness/baseline.py`)

- **Brute force**: enumerate the whole catalog through L0 and L1, report how many pass, then fly the cheapest and lightest non-dominated few with the reference controller.
- **Human**: the reference design and controller.

Same metrics, same leaderboard, no LLM cost.

## Left out

- Genetic algorithm baseline: brute force covers a catalog this small.
- Real builds and published flight times: the catalog is one EDF hopper, so hobby multirotor builds do not map onto it. L4 stays empty.
- Auto-tuned controller: see above.
