# Thesis Context — CARLA RL Lane Keeping & Intersection Navigation

This file exists specifically so an AI assistant (or a person) writing the
thesis document has an accurate, complete picture of the project without
needing to read the whole codebase. It is kept up to date as the project
progresses — check the "Status as of" date at the top before relying on
anything here as current.

**Status as of: 2026-09-10** — see the changelog at the very bottom of
this file for what changed on each date. Read that before assuming
anything here (especially §5's numbers) is still current.

---

## 1. Project scope

Reinforcement learning for autonomous vehicle control in CARLA 0.9.15.
Started as pure lane-keeping on a straight highway segment (Town04), then
extended to full route planning through intersections plus traffic-light
compliance on an urban map (Town10HD_Opt) with real signalized
intersections. Four algorithms are supported via a pluggable registry:
**PPO, SAC, DDPG, TD3**, all via Stable-Baselines3. DQN is explicitly
**not** supported — it requires discrete actions, and this project's
action space is continuous 2D control (acceleration, steer). That's a
documented design choice, not a gap.

### 1.1 System capabilities — what it can actually do, end to end

Useful as a self-contained list for an abstract or introduction. Every
item here is implemented and verified (§5), not aspirational:

- Connects to a running CARLA 0.9.15 server and drives a real vehicle
  actor in synchronous, fixed-timestep physics (20 Hz) — not a
  simplified kinematic simulation.
- Given any two points in Town10HD_Opt's road network, plans a legal
  route between them (CARLA's `GlobalRoutePlanner`, A* over the road
  topology graph) and tracks the vehicle's progress along it every
  step, correctly disambiguating direction inside junctions where
  several physically-close lanes are not the one the route intends —
  the specific problem that makes naive nearest-lane tracking fail at
  intersections (§4).
- Reads ground-truth traffic-light state from CARLA and enforces
  stop-on-red compliance as a first-class part of the reward/
  termination logic (not just an observation the agent could ignore) —
  distinguishes correctly stopping-and-waiting from driving through,
  and from normal braking distance while still approaching a light
  (§2, `traffic_rules.py`).
- Trains a policy with any of 4 different RL algorithms (PPO, SAC,
  DDPG, TD3) against this environment via one shared training script
  (`agent/train.py`), with per-algorithm hyperparameters isolated in
  one config file, and can resume a stopped/interrupted training run
  from its last checkpoint without restarting the step counter.
- Evaluates a trained checkpoint deterministically over N episodes and
  reports mean/std reward, mean lateral distance, success rate, and a
  full termination-reason breakdown (`agent/evaluate.py`).
- Generates a full set of comparison tables and publication-style
  figures (9 plots + a summary CSV) across all 4 algorithms from raw
  training/eval logs, with no manual data wrangling
  (`scripts/generate_metrics.py`, `scripts/plot_metrics.py`).
- Widens its own off-road tolerance automatically based on how many
  real same-direction lanes the current road segment has (§3, point 4)
  — a two-lane one-way road doesn't terminate the episode the instant
  the agent drifts into the next lane over.
- Produces path-deviation visualizations (actual driven path vs.
  planned route, in 2D and literal 3D) and curated environment
  screenshots directly from a live checkpoint driving in CARLA (§5.6,
  §5.7) — not mockups, real driving output.

## 2. Architecture (module-by-module)

The codebase is deliberately modular — each file has one responsibility,
so a component (e.g. the traffic-light reader) can be swapped later
(e.g. for a vision-based detector) without touching the others.

- **`carla_env/env.py`** — `CarlaLaneKeepingEnv`, the Gymnasium
  environment. Owns the CARLA connection, synchronous-mode physics
  stepping, vehicle spawn/reset lifecycle, and orchestrates every other
  module each step. Nothing else in the project talks to CARLA directly.
- **`carla_env/observation.py`** — builds the **5D observation vector**:
  `[lateral_distance, heading_error, speed, steering, traffic_light]`.
  The first 4 dimensions are the original lane-keeping state; the 5th
  (`traffic_light`: -1 = must stop, +1 = clear) was added when
  intersection support was built. All values normalized to roughly
  `[-1, 1]`.
- **`carla_env/action.py`** — 2D continuous action
  `[acceleration, steer]` → CARLA `VehicleControl`, with an
  exponential-smoothing filter (`alpha=0.6`) so policy-sampling jitter
  doesn't become visible steering oscillation.
- **`carla_env/reward.py`** — dense reward: weighted sum of centering,
  speed-target (Gaussian), heading-alignment, action-smoothness, plus a
  linear forward-progress term. The progress term exists specifically
  because DDPG/TD3 (deterministic policies) discovered they could
  exploit the Gaussian speed reward's non-zero floor at v=0 and collapse
  to standing still; the progress term is exactly zero at v=0, closing
  that exploit. Termination is checked in strict priority order: stall →
  collision → red_light_violation → off_road → wrong_heading →
  destination_reached → timeout (truncation, not termination).
  `StallDetector` and `RedLightViolationDetector` are small per-episode
  state machines.
- **`carla_env/route_planner.py`** — `RoutePlanner`, wraps CARLA's own
  `GlobalRoutePlanner` (A* over the road topology graph) to compute one
  route per episode between a start and destination point. Tracks which
  route waypoint the vehicle is currently closest to every step via a
  robust local-search algorithm (see §4). This module is what makes
  intersection navigation well-defined: without it, "which lane is
  nearest" is ambiguous inside a junction where several crossing/turning
  lanes are all physically close.
- **`carla_env/traffic_rules.py`** — `get_traffic_light_affordance()`
  reads CARLA's own ground-truth traffic-light state
  (`vehicle.is_at_traffic_light()` / `get_traffic_light_state()`) into a
  small dataclass. `RedLightViolationDetector` flags a violation only at
  the moment the vehicle *exits* a red/yellow zone while still moving
  above a threshold speed — not for the whole time the light is red,
  which would incorrectly punish normal braking distance. This
  ground-truth reader is deliberately isolated behind one call site in
  `env.py`, designed to be swapped later for a vision-based light/sign
  detector without any other module changing. Same design principle
  applies to the route planner's output — a real route today, could be
  a learned/perception-based planner later.
- **`agent/algorithms.py`** — `ALGORITHMS` registry mapping an algorithm
  name to its SB3 class, plus `build_model()`, which translates
  `configs/config.yaml`'s per-algorithm block into SB3 constructor
  kwargs (on-policy PPO vs. off-policy SAC/DDPG/TD3's replay buffer, and
  Ornstein-Uhlenbeck action noise specifically for DDPG/TD3, since
  they're deterministic policies needing explicit exploration noise —
  SAC explores via entropy instead). Adding a new continuous-action
  algorithm is a registry entry + a config block; zero changes to
  `train.py`.
- **`agent/train.py`** — training entry point. Runs a `train_env` and an
  `eval_env` **simultaneously** against the same CARLA server
  (`spawn_index_offset` keeps them from contending for the same spawn
  point).
- **`agent/evaluate.py`** — standalone evaluation of a saved checkpoint:
  mean/std reward, mean lateral distance, `success_rate` (fraction of
  episodes ending in `destination_reached`), full termination-reason
  breakdown, written to CSV.
- **`agent/callbacks.py`** — SB3 callbacks: episode-level CSV/TensorBoard
  logging, checkpointing, best-model saving.
- **`agent/baseline.py`** — a deliberately simple hand-coded P-controller
  (steer toward the route waypoint, brake for red lights). **Not** a
  competitive baseline — it exists purely to smoke-test the environment
  pipeline visually without waiting on RL training, and as the natural
  "does the learned policy actually beat a naive controller" comparison
  point.
- **`configs/config.yaml`** — single source of truth for every
  hyperparameter (per-algorithm and shared: reward weights, map, spawn
  mode, etc).

## 3. Key design decisions worth explaining in the thesis

1. **Route-relative observation instead of nearest-lane lookup.**
   Originally `lateral_distance`/`heading_error` were measured against
   `carla_map.get_waypoint()` — literally "whichever lane is physically
   closest." Fine on a single road, but ambiguous at any junction
   (several crossing lanes are all "closest"). Switched to measuring
   against the closest waypoint **on the planned route** instead — this
   is what makes turning at an intersection well-defined at all.
2. **Ground-truth-first, swap-later for route/traffic-light info.** Both
   are read from CARLA's simulator-level ground truth right now, not
   perception — a deliberate staging decision to validate the
   control/RL problem is solvable before adding perception noise on
   top. Both are isolated behind a single call site each, specifically
   so a later perception-based version can replace them without
   touching `env.py`'s control flow, `observation.py`, or `reward.py`.
3. **A successful route completion must not be penalized like a
   failure.** Fixed bug: reaching the destination originally triggered
   the same -10 terminal penalty as a collision, since both set
   `terminated=True`. Fixed by explicitly excluding
   `destination_reached` from the terminal-penalty branch — it's the
   one terminal condition that's a *success*, not a failure.
4. **Multi-lane off-road tolerance.** Off-road termination originally
   used a single lane's width as the threshold regardless of how many
   real lanes the road has going the same direction — drifting into an
   adjacent same-direction lane (common right at intersections, where
   dedicated turn lanes widen the road) was incorrectly treated as
   leaving the road entirely. Fixed by counting same-direction lanes via
   CARLA's lane topology and widening *only* the termination threshold
   by that count — the centering reward is untouched, so the agent is
   still gently encouraged toward its own lane, just not killed for
   briefly being in the next lane over.

## 3.5 Exact reward formula and termination conditions (precise values, for the results/methodology chapter)

All values are `carla_env/reward.py`'s live defaults, confirmed against
`configs/config.yaml`'s current `reward:` block on 2026-09-08. If the
config changes later, re-check this section against it before citing it.

### Full reward formula

At every step, the scalar reward is:

```
r = w_center   * r_centering
  + w_speed    * r_speed
  + w_heading  * r_heading
  + w_smooth   * r_smoothness
  + w_progress * r_progress
  + r_terminal
  + r_step
```

with current weights: `w_center=1.0`, `w_speed=1.5`, `w_heading=0.5`,
`w_smooth=0.5`, `w_progress=1.0`, `step_penalty` (`r_step`, added every
step unconditionally) `=-0.1`, `terminal_penalty=-10.0`.

Component formulas already in the codebase (centering, speed, heading —
not repeated here since they predate this note) plus the two the thesis
text doesn't cover yet:

**Smoothness term** — `r_smoothness = compute_smoothness_reward(action_delta)`:

```
r_smoothness = max(0, 1 - sum(|action_delta|) / 4.0)
```

where `action_delta` is the current raw action (from the policy, before
`ActionProcessor`'s exponential smoothing is applied to it) minus the
previous step's raw action, a 2D vector `[Δacceleration, Δsteer]`.
Range `[0, 1]`; peak `1.0` when the action didn't change at all since the
last step; `0.0` when both action dimensions swing their full range in
one step (e.g. acceleration `-1→+1` **and** steer `-1→+1`
simultaneously: `|Δ|=2.0` each, sum `=4.0`). It penalizes the *raw*
action rather than the smoothed/applied one specifically so the policy
can't rely on `ActionProcessor`'s filter (alpha=0.6) to absorb jitter it
didn't need to output in the first place — it has to learn to be smooth
on its own.

**Forward-progress term** — `r_progress = compute_progress_reward(speed_kmh, target_speed_kmh)`:

```
r_progress = min(speed_kmh, target_speed_kmh) / target_speed_kmh
```

Range `[0, 1]`, linear in speed, capped at 1.0 once the vehicle reaches
`target_speed_kmh` (currently 30 km/h). Exactly `0.0` at `speed=0` —
this is the whole point of the term. **Why it exists**: the Gaussian
speed-reward component alone gives a small but non-zero reward
(~0.017) even at zero speed, which let DDPG/TD3 (deterministic
policies) discover that standing still while collecting the centering +
heading + smoothness rewards was a viable local optimum, since there
was no reward *floor* punishing zero speed specifically. The linear
progress term has no such floor — it is provably zero at zero speed, so
standing still can no longer be locally optimal purely on the reward's
own structure. PPO and SAC are largely unaffected by this term in
practice (they already explore via entropy/stochastic sampling and
don't collapse to standing still the way a deterministic policy can),
but the term applies to all four algorithms identically — it isn't
algorithm-conditional in the code, just observed to matter most for
DDPG/TD3.

**Terminal reward**: `r_terminal = terminal_penalty` (currently -10.0)
on the step the episode ends via any *failure* reason (see termination
list below) — explicitly **not** applied on `timeout` (ran out of steps
without failing) and **not** applied on `destination_reached` (the one
success case), even though both are `terminated=True`/definitively-ended
episodes in Gymnasium's sense. This distinction (`is_terminal_for_reward
= terminated and term_reason != "destination_reached"`, in `env.py`) was
a real bug fix — earlier, reaching the destination successfully still
triggered the same -10 penalty as a collision, since both simply set
`terminated=True`.

### Termination conditions — exact thresholds, in priority order

`check_termination()` evaluates these in order and returns on the first
match — if two conditions would both apply on the same step (e.g. a
collision at the exact moment of reaching the destination), the earlier
one in this list wins:

1. **Stall** — `StallDetector`: counts consecutive steps where
   `speed_kmh < stall_min_speed_kmh` (currently **2.0 km/h**). Once the
   count reaches `stall_patience_steps` (currently **100 steps**, i.e.
   5 simulated seconds at the fixed 20 Hz physics rate), the episode
   terminates with the full `terminal_penalty`. **Exemption**: the
   counter is reset to zero (not just paused) on any step where the
   traffic-light affordance's `must_stop` is `True` — i.e. the vehicle
   is correctly stopped at a red/yellow light. Without this exemption,
   an agent that properly waits at a red light for more than 5 seconds
   would get the same terminal penalty as an agent that's actually
   stuck, directly undermining the traffic-light-compliance objective.
2. **Collision** — from the CARLA collision sensor, no threshold (any
   contact event fires it).
3. **Red-light violation** — `RedLightViolationDetector`: NOT flagged
   for the whole time a light is red (which would incorrectly punish
   the vehicle's normal braking distance while slowing down for a
   light it hasn't reached yet). Instead it fires on the exact step the
   vehicle *exits* the light's trigger zone (`is_at_light` goes from
   `True` to `False`) while, on the **previous** step, it was inside the
   zone (`_was_at_light=True`) **and** required to stop
   (`_was_must_stop=True`), **and** its current speed exceeds
   `red_light_stop_speed_kmh` (currently **5.0 km/h**) — i.e. it drove
   through rather than having actually come to a stop. If the vehicle
   waited for the light to turn green before exiting the zone,
   `_was_must_stop` would already be `False` by the time it exits, so no
   violation is flagged.
4. **Off-road** — `|lateral_distance_m| >= max_lateral_m`. Base value
   `max_lateral_m = 3.5 m` (roughly one lane width on CARLA's default
   road geometry), **but this threshold is not fixed** — see the
   multi-lane note below.
5. **Wrong heading** — `|heading_error_rad| >= max_heading_deg` (default
   **90°**, converted to radians at call time) — vehicle pointing more
   than perpendicular to the road's intended direction.
6. **Destination reached** — checked *after* all failure conditions
   above, so any genuine failure on the same step still takes priority
   over a simultaneous "reached the destination" — a collision right at
   the endpoint is still scored as a collision, not a success.
7. **Timeout** (truncation, not termination — no penalty applied) —
   `step_count >= max_steps` (currently **1000 steps**, i.e. 50
   simulated seconds).

### Off-road threshold and multi-lane roads — did it change?

**Yes.** The *base* value (`max_lateral_m = 3.5 m`, one lane's width) is
unchanged, and it's still what the **centering reward** (`r_centering`)
uses — the agent is still continuously encouraged toward its own lane's
centerline regardless of road width. But the **termination** threshold
passed into `check_termination()` is now:

```
effective_max_lateral_m = max_lateral_m * same_direction_lanes
```

where `same_direction_lanes` (from
`RoutePlanner.count_same_direction_lanes()`) is computed fresh every
step from the *current* route waypoint's real CARLA lane topology
(walking `get_left_lane()`/`get_right_lane()` from that waypoint,
stopping at the boundary where `lane_id`'s sign flips — CARLA's
convention for "this is now the opposite-direction side of the road" —
and skipping non-`Driving` lane types like shoulders). On a standard
single-lane-per-direction road this evaluates to `1`, so the threshold
is unchanged (3.5 m). On a two-lane one-way road it's `2`, giving a 7.0
m termination threshold — enough that drifting into the adjacent
same-direction lane no longer instantly ends the episode, while still
terminating well before the vehicle could plausibly have left the
roadway onto the opposite side or off-road entirely. **Why**: without
this, a car drifting into an adjacent same-direction lane (common
specifically near intersections, where dedicated turn lanes widen the
road) was being terminated identically to a car that had genuinely left
the road, even though it was still safely on pavement going the correct
direction — confirmed as a real, frequent failure mode via live testing
before the fix.

## 4. Route-tracking algorithm (good candidate for a diagram/pseudocode)

Naive approach: find the route waypoint with minimum Euclidean distance
to the vehicle, either globally or in a forward search window. This
fails two ways, both found via live testing:

- On a dense street grid, a physically closer point on a **parallel
  street** (not the current route) can win, snapping tracking to a
  wrong, unrelated road segment.
- Even restricted to a forward window, a route that curves back near
  itself (a tight turn or loop) can have a later, out-of-sequence point
  be closer than the correct next point.

**Current algorithm**: walk forward from the last known route index,
tracking the running closest point, and stop once distance has failed
to improve for a small "patience" window (tolerates one irregular
waypoint without stopping too early). This can never jump ahead to an
out-of-sequence point no matter how spatially close it is, because it
never looks past where it stopped improving. It also checks a small
bounded window *behind* the last index, to recover if the vehicle drifts
backward (e.g. rolling back slightly on an incline while stopped at a
red light) — a purely forward-only search can never recover from that,
tracking would freeze permanently.

## 5. Current verified status

### Correctness (offline + live testing)
Full offline unit test suite covers: action smoothing, every reward
component and the termination priority chain, traffic-light violation
detection (including the "stop and wait" vs. "drive through" distinction
and the stall-detector's exemption while legitimately waiting at a red
light), the route-tracking algorithm's edge cases (parallel-street
rejection, loop-back-near-itself, backward-drift recovery, single-outlier
tolerance), multi-lane counting, and algorithm-registry construction for
all 4 algorithms from the real config.

A dedicated code-review pass (8 parallel review agents covering
correctness, duplication, efficiency, dead code, and convention
adherence) found and fixed 17 issues, including several real crash risks
that would have broken a long unattended training run (an edge case
where the planned route could come back empty and crash the next step
with no handling at all; a regression where spawn-retry logic could get
stuck retrying the same occupied point instead of trying alternatives)
and a significant metrics bug (`success_rate` was counting "ran out of
time" as success instead of "reached the destination" — would have
produced wrong numbers in a results table).

Live-verified on actual CARLA (Town10HD_Opt — Town03, the originally
intended map, segfaults this machine's CARLA install on load; confirmed
via a bare CARLA API call outside any project code, so it's an
engine/GPU-driver issue, not a code bug): route generation and
spawn-exactly-on-route-start works across hundreds of randomly generated
routes with zero crashes.

### Training results — all four algorithms, equal 150,000-step budget (2026-09-08)

All four algorithms were trained to the **same total of 150,000 environment
steps**, then evaluated identically (`agent/evaluate.py`, 10 deterministic
episodes each). DDPG and TD3 reached 150k via an initial 80,000-step run
(chosen because their `learning_starts=50000` means they need at least
that just to start learning at all) followed by a 70,000-step continuation
(`--resume`, `reset_num_timesteps=False` so the step counter continues
rather than restarting) — this two-stage process is why their per-run
elapsed times differ from PPO/SAC's single continuous run, but the final
checkpoints are trained on an equal step budget, which is what matters
for a fair comparison. PPO and SAC trained in one continuous 150,000-step
run each (they don't have DDPG/TD3's warm-up requirement).

| Algorithm | Mean reward | Mean lateral distance | Success rate | Mean episode length |
|---|---|---|---|---|
| PPO | 617.87 (±420.55) | 0.87 m | 10.0% | 224.6 steps |
| SAC | 1784.60 (±973.86) | **0.56 m** | **40.0%** | 512.0 steps |
| DDPG | 1304.64 (±1051.85) | 1.76 m | 30.0% | 424.4 steps |
| TD3 | **2613.62** (±1066.66) | 0.56 m | 30.0% | **823.1 steps** |

Notable, non-obvious findings — worth discussing directly in the thesis
rather than picking one "winner" narrative:
- **TD3 has the highest mean reward and by far the longest surviving
  episodes**, but only middling success rate (tied with DDPG).
- **SAC has the highest success rate** (destination actually reached) and
  ties TD3 for best lateral centering precision, with a much shorter
  mean episode length than TD3 — suggesting SAC either reaches the
  destination faster on average or fails faster on the episodes it
  doesn't complete (worth a closer look at per-episode data if this goes
  in the thesis).
- **DDPG has by far the worst centering (1.76 m mean lateral distance)** —
  visibly noisier control than the other three, consistent with DDPG
  being the "weakest" of the deterministic-policy pair in most published
  comparisons.
- **PPO evaluates worst here despite having the best absolute
  `destination_reached` count during training** (23 successes across
  151,552 training steps, more than any other algorithm's raw count).
  The deterministic evaluation policy specifically struggles with red
  lights — 7 of 10 eval episodes at the earlier 150k checkpoint ended in
  `red_light_violation` (see the per-checkpoint breakdown a few
  paragraphs below for the exact numbers this claim is based on).
  Plausible explanation: PPO's stochastic training-time policy and its
  deterministic (mean-action) evaluation policy can behave meaningfully
  differently, and/or this is small-sample (10-episode) noise — this is
  a real, reportable finding, not an error, but shouldn't be
  overinterpreted from 10 episodes alone.

**Full per-checkpoint breakdown** (training-time termination counts +
evaluation results), including the intermediate 80,000-step DDPG/TD3
checkpoints (superseded by the 150k numbers above for the "final"
comparison, but a legitimate data point for a learning-curve figure):

- **DDPG @ 80k** (`results/checkpoints/ddpg/ddpg_lane_keeping_20260908_175251/`): training breakdown over 681 episodes — stall 39.9%, wrong_heading 29.4%, off_road 14.8%, collision 11.2%, red_light_violation 3.1%, destination_reached 1.2% (8), timeout 0.4%. Eval: reward 2025.46 (±1156.28), lateral 1.12 m, success 20.0%, ep. length 688.1.
- **DDPG @ 150k** (`.../ddpg_lane_keeping_20260908_200736/`): the +70k continuation segment alone — collision 19.1%, off_road 15.3%, stall 37.8%, red_light_violation 9.6%, destination_reached 3.8%, timeout 12.4%. Eval (final, reported in the table above): reward 1304.64, lateral 1.76 m, success 30.0%, ep. length 424.4.
- **TD3 @ 80k** (`.../td3_lane_keeping_20260908_181222/`): training breakdown over the full 80k — collision 11.9%, off_road 21.5%, wrong_heading 4.8%, stall 45.5%, red_light_violation 11.7%, destination_reached 2.7% (13), timeout 1.9%. Eval: reward 2160.24 (±782.96), lateral 1.50 m, success 40.0%, ep. length 782.3.
- **TD3 @ 150k** (`.../td3_lane_keeping_20260908_203207/`): the +70k continuation segment alone — collision 19.0%, off_road 26.4%, stall 8.0% (big drop from the 80k segment), red_light_violation 23.6%, destination_reached 8.6% (15 — more than double the rate of the first segment), timeout 14.4%. Eval (final, reported in the table above): reward 2613.62, lateral 0.56 m, success 30.0%, ep. length 823.1.
- **PPO @ 150k** (`.../ppo_lane_keeping_20260908_183240/`, 151,552 steps, single continuous run): training breakdown over 371 episodes — collision 29.5%, off_road 26.2%, red_light_violation 18.9%, timeout 14.3%, destination_reached 6.2% (23 — highest absolute count of all checkpoints), stall only 3.8% (PPO/SAC don't get the DDPG/TD3 stand-still exploit — they explore via entropy, not action noise). Eval (final, reported in the table above): reward 617.87, lateral 0.87 m, success 10.0%, ep. length 224.6, **7/10 episodes ended in red_light_violation**.
- **SAC @ 150k** (`.../sac_lane_keeping_20260908_190524/`, single continuous run, 60.3 min — noticeably slower wall-clock than the others at the same step count, SAC's entropy-tuning does more compute per step): training breakdown — red_light_violation 36.6% (100, highest rate of all), timeout 27.8%, destination_reached 12.8% (35 — highest absolute count of all checkpoints), collision 9.5%, stall 7.0%, off_road 5.1%, wrong_heading 1.1%. Eval (final, reported in the table above): reward 1784.60, lateral 0.56 m, success 40.0%, ep. length 512.0.

There was also an earlier, separate PPO run for **250,000 steps**
(interrupted at episode 442, not completed to the full target — a
machine handoff interrupted it) that showed periodic-evaluation reward
improving then dipping across 3 eval checkpoints (907 → 1080 → 558 mean
reward) — real, noisy early-training data, kept here as a longer-horizon
data point but **not** part of the equal-150k-budget comparison above
(different total step count, different run).

**Important honesty note for the thesis**: none of the above are the
full 500,000-timestep runs that `configs/config.yaml`'s
`training.total_timesteps` default represents as the intended final run
length. The 150k-step comparison above is real, fairly-compared data —
useful for showing relative algorithm behavior and as an intermediate
result — but not the final thesis-quality numbers. Whether to run the
full 500k for each algorithm before finalizing thesis numbers is an open
decision.

### 5.5 Two-axis comparison framework: speed-to-good-policy vs. reliability-of-found-policy

A single "winner" narrative undersells this comparison — the four
algorithms trade off along two genuinely different axes, both worth
reporting explicitly rather than collapsing into one ranking.

**Axis 1 — how fast training finds a capable policy (sample efficiency).**
`scripts/generate_metrics.py` tracks the training step at which the
20-episode rolling-mean *training-time* reward first crosses 2,500
(`SAMPLE_EFF_THRESHOLD`, see `results/metrics_comparison.csv`'s
`sample_eff_step` column):

| Algorithm | Step reward first crosses 2,500 |
|---|---|
| SAC | **40,097** (~27% of the 150k budget) |
| PPO | not reached within 150k |
| DDPG | not reached within 150k |
| TD3 | not reached within 150k |

SAC is the clear standout here — it reaches a sustained high-reward
policy in about a quarter of the training budget the other three never
sustain at all within 150k steps. This tracks with SAC's off-policy
replay buffer (reuses old transitions instead of discarding them like
PPO's on-policy updates) and automatic entropy tuning.

**Axis 2 — how reliable/safe the final policy is once training stops.**
This is not the same question as Axis 1 — a policy can reach a high
*training-time* reward quickly and still be inconsistent or unsafe at
evaluation time. Three signals, read together:

| Algorithm | Success rate | Reward std (±) | Worst-case lateral distance |
|---|---|---|---|
| PPO  | 10.0% | 420.55  | **1.12 m (best worst-case)** |
| SAC  | **40.0%** | 973.86  | 1.05 m |
| DDPG | 30.0% | 1051.85 | 2.37 m (worst) |
| TD3  | 30.0% | **1066.66 (most variance)** | 2.11 m |

Reading this honestly: **SAC is the strongest on both axes** — fastest
to a good policy *and* the most reliable one, which is a legitimate,
reportable conclusion. TD3 has the highest *mean* eval reward (§5's
table) but the most run-to-run variance and a worse worst-case lateral
excursion — a policy that occasionally drives very well but is less
consistently safe. DDPG is weakest on the safety axis specifically
(worst mean *and* worst-case lateral distance), consistent with its
known real-world reputation as the least stable of the four. PPO is the
outlier: worst success rate by far, but paradoxically the *tightest*
worst-case lateral control — when PPO's evaluated policy fails, §5
already establishes it's overwhelmingly failing on `red_light_violation`
(7/10 episodes at the final checkpoint) rather than losing lane control,
i.e. a rule-compliance failure mode, not a driving-precision one.

**Recommended framing for the results chapter**: present §5's headline
table first, then this two-axis breakdown as the deeper analysis —
"which algorithm learns fastest" and "which algorithm's final policy is
most trustworthy" are different questions with different answers here,
and that distinction is itself a legitimate finding, not a hedge.

### 5.6 Environment screenshots (for reader familiarization)

`results/screenshots/` (generated locally via `scripts/capture_screenshots.py`,
not committed to git — large binaries, regenerate on demand) holds 4
curated images of the live CARLA environment, captured with a trained
SAC policy actually driving (not random actions), for the thesis's
environment/setup section:

- `01_map_overview.png` — top-down establishing shot of Town10HD_Opt's
  central district (skyscrapers, waterfront) — gives the reader a sense
  of the environment's visual complexity and scale.
- `02_route_lane_keeping.png` — chase view of the ego vehicle centered
  in its lane, with the green line showing the planned route/target
  waypoint the observation is computed against (see §3.5/§4).
- `03_intersection_redlight.png` — chase view at an intersection where
  `traffic_light_must_stop` was True at capture time (visible traffic
  signal heads down the road) — illustrates the traffic-light
  affordance from §2/§3.
- `04_driver_pov.png` — windshield-height forward view, for a "what the
  road looks like" establishing shot.
- `05_hero_shot.png` — front-quarter angle of the vehicle at a
  crosswalk, warm sunset lighting — a clean "here's the car" shot for
  a title page or the environment section's opening figure.
- `06_curve_turn.png` — chase view mid-turn (large steering command),
  sunset lighting, route line visibly curving — a more dynamic
  complement to `02`'s straight-road shot.
- `07_aerial_establishing.png` — elevated 3/4 "drone" view of the
  vehicle approaching an intersection, sunset lighting — the most
  cinematic of the set, good as a section-opening or cover image.

### 5.7 Trajectory-vs-route figures (path deviation, not just aggregate stats)

`scripts/plot_trajectory.py` drives live episodes with the SAC checkpoint
(inference only -- no training/checkpoint changes) and plots the
vehicle's actual (x, y) path against the planned route, colored by
lateral deviation, in `results/plots/`:

- `trajectory_success_example.png` / `_3d.png` — a genuine
  `destination_reached` episode (436 steps, long route with a real
  curve). Mostly low deviation (green) with one brief spike to ~1.5m
  through the sharpest part of the curve, then recovers and completes
  cleanly -- a good "how well it actually tracks" figure, imperfections
  included rather than hidden.
- `trajectory_failure_example.png` / `_3d.png` — a genuine failure
  (`off_road`, terminated at step 33) — pairs honestly with the success
  case rather than only showing the best outcome. Both plots' color
  scale runs 0-1.5m lateral distance for a fair side-by-side comparison.
- The `_3d.png` variants plot the same path with simulation step as a
  third axis (not a bar/surface chart -- those are hard to read
  accurately due to occlusion; a single 3D line avoids that problem)
  with the planned route shown as a flat reference on the ground plane.

Because SAC's real success rate is ~40% (§5's table), the script tries
up to `MAX_EPISODES=6` fresh episodes looking for one genuine success
and one genuine failure rather than cherry-picking a single lucky run.

### 5.8 Data-integrity audit (2026-09-09) — read this before citing any figure

Before generating final figures, every data-loading path in
`scripts/generate_metrics.py` and `scripts/plot_metrics.py` was audited
end-to-end (prompted by a direct question: "is any of this actually
reliable?"). Five real bugs were found and fixed, not just cosmetic
issues — worth knowing about because they explain why some numbers
shifted slightly between early and final versions of the same figure,
and because the pattern (see the third bullet) is the kind of subtle,
non-crashing correctness bug worth discussing in a methodology/testing
section:

1. **Stale eval-CSV selection.** `load_eval_csv()`/`load_eval_data()`
   picked "whichever CSV file has the most episodes" rather than the
   most recent one. `results/logs/*/eval_runs/` holds old evaluation
   runs (one with 200 episodes) from months before the traffic-light
   feature existed — the old logic would have silently preferred a
   200-episode stale run over a fresh 10-episode one. Fixed to pick by
   filename timestamp.
2. **Old/new training-run mixing.** The training-curve loaders globbed
   *every* run directory ever created per algorithm (some going back to
   June, on a 4D pre-traffic-light observation space) and merged them
   by raw timestep. Since every fresh run's step counter restarts at 0,
   old and new runs' timesteps collide, risking incompatible data
   silently interleaved into one curve. Fixed by introducing
   `FINAL_COMPARISON_RUNS`, an explicit whitelist of the exact run
   directories behind the 150k-step comparison in §5's table.
3. **Radar chart reference-range miscalibration** — the most serious of
   the five, because it produced a plausible-looking chart with no
   error, just wrong information. `plot_radar_chart()`'s domain
   reference ranges (e.g. reward `[2500,3500]`) were calibrated for a
   fully-converged model. At this project's actual 150k-step values
   (reward 618–2614), 3 of 6 axes clipped to exactly 0.0 for *every*
   algorithm — e.g. PPO's 618 and TD3's 2614 reward, a real 4x
   difference, both rendered as an identical flat zero. Recalibrated to
   this project's actual achieved spread.
4. **`plot_speed_distribution()` re-implemented its own "most recent
   run" glob** instead of using the already-fixed loaders. It happened
   to pick the right directory by alphabetical luck, but for DDPG/TD3
   (whose 150k total spans two run directories) it silently read only
   the second, dropping the first 80k steps entirely — confirmed via
   episode counts (209/174 episodes in the continuation-only files,
   both under the requested "last 300"). Fixed to use the properly
   merged loader; TD3's plotted sample size corrected from n=84 to n=147.
5. Two "Final: X" annotation labels rendering on top of their own plot
   line (`training_curves.png`, `lateral_progress.png`) — cosmetic, fixed.

After all five fixes, `results/metrics_comparison.csv` was regenerated
and diffed byte-for-byte against the pre-fix version: **identical** —
confirming the underlying summary table was already correct and only
the *visualizations* built on top of it had these bugs. Every figure in
`results/plots/` was regenerated after the fixes and is current as of
this section's date.

## 6. Honest current limitations (do not oversell)

- **All four algorithms now have a complete, equal-budget (150,000 step)
  training run with a full evaluation pass** (see §5's table) — this
  limitation is resolved as of 2026-09-08 evening. The remaining gap is
  that 150k steps is still well short of the full 500,000-step runs
  `configs/config.yaml` specifies as the intended final length.
- The hand-coded baseline controller (not any RL agent) visibly fails at
  intersections/curves during manual observation — an **expected,
  already-understood limitation of that specific crude controller** (a
  fixed-gain P-controller has no lookahead/anticipation of an upcoming
  turn, and no smoothness optimization at all), not evidence of a bug in
  the environment or route logic. A trained RL policy has structural
  reasons to do better: it can learn anticipatory behavior from
  experience (which a fixed-gain controller cannot by construction), and
  the reward function explicitly penalizes jerky/large action changes
  (`w_smooth`) — the baseline optimizes for neither.
- Town03 (the originally intended intersection map) crashes this
  machine's CARLA install on load — an engine/GPU-driver issue, not a
  codebase bug. Town10HD_Opt is the working substitute (also has real
  intersections and traffic lights).
- Stop-sign detection is explicitly out of scope for the current pass —
  `traffic_rules.py` is structured so a sibling
  `get_stop_sign_affordance()`/`StopSignViolationDetector` pair could be
  added later in the same shape, but it hasn't been built.
- Camera/lidar sensors (a higher-dimensional observation extension) and
  a discretized-action variant for DQN are both documented as
  planned-but-not-built extensions, consistent with keeping the current
  thesis submission scoped to what's actually finished.

## 7. Possible future work (if the thesis wants this section)

- Full 500k-step training runs across all 4 algorithms for final,
  citable comparative results.
- Stop-sign detection (designed-for, not built).
- Vision-based traffic-light/lane detection replacing the current
  ground-truth reads (the architecture already isolates this behind one
  call site specifically for this swap).
- Fixing the Town03 crash to use the originally intended, richer
  intersection map.
- A stronger baseline controller (e.g. pure-pursuit with lookahead) for
  a more meaningful RL-vs-baseline comparison, since the current one is
  intentionally too crude to be a fair comparison point.

## 8. How these results compare to published CARLA RL literature (2026-09-09 research pass)

This section exists to answer one question honestly: **are this
project's numbers realistic, and how far are they from the field's
state of the art?** Three reference points, found via web search on
2026-09-09 — cite these directly in the thesis rather than presenting
the project's numbers in a vacuum.

**Closest direct comparison** — Alkhonain et al., *"A Comparative Study
of Deep Reinforcement Learning Algorithms for Urban Autonomous Driving:
Addressing the Geographic and Regulatory Challenges in CARLA"*, Applied
Sciences 15(12):6838, 2025 ([mdpi.com/2076-3417/15/12/6838](https://www.mdpi.com/2076-3417/15/12/6838),
[doi.org/10.3390/app15126838](https://doi.org/10.3390/app15126838)).
Compares DDPG, SAC, TD3, PPO, TQC and CrossQ head-to-head in CARLA over
**1,000,000 training steps** (~6.7× this project's 150k-step budget).
Findings that directly corroborate this project's own results:
- TQC and SAC (both off-policy, stochastic) achieved the best sample
  efficiency and route-completion performance; DDPG was the weakest,
  with a Route Completion rate of just 0.23 in challenging scenarios
  versus TQC's 0.91 — the same relative ranking (SAC strong, DDPG
  weakest) this project found independently.
- PPO showed "relatively irregular reward trends during training due to
  its on-policy nature, which precludes replay buffer usage, resulting
  in low sample efficiency" — visually exactly what this project's own
  `results/plots/training_curves.png` shows for PPO (the noisiest,
  least monotonic of the four curves), and consistent with PPO being
  the only algorithm here that never crosses the §5.5 sample-efficiency
  threshold within budget despite entropy-driven exploration.

**Upper-bound reference (not directly comparable)** — a TD3/SAC-based
"WAD" (waypoint/affordance-driven) agent reported 100% success on the
original CARLA benchmark and 82% on the harder NoCrash benchmark. This
number is real but not an apples-to-apples target: that system used
learned driving *affordances* (richer intermediate supervision than
this project's raw 5D state vector) and almost certainly a training
budget in the millions of steps, not 150k. Useful in the thesis only as
an "upper bound achievable with substantially more engineering and
compute," not as a bar this project's numbers should be judged against
directly.

**Field-level ceiling (different task entirely)** — the current CARLA
AD Leaderboard (2.1, sensor-rich, imitation/transformer-style methods,
not RL) tops out around a 90/100 "driving score" (e.g. TransFuser++,
FusionAssurance). This uses an entirely different metric (route
completion × infraction penalty, scripted routes with dense traffic)
and different method family — mentioned only to give the thesis a sense
of where the field's absolute ceiling sits, not as a comparison point.

**Honest conclusion for the thesis**: this project's *relative* findings
(SAC strongest overall, DDPG weakest on control precision, PPO's
on-policy sample inefficiency visible in its training curve) are
directionally consistent with a much larger, more heavily-resourced
published comparative study — which is meaningful independent
corroboration despite training at roughly 1/7th the step budget and
using a lower-dimensional observation space. The *absolute* numbers
(10–40% success rate here vs. 23–91% route completion in the cited
study, and up to 100% in the affordance-based upper-bound system) are
lower, and that gap is fully explained by the much smaller training
budget (150k vs. up to 1M+ steps) and the simpler 5D low-dimensional
observation versus richer affordance/sensor inputs elsewhere — not
evidence of a flawed method. This is the correct, defensible way to
frame "how close are we to the state of the art": right direction,
smaller scale, explicit about why.

## 9. Extensibility — how to add a new module without touching the rest

This is a deliberate architectural property, not an accident: every
major swap point below was designed in from the start (§2, §3 already
explain *why* for each), so this section is a consolidated, practical
"how to add X" reference — good material for a thesis section arguing
the system's design is sound for future extension, not just that it
works today.

| Want to add... | Where | What changes elsewhere |
|---|---|---|
| A new RL algorithm (any continuous-action SB3 algorithm) | One entry in `agent/algorithms.py`'s `ALGORITHMS` dict + a hyperparameter block in `configs/config.yaml` | **Nothing.** `agent/train.py`, `agent/evaluate.py`, the callbacks, and the metrics/plotting scripts are all algorithm-name-driven already. |
| Vision-based traffic-light detection (replacing ground truth) | A new function with `get_traffic_light_affordance()`'s exact signature/return type in `carla_env/traffic_rules.py`, swapped in at its single call site in `env.py` | **Nothing** in `observation.py`, `reward.py`, or the termination logic — they only see the `TrafficLightAffordance` object, not how it was produced. |
| A learned/perception-based route planner (replacing `GlobalRoutePlanner`) | A drop-in replacement for `RoutePlanner.plan_route()` returning the same `List[RouteWaypoint]` shape | **Nothing** downstream — `get_target_waypoint()`, the observation/reward code, and the multi-lane widening logic all consume the route by its waypoint list, not by how it was planned. |
| Stop-sign compliance | A sibling `get_stop_sign_affordance()` / `StopSignViolationDetector` pair in `traffic_rules.py`, same shape as the traffic-light pair (`reset()`/`update()`, called from `env.py`'s `step()`) | One new observation dimension (6D instead of 5D) and one new termination reason in `check_termination()`'s priority chain — structurally identical to how the traffic-light dimension was added originally. |
| Camera/lidar sensors (higher-dimensional observation) | A new sensor class in `carla_env/sensors.py` (same pattern as `CollisionSensor`: `__init__(world, vehicle)`, a callback, `destroy()`) | `observation.py`'s vector grows; anything reading the flat 5D vector directly (the algorithm registry doesn't — SB3 infers input size from `env.observation_space`) needs no change. |
| Discrete-action support for DQN | A new `CarlaLaneKeepingEnvDiscrete` variant (bucketizing the 2D continuous action into a discrete set) — `agent/algorithms.py`'s registry already has the DQN entry point designed in, just gated on this env variant existing | `train.py`/`evaluate.py` need no change — they're already environment-agnostic beyond the registry lookup. |
| A stronger baseline (e.g. pure-pursuit with lookahead) | A new file in `agent/` following `agent/baseline.py`'s shape (reads the same observation, returns the same 2D action) | Nothing — it's evaluated with the same `agent/evaluate.py` pipeline as any RL checkpoint. |

The common thread, worth stating explicitly in a design/architecture
chapter: **every extension point is defined by a data shape (an
observation vector, a `TrafficLightAffordance`, a `RouteWaypoint`
list, a 2D action), never by which specific piece of code produced
that data.** That's what makes each swap isolated to one file instead
of rippling through the codebase.

## 10. Real code excerpts (for accurate quoting/listings in the thesis)

Exact, current code — not paraphrased from memory — for the pieces
most worth citing directly as a code listing in the methodology
chapter. Pulled from the live files on 2026-09-10; if the thesis is
still being written much later, diff these against the actual files
before trusting them verbatim.

**The route-tracking algorithm** (`carla_env/route_planner.py`,
`RoutePlanner.get_closest_waypoint_index`) — the core loop behind §4's
prose description:

```python
def get_closest_waypoint_index(
    self, route, location, start_index=0,
    max_search_ahead=30, max_search_behind=5, patience=3,
):
    start_index = max(0, min(start_index, len(route) - 1))

    def dist_at(i):
        return self.distance(location, route[i].waypoint.transform.location)

    best_index = start_index
    best_distance = dist_at(start_index)

    # Small bounded backward check — recovers from minor backward drift
    # (e.g. rolling back slightly on an incline while stopped at a red
    # light) that a forward-only search could never recover from.
    behind_bound = max(0, start_index - max_search_behind)
    for i in range(start_index - 1, behind_bound - 1, -1):
        d = dist_at(i)
        if d < best_distance:
            best_distance = d
            best_index = i
        else:
            break

    # Forward walk, tolerating up to `patience` non-improving steps in a
    # row before concluding the local minimum has been passed. Never
    # looks past where it stopped improving, so it can't jump ahead to
    # an out-of-sequence point no matter how spatially close it is.
    search_end = min(start_index + max_search_ahead, len(route))
    stale_steps = 0
    for i in range(start_index + 1, search_end):
        d = dist_at(i)
        if d < best_distance:
            best_distance = d
            best_index = i
            stale_steps = 0
        else:
            stale_steps += 1
            if stale_steps >= patience:
                break

    return best_index
```

**The reward function's two novel terms** (`carla_env/reward.py`) —
exact formulas already given in prose in §3.5, here as the literal code:

```python
def compute_smoothness_reward(action_delta: np.ndarray) -> float:
    delta_magnitude = float(np.abs(action_delta).sum())
    return max(0.0, 1.0 - delta_magnitude / 4.0)

def compute_progress_reward(speed_kmh: float, target_speed_kmh: float) -> float:
    return float(min(speed_kmh, target_speed_kmh) / target_speed_kmh)
```

**Termination priority chain** (`carla_env/reward.py`,
`check_termination`) — the literal implementation of §3.5's ordered list:

```python
def check_termination(obs_data, collision_flag, step_count, max_steps=1000,
                       max_lateral_m=3.5, max_heading_deg=90.0,
                       stall_flag=False, red_light_violation=False,
                       destination_reached=False):
    if stall_flag:
        return True, False, "stall"
    if collision_flag:
        return True, False, "collision"
    if red_light_violation:
        return True, False, "red_light_violation"
    if abs(obs_data.lateral_distance_m) >= max_lateral_m:
        return True, False, "off_road"
    max_heading_rad = math.radians(max_heading_deg)
    if abs(obs_data.heading_error_rad) >= max_heading_rad:
        return True, False, "wrong_heading"
    if destination_reached:
        return True, False, "destination_reached"
    if step_count >= max_steps:
        return False, True, "timeout"
    return False, False, ""
```

**Red-light violation detection** (`carla_env/traffic_rules.py`,
`RedLightViolationDetector.update`) — "exiting the zone while still
moving," not "moving while a light is red":

```python
def update(self, affordance: TrafficLightAffordance, speed_kmh: float) -> bool:
    if not self.enabled:
        return False
    violated = (
        self._was_at_light
        and not affordance.is_at_light
        and self._was_must_stop
        and speed_kmh > self.stop_speed_kmh
    )
    self._was_at_light  = affordance.is_at_light
    self._was_must_stop = affordance.must_stop
    return violated
```

**Multi-lane off-road tolerance in context** (`carla_env/env.py`,
inside `step()`) — shows exactly how the centering reward and the
termination threshold deliberately use different values:

```python
same_direction_lanes = self._route_planner.count_same_direction_lanes(
    target_waypoint.waypoint
)
effective_max_lateral_m = self.reward_config.max_lateral_m * same_direction_lanes

terminated, truncated, term_reason = check_termination(
    obs_data            = obs_data,
    collision_flag      = collision_flag,
    step_count          = self._step_count,
    max_steps           = self.max_steps,
    max_lateral_m       = effective_max_lateral_m,   # widened
    stall_flag          = stall_flag,
    red_light_violation = red_light_violation,
    destination_reached = destination_reached,
)
# compute_reward() below still uses self.reward_config.max_lateral_m
# (the *base*, un-widened value) for the centering term — the agent is
# still gently pulled toward its own lane regardless of road width.
```

**The algorithm registry pattern** (`agent/algorithms.py`) — why
adding an algorithm needs no changes to `train.py`:

```python
ALGORITHMS = {
    "ppo":  PPO,
    "sac":  SAC,
    "ddpg": DDPG,
    "td3":  TD3,
}

def _build_kwargs(algo_name, algo_cfg, action_space):
    if algo_name in _ON_POLICY_ALGOS:
        return dict(
            learning_rate=algo_cfg["learning_rate"], n_steps=algo_cfg["n_steps"],
            batch_size=algo_cfg["batch_size"], n_epochs=algo_cfg["n_epochs"],
            gamma=algo_cfg["gamma"], gae_lambda=algo_cfg["gae_lambda"],
            clip_range=algo_cfg["clip_range"], ent_coef=algo_cfg["ent_coef"],
            vf_coef=algo_cfg["vf_coef"], max_grad_norm=algo_cfg["max_grad_norm"],
            verbose=algo_cfg["verbose"],
        )
    # Off-policy (SAC/DDPG/TD3) share replay-buffer hyperparameters;
    # DDPG/TD3 additionally get action noise since they're deterministic
    # policies (SAC explores via entropy instead) -- see _NOISE_ALGOS
    # branch in the actual file for that part.
    kwargs = dict(
        learning_rate=algo_cfg["learning_rate"], buffer_size=algo_cfg["buffer_size"],
        learning_starts=algo_cfg["learning_starts"], batch_size=algo_cfg["batch_size"],
        tau=algo_cfg["tau"], gamma=algo_cfg["gamma"],
        train_freq=algo_cfg["train_freq"], gradient_steps=algo_cfg["gradient_steps"],
        verbose=algo_cfg["verbose"],
    )
    return kwargs
```

---

## Changelog (most recent first)

- **2026-09-10**: Fixed a structural bug in this file itself — §5.6/§5.7
  had been accidentally inserted after §7 instead of after §5.5 (still
  readable, just out of logical order). Added §1.1 (system capabilities
  summary), §9 (extensibility/how-to-add-a-module reference table), and
  §10 (real, current code excerpts for the pieces most worth quoting
  directly in the methodology chapter) — all requested so the
  Overleaf-writing session has the same depth of context as this one,
  without needing repo access itself.
- **2026-09-09**: Added §5.8 (data-integrity audit — 5 real bugs found
  and fixed across the metrics/plotting pipeline, see that section for
  the full list), §5.6/§5.7 (screenshots, trajectory-vs-route figures),
  §8 (comparison against published CARLA RL literature).
- **2026-09-08**: Added §3.5 (exact reward formula and termination
  thresholds), completed the equal-150k-step 4-algorithm comparison
  (§5), added §5.5 (two-axis speed-vs-reliability framework).
- **2026-09-07**: 17-issue code-review pass, multi-lane off-road
  tolerance fix, initial version of this file.

---

*This file is maintained by the Claude Code session doing the
implementation work. If something here looks stale or you need more
detail than it covers (exact hyperparameter values, specific commit
history, exact reward formula with weights, etc.), the codebase itself
is the source of truth — `configs/config.yaml` for hyperparameters,
`carla_env/reward.py` for the exact reward formula, `git log` for
development history.*
