# Thesis Context — CARLA RL Lane Keeping & Intersection Navigation

This file exists specifically so an AI assistant (or a person) writing the
thesis document has an accurate, complete picture of the project without
needing to read the whole codebase. It is kept up to date as the project
progresses — check the "Status as of" date at the top before relying on
anything here as current.

**Status as of: 2026-09-08** (updated same day as written)

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

---

*This file is maintained by the Claude Code session doing the
implementation work. If something here looks stale or you need more
detail than it covers (exact hyperparameter values, specific commit
history, exact reward formula with weights, etc.), the codebase itself
is the source of truth — `configs/config.yaml` for hyperparameters,
`carla_env/reward.py` for the exact reward formula, `git log` for
development history.*
