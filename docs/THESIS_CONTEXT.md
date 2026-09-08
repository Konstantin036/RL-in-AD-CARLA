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

### Training results so far (all four algorithms, in progress today)

| Algorithm | Steps trained | Result |
|---|---|---|
| PPO | 40,960 (×2 runs) + 250,000 (interrupted at ep. 442) | Clear learning signal: `destination_reached` episodes appearing, speed climbing toward the 30 km/h target, episode length growing. Periodic evals during the 250k run showed reward improving then dipping (907→1080→558 mean reward across 3 eval checkpoints) — real, noisy early-training data, not a clean monotonic curve yet. |
| SAC | 3,000 (smoke test only, more runs pending today) | Clean run, no crashes. Full verification run in progress. |
| DDPG | 80,000 (complete) | 681 episodes, 17.4 min. Termination breakdown: stall 39.9%, wrong_heading 29.4%, off_road 14.8%, collision 11.2%, red_light_violation 3.1%, destination_reached 1.2% (8 real successes), timeout 0.4%. High stall rate matches DDPG's well-documented tendency toward a stand-still local optimum (this is why `learning_starts` was raised to 50,000 for DDPG/TD3 earlier in the project). `evaluate.py` run on the resulting checkpoint (10 episodes, deterministic): mean reward 2025.46 (±1156.28), mean lateral distance 1.12 m, **success rate 20.0%** (2/10 reached destination), mean episode length 688 steps. |
| TD3 | In progress as this file is written | — |

**Important honesty note for the thesis**: none of these are the full
500,000-timestep runs that `configs/config.yaml`'s `training.total_timesteps`
default represents as the intended final run length. Today's runs
(80k-250k depending on algorithm) are verification/comparison runs to
confirm the pipeline and get an early read on relative behavior — not
final results. Whether to run the full 500k for each algorithm before
finalizing thesis numbers is an open decision.

## 6. Honest current limitations (do not oversell)

- **Only DDPG has a complete non-trivial training run with a full
  evaluation pass as of this writing** (see table above) — the other
  three are either smoke-tested only or in progress.
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
