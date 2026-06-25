# Action Smoothness Reward Penalty — Design

## Problem

Live training observation (SAC, ~240k steps in): the agent achieves excellent
lateral-distance accuracy (0.03-0.04m mean) and stays centered for 100% of
recent episodes (all `timeout`, zero collisions), but the steering it outputs
is violently jittery — swinging from -1.0 to +0.97 across consecutive steps,
in **both** deterministic and stochastic action-selection modes. Confirmed by
loading the latest checkpoint (`results/checkpoints/sac/sac_lane_keeping_240000_steps.zip`)
and inspecting raw action sequences directly.

Applying the existing action smoothing filter
(`carla_env/action.py`'s `ActionSmoother`, `alpha=0.6`) to a sample sequence
only partially dampens this: raw steering jitter of ~0.49 (mean
|step-to-step delta|) becomes ~0.27 after smoothing — still large,
oscillating swings reaching the vehicle, not gentle correction.

Root cause: `carla_env/reward.py`'s reward function
(`w_center * r_centering + w_speed * r_speed + w_heading * r_heading +
r_terminal + r_step`) has no term that depends on the action at all. The
agent has discovered that rapid steering dither, combined with vehicle
inertia and the existing smoothing filter, nets out to a stable centered
trajectory — which is "successful" by the current objective even though
it looks (and would feel, in a real car) unacceptably jerky. No amount of
additional training under this reward function will produce smoother
driving, because smoothness is never rewarded or penalized.

## Goal

Add a smoothness term to the reward function that directly penalizes large
step-to-step changes in the policy's raw action (both acceleration and
steering), so the agent is rewarded for smooth control inputs, not just net
lane position.

## Design

### 1. `carla_env/reward.py` — new component function

```python
def compute_smoothness_reward(action_delta: np.ndarray) -> float:
    """
    Reward for smooth (low-jitter) control inputs.

    Formula: 1.0 - sum(|action_delta|) / 4.0
    Range:   [0.0, 1.0]
    Peak:    1.0 when action_delta is exactly zero (no change since
             last step)
    Zero:    0.0 when both action dimensions swing the full range in
             one step (e.g. acceleration -1 -> +1 AND steer -1 -> +1
             simultaneously: |Δ|=2.0 each, sum=4.0)

    Why penalize the raw action instead of the smoothed/applied one?
        Penalizing the raw action teaches the policy itself to be
        smooth, rather than rewarding it for relying on the existing
        ActionSmoother (carla_env/action.py, alpha=0.6) to absorb
        jitter it could have avoided outputting in the first place.
    """
    delta_magnitude = float(np.abs(action_delta).sum())
    return max(0.0, 1.0 - delta_magnitude / 4.0)
```

### 2. `compute_reward()` — new required parameter

`compute_reward(obs_data, is_terminal, action_delta, cfg=None)`. The
weighted sum gains `+ cfg.w_smooth * r_smoothness`. `RewardInfo` gains an
`r_smoothness: float` field (and its `__repr__` includes it), matching how
`r_centering`/`r_speed`/`r_heading` are already broken out for logging.

This is a breaking signature change — every existing caller of
`compute_reward()` (currently only `carla_env/env.py:step()` and
`scripts/test_reward.py`'s tests) must be updated to pass `action_delta`.

### 3. `carla_env/env.py` — track the previous raw action

New instance attribute `self._previous_raw_action`, reset to
`np.zeros(2, dtype=np.float32)` in `reset()` (mirroring how
`ActionSmoother` itself starts its internal smoothing state at zero — same
convention, same file family). In `step(action)`:

```python
action_delta = action - self._previous_raw_action
self._previous_raw_action = np.asarray(action, dtype=np.float32).copy()
```

computed once, passed into `compute_reward(..., action_delta=action_delta, ...)`.

### 4. `RewardConfig` and `configs/config.yaml` — new weight, toggleable like `step_penalty`

`RewardConfig` gains `w_smooth: float = 0.5` (dataclass default, matching
the project's existing pattern of giving every weight a sensible default).
`configs/config.yaml`'s `reward:` block gains:

```yaml
w_smooth:         0.5         # weight: action smoothness penalty
                               #   (set to 0.0 to disable)
```

Setting `w_smooth: 0.0` turns the term off completely (no code change, no
new toggle flag needed) — multiplying any value by a zero weight removes
its contribution from the total, exactly how `step_penalty` is already
documented ("Set to 0.0 to disable.") in `RewardConfig`'s docstring. This
keeps the modularity pattern consistent across the project: just like
`agent/algorithms.py`'s registry lets you switch RL algorithms via one
config field, this lets you switch the smoothness penalty on/off via one
config field, without touching code or restarting with a different flag.

`0.5` matches `w_heading`'s magnitude — meaningful enough to shape
behavior, not so large it swamps centering/speed/heading.

## Out of scope

- No change to `carla_env/action.py`'s existing `ActionSmoother`
  (`alpha=0.6`) — the smoothing filter and the new reward penalty are
  complementary, not redundant: smoothing limits how much jitter reaches
  the vehicle regardless of what the policy outputs; the reward penalty
  gives the policy an incentive not to output jitter in the first place.
- No retroactive change to already-saved checkpoints or already-collected
  training data — this changes the reward signal going forward only.
- No automatic restart of the currently-running live training session.
  The current run's process already has the old `reward.py` loaded in
  memory; this change takes effect on the next fresh `agent/train.py`
  invocation.

## Testing

`compute_smoothness_reward()` is a pure function (no CARLA dependency) —
fully offline-testable in `scripts/test_reward.py`, following the same
pattern as the existing centering/speed/heading component tests:

1. Zero delta (`action_delta = [0.0, 0.0]`) → exactly `1.0`.
2. Maximum delta (`action_delta = [2.0, 2.0]`, i.e. both dims flipped
   from one extreme to the other) → exactly `0.0`.
3. A representative intermediate delta (e.g. `[0.5, -0.3]`, magnitude
   sum `0.8`) → `1.0 - 0.8/4.0 = 0.8`.
4. `compute_reward()`'s full weighted-sum integration test (already
   exists in `scripts/test_reward.py` as "Full reward at representative
   states") gets a new representative state exercising a nonzero
   `action_delta`, confirming the smoothness term is actually wired into
   the total.
5. Toggle test: call `compute_reward()` with a large nonzero
   `action_delta` once with `cfg.w_smooth=0.5` and once with
   `cfg.w_smooth=0.0`, confirming the two totals differ by exactly
   `0.5 * r_smoothness` — i.e. the weight genuinely turns the term's
   contribution on/off, matching `step_penalty`'s existing toggle
   convention.

Live verification (since this changes training dynamics, not just a pure
function): start a fresh short training run (a few thousand timesteps) and
confirm via `episode_log.csv`/console output that training still runs
without error and produces sane reward magnitudes (the new `w_smooth *
r_smoothness` term is bounded `[0, 0.5]` per step, comparable in scale to
the existing terms, so total reward magnitudes should look similar to
before, not wildly different). Full evidence of *smoother driving* will
only emerge after a complete training run, not from this verification
step alone.
