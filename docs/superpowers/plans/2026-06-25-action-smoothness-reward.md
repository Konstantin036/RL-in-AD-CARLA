# Action Smoothness Reward Penalty Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reward term that penalizes jerky step-to-step changes in the agent's raw action (acceleration and steering both), toggleable via a single config weight (`w_smooth: 0.0` disables it, matching the project's existing `step_penalty` convention), and trackable in TensorBoard/CSV like every other reward component.

**Architecture:** A new pure function `compute_smoothness_reward()` in `carla_env/reward.py`, wired into `compute_reward()`'s existing weighted-sum pattern via a new required `action_delta` parameter. `carla_env/env.py` tracks the previous raw action across steps and computes the delta. `agent/callbacks.py` gains a mean-smoothness metric alongside the existing mean-lateral-distance/mean-speed tracking. `configs/config.yaml` gains one new weight.

**Tech Stack:** Python 3.7.16, no new dependencies (uses existing `numpy`).

## Global Constraints

- Python 3.7 compatible syntax only.
- Use `logger` (not `print`) inside `carla_env/` and `agent/` modules; use `print` inside `scripts/`.
- Every non-terminal reward component stays normalized to `[0, 1]` (existing design goal stated in `carla_env/reward.py`'s module docstring) — `compute_smoothness_reward()` must follow this convention.
- `w_smooth: 0.0` must fully disable the term's contribution — no separate boolean flag, reusing the project's existing weight-as-toggle convention (already documented for `step_penalty`).
- This changes the reward signal going forward only — no retroactive change to saved checkpoints, and the currently-running live training process (already has the old `reward.py` loaded) is unaffected until its next restart.
- Live verification (Task 6) must not call `.close()` on any `CarlaLaneKeepingEnv` instance while other training may be using the same CARLA server in synchronous mode — `close()` disables synchronous mode for the *entire shared world*, not just the caller, which previously broke a live training run. Use a spawn point not in use by any other active run, and let the verification script exit without closing.

---

### Task 1: `compute_smoothness_reward()` pure function

**Files:**
- Modify: `carla_env/reward.py:60-80` (`RewardConfig` — add `w_smooth` field and tuning-guidance line)
- Modify: `carla_env/reward.py:94-109` (`RewardInfo` — add `r_smoothness` field)
- Modify: `carla_env/reward.py` (add `compute_smoothness_reward()` function, after `compute_heading_reward()`)
- Test: `scripts/test_reward.py` (append a new test function)

**Interfaces:**
- Produces: `compute_smoothness_reward(action_delta: np.ndarray) -> float`, range `[0.0, 1.0]`. `RewardConfig.w_smooth: float` (default `0.5`). `RewardInfo.r_smoothness: float`. Consumed by Task 2's `compute_reward()`.

- [ ] **Step 1: Write the failing test**

In `scripts/test_reward.py`, change:

```python
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from carla_env.reward import (
    RewardConfig,
    RewardInfo,
    compute_centering_reward,
    compute_speed_reward,
    compute_heading_reward,
    compute_reward,
    check_termination,
)
from carla_env.observation import ObservationData
```

to:

```python
import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from carla_env.reward import (
    RewardConfig,
    RewardInfo,
    compute_centering_reward,
    compute_speed_reward,
    compute_heading_reward,
    compute_smoothness_reward,
    compute_reward,
    check_termination,
)
from carla_env.observation import ObservationData
```

Then change:

```python
    print("  ✓ PASSED")


# ── Test 4: Full reward at representative states ───────────────────────────────

def test_full_reward():
    sep("4. Full reward at representative states")
```

to:

```python
    print("  ✓ PASSED")


# ── Test: Smoothness reward ────────────────────────────────────────────────────

def test_smoothness():
    sep("3b. Smoothness reward  r = 1 - sum(|action_delta|) / 4.0")
    cases = [
        # (action_delta, expected, description)
        (np.array([0.0,  0.0]), 1.0, "no change — perfectly smooth"),
        (np.array([2.0,  2.0]), 0.0, "both dims flipped full range"),
        (np.array([0.5, -0.3]), 0.8, "moderate change"),
        (np.array([1.0,  0.0]), 0.5, "one dim flipped halfway"),
        (np.array([3.0,  3.0]), 0.0, "beyond max — clipped at 0.0"),
    ]
    for action_delta, expected, desc in cases:
        result = compute_smoothness_reward(action_delta)
        status = "✓" if abs(result - expected) < 1e-6 else "✗"
        print(f"  delta={action_delta}  expected={expected:.3f}  got={result:.3f}  {status}  {desc}")
        assert abs(result - expected) < 1e-6, f"{desc}: expected {expected}, got {result}"
    print("  ✓ PASSED")


# ── Test 4: Full reward at representative states ───────────────────────────────

def test_full_reward():
    sep("4. Full reward at representative states")
```

Update `main()` to call it (insert between `test_heading()` and `test_full_reward()`):

```python
def main():
    print("=" * 60)
    print("  Phase 5 — Reward Function Unit Tests")
    print("  (No CARLA connection required)")
    print("=" * 60)

    test_centering()
    test_speed()
    test_heading()
    test_smoothness()
    test_full_reward()
    test_termination()
    test_reward_range()

    print(f"\n{'='*60}")
    print("  All reward tests passed.")
    print("  Ready to move to Phase 6 — Gym Environment.")
    print(f"{'='*60}\n")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_reward.py`
Expected: `ImportError: cannot import name 'compute_smoothness_reward' from 'carla_env.reward'`

- [ ] **Step 3: Add `w_smooth` to `RewardConfig`**

In `carla_env/reward.py`, change:

```python
    # ── Heading term ───────────────────────────────────────────────────────────
    w_heading: float = 0.5          # weight for heading alignment reward

    # ── Terminal penalty ────────────────────────────────────────────────────────
```

to:

```python
    # ── Heading term ───────────────────────────────────────────────────────────
    w_heading: float = 0.5          # weight for heading alignment reward

    # ── Smoothness term ────────────────────────────────────────────────────────
    # Penalizes large step-to-step changes in the raw action (acceleration
    # and steering both). Set to 0.0 to disable.
    w_smooth: float = 0.5           # weight for action smoothness penalty

    # ── Terminal penalty ────────────────────────────────────────────────────────
```

- [ ] **Step 4: Update the tuning-guidance docstring**

In `carla_env/reward.py`, change:

```python
    Common tuning guidance:
        Agent stands still     → increase w_speed or target_speed
        Agent drives too fast  → decrease target_speed or sigma_speed
        Agent cuts corners     → increase w_heading
        Agent hugs one side    → increase w_center
    """
```

to:

```python
    Common tuning guidance:
        Agent stands still     → increase w_speed or target_speed
        Agent drives too fast  → decrease target_speed or sigma_speed
        Agent cuts corners     → increase w_heading
        Agent hugs one side    → increase w_center
        Agent steers jerkily   → increase w_smooth (0.0 disables it)
    """
```

- [ ] **Step 5: Add `r_smoothness` to `RewardInfo`**

In `carla_env/reward.py`, change:

```python
    total: float         # final scalar reward sent to the agent
    r_centering: float   # centering component (before weighting)
    r_speed: float       # speed component (before weighting)
    r_heading: float     # heading component (before weighting)
    r_terminal: float    # terminal penalty (0 unless episode ended)
    r_step: float        # step penalty
    is_terminal: bool    # True if episode ended this step

    def __repr__(self) -> str:
        return (
            f"Reward(total={self.total:+.3f} | "
            f"center={self.r_centering:.3f} "
            f"speed={self.r_speed:.3f} "
            f"heading={self.r_heading:.3f} "
            f"terminal={self.r_terminal:.1f})"
        )
```

to:

```python
    total: float         # final scalar reward sent to the agent
    r_centering: float   # centering component (before weighting)
    r_speed: float       # speed component (before weighting)
    r_heading: float     # heading component (before weighting)
    r_smoothness: float  # smoothness component (before weighting)
    r_terminal: float    # terminal penalty (0 unless episode ended)
    r_step: float        # step penalty
    is_terminal: bool    # True if episode ended this step

    def __repr__(self) -> str:
        return (
            f"Reward(total={self.total:+.3f} | "
            f"center={self.r_centering:.3f} "
            f"speed={self.r_speed:.3f} "
            f"heading={self.r_heading:.3f} "
            f"smooth={self.r_smoothness:.3f} "
            f"terminal={self.r_terminal:.1f})"
        )
```

- [ ] **Step 6: Add `compute_smoothness_reward()`**

In `carla_env/reward.py`, change:

```python
    normalized = abs(heading_error_rad) / math.pi
    return float(max(0.0, 1.0 - normalized))


# ── Main reward function ───────────────────────────────────────────────────────
```

to:

```python
    normalized = abs(heading_error_rad) / math.pi
    return float(max(0.0, 1.0 - normalized))


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
        carla_env/action.py's ActionSmoother (alpha=0.6) already limits
        how much jitter reaches the vehicle, but the policy itself can
        still rely on that filter to absorb jitter it didn't need to
        output in the first place. Penalizing the raw action teaches
        the policy to be smooth on its own.
    """
    delta_magnitude = float(np.abs(action_delta).sum())
    return max(0.0, 1.0 - delta_magnitude / 4.0)


# ── Main reward function ───────────────────────────────────────────────────────
```

- [ ] **Step 7: Run test to verify it passes**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_reward.py`
Expected: this will still fail at this point with a different error — `compute_reward()` (called by `test_full_reward`/`test_reward_range`, both run by `main()` before this task is done) doesn't yet accept `action_delta`. That's expected; Task 2 fixes it. For this task's own verification, run just the new test in isolation:

```bash
/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python -c "
import sys; sys.path.insert(0, '.')
from scripts.test_reward import test_smoothness
test_smoothness()
"
```
Expected: prints all 5 cases with `✓` and ends with `  ✓ PASSED`.

- [ ] **Step 8: Commit**

```bash
cd /home/rtrk/konstantin/diplomski/carla_rl_project
git add carla_env/reward.py scripts/test_reward.py
git commit -m "Add compute_smoothness_reward() pure function and RewardConfig.w_smooth"
```

---

### Task 2: Wire `action_delta` into `compute_reward()`

**Files:**
- Modify: `carla_env/reward.py:1-32` (module docstring — reward formula)
- Modify: `carla_env/reward.py:179-245` (`compute_reward()`)
- Modify: `scripts/test_reward.py` (update all existing `compute_reward()` call sites)

**Interfaces:**
- Consumes: `compute_smoothness_reward()`, `RewardConfig.w_smooth`, `RewardInfo.r_smoothness` from Task 1.
- Produces: `compute_reward(obs_data, is_terminal: bool, action_delta: np.ndarray, cfg: RewardConfig = None) -> tuple`. This is a breaking signature change (new required positional parameter inserted before `cfg`) — every caller must be updated. Consumed by Task 3 (`carla_env/env.py`).

- [ ] **Step 1: Update the module docstring's reward formula**

In `carla_env/reward.py`, change:

```python
Reward formula:
    r = w_center * r_centering
      + w_speed  * r_speed
      + w_heading * r_heading
      + r_terminal  (only on terminal steps)

Where:
    r_centering = 1.0 - |lateral_distance| / MAX_LATERAL_DISTANCE
    r_speed     = exp(-((speed - target_speed)^2) / (2 * sigma^2))
    r_heading   = 1.0 - |heading_error| / pi
    r_terminal  = TERMINAL_PENALTY  (large negative, episode-ending events)
```

to:

```python
Reward formula:
    r = w_center * r_centering
      + w_speed  * r_speed
      + w_heading * r_heading
      + w_smooth * r_smoothness
      + r_terminal  (only on terminal steps)

Where:
    r_centering  = 1.0 - |lateral_distance| / MAX_LATERAL_DISTANCE
    r_speed      = exp(-((speed - target_speed)^2) / (2 * sigma^2))
    r_heading    = 1.0 - |heading_error| / pi
    r_smoothness = 1.0 - sum(|action_delta|) / 4.0
    r_terminal   = TERMINAL_PENALTY  (large negative, episode-ending events)
```

- [ ] **Step 2: Add `action_delta` to `compute_reward()`'s signature and body**

In `carla_env/reward.py`, change:

```python
def compute_reward(
    obs_data,           # ObservationData from observation.py
    is_terminal: bool,  # True if the episode is ending this step
    cfg: RewardConfig = None,
) -> tuple:
    """
    Compute the full reward for one step.

    This is called by env.py at every step(), after the world ticks
    and the new observation is computed.

    Args:
        obs_data:    ObservationData (lateral_distance_m, heading_error_rad,
                     speed_kmh, steering) — from observation.py
        is_terminal: True if the episode ends after this step
                     (collision, off-road, or timeout)
        cfg:         RewardConfig — defaults to RewardConfig() if None

    Returns:
        reward:      float — scalar reward for the agent
        info:        RewardInfo — breakdown for logging
    """
    if cfg is None:
        cfg = RewardConfig()

    # ── Compute individual components ──────────────────────────────────────────
    r_centering = compute_centering_reward(
        obs_data.lateral_distance_m,
        cfg.max_lateral_m,
    )

    r_speed = compute_speed_reward(
        obs_data.speed_kmh,
        cfg.target_speed_kmh,
        cfg.sigma_speed,
    )

    r_heading = compute_heading_reward(obs_data.heading_error_rad)

    # ── Terminal penalty ────────────────────────────────────────────────────────
    # Only applied on the step where the episode ends badly.
    # Not applied on timeout (episode ran for max steps without crashing).
    r_terminal = cfg.terminal_penalty if is_terminal else 0.0

    # ── Step penalty ────────────────────────────────────────────────────────────
    r_step = cfg.step_penalty

    # ── Weighted sum ────────────────────────────────────────────────────────────
    total = (
        cfg.w_center  * r_centering
        + cfg.w_speed   * r_speed
        + cfg.w_heading * r_heading
        + r_terminal
        + r_step
    )

    info = RewardInfo(
        total=total,
        r_centering=r_centering,
        r_speed=r_speed,
        r_heading=r_heading,
        r_terminal=r_terminal,
        r_step=r_step,
        is_terminal=is_terminal,
    )

    return float(total), info
```

to:

```python
def compute_reward(
    obs_data,                 # ObservationData from observation.py
    is_terminal: bool,        # True if the episode is ending this step
    action_delta: np.ndarray, # current raw action minus previous raw action, shape (2,)
    cfg: RewardConfig = None,
) -> tuple:
    """
    Compute the full reward for one step.

    This is called by env.py at every step(), after the world ticks
    and the new observation is computed.

    Args:
        obs_data:     ObservationData (lateral_distance_m, heading_error_rad,
                      speed_kmh, steering) — from observation.py
        is_terminal:  True if the episode ends after this step
                      (collision, off-road, or timeout)
        action_delta: current raw action minus the previous raw action,
                      shape (2,) — see compute_smoothness_reward()
        cfg:          RewardConfig — defaults to RewardConfig() if None

    Returns:
        reward:      float — scalar reward for the agent
        info:        RewardInfo — breakdown for logging
    """
    if cfg is None:
        cfg = RewardConfig()

    # ── Compute individual components ──────────────────────────────────────────
    r_centering = compute_centering_reward(
        obs_data.lateral_distance_m,
        cfg.max_lateral_m,
    )

    r_speed = compute_speed_reward(
        obs_data.speed_kmh,
        cfg.target_speed_kmh,
        cfg.sigma_speed,
    )

    r_heading = compute_heading_reward(obs_data.heading_error_rad)

    r_smoothness = compute_smoothness_reward(action_delta)

    # ── Terminal penalty ────────────────────────────────────────────────────────
    # Only applied on the step where the episode ends badly.
    # Not applied on timeout (episode ran for max steps without crashing).
    r_terminal = cfg.terminal_penalty if is_terminal else 0.0

    # ── Step penalty ────────────────────────────────────────────────────────────
    r_step = cfg.step_penalty

    # ── Weighted sum ────────────────────────────────────────────────────────────
    total = (
        cfg.w_center  * r_centering
        + cfg.w_speed   * r_speed
        + cfg.w_heading * r_heading
        + cfg.w_smooth  * r_smoothness
        + r_terminal
        + r_step
    )

    info = RewardInfo(
        total=total,
        r_centering=r_centering,
        r_speed=r_speed,
        r_heading=r_heading,
        r_smoothness=r_smoothness,
        r_terminal=r_terminal,
        r_step=r_step,
        is_terminal=is_terminal,
    )

    return float(total), info
```

- [ ] **Step 3: Update `test_full_reward()`'s call sites**

In `scripts/test_reward.py`, change:

```python
    for lat, hdg_deg, spd, is_term, desc in scenarios:
        obs = ObservationData(
            lateral_distance_m=lat,
            heading_error_rad=math.radians(hdg_deg),
            speed_kmh=spd,
            steering=0.0,
        )
        reward, info = compute_reward(obs, is_terminal=is_term, cfg=cfg)
```

to:

```python
    for lat, hdg_deg, spd, is_term, desc in scenarios:
        obs = ObservationData(
            lateral_distance_m=lat,
            heading_error_rad=math.radians(hdg_deg),
            speed_kmh=spd,
            steering=0.0,
        )
        reward, info = compute_reward(
            obs, is_terminal=is_term, action_delta=np.zeros(2), cfg=cfg,
        )
```

Then change:

```python
    print("\n  Best possible reward (no step penalty):")
    obs_best = ObservationData(0.0, 0.0, 30.0, 0.0)
    r_best, _ = compute_reward(obs_best, False, cfg)
    print(f"    w_center({cfg.w_center}) * 1.0 "
          f"+ w_speed({cfg.w_speed}) * 1.0 "
          f"+ w_heading({cfg.w_heading}) * 1.0 "
          f"+ step_penalty({cfg.step_penalty})")
    print(f"    = {r_best:+.3f}")
    print("  ✓ PASSED")
```

to:

```python
    print("\n  Best possible reward (no step penalty):")
    obs_best = ObservationData(0.0, 0.0, 30.0, 0.0)
    r_best, _ = compute_reward(obs_best, False, action_delta=np.zeros(2), cfg=cfg)
    print(f"    w_center({cfg.w_center}) * 1.0 "
          f"+ w_speed({cfg.w_speed}) * 1.0 "
          f"+ w_heading({cfg.w_heading}) * 1.0 "
          f"+ w_smooth({cfg.w_smooth}) * 1.0 "
          f"+ step_penalty({cfg.step_penalty})")
    print(f"    = {r_best:+.3f}")
    print("  ✓ PASSED")
```

- [ ] **Step 4: Update `test_reward_range()`'s call site to also randomize `action_delta`**

In `scripts/test_reward.py`, change:

```python
    for _ in range(n_samples):
        obs = ObservationData(
            lateral_distance_m=random.uniform(-4.0,  4.0),
            heading_error_rad =random.uniform(-math.pi, math.pi),
            speed_kmh         =random.uniform(0.0,   80.0),
            steering          =random.uniform(-1.0,   1.0),
        )
        is_terminal = random.random() < 0.05
        r, _ = compute_reward(obs, is_terminal, cfg)
        min_r = min(min_r, r)
        max_r = max(max_r, r)

    print(f"  Sampled {n_samples} random states")
    print(f"  Min reward: {min_r:+.3f}")
    print(f"  Max reward: {max_r:+.3f}")
    print(f"  Expected:   min ≈ {cfg.terminal_penalty + cfg.step_penalty:.1f},  "
          f"max ≈ {cfg.w_center + cfg.w_speed + cfg.w_heading + cfg.step_penalty:.2f}")
    assert min_r >= cfg.terminal_penalty - 0.1, "Min reward unexpectedly low"
    assert max_r <= cfg.w_center + cfg.w_speed + cfg.w_heading + 0.1, "Max reward unexpectedly high"
    print("  ✓ PASSED")
```

to:

```python
    for _ in range(n_samples):
        obs = ObservationData(
            lateral_distance_m=random.uniform(-4.0,  4.0),
            heading_error_rad =random.uniform(-math.pi, math.pi),
            speed_kmh         =random.uniform(0.0,   80.0),
            steering          =random.uniform(-1.0,   1.0),
        )
        is_terminal = random.random() < 0.05
        action_delta = np.array([
            random.uniform(-2.0, 2.0),
            random.uniform(-2.0, 2.0),
        ])
        r, _ = compute_reward(obs, is_terminal, action_delta, cfg)
        min_r = min(min_r, r)
        max_r = max(max_r, r)

    max_possible = cfg.w_center + cfg.w_speed + cfg.w_heading + cfg.w_smooth + cfg.step_penalty
    print(f"  Sampled {n_samples} random states")
    print(f"  Min reward: {min_r:+.3f}")
    print(f"  Max reward: {max_r:+.3f}")
    print(f"  Expected:   min ≈ {cfg.terminal_penalty + cfg.step_penalty:.1f},  "
          f"max ≈ {max_possible:.2f}")
    assert min_r >= cfg.terminal_penalty - 0.1, "Min reward unexpectedly low"
    assert max_r <= max_possible + 0.1, "Max reward unexpectedly high"
    print("  ✓ PASSED")
```

- [ ] **Step 5: Add the toggle test**

In `scripts/test_reward.py`, change:

```python
    print("  ✓ PASSED")


# ── Test 4: Full reward at representative states ───────────────────────────────

def test_full_reward():
    sep("4. Full reward at representative states")
```

to:

```python
    print("  ✓ PASSED")


def test_smoothness_toggle():
    sep("3c. w_smooth=0.0 fully disables the smoothness term's contribution")
    obs = ObservationData(
        lateral_distance_m=0.0,
        heading_error_rad=0.0,
        speed_kmh=30.0,
        steering=0.0,
    )
    action_delta = np.array([1.5, -1.0])  # a large, nonzero change

    cfg_on = RewardConfig(w_smooth=0.5)
    cfg_off = RewardConfig(w_smooth=0.0)

    reward_on, info_on = compute_reward(obs, False, action_delta, cfg_on)
    reward_off, info_off = compute_reward(obs, False, action_delta, cfg_off)

    # The raw component is identical either way — only its weighted
    # contribution to the total changes.
    assert abs(info_on.r_smoothness - info_off.r_smoothness) < 1e-9
    expected_diff = 0.5 * info_on.r_smoothness
    actual_diff = reward_on - reward_off
    print(f"  reward with w_smooth=0.5: {reward_on:+.4f}")
    print(f"  reward with w_smooth=0.0: {reward_off:+.4f}")
    print(f"  difference: {actual_diff:.4f}  (expected: {expected_diff:.4f})")
    assert abs(actual_diff - expected_diff) < 1e-6, \
        f"Expected difference {expected_diff}, got {actual_diff}"
    print("  ✓ PASSED")


# ── Test 4: Full reward at representative states ───────────────────────────────

def test_full_reward():
    sep("4. Full reward at representative states")
```

- [ ] **Step 6: Update `main()` to call the two new tests**

In `scripts/test_reward.py`, change:

```python
    test_centering()
    test_speed()
    test_heading()
    test_smoothness()
    test_full_reward()
    test_termination()
    test_reward_range()
```

to:

```python
    test_centering()
    test_speed()
    test_heading()
    test_smoothness()
    test_smoothness_toggle()
    test_full_reward()
    test_termination()
    test_reward_range()
```

- [ ] **Step 7: Run the full test file to verify everything passes**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_reward.py`
Expected: `All reward tests passed.` with no errors, including the new `3b.` and `3c.` sections.

- [ ] **Step 8: Run the rest of the offline suite to confirm no regressions**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_spawn.py && /home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_action.py && /home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_algorithms.py`
Expected: all pass (these don't touch `reward.py`'s signature, but confirm nothing else broke).

- [ ] **Step 9: Commit**

```bash
cd /home/rtrk/konstantin/diplomski/carla_rl_project
git add carla_env/reward.py scripts/test_reward.py
git commit -m "Wire action_delta and w_smooth into compute_reward()"
```

---

### Task 3: Track the previous raw action in `carla_env/env.py`

**Files:**
- Modify: `carla_env/env.py:223-224` (`__init__` — internal state)
- Modify: `carla_env/env.py:480` area (`reset()` — reset tracking)
- Modify: `carla_env/env.py:518-553` area (`step()` — compute delta, pass to `compute_reward()`, add info key)

**Interfaces:**
- Consumes: `compute_reward(obs_data, is_terminal, action_delta, cfg)` from Task 2.
- Produces: `info["reward_smoothness"]` key in `step()`'s returned info dict — consumed by Task 4 (`agent/callbacks.py`).

- [ ] **Step 1: Add `_previous_raw_action` to `__init__`'s internal state**

In `carla_env/env.py`, change:

```python
        self._vehicle             = None   # carla.Vehicle (ego)
        self._collision_sensor    = None   # CollisionSensor wrapper
        self._spawn_points        = []     # list of carla.Transform
        self._last_spawn_transform = None  # carla.Transform vehicle was spawned at
```

to:

```python
        self._vehicle             = None   # carla.Vehicle (ego)
        self._collision_sensor    = None   # CollisionSensor wrapper
        self._spawn_points        = []     # list of carla.Transform
        self._last_spawn_transform = None  # carla.Transform vehicle was spawned at
        self._previous_raw_action  = np.zeros(2, dtype=np.float32)  # for smoothness reward
```

- [ ] **Step 2: Reset it at the start of each episode**

In `carla_env/env.py`, change:

```python
        # ── Reset action smoother ──────────────────────────────────────────────
        # Critical: without this, the smoother carries state from the
        # last episode into the new one.
        self._action_processor.reset()
```

to:

```python
        # ── Reset action smoother ──────────────────────────────────────────────
        # Critical: without this, the smoother carries state from the
        # last episode into the new one.
        self._action_processor.reset()

        # ── Reset smoothness-reward tracking ────────────────────────────────────
        # Same reasoning as the action smoother: without this, the first
        # action of a new episode would be compared against the last
        # action of the *previous* episode.
        self._previous_raw_action = np.zeros(2, dtype=np.float32)
```

- [ ] **Step 3: Compute the delta and pass it to `compute_reward()`**

In `carla_env/env.py`, change:

```python
        # Terminal penalty only on agent failure, not on timeout
        is_terminal_for_reward = terminated   # not truncated

        # ── Compute reward ─────────────────────────────────────────────────────
        reward, reward_info = compute_reward(
            obs_data    = obs_data,
            is_terminal = is_terminal_for_reward,
            cfg         = self.reward_config,
        )
```

to:

```python
        # Terminal penalty only on agent failure, not on timeout
        is_terminal_for_reward = terminated   # not truncated

        # ── Compute action delta for the smoothness reward ─────────────────────
        action_array = np.asarray(action, dtype=np.float32)
        action_delta = action_array - self._previous_raw_action
        self._previous_raw_action = action_array.copy()

        # ── Compute reward ─────────────────────────────────────────────────────
        reward, reward_info = compute_reward(
            obs_data     = obs_data,
            is_terminal  = is_terminal_for_reward,
            action_delta = action_delta,
            cfg          = self.reward_config,
        )
```

- [ ] **Step 4: Add `reward_smoothness` to the info dict**

In `carla_env/env.py`, change:

```python
            # Reward breakdown
            "reward_total":     reward_info.total,
            "reward_centering": reward_info.r_centering,
            "reward_speed":     reward_info.r_speed,
            "reward_heading":   reward_info.r_heading,
            "reward_terminal":  reward_info.r_terminal,
```

to:

```python
            # Reward breakdown
            "reward_total":      reward_info.total,
            "reward_centering":  reward_info.r_centering,
            "reward_speed":      reward_info.r_speed,
            "reward_heading":    reward_info.r_heading,
            "reward_smoothness": reward_info.r_smoothness,
            "reward_terminal":   reward_info.r_terminal,
```

- [ ] **Step 5: Run the offline suite to confirm no regressions**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_spawn.py && /home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_action.py && /home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_reward.py && /home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_algorithms.py`
Expected: all pass. (`carla_env/env.py` itself has no offline tests — its CARLA-dependent code paths are only reachable live, consistent with the rest of this file. Live verification is Task 6.)

- [ ] **Step 6: Commit**

```bash
cd /home/rtrk/konstantin/diplomski/carla_rl_project
git add carla_env/env.py
git commit -m "Track previous raw action and feed action_delta into compute_reward()"
```

---

### Task 4: Track mean smoothness in `agent/callbacks.py`

**Files:**
- Modify: `agent/callbacks.py:51-101` (docstring + `__init__` + `_on_training_start`)
- Modify: `agent/callbacks.py:106-185` (`_on_step`)

**Interfaces:**
- Consumes: `info["reward_smoothness"]` from Task 3.
- Produces: `episode/mean_smoothness` TensorBoard scalar, `mean_smoothness` CSV column — both follow the exact existing pattern of `episode/mean_lat_dist`/`mean_lateral_dist`.

- [ ] **Step 1: Update the class docstring**

In `agent/callbacks.py`, change:

```python
    Metrics logged per episode:
        - episode_reward      total undiscounted return
        - episode_length      number of steps
        - termination_reason  collision / off_road / wrong_heading / timeout
        - mean lateral distance
        - mean speed
```

to:

```python
    Metrics logged per episode:
        - episode_reward      total undiscounted return
        - episode_length      number of steps
        - termination_reason  collision / off_road / wrong_heading / timeout
        - mean lateral distance
        - mean speed
        - mean smoothness (1.0 = perfectly smooth control, 0.0 = max jitter)
```

- [ ] **Step 2: Add the accumulator**

In `agent/callbacks.py`, change:

```python
        # Running episode accumulators
        self._ep_reward      = 0.0
        self._ep_steps       = 0
        self._ep_lat_dists   = []
        self._ep_speeds      = []
        self._ep_count       = 0
        self._training_start = None
```

to:

```python
        # Running episode accumulators
        self._ep_reward       = 0.0
        self._ep_steps        = 0
        self._ep_lat_dists    = []
        self._ep_speeds       = []
        self._ep_smoothness   = []
        self._ep_count        = 0
        self._training_start  = None
```

- [ ] **Step 3: Add the CSV header column**

In `agent/callbacks.py`, change:

```python
            writer.writerow([
                "episode",
                "timestep",
                "episode_reward",
                "episode_length",
                "mean_lateral_dist",
                "mean_speed_kmh",
                "termination_reason",
                "elapsed_seconds",
            ])
```

to:

```python
            writer.writerow([
                "episode",
                "timestep",
                "episode_reward",
                "episode_length",
                "mean_lateral_dist",
                "mean_speed_kmh",
                "mean_smoothness",
                "termination_reason",
                "elapsed_seconds",
            ])
```

- [ ] **Step 4: Accumulate per-step and compute the episode mean**

In `agent/callbacks.py`, change:

```python
            lat = info.get("lateral_distance", 0.0)
            spd = info.get("speed_kmh",        0.0)
            self._ep_lat_dists.append(abs(lat))
            self._ep_speeds.append(spd)
```

to:

```python
            lat = info.get("lateral_distance",  0.0)
            spd = info.get("speed_kmh",         0.0)
            smooth = info.get("reward_smoothness", 0.0)
            self._ep_lat_dists.append(abs(lat))
            self._ep_speeds.append(spd)
            self._ep_smoothness.append(smooth)
```

Then change:

```python
                # Compute episode summary stats
                mean_lat = float(np.mean(self._ep_lat_dists)) if self._ep_lat_dists else 0.0
                mean_spd = float(np.mean(self._ep_speeds))    if self._ep_speeds    else 0.0
                elapsed  = time.time() - self._training_start
```

to:

```python
                # Compute episode summary stats
                mean_lat     = float(np.mean(self._ep_lat_dists))  if self._ep_lat_dists  else 0.0
                mean_spd     = float(np.mean(self._ep_speeds))     if self._ep_speeds     else 0.0
                mean_smooth  = float(np.mean(self._ep_smoothness)) if self._ep_smoothness else 0.0
                elapsed      = time.time() - self._training_start
```

- [ ] **Step 5: Log to TensorBoard**

In `agent/callbacks.py`, change:

```python
                self.logger.record("episode/reward",       self._ep_reward)
                self.logger.record("episode/length",       self._ep_steps)
                self.logger.record("episode/mean_lat_dist",mean_lat)
                self.logger.record("episode/mean_speed",   mean_spd)
                self.logger.record("episode/count",        self._ep_count)
```

to:

```python
                self.logger.record("episode/reward",         self._ep_reward)
                self.logger.record("episode/length",         self._ep_steps)
                self.logger.record("episode/mean_lat_dist",  mean_lat)
                self.logger.record("episode/mean_speed",     mean_spd)
                self.logger.record("episode/mean_smoothness",mean_smooth)
                self.logger.record("episode/count",          self._ep_count)
```

- [ ] **Step 6: Write to CSV**

In `agent/callbacks.py`, change:

```python
                    writer.writerow([
                        self._ep_count,
                        self.num_timesteps,
                        round(self._ep_reward, 4),
                        self._ep_steps,
                        round(mean_lat, 4),
                        round(mean_spd, 4),
                        reason,
                        round(elapsed, 1),
                    ])
```

to:

```python
                    writer.writerow([
                        self._ep_count,
                        self.num_timesteps,
                        round(self._ep_reward, 4),
                        self._ep_steps,
                        round(mean_lat, 4),
                        round(mean_spd, 4),
                        round(mean_smooth, 4),
                        reason,
                        round(elapsed, 1),
                    ])
```

- [ ] **Step 7: Reset the accumulator for the next episode**

In `agent/callbacks.py`, change:

```python
                # Reset accumulators for next episode
                self._ep_reward    = 0.0
                self._ep_steps     = 0
                self._ep_lat_dists = []
                self._ep_speeds    = []
```

to:

```python
                # Reset accumulators for next episode
                self._ep_reward      = 0.0
                self._ep_steps       = 0
                self._ep_lat_dists   = []
                self._ep_speeds      = []
                self._ep_smoothness  = []
```

- [ ] **Step 8: Sanity-check the module still imports cleanly**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python -c "import sys; sys.path.insert(0, '.'); import agent.callbacks"`
Expected: no output, exit code 0.

- [ ] **Step 9: Commit**

```bash
cd /home/rtrk/konstantin/diplomski/carla_rl_project
git add agent/callbacks.py
git commit -m "Track mean smoothness reward in EpisodeLoggerCallback"
```

---

### Task 5: Add `w_smooth` to `configs/config.yaml`

**Files:**
- Modify: `configs/config.yaml:33-41`

**Interfaces:**
- Consumes: nothing new — `RewardConfig.w_smooth` already defaults to `0.5` (Task 1), this just makes the value explicit and documented in the single source of truth for hyperparameters, matching every other reward weight.

- [ ] **Step 1: Add the field**

In `configs/config.yaml`, change:

```yaml
# ── Reward function ───────────────────────────────────────────
reward:
  w_center:         1.0         # weight: centering reward
  w_speed:          1.5         # weight: speed reward
  w_heading:        0.5         # weight: heading alignment reward
  target_speed_kmh: 30.0        # desired cruising speed
  sigma_speed:      10.0        # Gaussian width for speed reward
  terminal_penalty: -10.0       # penalty for collision / off-road
  step_penalty:     -0.1        # small penalty per step
```

to:

```yaml
# ── Reward function ───────────────────────────────────────────
reward:
  w_center:         1.0         # weight: centering reward
  w_speed:          1.5         # weight: speed reward
  w_heading:        0.5         # weight: heading alignment reward
  w_smooth:         0.5         # weight: action smoothness penalty
                                 #   (set to 0.0 to disable)
  target_speed_kmh: 30.0        # desired cruising speed
  sigma_speed:      10.0        # Gaussian width for speed reward
  terminal_penalty: -10.0       # penalty for collision / off-road
  step_penalty:     -0.1        # small penalty per step
```

- [ ] **Step 2: Confirm `agent/train.py` already reads this correctly**

`agent/train.py`'s `make_env()` already builds `RewardConfig` from `reward_cfg["w_center"]` etc. (existing code, unmodified by this plan) — but it currently does NOT pass `w_smooth` through, since that dict-building code predates this field. Check the current code:

Run: `grep -n "RewardConfig(" agent/train.py`

You should see something like:
```python
    rc = RewardConfig(
        w_center         = reward_cfg["w_center"],
        w_speed          = reward_cfg["w_speed"],
        w_heading        = reward_cfg["w_heading"],
        target_speed_kmh = reward_cfg["target_speed_kmh"],
        sigma_speed      = reward_cfg["sigma_speed"],
        terminal_penalty = reward_cfg["terminal_penalty"],
        step_penalty     = reward_cfg["step_penalty"],
    )
```

Change it to:
```python
    rc = RewardConfig(
        w_center         = reward_cfg["w_center"],
        w_speed          = reward_cfg["w_speed"],
        w_heading        = reward_cfg["w_heading"],
        w_smooth         = reward_cfg["w_smooth"],
        target_speed_kmh = reward_cfg["target_speed_kmh"],
        sigma_speed      = reward_cfg["sigma_speed"],
        terminal_penalty = reward_cfg["terminal_penalty"],
        step_penalty     = reward_cfg["step_penalty"],
    )
```

Without this, `configs/config.yaml`'s `w_smooth` value would be silently ignored and `RewardConfig`'s dataclass default (`0.5`) would always be used instead — defeating the point of making it configurable.

- [ ] **Step 3: Sanity-check the script still imports and parses config cleanly**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python -c "
import sys; sys.path.insert(0, '.')
import yaml
from carla_env.reward import RewardConfig
cfg = yaml.safe_load(open('configs/config.yaml'))
rc = RewardConfig(**{k: v for k, v in cfg['reward'].items() if k in RewardConfig.__dataclass_fields__})
print(rc)
"`
Expected: prints a `RewardConfig(...)` instance with `w_smooth=0.5` visible, no errors.

- [ ] **Step 4: Commit**

```bash
cd /home/rtrk/konstantin/diplomski/carla_rl_project
git add configs/config.yaml agent/train.py
git commit -m "Add w_smooth to config.yaml and wire it through agent/train.py"
```

---

### Task 6: Live verification against a running CARLA server

**Files:** none (verification only — no source changes expected unless a bug surfaces, in which case fix it in the relevant task's files and re-run)

**Interfaces:** Exercises the full `carla_env/env.py` `step()`/`reset()` path with the new smoothness reward wired in, end to end.

- [ ] **Step 1: Confirm CARLA is reachable**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/verify_carla.py`
Expected: connects successfully.

- [ ] **Step 2: Check whether another training run is currently using the CARLA server**

Run: `ps aux | grep "agent/train.py" | grep -v grep`

If a process is running, note its `--algo` and check `configs/config.yaml`'s `env.spawn_index` value at that time — your verification must use a **different** spawn index (e.g. `158`, confirmed safe and clear of any junction for 450m in earlier work on this project) to avoid contending with the live run's `train_env`/`eval_env` (which use `spawn_index` and `spawn_index + 1` respectively).

- [ ] **Step 3: Run a short verification script — do NOT call `.close()`**

```bash
/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python -c "
import sys
sys.path.insert(0, '.')
import logging
logging.basicConfig(level=logging.WARNING)
import numpy as np

from carla_env.env import CarlaLaneKeepingEnv

# Use a spawn point not in use by any other active run (see Step 2).
env = CarlaLaneKeepingEnv(map_name='Town04', max_steps=50, spawn_index=158, verbose=False)
obs, info = env.reset()

actions = [
    np.array([1.0,  1.0], dtype=np.float32),   # large jump from zero
    np.array([1.0,  1.0], dtype=np.float32),   # repeat — should be smooth now
    np.array([-1.0, -1.0], dtype=np.float32),  # large jump again
    np.array([0.0,  0.0], dtype=np.float32),   # another large jump
]
for i, action in enumerate(actions):
    obs, reward, terminated, truncated, info = env.step(action)
    print(f'step {i}: action={action}  reward_smoothness={info[\"reward_smoothness\"]:.4f}  reward_total={info[\"reward_total\"]:.4f}')

# Deliberately not calling env.close() — see this plan's Global Constraints
# and docs/superpowers/specs/2026-06-25-action-smoothness-reward-design.md
# for why: close() disables synchronous mode for the entire shared CARLA
# world, which previously broke a separate live training run.
"
```

Expected output: 4 lines. Step 0 (`[1,1]` from a zero starting `_previous_raw_action`) should show a low `reward_smoothness` (large delta: `|1-0|+|1-0|=2.0` → `1.0 - 2.0/4.0 = 0.5`). Step 1 (repeating `[1,1]`, zero delta) should show `reward_smoothness=1.0000` (perfectly smooth — no change). Step 2 (`[1,1]` → `[-1,-1]`, delta magnitude `4.0`) should show `reward_smoothness=0.0000` (the maximum-jitter case). Step 3 (`[-1,-1]` → `[0,0]`, delta magnitude `2.0`) should show `reward_smoothness=0.5000` again.

- [ ] **Step 4: Confirm the live training run (if any) is unaffected**

If a training process was running in Step 2, check it's still alive and its episode log is still being written:

```bash
ps aux | grep "agent/train.py" | grep -v grep
tail -3 results/logs/<algo>/<run_name>/episode_log.csv
```

Expected: process still running, log file timestamp recent (within the last minute or two).

- [ ] **Step 5: Record results**

No commit needed for this task (no source files changed) unless a bug was found and fixed — in that case, the fix belongs in whichever earlier task's files it touches, and should be committed there with a clear message, then re-verify.

---

## Plan Self-Review Notes

- **Spec coverage:** Pure function + toggle (Task 1, 5), `compute_reward()` wiring (Task 2), `env.py` tracking (Task 3), TensorBoard/CSV trackability — not explicitly required by the spec but directly serves the user's stated "automatize and modularize so we can better track the project" goal, added as Task 4 following the file's existing established pattern exactly. Config field (Task 5) includes the easily-missed `agent/train.py`'s `RewardConfig(...)` call site, which the original spec did not explicitly call out but which is required for the config value to actually take effect — caught during this plan's file reading. Live verification (Task 6) explicitly carries forward the lesson from the earlier sync-mode incident (documented in Global Constraints).
- **Placeholder scan:** no TBD/TODO; every step has runnable code or an exact command with expected output.
- **Type consistency:** `compute_smoothness_reward(action_delta: np.ndarray) -> float`, `compute_reward(obs_data, is_terminal, action_delta, cfg=None)`, `RewardInfo.r_smoothness`, `info["reward_smoothness"]` — all consistent in name and position across every task that defines, calls, or consumes them.
