# Model Evaluation Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `agent/evaluate.py`, a standalone script that loads a saved
checkpoint for any supported algorithm (PPO/SAC/DDPG/TD3), runs it
deterministically for N episodes, and reports reward/lateral-distance/
success-rate/termination-reason metrics to console and CSV.

**Architecture:** Small, independently-callable functions
(`build_env`, `load_model`, `run_episode`, `run_evaluation`,
`compute_summary`, `write_csv`, `print_summary`) wired together by a thin
`main()`. A small DRY refactor extracts the YAML→`RewardConfig` mapping
(currently duplicated inline in `train.py`) into `RewardConfig.from_dict()`,
used by both `train.py` and the new script.

**Tech Stack:** Python 3.7, Stable-Baselines3 2.0.0, Gymnasium 0.28.1,
PyYAML, the project's existing `carla_env`/`agent` packages.

## Global Constraints

- Python 3.7-compatible syntax only — no `list[int]`, no walrus operator.
- All CARLA imports stay inside functions, not at module top level —
  `carla_env/env.py` already follows this; `agent/evaluate.py` must too,
  so its pure functions (`compute_summary`, `load_model`'s error path)
  stay testable without CARLA installed.
- Use `logger` (not `print`) inside `carla_env/` and `agent/` modules —
  `agent/evaluate.py` lives in `agent/`, so status/progress messages use
  `logging`, matching `agent/train.py`'s existing setup. `print_summary()`
  is the one designed exception: its whole job is producing the
  human-readable report, so it uses `print` for that report's body only.
- Reward weights belong only in `configs/config.yaml`, never hardcoded —
  already satisfied by reusing `RewardConfig.from_dict(cfg["reward"])`.
- `agent/evaluate.py` must never be run while a `train.py` process is
  connected to the same CARLA server (see design's Operational
  Constraint) — this is a documented, manual precaution, not enforced in
  code.
- An episode is "successful" iff its `termination_reason == "timeout"`
  (matches `check_termination()`'s existing `terminated`/`truncated`
  split in `carla_env/reward.py`).

---

### Task 1: Extract `RewardConfig.from_dict()` and use it in `train.py`

**Files:**
- Modify: `carla_env/reward.py` (add classmethod after the `RewardConfig`
  dataclass body, before the blank line that precedes `# ── Reward
  component data class`)
- Modify: `agent/train.py:108-117` (the inline `RewardConfig(...)` call
  inside `make_env()`)
- Modify: `scripts/test_reward.py` (add one test, wire into `main()`)

**Interfaces:**
- Produces: `RewardConfig.from_dict(reward_cfg: dict) -> RewardConfig` —
  used by Task 5's `build_env()` and by `train.py`'s `make_env()`.

- [ ] **Step 1: Write the failing test**

Add this test function to `scripts/test_reward.py`, placed after
`test_smoothness_toggle()` and before `test_full_reward()`:

```python
# ── Test: RewardConfig.from_dict() ─────────────────────────────────────────────

def test_reward_config_from_dict():
    sep("RewardConfig.from_dict()")
    reward_cfg = {
        "w_center":         1.0,
        "w_speed":          1.5,
        "w_heading":        0.5,
        "w_smooth":         0.5,
        "target_speed_kmh": 30.0,
        "sigma_speed":      10.0,
        "terminal_penalty": -10.0,
        "step_penalty":     -0.1,
    }
    rc = RewardConfig.from_dict(reward_cfg)
    assert rc.w_center == 1.0
    assert rc.w_speed == 1.5
    assert rc.w_heading == 0.5
    assert rc.w_smooth == 0.5
    assert rc.target_speed_kmh == 30.0
    assert rc.sigma_speed == 10.0
    assert rc.terminal_penalty == -10.0
    assert rc.step_penalty == -0.1
    print("  RewardConfig.from_dict() ->", rc)
    print("  ✓ PASSED")
```

Also add the call to `main()`, right after `test_smoothness_toggle()`:

```python
    test_smoothness_toggle()
    test_reward_config_from_dict()
    test_full_reward()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_reward.py`
Expected: `AttributeError: type object 'RewardConfig' has no attribute 'from_dict'`

- [ ] **Step 3: Write minimal implementation**

In `carla_env/reward.py`, add this classmethod as the last member of the
`RewardConfig` dataclass (after the `step_penalty` field, still inside the
`class RewardConfig:` body, before the blank lines and the `# ── Reward
component data class` section header):

```python
    @classmethod
    def from_dict(cls, reward_cfg: dict) -> "RewardConfig":
        """
        Build a RewardConfig from configs/config.yaml's `reward:` block.

        Single source of truth for the YAML-dict -> dataclass mapping —
        both agent/train.py and agent/evaluate.py call this so the
        mapping only has to be correct in one place.
        """
        return cls(
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

Note: `@classmethod` needs `from dataclasses import dataclass` already
imported (it is, at the top of `reward.py`) — no new imports required.

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_reward.py`
Expected: all tests print `✓ PASSED`, ending with `All reward tests passed.`

- [ ] **Step 5: Update `agent/train.py` to use it and verify offline tests still pass**

In `agent/train.py`, find this block inside `make_env()` (around line
108-117):

```python
    # Build RewardConfig from YAML values
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

Replace it with:

```python
    # Build RewardConfig from YAML values
    rc = RewardConfig.from_dict(reward_cfg)
```

Run the existing offline algorithm tests to confirm nothing broke (this
file's other logic is unchanged, but `RewardConfig` construction is on the
path `build_model()`'s test exercises indirectly through `make_env()`-like
config loading):

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_algorithms.py`
Expected: ends with `ALL TESTS PASSED`

- [ ] **Step 6: Commit**

```bash
git add carla_env/reward.py agent/train.py scripts/test_reward.py
git commit -m "Extract RewardConfig.from_dict() and use it in train.py's make_env()"
```

---

### Task 2: `EpisodeResult`, `EvaluationSummary`, and `compute_summary()`

**Files:**
- Create: `agent/evaluate.py`
- Create: `scripts/test_evaluate.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces:
  - `EpisodeResult` dataclass: `episode_num: int`, `reward: float`,
    `length: int`, `mean_lateral_distance: float`,
    `termination_reason: str`
  - `EvaluationSummary` dataclass: `n_episodes: int`, `mean_reward: float`,
    `std_reward: float`, `mean_lateral_distance: float`,
    `success_rate: float`, `mean_length: float`,
    `termination_counts: dict`
  - `compute_summary(results: List[EpisodeResult]) -> EvaluationSummary`
  - Used by Task 4's `write_csv()`/`print_summary()` and Task 5's `main()`.

- [ ] **Step 1: Write the failing test**

Create `scripts/test_evaluate.py`:

```python
"""
test_evaluate.py
-----------------
Offline unit tests for agent/evaluate.py (no CARLA required).

Run from project root:
    python scripts/test_evaluate.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.evaluate import EpisodeResult, EvaluationSummary, compute_summary


def sep(title=""):
    print(f"\n{'─'*60}")
    if title:
        print(f"  {title}")
        print(f"{'─'*60}")


# ── Test 1: compute_summary() on a known set of episodes ──────────────────────

def test_compute_summary_basic():
    sep("compute_summary() — mixed timeout/collision episodes")
    results = [
        EpisodeResult(episode_num=1, reward=100.0, length=1000,
                      mean_lateral_distance=0.10, termination_reason="timeout"),
        EpisodeResult(episode_num=2, reward=80.0, length=1000,
                      mean_lateral_distance=0.20, termination_reason="timeout"),
        EpisodeResult(episode_num=3, reward=20.0, length=400,
                      mean_lateral_distance=0.50, termination_reason="collision"),
        EpisodeResult(episode_num=4, reward=90.0, length=1000,
                      mean_lateral_distance=0.15, termination_reason="timeout"),
    ]
    summary = compute_summary(results)

    assert summary.n_episodes == 4
    assert summary.mean_reward == 72.5            # (100+80+20+90)/4
    expected_std = 31.12474899             # population std of [100,80,20,90]
    assert abs(summary.std_reward - expected_std) < 1e-6
    expected_mean_lat = (0.10 + 0.20 + 0.50 + 0.15) / 4
    assert abs(summary.mean_lateral_distance - expected_mean_lat) < 1e-9
    assert summary.success_rate == 0.75            # 3 of 4 are "timeout"
    assert summary.mean_length == 850.0            # (1000+1000+400+1000)/4
    assert summary.termination_counts == {"timeout": 3, "collision": 1}
    print("  summary:", summary)
    print("  ✓ PASSED")


# ── Test 2: compute_summary() with no collisions at all ────────────────────────

def test_compute_summary_all_success():
    sep("compute_summary() — all episodes successful")
    results = [
        EpisodeResult(episode_num=1, reward=50.0, length=1000,
                      mean_lateral_distance=0.05, termination_reason="timeout"),
        EpisodeResult(episode_num=2, reward=50.0, length=1000,
                      mean_lateral_distance=0.05, termination_reason="timeout"),
    ]
    summary = compute_summary(results)

    assert summary.success_rate == 1.0
    assert summary.std_reward == 0.0
    assert summary.termination_counts == {"timeout": 2}
    print("  summary:", summary)
    print("  ✓ PASSED")


# ── Test 3: compute_summary() distinguishes all three failure reasons ─────────

def test_compute_summary_all_failure_reasons():
    sep("compute_summary() — collision, off_road, wrong_heading all counted")
    results = [
        EpisodeResult(episode_num=1, reward=10.0, length=100,
                      mean_lateral_distance=1.0, termination_reason="collision"),
        EpisodeResult(episode_num=2, reward=10.0, length=100,
                      mean_lateral_distance=1.0, termination_reason="off_road"),
        EpisodeResult(episode_num=3, reward=10.0, length=100,
                      mean_lateral_distance=1.0, termination_reason="wrong_heading"),
    ]
    summary = compute_summary(results)

    assert summary.success_rate == 0.0
    assert summary.termination_counts == {
        "collision": 1, "off_road": 1, "wrong_heading": 1,
    }
    print("  summary:", summary)
    print("  ✓ PASSED")


def main():
    print("=" * 60)
    print("  EVALUATE.PY OFFLINE TESTS")
    print("  (No CARLA connection required)")
    print("=" * 60)

    test_compute_summary_basic()
    test_compute_summary_all_success()
    test_compute_summary_all_failure_reasons()

    print(f"\n{'='*60}")
    print("  All evaluate.py tests passed.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_evaluate.py`
Expected: `ModuleNotFoundError: No module named 'agent.evaluate'` (or
`ImportError: cannot import name 'EpisodeResult'` if the file doesn't
exist yet)

- [ ] **Step 3: Write minimal implementation**

Create `agent/evaluate.py` with this content (this task only needs the
dataclasses and `compute_summary()` — later tasks append more to this same
file):

```python
"""
evaluate.py
-----------
Standalone evaluation script — load a saved checkpoint for any supported
algorithm (PPO/SAC/DDPG/TD3) and measure how well it drives.

Unlike agent/train.py's built-in EvalCallback (which only runs against
whatever model is currently in memory, every eval_freq steps, and stops
existing once training ends), this script can be pointed at any saved
.zip checkpoint, any time, independent of training.

IMPORTANT — do not run this while training is live:
    This script connects its own CARLA client to the same server
    agent/train.py uses, and calls env.close() on exit (which disables
    CARLA's synchronous mode). Running this alongside a live train.py
    process risks the same world-disruption crash documented in
    carla_env/env.py — both clients ticking/closing the same shared
    world at once. Only run this when no train.py process is connected
    to the CARLA server.

How to run (once training is stopped):
    python agent/evaluate.py --algo sac --checkpoint results/checkpoints/sac/.../best_model/best_model.zip --episodes 20

What it produces:
    Console summary: mean/std reward, mean lateral distance, success
    rate, mean episode length, termination-reason counts.
    CSV: results/logs/<algo>/eval_runs/eval_<checkpoint_stem>_<timestamp>.csv
"""

import os
import sys
import logging
from dataclasses import dataclass
from typing import List

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ── Result types ────────────────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    """One evaluation episode's outcome."""
    episode_num: int
    reward: float
    length: int
    mean_lateral_distance: float   # mean(|lateral_distance_m|) over the episode
    termination_reason: str        # "timeout" | "collision" | "off_road" | "wrong_heading"


@dataclass
class EvaluationSummary:
    """Aggregate statistics across a full evaluation run."""
    n_episodes: int
    mean_reward: float
    std_reward: float
    mean_lateral_distance: float
    success_rate: float            # fraction with termination_reason == "timeout"
    mean_length: float
    termination_counts: dict       # e.g. {"timeout": 18, "collision": 2}


# ── Pure aggregation function (no CARLA, no I/O — easy to test offline) ────────

def compute_summary(results: List[EpisodeResult]) -> EvaluationSummary:
    """
    Reduce a list of EpisodeResult into one EvaluationSummary.

    Pure function: same input always produces the same output, no side
    effects. This is what makes it testable without CARLA — see
    scripts/test_evaluate.py.
    """
    n = len(results)
    rewards = [r.reward for r in results]
    lateral_distances = [r.mean_lateral_distance for r in results]
    lengths = [r.length for r in results]

    mean_reward = sum(rewards) / n
    variance = sum((r - mean_reward) ** 2 for r in rewards) / n
    std_reward = variance ** 0.5

    termination_counts = {}
    for r in results:
        termination_counts[r.termination_reason] = (
            termination_counts.get(r.termination_reason, 0) + 1
        )

    success_count = termination_counts.get("timeout", 0)

    return EvaluationSummary(
        n_episodes=n,
        mean_reward=mean_reward,
        std_reward=std_reward,
        mean_lateral_distance=sum(lateral_distances) / n,
        success_rate=success_count / n,
        mean_length=sum(lengths) / n,
        termination_counts=termination_counts,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_evaluate.py`
Expected: all three tests print `✓ PASSED`, ending with `All evaluate.py
tests passed.`

- [ ] **Step 5: Commit**

```bash
git add agent/evaluate.py scripts/test_evaluate.py
git commit -m "Add EpisodeResult, EvaluationSummary, and compute_summary() to evaluate.py"
```

---

### Task 3: `load_model()`

**Files:**
- Modify: `agent/evaluate.py` (append below `compute_summary()`)
- Modify: `scripts/test_evaluate.py` (append new tests)

**Interfaces:**
- Consumes: `agent.algorithms.ALGORITHMS` (dict of `{"ppo": PPO, "sac":
  SAC, "ddpg": DDPG, "td3": TD3}`, already defined in
  `agent/algorithms.py`).
- Produces: `load_model(algo_name: str, checkpoint_path: str)` — returns
  an SB3 model instance; used by Task 5's `main()`.

- [ ] **Step 1: Write the failing test**

Add to `scripts/test_evaluate.py`, after the existing imports line, add:

```python
from agent.evaluate import (
    EpisodeResult, EvaluationSummary, compute_summary, load_model,
)
```

(replacing the existing single-line import of those first three names).

Then add this test function, placed after `test_compute_summary_all_failure_reasons()`:

```python
# ── Test 4: load_model() rejects an unknown algorithm before touching CARLA ───

def test_load_model_unknown_algo():
    sep("load_model() — unknown algorithm raises before any CARLA/file access")
    try:
        load_model("dqn", "/nonexistent/path.zip")
        assert False, "Expected ValueError for unknown algo 'dqn'"
    except ValueError as e:
        assert "dqn" in str(e)
        print("  Correctly raised:", e)
        print("  ✓ PASSED")
```

Add the call to `main()`:

```python
    test_compute_summary_all_failure_reasons()
    test_load_model_unknown_algo()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_evaluate.py`
Expected: `ImportError: cannot import name 'load_model' from 'agent.evaluate'`

- [ ] **Step 3: Write minimal implementation**

Append to `agent/evaluate.py`, after `compute_summary()`'s closing line:

```python
# ── Model loading ────────────────────────────────────────────────────────────────

def load_model(algo_name: str, checkpoint_path: str):
    """
    Load a saved SB3 checkpoint for the given algorithm.

    No `env` argument is passed to `.load()` — this script only calls
    `model.predict()`, which needs the policy's weights, not a live
    training environment. Validating algo_name happens before this
    function touches the filesystem or CARLA, so a typo in --algo fails
    immediately with a clear message instead of partway through setup.

    Parameters
    ----------
    algo_name       : one of agent.algorithms.ALGORITHMS keys
    checkpoint_path : path to a saved .zip file

    Returns
    -------
    An SB3 BaseAlgorithm subclass instance (PPO, SAC, DDPG, or TD3).
    """
    from agent.algorithms import ALGORITHMS

    if algo_name not in ALGORITHMS:
        raise ValueError(
            "Unknown algorithm '{}'. Available: {}".format(
                algo_name, sorted(ALGORITHMS.keys())
            )
        )

    algo_cls = ALGORITHMS[algo_name]
    return algo_cls.load(checkpoint_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_evaluate.py`
Expected: all four tests print `✓ PASSED`

- [ ] **Step 5: Commit**

```bash
git add agent/evaluate.py scripts/test_evaluate.py
git commit -m "Add load_model() to evaluate.py"
```

---

### Task 4: `write_csv()` and `print_summary()`

**Files:**
- Modify: `agent/evaluate.py` (append below `load_model()`)
- Modify: `scripts/test_evaluate.py` (append new tests)

**Interfaces:**
- Consumes: `EpisodeResult`, `EvaluationSummary` (from Task 2).
- Produces: `write_csv(results: List[EpisodeResult], path: str) -> None`
  and `print_summary(summary: EvaluationSummary) -> None`; both used by
  Task 5's `main()`.

- [ ] **Step 1: Write the failing test**

Update the import line in `scripts/test_evaluate.py` to:

```python
from agent.evaluate import (
    EpisodeResult, EvaluationSummary, compute_summary, load_model,
    write_csv, print_summary,
)
```

Add `import tempfile` and `import csv` near the top of `scripts/test_evaluate.py`
(alongside the existing `import sys` / `import os`).

Add these two test functions after `test_load_model_unknown_algo()`:

```python
# ── Test 5: write_csv() produces one row per episode ───────────────────────────

def test_write_csv():
    sep("write_csv() — one row per episode, correct columns")
    results = [
        EpisodeResult(episode_num=1, reward=100.0, length=1000,
                      mean_lateral_distance=0.10, termination_reason="timeout"),
        EpisodeResult(episode_num=2, reward=20.0, length=400,
                      mean_lateral_distance=0.50, termination_reason="collision"),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "eval_test.csv")
        write_csv(results, csv_path)

        assert os.path.exists(csv_path)
        with open(csv_path, "r") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 2
        assert rows[0]["episode_num"] == "1"
        assert rows[0]["reward"] == "100.0"
        assert rows[0]["length"] == "1000"
        assert rows[0]["mean_lateral_distance"] == "0.1"
        assert rows[0]["termination_reason"] == "timeout"
        assert rows[1]["termination_reason"] == "collision"
    print("  CSV written and verified at:", csv_path)
    print("  ✓ PASSED")


# ── Test 6: print_summary() runs without error and reports key numbers ────────

def test_print_summary():
    sep("print_summary() — produces readable console output")
    summary = EvaluationSummary(
        n_episodes=4, mean_reward=72.5, std_reward=31.79,
        mean_lateral_distance=0.2375, success_rate=0.75, mean_length=850.0,
        termination_counts={"timeout": 3, "collision": 1},
    )
    print_summary(summary)   # smoke test: must not raise
    print("  ✓ PASSED")
```

Add both calls to `main()`:

```python
    test_load_model_unknown_algo()
    test_write_csv()
    test_print_summary()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_evaluate.py`
Expected: `ImportError: cannot import name 'write_csv' from 'agent.evaluate'`

- [ ] **Step 3: Write minimal implementation**

Append to `agent/evaluate.py`, after `load_model()`:

```python
# ── Output: CSV and console report ────────────────────────────────────────────────

def write_csv(results: List[EpisodeResult], path: str) -> None:
    """Write one row per EpisodeResult to a CSV at the given path."""
    import csv

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode_num", "reward", "length",
            "mean_lateral_distance", "termination_reason",
        ])
        for r in results:
            writer.writerow([
                r.episode_num, r.reward, r.length,
                r.mean_lateral_distance, r.termination_reason,
            ])
    logger.info(f"Wrote {len(results)} episode rows to: {path}")


def print_summary(summary: EvaluationSummary) -> None:
    """Print a human-readable evaluation report to the console."""
    print("\n" + "=" * 55)
    print("  EVALUATION SUMMARY")
    print("=" * 55)
    print(f"  Episodes:              {summary.n_episodes}")
    print(f"  Mean reward:           {summary.mean_reward:.2f} (+/- {summary.std_reward:.2f})")
    print(f"  Mean lateral distance: {summary.mean_lateral_distance:.4f} m")
    print(f"  Success rate:          {summary.success_rate * 100:.1f}%")
    print(f"  Mean episode length:   {summary.mean_length:.1f} steps")
    print("  Termination reasons:")
    for reason, count in sorted(summary.termination_counts.items()):
        print(f"    {reason:15s} {count}")
    print("=" * 55 + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_evaluate.py`
Expected: all six tests print `✓ PASSED`

- [ ] **Step 5: Commit**

```bash
git add agent/evaluate.py scripts/test_evaluate.py
git commit -m "Add write_csv() and print_summary() to evaluate.py"
```

---

### Task 5: `build_env()`, `run_episode()`, `run_evaluation()`, and CLI `main()`

**Files:**
- Modify: `agent/evaluate.py` (append below `print_summary()`)

**Interfaces:**
- Consumes: `CarlaLaneKeepingEnv` (from `carla_env/env.py`, constructor
  signature `host, port, map_name, max_steps, reward_config,
  action_smooth, seed, spawn_index, spawn_index_offset, verbose` — see
  `agent/train.py`'s `make_env()` for the exact call this mirrors);
  `RewardConfig.from_dict()` (Task 1); `load_model()` (Task 3);
  `compute_summary()`, `write_csv()`, `print_summary()` (Tasks 2 and 4).
- Produces: `build_env(cfg: dict) -> CarlaLaneKeepingEnv`,
  `run_episode(env, model, episode_num: int) -> EpisodeResult`,
  `run_evaluation(env, model, n_episodes: int) -> List[EpisodeResult]`,
  and a CLI entry point (`if __name__ == "__main__":`).

This task requires a live CARLA server, so it is verified manually rather
than with an automated test (consistent with how `carla_env/env.py`'s
integration behavior is verified — see `scripts/test_env.py`'s manual-run
convention).

- [ ] **Step 1: Implement `build_env()`, `run_episode()`, `run_evaluation()`, and `main()`**

Append to `agent/evaluate.py`, after `print_summary()`:

```python
# ── Environment construction ──────────────────────────────────────────────────

def build_env(cfg: dict):
    """
    Build a single, unwrapped CarlaLaneKeepingEnv from config.

    Unlike agent/train.py's make_env(), this does not wrap with
    Monitor/DummyVecEnv (those exist for SB3's training internals, not
    needed for a plain evaluation loop) and uses spawn_index_offset=0
    (a single env here, so there's no train/eval spawn contention to
    avoid).
    """
    from carla_env.env import CarlaLaneKeepingEnv
    from carla_env.reward import RewardConfig

    env_cfg = cfg["env"]
    rc = RewardConfig.from_dict(cfg["reward"])

    return CarlaLaneKeepingEnv(
        host          = env_cfg["host"],
        port          = env_cfg["port"],
        map_name      = env_cfg["map_name"],
        max_steps     = env_cfg["max_steps"],
        reward_config = rc,
        action_smooth = env_cfg["action_smooth"],
        seed          = env_cfg["seed"],
        spawn_index   = env_cfg.get("spawn_index"),
        spawn_index_offset = 0,
        verbose       = False,
    )


# ── Episode runner ────────────────────────────────────────────────────────────────

def run_episode(env, model, episode_num: int) -> EpisodeResult:
    """
    Run exactly one episode with the model's deterministic (mean) action,
    until the environment reports terminated or truncated.
    """
    obs, _info = env.reset()
    total_reward = 0.0
    lateral_distances = []
    step_count = 0
    termination_reason = ""

    terminated = False
    truncated = False
    while not terminated and not truncated:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        lateral_distances.append(abs(info["lateral_distance"]))
        step_count += 1
        termination_reason = info["termination_reason"]

    mean_lateral = sum(lateral_distances) / len(lateral_distances)

    return EpisodeResult(
        episode_num=episode_num,
        reward=total_reward,
        length=step_count,
        mean_lateral_distance=mean_lateral,
        termination_reason=termination_reason,
    )


def run_evaluation(env, model, n_episodes: int) -> List[EpisodeResult]:
    """
    Run n_episodes evaluation episodes and return the list of results.

    Extension point for future work: a multi-checkpoint comparison script
    can call this once per checkpoint and combine the results, without
    touching run_episode() or compute_summary().
    """
    results = []
    for i in range(1, n_episodes + 1):
        result = run_episode(env, model, episode_num=i)
        logger.info(
            f"Episode {i}/{n_episodes}: reward={result.reward:.2f} "
            f"length={result.length} "
            f"mean_lateral_distance={result.mean_lateral_distance:.4f} "
            f"reason={result.termination_reason}"
        )
        results.append(result)
    return results


# ── CLI entry point ────────────────────────────────────────────────────────────────

def main():
    import argparse
    import time
    import yaml

    parser = argparse.ArgumentParser(
        description="Evaluate a saved RL checkpoint by running it for N "
                     "deterministic episodes. Only run this when no "
                     "train.py process is connected to the same CARLA "
                     "server (see this file's module docstring)."
    )
    parser.add_argument("--algo", required=True, choices=["ppo", "sac", "ddpg", "td3"],
                         help="Algorithm the checkpoint was trained with.")
    parser.add_argument("--checkpoint", required=True,
                         help="Path to a saved .zip checkpoint.")
    parser.add_argument("--episodes", type=int, default=20,
                         help="Number of evaluation episodes to run (default: 20).")
    parser.add_argument("--config", default="configs/config.yaml",
                         help="Path to config.yaml (default: configs/config.yaml).")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    logger.warning(
        "evaluate.py is about to connect to CARLA. Do not run this while "
        "a train.py process is connected to the same server (see this "
        "file's module docstring)."
    )

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model = load_model(args.algo, args.checkpoint)
    logger.info(f"Loaded {args.algo.upper()} model from: {args.checkpoint}")

    env = build_env(cfg)
    try:
        results = run_evaluation(env, model, args.episodes)
    finally:
        env.close()

    summary = compute_summary(results)
    print_summary(summary)

    checkpoint_stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(
        cfg["paths"]["log_dir"], args.algo, "eval_runs",
        f"eval_{checkpoint_stem}_{timestamp}.csv",
    )
    write_csv(results, csv_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify offline tests still pass (this task adds no new offline-testable logic, but must not break Tasks 1-4)**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_evaluate.py`
Expected: all six tests still print `✓ PASSED`

- [ ] **Step 3: Manual verification with a live CARLA server**

This step requires CARLA running and **no `train.py` process connected**.
Confirm no training is live first:

Run: `ps aux | grep agent/train.py | grep -v grep`
Expected: no output (if there is output, STOP — do not proceed until that
training run is stopped or finished)

Then, with CARLA running standalone, run the script against a real saved
checkpoint, e.g.:

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python agent/evaluate.py --algo sac --checkpoint results/checkpoints/sac/sac_lane_keeping_20260626_111850/final_model.zip --episodes 5`

Expected: console output ending with an `EVALUATION SUMMARY` block showing
5 episodes, a success rate, and a termination-reason table; a CSV file
created under `results/logs/sac/eval_runs/`; the CARLA world left in a
clean state (synchronous mode disabled, no leftover vehicle) afterward.

- [ ] **Step 4: Commit**

```bash
git add agent/evaluate.py
git commit -m "Add build_env(), run_episode(), run_evaluation(), and CLI to evaluate.py"
```

---

## Self-Review Notes

- **Spec coverage:** Operational constraint (Task 5 main() warning +
  Step 3's live-process check) — covered. `RewardConfig.from_dict()`
  refactor — Task 1. All seven functions from the design's function list
  — covered across Tasks 2-5. CSV path/format and console summary format
  — Task 4. Success-rate definition — `compute_summary()` in Task 2 keys
  off `"timeout"` exactly as specified. Error handling for unknown
  `--algo` and missing `--checkpoint` — Task 3 (`load_model`) and Task 5
  (`main()`'s `FileNotFoundError` check), both before any CARLA
  connection. Non-Goals (plots, multi-checkpoint comparison) — explicitly
  not built, consistent with the spec.
- **Placeholder scan:** none found — every step has complete code.
- **Type consistency:** `EpisodeResult`/`EvaluationSummary` field names
  and types are defined once in Task 2 and used identically in Tasks 4
  and 5 (`reward`, `length`, `mean_lateral_distance`, `termination_reason`,
  `termination_counts`, etc.) — no renames across tasks.
