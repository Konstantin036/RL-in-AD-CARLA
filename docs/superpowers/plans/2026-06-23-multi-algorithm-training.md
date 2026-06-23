# Multi-Algorithm Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `agent/train.py` train PPO, SAC, DDPG, or TD3 against `CarlaLaneKeepingEnv` by selecting an algorithm via config/CLI, with each algorithm's checkpoints and logs isolated in their own results subdirectory.

**Architecture:** A new `agent/algorithms.py` module is the single source of truth for which algorithms exist and how to build them (`ALGORITHMS` registry + `build_model()`). `agent/train.py` is generalized to call this registry instead of hardcoding `PPO(...)`, and to scope its output paths by algorithm name. `configs/config.yaml` gains a top-level `algo:` field and one hyperparameter block per algorithm. `carla_env/` and `agent/callbacks.py` are untouched — they were already algorithm-agnostic.

**Tech Stack:** Python 3.7.16, Stable-Baselines3 2.0.0, Gymnasium 0.28.1, PyYAML, CARLA 0.9.15 (server running at localhost:2000 during this implementation).

## Global Constraints

- Python 3.7 compatible syntax only — no `list[int]`-style built-in generics; use `typing.Optional`, `typing.List`, etc.
- All CARLA imports stay inside functions, never at module top level (per existing `carla_env/env.py` pattern) — `agent/algorithms.py` does not import CARLA at all, since it only talks to SB3.
- Use `logger` (not `print`) inside `agent/` modules; use `print` inside `scripts/` (existing project convention, see `CLAUDE.md`).
- Never hardcode reward weights or hyperparameters in code — they live in `configs/config.yaml`.
- DQN is explicitly out of scope for this plan (continuous action space; see spec's "Future extension: DQN" section) — do not add a `"dqn"` entry to `ALGORITHMS`.
- Installed versions to target exactly: `stable-baselines3==2.0.0`, `gymnasium==0.28.1`, run via `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python`.
- CARLA server is running and reachable at `localhost:2000` for live smoke tests in Task 6.

---

### Task 1: Algorithm registry skeleton

**Files:**
- Create: `agent/algorithms.py`
- Test: `scripts/test_algorithms.py`

**Interfaces:**
- Produces: `ALGORITHMS` (dict, str → SB3 algorithm class), `get_run_prefix(algo_name: str) -> str`. Both are imported by later tasks and by `agent/train.py` (Task 5).

- [ ] **Step 1: Write the failing test**

Create `scripts/test_algorithms.py`:

```python
"""
test_algorithms.py
-------------------
Offline unit tests for agent/algorithms.py (no CARLA required).

Run from anywhere:
    python scripts/test_algorithms.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.algorithms import ALGORITHMS, get_run_prefix


def separator(title=""):
    print(f"\n{'─'*55}")
    if title:
        print(f"  {title}")
        print(f"{'─'*55}")


def test_registry_contents():
    separator("1. Registry contains exactly the supported algorithms")
    expected = {"ppo", "sac", "ddpg", "td3"}
    assert set(ALGORITHMS.keys()) == expected, (
        f"Expected {expected}, got {set(ALGORITHMS.keys())}"
    )
    assert "dqn" not in ALGORITHMS, "DQN must not be registered (see spec)"
    print(f"  Registered algorithms: {sorted(ALGORITHMS.keys())}")
    print("  ✓ PASSED")


def test_run_prefix():
    separator("2. get_run_prefix() naming")
    assert get_run_prefix("ppo") == "ppo_lane_keeping"
    assert get_run_prefix("sac") == "sac_lane_keeping"
    assert get_run_prefix("ddpg") == "ddpg_lane_keeping"
    assert get_run_prefix("td3") == "td3_lane_keeping"
    print("  ppo  ->", get_run_prefix("ppo"))
    print("  sac  ->", get_run_prefix("sac"))
    print("  ddpg ->", get_run_prefix("ddpg"))
    print("  td3  ->", get_run_prefix("td3"))
    print("  ✓ PASSED")


def test_run_prefix_unknown_algo():
    separator("3. get_run_prefix() rejects unknown algorithms")
    try:
        get_run_prefix("dqn")
        raise AssertionError("Expected ValueError for unknown algorithm")
    except ValueError as e:
        print(f"  Raised ValueError as expected: {e}")
    print("  ✓ PASSED")


if __name__ == "__main__":
    test_registry_contents()
    test_run_prefix()
    test_run_prefix_unknown_algo()
    print(f"\n{'='*55}")
    print("  ALL TESTS PASSED")
    print(f"{'='*55}\n")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_algorithms.py`
Expected: `ModuleNotFoundError: No module named 'agent.algorithms'`

- [ ] **Step 3: Write minimal implementation**

Create `agent/algorithms.py`:

```python
"""
algorithms.py
-------------
Algorithm registry — single source of truth for which RL algorithms
this project supports and how to construct them.

Why this exists:
    agent/train.py used to hardcode PPO directly. To compare multiple
    algorithms (PPO, SAC, DDPG, TD3) on the same CARLA environment
    without duplicating train.py per algorithm, all algorithm-specific
    construction logic lives here behind one function: build_model().

Adding a new algorithm later:
    1. Import its SB3 class and add it to ALGORITHMS below.
    2. Add a branch in _build_kwargs() if it needs hyperparameters not
       already handled by the on-policy / off-policy / noise branches.
    3. Add its hyperparameter block to configs/config.yaml.
    No changes to train.py are needed.

Note on DQN:
    Deliberately not registered here. DQN requires a discrete action
    space; this project's action space is continuous (Box(2,)). See
    docs/superpowers/specs/2026-06-23-multi-algorithm-training-design.md
    for the planned path to add it via a separate discretized env.
"""

from stable_baselines3 import PPO, SAC, DDPG, TD3


# ── Registry ────────────────────────────────────────────────────────────────────

ALGORITHMS = {
    "ppo":  PPO,
    "sac":  SAC,
    "ddpg": DDPG,
    "td3":  TD3,
}


def get_run_prefix(algo_name: str) -> str:
    """
    Return the filename/run-name prefix for an algorithm, e.g.
    "ppo" -> "ppo_lane_keeping". Used for checkpoint name_prefix and
    run_name in agent/train.py.
    """
    if algo_name not in ALGORITHMS:
        raise ValueError(
            "Unknown algorithm '{}'. Available: {}".format(
                algo_name, sorted(ALGORITHMS.keys())
            )
        )
    return "{}_lane_keeping".format(algo_name)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_algorithms.py`
Expected: `ALL TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
cd /home/rtrk/konstantin/diplomski/carla_rl_project
git add agent/algorithms.py scripts/test_algorithms.py
git commit -m "Add algorithm registry skeleton (ALGORITHMS, get_run_prefix)"
```

---

### Task 2: `build_model()` — fresh model construction

**Files:**
- Modify: `agent/algorithms.py` (append `_build_kwargs`, `build_model`)
- Test: `scripts/test_algorithms.py` (append dummy env + construction tests)

**Interfaces:**
- Consumes: `ALGORITHMS` from Task 1.
- Produces: `build_model(algo_name: str, cfg: dict, env, tensorboard_log: str, resume_path: Optional[str] = None, seed: Optional[int] = None)` returning an SB3 `BaseAlgorithm` instance. `cfg` must contain `cfg[algo_name]` with the keys listed in Step 3 below. Used by Task 3 (resume), Task 4 (real config), and Task 5 (`train.py`).

- [ ] **Step 1: Write the failing test**

Append to `scripts/test_algorithms.py` (add these imports at the top alongside the existing ones, and add the new test functions before the `if __name__ == "__main__":` block):

```python
import numpy as np
import gymnasium as gym

from agent.algorithms import build_model
from carla_env.action import get_action_space
from carla_env.observation import get_observation_space
```

```python
class DummyContinuousEnv(gym.Env):
    """
    Minimal CARLA-free stand-in for CarlaLaneKeepingEnv, used only to
    test that build_model() wires SB3 hyperparameters correctly without
    needing a running CARLA server. Matches the real env's spaces
    exactly (carla_env.observation / carla_env.action are pure functions
    with no CARLA dependency, so reusing them here keeps the test
    spaces honest).
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = get_observation_space()
        self.action_space = get_action_space()
        self._step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        return self.observation_space.sample() * 0.0, {}

    def step(self, action):
        self._step_count += 1
        obs = self.observation_space.sample() * 0.0
        terminated = False
        truncated = self._step_count >= 16
        return obs, 0.0, terminated, truncated, {}


# Small hyperparameter sets — enough to exercise a few real gradient
# updates per algorithm without the test taking more than a few seconds.
_TEST_CFG = {
    "ppo": {
        "learning_rate": 3e-4, "n_steps": 8, "batch_size": 4,
        "n_epochs": 2, "gamma": 0.99, "gae_lambda": 0.95,
        "clip_range": 0.2, "ent_coef": 0.01, "vf_coef": 0.5,
        "max_grad_norm": 0.5, "verbose": 0,
    },
    "sac": {
        "learning_rate": 3e-4, "buffer_size": 200, "learning_starts": 4,
        "batch_size": 4, "tau": 0.005, "gamma": 0.99, "train_freq": 1,
        "gradient_steps": 1, "verbose": 0,
    },
    "ddpg": {
        "learning_rate": 1e-3, "buffer_size": 200, "learning_starts": 4,
        "batch_size": 4, "tau": 0.005, "gamma": 0.99, "train_freq": 1,
        "gradient_steps": 1, "action_noise_sigma": 0.1, "verbose": 0,
    },
    "td3": {
        "learning_rate": 1e-3, "buffer_size": 200, "learning_starts": 4,
        "batch_size": 4, "tau": 0.005, "gamma": 0.99, "train_freq": 1,
        "gradient_steps": 1, "action_noise_sigma": 0.1, "verbose": 0,
    },
}


def test_build_model_and_learn_for_each_algo():
    separator("4. build_model() constructs and trains briefly for each algo")
    for algo_name in ["ppo", "sac", "ddpg", "td3"]:
        env = DummyContinuousEnv()
        cfg = {algo_name: _TEST_CFG[algo_name]}
        model = build_model(
            algo_name=algo_name,
            cfg=cfg,
            env=env,
            tensorboard_log=None,
            seed=0,
        )
        assert isinstance(model, ALGORITHMS[algo_name]), (
            f"Expected {ALGORITHMS[algo_name]}, got {type(model)}"
        )
        # Run a handful of timesteps to confirm the kwargs actually
        # produce a working train loop, not just a constructible object.
        model.learn(total_timesteps=16, progress_bar=False)
        print(f"  {algo_name}: built {type(model).__name__} and ran learn() OK")
    print("  ✓ PASSED")
```

Also import `ALGORITHMS` (already imported in Task 1's test) and add the new test call to the `__main__` block:

```python
if __name__ == "__main__":
    test_registry_contents()
    test_run_prefix()
    test_run_prefix_unknown_algo()
    test_build_model_and_learn_for_each_algo()
    print(f"\n{'='*55}")
    print("  ALL TESTS PASSED")
    print(f"{'='*55}\n")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_algorithms.py`
Expected: `ImportError: cannot import name 'build_model' from 'agent.algorithms'`

- [ ] **Step 3: Write minimal implementation**

Append to `agent/algorithms.py` (after the existing `get_run_prefix` function):

```python
import numpy as np
from typing import Optional
from stable_baselines3.common.noise import NormalActionNoise


# Algorithms whose policy is deterministic and therefore need explicit
# action noise injected for exploration. SAC explores via its stochastic
# policy and does not need this.
_NOISE_ALGOS = {"ddpg", "td3"}

# On-policy algorithms collect a fresh rollout buffer every update and
# use different hyperparameters than off-policy (replay buffer) ones.
_ON_POLICY_ALGOS = {"ppo"}


def _build_kwargs(algo_name: str, algo_cfg: dict, action_space) -> dict:
    """
    Translate a config.yaml algorithm block into SB3 constructor kwargs
    (everything except policy/env/tensorboard_log/seed, which build_model
    adds separately).
    """
    if algo_name in _ON_POLICY_ALGOS:
        return dict(
            learning_rate=algo_cfg["learning_rate"],
            n_steps=algo_cfg["n_steps"],
            batch_size=algo_cfg["batch_size"],
            n_epochs=algo_cfg["n_epochs"],
            gamma=algo_cfg["gamma"],
            gae_lambda=algo_cfg["gae_lambda"],
            clip_range=algo_cfg["clip_range"],
            ent_coef=algo_cfg["ent_coef"],
            vf_coef=algo_cfg["vf_coef"],
            max_grad_norm=algo_cfg["max_grad_norm"],
            verbose=algo_cfg["verbose"],
        )

    # Off-policy: SAC, DDPG, TD3 share the replay-buffer hyperparameters.
    kwargs = dict(
        learning_rate=algo_cfg["learning_rate"],
        buffer_size=algo_cfg["buffer_size"],
        learning_starts=algo_cfg["learning_starts"],
        batch_size=algo_cfg["batch_size"],
        tau=algo_cfg["tau"],
        gamma=algo_cfg["gamma"],
        train_freq=algo_cfg["train_freq"],
        gradient_steps=algo_cfg["gradient_steps"],
        verbose=algo_cfg["verbose"],
    )

    if algo_name in _NOISE_ALGOS:
        n_actions = action_space.shape[-1]
        sigma = algo_cfg["action_noise_sigma"]
        kwargs["action_noise"] = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=sigma * np.ones(n_actions),
        )

    return kwargs


def build_model(
    algo_name: str,
    cfg: dict,
    env,
    tensorboard_log: str,
    resume_path: Optional[str] = None,
    seed: Optional[int] = None,
):
    """
    Construct (or resume) an SB3 model for the given algorithm.

    Parameters
    ----------
    algo_name       : one of ALGORITHMS keys, e.g. "ppo", "sac"
    cfg             : the full config.yaml dict (must contain cfg[algo_name]
                       unless resume_path is given)
    env             : the (possibly vectorized) training environment
    tensorboard_log : directory for TensorBoard logs
    resume_path     : if given, load this checkpoint instead of building fresh
    seed            : RNG seed (ignored when resuming)

    Returns
    -------
    An SB3 BaseAlgorithm subclass instance (PPO, SAC, DDPG, or TD3).
    """
    if algo_name not in ALGORITHMS:
        raise ValueError(
            "Unknown algorithm '{}'. Available: {}".format(
                algo_name, sorted(ALGORITHMS.keys())
            )
        )

    algo_cls = ALGORITHMS[algo_name]

    if resume_path is not None:
        return algo_cls.load(
            resume_path,
            env=env,
            tensorboard_log=tensorboard_log,
        )

    kwargs = _build_kwargs(algo_name, cfg[algo_name], env.action_space)

    return algo_cls(
        policy="MlpPolicy",
        env=env,
        tensorboard_log=tensorboard_log,
        seed=seed,
        **kwargs
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_algorithms.py`
Expected: `ALL TESTS PASSED` (the build+learn loop for all 4 algorithms should finish in a few seconds since the dummy env has no CARLA latency).

- [ ] **Step 5: Commit**

```bash
cd /home/rtrk/konstantin/diplomski/carla_rl_project
git add agent/algorithms.py scripts/test_algorithms.py
git commit -m "Add build_model() with on-policy/off-policy/noise hyperparameter wiring"
```

---

### Task 3: `build_model()` — resume from checkpoint

**Files:**
- Modify: `scripts/test_algorithms.py` (append resume test only — `build_model`'s resume branch was already written in Task 2, this task verifies it)

**Interfaces:**
- Consumes: `build_model(..., resume_path=...)` from Task 2.

- [ ] **Step 1: Write the failing test**

Append to `scripts/test_algorithms.py`, add `import tempfile` to the top imports, and add this test function before `__main__`:

```python
def test_build_model_resume():
    separator("5. build_model() resumes a saved checkpoint")
    env = DummyContinuousEnv()
    cfg = {"sac": _TEST_CFG["sac"]}

    fresh_model = build_model(
        algo_name="sac", cfg=cfg, env=env, tensorboard_log=None, seed=0,
    )
    fresh_model.learn(total_timesteps=8, progress_bar=False)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "sac_checkpoint")
        fresh_model.save(save_path)

        resumed_model = build_model(
            algo_name="sac",
            cfg=cfg,
            env=DummyContinuousEnv(),
            tensorboard_log=None,
            resume_path=save_path,
        )
        assert isinstance(resumed_model, ALGORITHMS["sac"])

        sample_obs = env.observation_space.sample()
        action, _ = resumed_model.predict(sample_obs, deterministic=True)
        assert action.shape == (2,), f"Expected action shape (2,), got {action.shape}"
        print(f"  Resumed SAC model predicted action: {action}")
    print("  ✓ PASSED")
```

Add the call in `__main__`:

```python
if __name__ == "__main__":
    test_registry_contents()
    test_run_prefix()
    test_run_prefix_unknown_algo()
    test_build_model_and_learn_for_each_algo()
    test_build_model_resume()
    print(f"\n{'='*55}")
    print("  ALL TESTS PASSED")
    print(f"{'='*55}\n")
```

- [ ] **Step 2: Run test to verify it fails first if resume were broken**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_algorithms.py`
Expected: Since `build_model`'s resume branch already exists from Task 2, this should actually **PASS** immediately. This step exists to confirm that — if it fails, it means Task 2's resume branch has a bug (e.g. `algo_cls.load` argument mismatch); fix `agent/algorithms.py` before continuing.

- [ ] **Step 3: Confirm implementation (no new code expected)**

No changes expected to `agent/algorithms.py`. If Step 2 failed, the fix belongs in the `resume_path is not None` branch of `build_model` written in Task 2.

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_algorithms.py`
Expected: `ALL TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
cd /home/rtrk/konstantin/diplomski/carla_rl_project
git add scripts/test_algorithms.py
git commit -m "Verify build_model() checkpoint resume path with a regression test"
```

---

### Task 4: `configs/config.yaml` — algo field and per-algorithm blocks

**Files:**
- Modify: `configs/config.yaml`
- Modify: `scripts/test_algorithms.py` (append real-config construction test)

**Interfaces:**
- Produces: `cfg["algo"]` (str, default `"ppo"`), `cfg["sac"]`, `cfg["ddpg"]`, `cfg["td3"]` blocks (dicts) alongside the existing `cfg["ppo"]`. Consumed by `agent/train.py` in Task 5.

- [ ] **Step 1: Write the failing test**

Append to `scripts/test_algorithms.py`, add `import yaml` to the top imports, and add this test before `__main__`:

```python
def test_real_config_builds_every_algorithm():
    separator("6. configs/config.yaml has a valid block for every algorithm")
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    assert "algo" in cfg, "config.yaml must have a top-level 'algo' field"
    assert cfg["algo"] in ALGORITHMS, f"cfg['algo']={cfg['algo']!r} is not a registered algorithm"

    for algo_name in ALGORITHMS:
        assert algo_name in cfg, f"config.yaml is missing a '{algo_name}:' block"
        env = DummyContinuousEnv()
        model = build_model(
            algo_name=algo_name, cfg=cfg, env=env, tensorboard_log=None, seed=0,
        )
        assert isinstance(model, ALGORITHMS[algo_name])
        print(f"  {algo_name}: real config.yaml block builds {type(model).__name__} OK")
    print("  ✓ PASSED")
```

Add the call in `__main__`:

```python
if __name__ == "__main__":
    test_registry_contents()
    test_run_prefix()
    test_run_prefix_unknown_algo()
    test_build_model_and_learn_for_each_algo()
    test_build_model_resume()
    test_real_config_builds_every_algorithm()
    print(f"\n{'='*55}")
    print("  ALL TESTS PASSED")
    print(f"{'='*55}\n")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_algorithms.py`
Expected: `AssertionError: config.yaml must have a top-level 'algo' field`

- [ ] **Step 3: Add the `algo` field**

In `configs/config.yaml`, change:

```yaml
# ── Environment ───────────────────────────────────────────────
env:
```

to:

```yaml
# ── Algorithm selection ─────────────────────────────────────────
# Which RL algorithm to train with. Overridden by `--algo` on the CLI.
# One of: ppo, sac, ddpg, td3. (DQN is not supported — this project's
# action space is continuous; see docs/superpowers/specs/2026-06-23-
# multi-algorithm-training-design.md for why and the planned path to
# add it.)
algo: ppo

# ── Environment ───────────────────────────────────────────────
env:
```

- [ ] **Step 4: Replace the `paths` block (remove the now-unused `best_model` key)**

In `configs/config.yaml`, change:

```yaml
# ── Paths ─────────────────────────────────────────────────────
paths:
  log_dir:          "results/logs"
  checkpoint_dir:   "results/checkpoints"
  plot_dir:         "results/plots"
  best_model:       "results/checkpoints/best_model"
```

to:

```yaml
# ── Paths ─────────────────────────────────────────────────────
# These are base directories. agent/train.py joins each with the
# current algorithm name, e.g. results/checkpoints/sac/, so runs for
# different algorithms never collide or overwrite each other.
# best_model is not listed separately — it's {checkpoint_dir}/{algo}/best_model.
paths:
  log_dir:          "results/logs"
  checkpoint_dir:   "results/checkpoints"
  plot_dir:         "results/plots"
```

- [ ] **Step 5: Append the `sac`, `ddpg`, `td3` hyperparameter blocks**

In `configs/config.yaml`, change:

```yaml
# ── Training ──────────────────────────────────────────────────
training:
```

to:

```yaml
# ── SAC hyperparameters ────────────────────────────────────────
# Off-policy, stochastic policy — explores via entropy, no action
# noise needed. Starting points from SB3 defaults; not yet
# benchmarked against CARLA, adjust based on experiments.
# See: https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
sac:
  learning_rate:    0.0003       # Adam optimizer learning rate
  buffer_size:      200000       # replay buffer capacity (transitions)
  learning_starts:  1000         # steps of random actions before training starts
  batch_size:       256          # minibatch size for gradient updates
  tau:              0.005        # target network soft-update rate
  gamma:            0.99         # discount factor
  train_freq:       1            # train every N environment steps
  gradient_steps:   1            # gradient updates per training call
  verbose:          1            # SB3 verbosity (0=silent, 1=info, 2=debug)

# ── DDPG hyperparameters ───────────────────────────────────────
# Off-policy, deterministic policy — needs explicit action noise for
# exploration (unlike SAC). Starting points from SB3 defaults; not yet
# benchmarked against CARLA, adjust based on experiments.
# See: https://stable-baselines3.readthedocs.io/en/master/modules/ddpg.html
ddpg:
  learning_rate:      0.001      # Adam optimizer learning rate
  buffer_size:        200000     # replay buffer capacity (transitions)
  learning_starts:    1000       # steps of random actions before training starts
  batch_size:         256        # minibatch size for gradient updates
  tau:                0.005      # target network soft-update rate
  gamma:              0.99       # discount factor
  train_freq:         1          # train every N environment steps
  gradient_steps:     1          # gradient updates per training call
  action_noise_sigma: 0.1        # stddev of Gaussian exploration noise
  verbose:            1          # SB3 verbosity (0=silent, 1=info, 2=debug)

# ── TD3 hyperparameters ────────────────────────────────────────
# Off-policy, deterministic policy (DDPG's successor) — also needs
# explicit action noise. Starting points from SB3 defaults; not yet
# benchmarked against CARLA, adjust based on experiments.
# See: https://stable-baselines3.readthedocs.io/en/master/modules/td3.html
td3:
  learning_rate:      0.001      # Adam optimizer learning rate
  buffer_size:        200000     # replay buffer capacity (transitions)
  learning_starts:    1000       # steps of random actions before training starts
  batch_size:         256        # minibatch size for gradient updates
  tau:                0.005      # target network soft-update rate
  gamma:              0.99       # discount factor
  train_freq:         1          # train every N environment steps
  gradient_steps:     1          # gradient updates per training call
  action_noise_sigma: 0.1        # stddev of Gaussian exploration noise
  verbose:            1          # SB3 verbosity (0=silent, 1=info, 2=debug)

# ── Training ──────────────────────────────────────────────────
training:
```

- [ ] **Step 6: Run test to verify it passes**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/test_algorithms.py`
Expected: `ALL TESTS PASSED`

- [ ] **Step 7: Commit**

```bash
cd /home/rtrk/konstantin/diplomski/carla_rl_project
git add configs/config.yaml scripts/test_algorithms.py
git commit -m "Add algo field and SAC/DDPG/TD3 hyperparameter blocks to config.yaml"
```

---

### Task 5: Generalize `agent/train.py` to use the algorithm registry

**Files:**
- Modify: `agent/train.py:1-302` (full file rewritten in place)

**Interfaces:**
- Consumes: `agent.algorithms.{ALGORITHMS, build_model, get_run_prefix}` (Tasks 1–2), `cfg["algo"]` and `cfg[algo_name]` from `configs/config.yaml` (Task 4).
- Produces: `train(config_path, total_timesteps=None, resume_path=None, algo=None)` — same return shape as before (`model, run_log_dir`), now algorithm-scoped paths.

- [ ] **Step 1: Update imports**

In `agent/train.py`, change:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from carla_env.env    import CarlaLaneKeepingEnv
from carla_env.reward import RewardConfig
from agent.callbacks  import EpisodeLoggerCallback
```

to:

```python
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from carla_env.env     import CarlaLaneKeepingEnv
from carla_env.reward  import RewardConfig
from agent.callbacks   import EpisodeLoggerCallback
from agent.algorithms  import ALGORITHMS, build_model, get_run_prefix
```

- [ ] **Step 2: Make `make_env` accept an explicit log directory**

In `agent/train.py`, change:

```python
def make_env(cfg: dict, seed: int = 0):
    """
    Factory function that creates and wraps the CARLA environment.

    We wrap with Monitor so SB3 can track episode rewards and lengths
    automatically. Monitor writes to a .csv file in the log directory.

    Why a factory function?
        SB3's DummyVecEnv expects a callable that returns an env,
        not the env itself. This pattern also makes it easy to create
        multiple parallel environments later.
    """
```

to:

```python
def make_env(cfg: dict, log_dir: str, seed: int = 0):
    """
    Factory function that creates and wraps the CARLA environment.

    We wrap with Monitor so SB3 can track episode rewards and lengths
    automatically. Monitor writes to a .csv file in the log directory.

    Why a factory function?
        SB3's DummyVecEnv expects a callable that returns an env,
        not the env itself. This pattern also makes it easy to create
        multiple parallel environments later.

    Why log_dir as a parameter instead of reading cfg["paths"]["log_dir"]?
        The caller (train()) computes an algorithm-scoped log directory
        (e.g. results/logs/sac/) so runs for different algorithms never
        collide. Reading it directly from cfg here would lose that.
    """
```

Then, later in the same function, change:

```python
    # Monitor wrapper: records episode reward/length to CSV
    # and makes them available to SB3's logging system
    log_dir = cfg["paths"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, filename=os.path.join(log_dir, f"monitor_{seed}"))
```

to:

```python
    # Monitor wrapper: records episode reward/length to CSV
    # and makes them available to SB3's logging system
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, filename=os.path.join(log_dir, f"monitor_{seed}"))
```

- [ ] **Step 3: Resolve the algorithm name and scoped paths at the top of `train()`**

In `agent/train.py`, change the function signature and the start of its body:

```python
def train(config_path: str, total_timesteps: int = None, resume_path: str = None):
    """
    Full PPO training pipeline.

    Args:
        config_path:      path to configs/config.yaml
        total_timesteps:  override config value (useful for quick tests)
        resume_path:      path to a saved model to resume training from
    """

    # ── Load config ────────────────────────────────────────────────────────────
    cfg = load_config(config_path)

    # Command-line overrides
    if total_timesteps is not None:
        cfg["training"]["total_timesteps"] = total_timesteps

    # ── Create output directories ──────────────────────────────────────────────
    for key in ["log_dir", "checkpoint_dir", "plot_dir"]:
        os.makedirs(cfg["paths"][key], exist_ok=True)

    # ── Timestamped run name ───────────────────────────────────────────────────
    # Each training run gets a unique name so logs don't overwrite each other.
    run_name    = f"ppo_lane_keeping_{time.strftime('%Y%m%d_%H%M%S')}"
    run_log_dir = os.path.join(cfg["paths"]["log_dir"], run_name)
    os.makedirs(run_log_dir, exist_ok=True)
```

to:

```python
def train(
    config_path: str,
    total_timesteps: int = None,
    resume_path: str = None,
    algo: str = None,
):
    """
    Full multi-algorithm training pipeline (PPO, SAC, DDPG, TD3).

    Args:
        config_path:      path to configs/config.yaml
        total_timesteps:  override config value (useful for quick tests)
        resume_path:      path to a saved model to resume training from
        algo:             algorithm name, overrides cfg["algo"] if given
    """

    # ── Load config ────────────────────────────────────────────────────────────
    cfg = load_config(config_path)

    # Command-line overrides
    if total_timesteps is not None:
        cfg["training"]["total_timesteps"] = total_timesteps

    algo_name = algo or cfg.get("algo", "ppo")
    if algo_name not in ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm '{algo_name}'. "
            f"Available: {sorted(ALGORITHMS.keys())}"
        )
    cfg["algo"] = algo_name
    run_prefix  = get_run_prefix(algo_name)
    logger.info(f"Algorithm: {algo_name}")

    # ── Algorithm-scoped output directories ───────────────────────────────────
    # Each algorithm gets its own subdirectory so runs never collide or
    # overwrite each other's checkpoints/logs.
    checkpoint_dir = os.path.join(cfg["paths"]["checkpoint_dir"], algo_name)
    log_dir        = os.path.join(cfg["paths"]["log_dir"], algo_name)
    plot_dir       = os.path.join(cfg["paths"]["plot_dir"], algo_name)
    best_model_dir = os.path.join(checkpoint_dir, "best_model")

    for d in [checkpoint_dir, log_dir, plot_dir, best_model_dir]:
        os.makedirs(d, exist_ok=True)

    # ── Timestamped run name ───────────────────────────────────────────────────
    # Each training run gets a unique name so logs don't overwrite each other.
    run_name    = f"{run_prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_log_dir = os.path.join(log_dir, run_name)
    os.makedirs(run_log_dir, exist_ok=True)
```

- [ ] **Step 4: Use the scoped log dir when creating environments**

In `agent/train.py`, change:

```python
    # ── Create training environment ────────────────────────────────────────────
    logger.info("Creating training environment ...")
    # DummyVecEnv wraps the env in a vectorized interface that SB3 expects.
    # We use 1 environment (DummyVecEnv with n=1) because CARLA is heavy.
    # Multi-environment training would require multiple CARLA instances.
    train_env = DummyVecEnv([lambda: make_env(cfg, seed=0)])

    # ── Create evaluation environment ─────────────────────────────────────────
    # A separate environment for periodic evaluation during training.
    # This gives us unbiased performance estimates on fresh episodes.
    logger.info("Creating evaluation environment ...")
    eval_env = DummyVecEnv([lambda: make_env(cfg, seed=100)])
```

to:

```python
    # ── Create training environment ────────────────────────────────────────────
    logger.info("Creating training environment ...")
    # DummyVecEnv wraps the env in a vectorized interface that SB3 expects.
    # We use 1 environment (DummyVecEnv with n=1) because CARLA is heavy.
    # Multi-environment training would require multiple CARLA instances.
    train_env = DummyVecEnv([lambda: make_env(cfg, log_dir=log_dir, seed=0)])

    # ── Create evaluation environment ─────────────────────────────────────────
    # A separate environment for periodic evaluation during training.
    # This gives us unbiased performance estimates on fresh episodes.
    logger.info("Creating evaluation environment ...")
    eval_env = DummyVecEnv([lambda: make_env(cfg, log_dir=log_dir, seed=100)])
```

- [ ] **Step 5: Replace the hardcoded PPO construction with the registry call**

In `agent/train.py`, change:

```python
    # ── Build PPO agent ────────────────────────────────────────────────────────
    ppo_cfg = cfg["ppo"]

    if resume_path is not None:
        # Resume training from a saved checkpoint
        logger.info(f"Resuming from: {resume_path}")
        model = PPO.load(
            resume_path,
            env=train_env,
            tensorboard_log=run_log_dir,
        )
    else:
        # Fresh training run
        model = PPO(
            policy         = "MlpPolicy",   # Multi-layer perceptron policy
                                            # appropriate for our 4D obs space
            env            = train_env,
            learning_rate  = ppo_cfg["learning_rate"],
            n_steps        = ppo_cfg["n_steps"],
            batch_size     = ppo_cfg["batch_size"],
            n_epochs       = ppo_cfg["n_epochs"],
            gamma          = ppo_cfg["gamma"],
            gae_lambda     = ppo_cfg["gae_lambda"],
            clip_range     = ppo_cfg["clip_range"],
            ent_coef       = ppo_cfg["ent_coef"],
            vf_coef        = ppo_cfg["vf_coef"],
            max_grad_norm  = ppo_cfg["max_grad_norm"],
            verbose        = ppo_cfg["verbose"],
            tensorboard_log= run_log_dir,
            seed           = cfg["env"]["seed"],
        )

    logger.info(f"PPO policy network:\n{model.policy}")
```

to:

```python
    # ── Build agent (PPO / SAC / DDPG / TD3 via the registry) ─────────────────
    if resume_path is not None:
        logger.info(f"Resuming from: {resume_path}")

    model = build_model(
        algo_name       = algo_name,
        cfg             = cfg,
        env             = train_env,
        tensorboard_log = run_log_dir,
        resume_path     = resume_path,
        seed            = cfg["env"]["seed"],
    )

    logger.info(f"{algo_name.upper()} policy network:\n{model.policy}")
```

- [ ] **Step 6: Scope the checkpoint and eval callbacks**

In `agent/train.py`, change:

```python
    # 2. Checkpoint — saves model every save_freq steps
    checkpoint_cb = CheckpointCallback(
        save_freq   = cfg["training"]["save_freq"],
        save_path   = cfg["paths"]["checkpoint_dir"],
        name_prefix = "ppo_lane_keeping",
        verbose     = 1,
    )

    # 3. Evaluation — runs eval_episodes on the eval env periodically,
    #    saves the best model seen so far
    eval_cb = EvalCallback(
        eval_env           = eval_env,
        best_model_save_path = cfg["paths"]["best_model"],
        log_path           = os.path.join(run_log_dir, "eval"),
        eval_freq          = cfg["training"]["eval_freq"],
        n_eval_episodes    = cfg["training"]["eval_episodes"],
        deterministic      = True,   # use mean action, not sampled
        verbose            = 1,
    )
```

to:

```python
    # 2. Checkpoint — saves model every save_freq steps
    checkpoint_cb = CheckpointCallback(
        save_freq   = cfg["training"]["save_freq"],
        save_path   = checkpoint_dir,
        name_prefix = run_prefix,
        verbose     = 1,
    )

    # 3. Evaluation — runs eval_episodes on the eval env periodically,
    #    saves the best model seen so far
    eval_cb = EvalCallback(
        eval_env           = eval_env,
        best_model_save_path = best_model_dir,
        log_path           = os.path.join(run_log_dir, "eval"),
        eval_freq          = cfg["training"]["eval_freq"],
        n_eval_episodes    = cfg["training"]["eval_episodes"],
        deterministic      = True,   # use mean action, not sampled
        verbose            = 1,
    )
```

- [ ] **Step 7: Save the final model under the scoped checkpoint dir**

In `agent/train.py`, change:

```python
    # ── Save final model ───────────────────────────────────────────────────────
    final_path = os.path.join(cfg["paths"]["checkpoint_dir"], "final_model")
    model.save(final_path)
    logger.info(f"Final model saved to: {final_path}")
```

to:

```python
    # ── Save final model ───────────────────────────────────────────────────────
    final_path = os.path.join(checkpoint_dir, "final_model")
    model.save(final_path)
    logger.info(f"Final model saved to: {final_path}")
```

- [ ] **Step 8: Add the `--algo` CLI flag**

In `agent/train.py`, change:

```python
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a PPO agent for CARLA lane keeping"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total_timesteps from config (useful for quick tests)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to saved model to resume training from",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        config_path      = args.config,
        total_timesteps  = args.timesteps,
        resume_path      = args.resume,
    )
```

to:

```python
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an RL agent (PPO/SAC/DDPG/TD3) for CARLA lane keeping"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default=None,
        choices=sorted(ALGORITHMS.keys()),
        help="RL algorithm to train with (overrides config.yaml's 'algo' field)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total_timesteps from config (useful for quick tests)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to saved model to resume training from",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        config_path      = args.config,
        total_timesteps  = args.timesteps,
        resume_path      = args.resume,
        algo             = args.algo,
    )
```

- [ ] **Step 9: Update the module docstring**

In `agent/train.py`, change the docstring header:

```python
"""
train.py
--------
Phase 8 — PPO Training Entry Point

Purpose:
    Load configuration, create the CARLA environment, set up PPO,
    attach callbacks, and run training.

How to run:
    python agent/train.py
    python agent/train.py --config configs/config.yaml
    python agent/train.py --timesteps 100000   (quick test run)
    python agent/train.py --resume results/checkpoints/best_model
```

to:

```python
"""
train.py
--------
Phase 8 — Multi-Algorithm Training Entry Point

Purpose:
    Load configuration, create the CARLA environment, build the
    selected RL algorithm (PPO, SAC, DDPG, or TD3) via the registry in
    agent/algorithms.py, attach callbacks, and run training.

How to run:
    python agent/train.py                          (uses cfg["algo"], default "ppo")
    python agent/train.py --algo sac
    python agent/train.py --algo ddpg --timesteps 10000   (quick test run)
    python agent/train.py --algo td3 --resume results/checkpoints/td3/best_model
```

- [ ] **Step 10: Sanity-check the script still parses and imports cleanly**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python -c "import sys; sys.path.insert(0, '.'); import agent.train"`
Expected: no output, exit code 0 (a clean import with no syntax/import errors).

- [ ] **Step 11: Commit**

```bash
cd /home/rtrk/konstantin/diplomski/carla_rl_project
git add agent/train.py
git commit -m "Generalize train.py to select algorithm via registry and scope output paths per algorithm"
```

---

### Task 6: Live CARLA smoke tests for all four algorithms

**Files:** none (verification only — no source changes expected unless a bug surfaces, in which case fix it in `agent/train.py` or `agent/algorithms.py` and re-run)

**Interfaces:** Exercises `agent/train.py`'s CLI end-to-end against the running CARLA server at `localhost:2000`.

- [ ] **Step 1: Confirm CARLA is reachable**

Run: `/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python scripts/verify_carla.py`
Expected: connects successfully (matches the check already done during planning — server version 0.9.15).

- [ ] **Step 2: Smoke-test PPO**

Run:
```bash
cd /home/rtrk/konstantin/diplomski/carla_rl_project
/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python agent/train.py --algo ppo --timesteps 300
```
Expected: completes without traceback; ends with `Training complete.`
Then verify: `ls results/checkpoints/ppo/ results/logs/ppo/` — both directories exist and are non-empty (checkpoint `.zip` files, a timestamped run directory, `best_model/`).

- [ ] **Step 3: Smoke-test SAC**

Run:
```bash
/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python agent/train.py --algo sac --timesteps 300
```
Expected: completes without traceback.
Then verify: `ls results/checkpoints/sac/ results/logs/sac/` — both populated, and distinct from `ppo/`'s contents (different filenames, e.g. `sac_lane_keeping_*`).

- [ ] **Step 4: Smoke-test DDPG**

Run:
```bash
/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python agent/train.py --algo ddpg --timesteps 300
```
Expected: completes without traceback.
Then verify: `ls results/checkpoints/ddpg/ results/logs/ddpg/` — both populated.

- [ ] **Step 5: Smoke-test TD3**

Run:
```bash
/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python agent/train.py --algo td3 --timesteps 300
```
Expected: completes without traceback.
Then verify: `ls results/checkpoints/td3/ results/logs/td3/` — both populated.

- [ ] **Step 6: Smoke-test resume, using the SAC run from Step 3**

Run:
```bash
/home/rtrk/konstantin/miniconda3/envs/carla915/bin/python agent/train.py --algo sac --timesteps 100 --resume results/checkpoints/sac/final_model
```
Expected: log line `Resuming from: results/checkpoints/sac/final_model`, completes without traceback, and writes a new timestamped run directory under `results/logs/sac/`.

- [ ] **Step 7: Record results**

No commit needed for this task (no source files changed). If any step required a fix, that fix should already have been committed as part of Task 5; amend here only if a genuinely new bug was found and fixed — in that case:

```bash
cd /home/rtrk/konstantin/diplomski/carla_rl_project
git add -A
git commit -m "Fix issue found during live CARLA smoke tests across PPO/SAC/DDPG/TD3"
```

---

### Task 7: Migrate existing PPO results into the per-algorithm layout

**Files:** filesystem only (`results/` is gitignored — no git changes here)

- [ ] **Step 1: Inspect current contents**

Run:
```bash
cd /home/rtrk/konstantin/diplomski/carla_rl_project
ls results/checkpoints/ results/logs/
```
Expected (from the original project state): checkpoint `.zip` files and `best_model/` directly under `results/checkpoints/`; `monitor_*.csv` and `ppo_lane_keeping_*` run directories directly under `results/logs/`. (Task 6 will have already created `results/checkpoints/ppo/` and `results/logs/ppo/` with new runs — this step moves the *original*, pre-existing files into those same directories.)

- [ ] **Step 2: Move pre-existing checkpoints into `results/checkpoints/ppo/`**

Run:
```bash
mkdir -p results/checkpoints/ppo
for f in results/checkpoints/*.zip; do
  [ -e "$f" ] && mv "$f" results/checkpoints/ppo/
done
[ -d results/checkpoints/best_model ] && mv results/checkpoints/best_model results/checkpoints/ppo/best_model
```

- [ ] **Step 3: Move pre-existing logs into `results/logs/ppo/`**

Run:
```bash
mkdir -p results/logs/ppo
for d in results/logs/ppo_lane_keeping_*; do
  [ -e "$d" ] && mv "$d" results/logs/ppo/
done
for f in results/logs/monitor_*.csv; do
  [ -e "$f" ] && mv "$f" results/logs/ppo/
done
```

- [ ] **Step 4: Verify the migration**

Run: `ls results/checkpoints/ results/logs/`
Expected: `results/checkpoints/` contains only algorithm subdirectories (`ppo/`, plus `sac/`, `ddpg/`, `td3/` from Task 6); same for `results/logs/`. No loose `.zip`, `.csv`, or run-name directories remain at the top level.

No commit needed — `results/` is gitignored.

---

### Task 8: Update `CLAUDE.md` documentation

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the "What this project is" intro**

In `CLAUDE.md`, change:

```markdown
A reinforcement learning system that trains a car to keep its lane
in the CARLA driving simulator. The agent uses PPO (Proximal Policy
Optimization) from Stable-Baselines3. The observation is a 4D
low-dimensional state vector. The action is 2D continuous control.
```

to:

```markdown
A reinforcement learning system that trains a car to keep its lane
in the CARLA driving simulator. The agent can be trained with PPO,
SAC, DDPG, or TD3 from Stable-Baselines3, selected via a pluggable
algorithm registry (`agent/algorithms.py`) — see "Switching algorithms"
below. The observation is a 4D low-dimensional state vector. The
action is 2D continuous control.
```

- [ ] **Step 2: Update the project structure diagram**

In `CLAUDE.md`, change:

```markdown
├── agent/                      # RL training pipeline
│   ├── __init__.py
│   ├── train.py                # PPO training entry point
│   ├── evaluate.py             # Evaluation script (Phase 9 — not yet built)
│   └── callbacks.py            # SB3 callbacks: episode logger, checkpoints
```

to:

```markdown
├── agent/                      # RL training pipeline
│   ├── __init__.py
│   ├── algorithms.py           # Algorithm registry: PPO/SAC/DDPG/TD3 -> SB3 classes
│   ├── train.py                # Multi-algorithm training entry point
│   ├── evaluate.py             # Evaluation script (Phase 9 — not yet built)
│   └── callbacks.py            # SB3 callbacks: episode logger, checkpoints
```

And change:

```markdown
├── results/                    # Generated at runtime — not committed to git
│   ├── logs/                   # TensorBoard logs + episode CSV
│   ├── checkpoints/            # Saved model weights
│   └── plots/                  # Evaluation plots (Phase 9)
```

to:

```markdown
├── results/                    # Generated at runtime — not committed to git
│   ├── logs/{algo}/            # TensorBoard logs + episode CSV, per algorithm
│   ├── checkpoints/{algo}/     # Saved model weights, per algorithm
│   └── plots/{algo}/           # Evaluation plots (Phase 9), per algorithm
```

- [ ] **Step 3: Add a "Switching algorithms" section**

In `CLAUDE.md`, immediately after the "### Action vector (2D)" section and before "### Reward function", insert:

```markdown
### Switching algorithms

`agent/algorithms.py` holds the `ALGORITHMS` registry (`ppo`, `sac`,
`ddpg`, `td3` → their SB3 classes) and `build_model()`, which knows how
to translate each algorithm's `configs/config.yaml` block into SB3
constructor kwargs (on-policy PPO vs. off-policy SAC/DDPG/TD3, plus
action noise for the deterministic policies DDPG/TD3).

Select an algorithm via `configs/config.yaml`'s top-level `algo:` field,
or override per run with `--algo`:

```bash
python agent/train.py --algo sac
```

Each algorithm's checkpoints and logs live in their own subdirectory
(`results/checkpoints/{algo}/`, `results/logs/{algo}/`) so runs never
collide.

**DQN is not supported** — it requires a discrete action space, and this
project's action space is continuous. Adding it would require a separate
discretized environment variant; see
`docs/superpowers/specs/2026-06-23-multi-algorithm-training-design.md`
for the planned approach.

Adding a new continuous-action algorithm later: add it to `ALGORITHMS`,
extend `_build_kwargs()` if it needs hyperparameters not already handled,
and add its block to `configs/config.yaml`. No changes to `train.py`
are needed.
```

- [ ] **Step 4: Update "How to run things"**

In `CLAUDE.md`, change:

```markdown
# Train PPO agent (CARLA must be running)
python agent/train.py
python agent/train.py --timesteps 10000    # quick test
python agent/train.py --resume results/checkpoints/best_model
```

to:

```markdown
# Train an agent (CARLA must be running) — algo defaults to config.yaml's `algo:` field
python agent/train.py
python agent/train.py --algo sac
python agent/train.py --algo ddpg --timesteps 10000    # quick test
python agent/train.py --algo td3 --resume results/checkpoints/td3/best_model

# Run offline algorithm-wiring tests (no CARLA needed)
python scripts/test_algorithms.py
```

- [ ] **Step 5: Update "Current training config" example**

In `CLAUDE.md`, change:

```markdown
## Current training config (configs/config.yaml)

```yaml
env:
  map_name:     Town04      # highway loop, no intersections
  max_steps:    1000        # 50 simulated seconds per episode
  spawn_index:  0           # fixed spawn point for reproducibility

reward:
  w_speed:      1.5         # high weight to prevent standing still
  step_penalty: -0.1        # encourages efficient driving

ppo:
  learning_rate: 0.0003
  n_steps:       2048
  batch_size:    64
  gamma:         0.99

training:
  total_timesteps: 500000
```
```

to:

```markdown
## Current training config (configs/config.yaml)

```yaml
algo: ppo                   # ppo | sac | ddpg | td3 — overridden by --algo

env:
  map_name:     Town04      # highway loop, no intersections
  max_steps:    1000        # 50 simulated seconds per episode
  spawn_index:  0           # fixed spawn point for reproducibility

reward:
  w_speed:      1.5         # high weight to prevent standing still
  step_penalty: -0.1        # encourages efficient driving

ppo:
  learning_rate: 0.0003
  n_steps:       2048
  batch_size:    64
  gamma:         0.99

sac:
  learning_rate: 0.0003
  buffer_size:   200000
  batch_size:    256

training:
  total_timesteps: 500000
```
```

- [ ] **Step 6: Update "What has NOT been built yet"**

In `CLAUDE.md`, the DQN/discretized-action note belongs here. Change:

```markdown
- **Camera/lidar sensors** — `carla_env/sensors.py` currently only
  has CollisionSensor. CameraSensor and LidarSensor can be added
  for a high-dimensional observation extension.
```

to:

```markdown
- **Camera/lidar sensors** — `carla_env/sensors.py` currently only
  has CollisionSensor. CameraSensor and LidarSensor can be added
  for a high-dimensional observation extension.

- **DQN support** — requires a discretized-action environment variant
  (e.g. `CarlaLaneKeepingEnvDiscrete`) since DQN only supports discrete
  action spaces. The algorithm registry in `agent/algorithms.py` is
  designed so DQN can be added as a one-line registry entry once that
  env variant exists. See the design spec's "Future extension: DQN"
  section.
```

- [ ] **Step 7: Commit**

```bash
cd /home/rtrk/konstantin/diplomski/carla_rl_project
git add CLAUDE.md
git commit -m "Document multi-algorithm training pipeline in CLAUDE.md"
```

---

## Plan Self-Review Notes

- **Spec coverage:** Registry (Task 1–2), resume (Task 3), config blocks (Task 4), `train.py` wiring (Task 5), live verification (Task 6), results migration (Task 7), docs (Task 8) — all spec sections have a corresponding task. DQN is explicitly *not* implemented per the spec's "Future extension" section; Task 1's test asserts it's absent from the registry to guard against accidental scope creep.
- **Type consistency:** `build_model(algo_name, cfg, env, tensorboard_log, resume_path=None, seed=None)` signature is identical across Tasks 2, 3, 5, and its test usages. `ALGORITHMS`, `get_run_prefix` likewise used consistently.
- **Placeholder scan:** no TBD/TODO; every step has runnable code or an exact command with expected output.
