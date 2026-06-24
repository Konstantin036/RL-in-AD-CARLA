"""
test_algorithms.py
-------------------
Offline unit tests for agent/algorithms.py (no CARLA required).

Run from anywhere:
    python scripts/test_algorithms.py
"""

import sys
import os
import tempfile
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import gymnasium as gym

from agent.algorithms import ALGORITHMS, get_run_prefix, build_model
from carla_env.action import get_action_space
from carla_env.observation import get_observation_space


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
