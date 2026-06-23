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
