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

import numpy as np
from typing import Optional
from stable_baselines3 import PPO, SAC, DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise


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
