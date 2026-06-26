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
    python agent/train.py --algo td3 --resume results/checkpoints/td3/td3_lane_keeping_20260101_120000/best_model

What it produces:
    results/logs/<algo>/           TensorBoard logs + episode CSV
    results/checkpoints/<algo>/<run_name>/    Model checkpoints every save_freq steps
    results/checkpoints/<algo>/<run_name>/best_model  Best model by mean eval reward
    results/checkpoints/<algo>/<run_name>/final_model.zip  Model at end of this run

Monitor training with TensorBoard:
    tensorboard --logdir results/logs
    Then open http://localhost:6006 in your browser.

Thesis note:
    This script is your main experimental entry point.
    Each run produces a timestamped log directory so you can compare
    multiple runs side by side in TensorBoard.
"""

import os
import sys
import time
import argparse
import logging

# ── Path setup ─────────────────────────────────────────────────────────────────
# Add project root to path so all carla_env imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np

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


# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ── Config loader ──────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Load YAML config file and return as nested dict."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Config loaded from: {config_path}")
    return cfg


# ── Environment factory ────────────────────────────────────────────────────────

def make_env(cfg: dict, log_dir: str, seed: int = 0, is_eval: bool = False):
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

    Why is_eval?
        train() runs a train_env and an eval_env simultaneously against
        the same CARLA server. If both used the identical deterministic
        spawn_index, whichever one currently has a live vehicle there
        would permanently block the other (confirmed via live testing).
        is_eval shifts the eval environment to a different spawn point
        via spawn_index_offset, so both stay deterministic without
        contending for the same spot.
    """
    env_cfg    = cfg["env"]
    reward_cfg = cfg["reward"]

    # Build RewardConfig from YAML values
    rc = RewardConfig.from_dict(reward_cfg)

    env = CarlaLaneKeepingEnv(
        host          = env_cfg["host"],
        port          = env_cfg["port"],
        map_name      = env_cfg["map_name"],
        max_steps     = env_cfg["max_steps"],
        reward_config = rc,
        action_smooth = env_cfg["action_smooth"],
        seed               = env_cfg["seed"] + seed,
        spawn_index        = env_cfg.get("spawn_index"),
        spawn_index_offset = 1 if is_eval else 0,
        verbose            = False,
    )

    # Monitor wrapper: records episode reward/length to CSV
    # and makes them available to SB3's logging system
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, filename=os.path.join(log_dir, f"monitor_{seed}"))

    return env


# ── Main training function ─────────────────────────────────────────────────────

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

    # ── Timestamped run name ───────────────────────────────────────────────────
    # Each training run gets a unique name, and checkpoints/best_model are
    # scoped under it (not just under the algorithm) — otherwise two
    # separate runs of the same algorithm that both reach e.g. 50,000
    # steps would silently overwrite each other's checkpoint files.
    run_name           = f"{run_prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_log_dir        = os.path.join(log_dir, run_name)
    run_checkpoint_dir = os.path.join(checkpoint_dir, run_name)
    best_model_dir     = os.path.join(run_checkpoint_dir, "best_model")

    for d in [run_checkpoint_dir, run_log_dir, plot_dir, best_model_dir]:
        os.makedirs(d, exist_ok=True)

    logger.info(f"Run name: {run_name}")
    logger.info(f"Log dir:  {run_log_dir}")
    logger.info(f"Checkpoint dir: {run_checkpoint_dir}")

    # Save config copy to run directory for reproducibility
    import shutil
    shutil.copy(config_path, os.path.join(run_log_dir, "config.yaml"))

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
    eval_env = DummyVecEnv([lambda: make_env(cfg, log_dir=log_dir, seed=100, is_eval=True)])

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

    # ── Build callbacks ────────────────────────────────────────────────────────

    # 1. Episode logger — writes per-episode metrics to CSV and TensorBoard
    episode_logger = EpisodeLoggerCallback(
        log_dir = run_log_dir,
        verbose = 1,
    )

    # 2. Checkpoint — saves model every save_freq steps
    checkpoint_cb = CheckpointCallback(
        save_freq   = cfg["training"]["save_freq"],
        save_path   = run_checkpoint_dir,
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

    # Combine all callbacks into one list
    callbacks = CallbackList([episode_logger, checkpoint_cb, eval_cb])

    # ── Train ──────────────────────────────────────────────────────────────────
    total_steps = cfg["training"]["total_timesteps"]
    logger.info(f"Starting training: {total_steps:,} timesteps")
    logger.info(f"Monitor with TensorBoard:")
    logger.info(f"  tensorboard --logdir {cfg['paths']['log_dir']}")

    try:
        model.learn(
            total_timesteps  = total_steps,
            callback         = callbacks,
            log_interval     = cfg["training"]["log_interval"],
            reset_num_timesteps = (resume_path is None),
            progress_bar     = True,
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")

    # ── Save final model ───────────────────────────────────────────────────────
    final_path = os.path.join(run_checkpoint_dir, "final_model")
    model.save(final_path)
    logger.info(f"Final model saved to: {final_path}")

    # ── Cleanup ────────────────────────────────────────────────────────────────
    train_env.close()
    eval_env.close()
    logger.info("Training complete.")

    return model, run_log_dir


# ── Entry point ────────────────────────────────────────────────────────────────

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
