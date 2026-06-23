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

What it produces:
    results/logs/           TensorBoard logs + episode CSV
    results/checkpoints/    Model checkpoints every save_freq steps
    results/checkpoints/best_model  Best model by mean eval reward

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
    env_cfg    = cfg["env"]
    reward_cfg = cfg["reward"]

    # Build RewardConfig from YAML values
    rc = RewardConfig(
        w_center         = reward_cfg["w_center"],
        w_speed          = reward_cfg["w_speed"],
        w_heading        = reward_cfg["w_heading"],
        target_speed_kmh = reward_cfg["target_speed_kmh"],
        sigma_speed      = reward_cfg["sigma_speed"],
        terminal_penalty = reward_cfg["terminal_penalty"],
        step_penalty     = reward_cfg["step_penalty"],
    )

    env = CarlaLaneKeepingEnv(
        host          = env_cfg["host"],
        port          = env_cfg["port"],
        map_name      = env_cfg["map_name"],
        max_steps     = env_cfg["max_steps"],
        reward_config = rc,
        action_smooth = env_cfg["action_smooth"],
        seed          = env_cfg["seed"] + seed,
        verbose       = False,
    )

    # Monitor wrapper: records episode reward/length to CSV
    # and makes them available to SB3's logging system
    log_dir = cfg["paths"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, filename=os.path.join(log_dir, f"monitor_{seed}"))

    return env


# ── Main training function ─────────────────────────────────────────────────────

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

    logger.info(f"Run name: {run_name}")
    logger.info(f"Log dir:  {run_log_dir}")

    # Save config copy to run directory for reproducibility
    import shutil
    shutil.copy(config_path, os.path.join(run_log_dir, "config.yaml"))

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

    # ── Build callbacks ────────────────────────────────────────────────────────

    # 1. Episode logger — writes per-episode metrics to CSV and TensorBoard
    episode_logger = EpisodeLoggerCallback(
        log_dir = run_log_dir,
        verbose = 1,
    )

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
    final_path = os.path.join(cfg["paths"]["checkpoint_dir"], "final_model")
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
