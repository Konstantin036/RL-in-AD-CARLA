"""
callbacks.py
------------
Phase 8 — Training Callbacks

Purpose:
    Custom Stable-Baselines3 callbacks that run during PPO training to:
        1. Log episode metrics to TensorBoard and CSV
        2. Save checkpoints periodically
        3. Save the best model based on mean episode reward
        4. Print a human-readable training summary

What is a callback?
    SB3 calls your callback at fixed points during training:
        on_step()         — after every environment step
        on_rollout_end()  — after collecting n_steps of experience
        on_training_end() — when total_timesteps is reached

    You use callbacks to inject custom logic without modifying SB3's
    training loop. Think of them as hooks.

Why not just use SB3's built-in logging?
    SB3 logs reward/episode_length by default, but we want richer data:
    - Reward component breakdown (centering, speed, heading)
    - Termination reason distribution (collision vs off_road vs timeout)
    - Custom metrics for your thesis plots

Thesis note:
    The CSV file produced by EpisodeLoggerCallback is what you will use
    to generate your thesis learning curves. Every episode's reward,
    length, and termination reason is recorded in a single file.
"""

import os
import csv
import time
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


# ── Episode logger ─────────────────────────────────────────────────────────────

class EpisodeLoggerCallback(BaseCallback):
    """
    Logs episode-level metrics after every episode ends.

    Writes to:
        - TensorBoard (viewable with `tensorboard --logdir results/logs`)
        - CSV file at results/logs/episode_log.csv

    Metrics logged per episode:
        - episode_reward      total undiscounted return
        - episode_length      number of steps
        - termination_reason  collision / off_road / wrong_heading / timeout
        - mean lateral distance
        - mean speed
        - mean smoothness (1.0 = perfectly smooth control, 0.0 = max jitter)

    Usage:
        callback = EpisodeLoggerCallback(log_dir="results/logs")
    """

    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir  = log_dir
        self.csv_path = os.path.join(log_dir, "episode_log.csv")

        # Running episode accumulators
        self._ep_reward       = 0.0
        self._ep_steps        = 0
        self._ep_lat_dists    = []
        self._ep_speeds       = []
        self._ep_smoothness   = []
        self._ep_count        = 0
        self._training_start  = None

        # Termination reason counters (for logging distribution)
        self._term_counts = {
            "collision":     0,
            "off_road":      0,
            "wrong_heading": 0,
            "timeout":       0,
            "other":         0,
        }

    def _on_training_start(self) -> None:
        """Called once before training begins. Set up CSV file."""
        os.makedirs(self.log_dir, exist_ok=True)
        self._training_start = time.time()

        # Write CSV header
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
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

        if self.verbose:
            print(f"[Logger] CSV log: {self.csv_path}")

    def _on_step(self) -> bool:
        """
        Called after every environment step.
        Accumulates per-step metrics and detects episode boundaries.

        Returns True to continue training, False to stop early.
        """
        # Read info dict from the environment
        # In SB3, self.locals["infos"] is a list (one per env in VecEnv)
        infos = self.locals.get("infos", [{}])
        dones = self.locals.get("dones", [False])

        for info, done in zip(infos, dones):
            # Accumulate step-level metrics
            self._ep_reward += info.get("reward_total",     0.0)
            self._ep_steps  += 1

            lat = info.get("lateral_distance",  0.0)
            spd = info.get("speed_kmh",         0.0)
            smooth = info.get("reward_smoothness", 0.0)
            self._ep_lat_dists.append(abs(lat))
            self._ep_speeds.append(spd)
            self._ep_smoothness.append(smooth)

            # Episode ended
            if done:
                self._ep_count += 1
                reason = info.get("termination_reason", "other") or "timeout"

                # Count termination reasons
                key = reason if reason in self._term_counts else "other"
                self._term_counts[key] += 1

                # Compute episode summary stats
                mean_lat     = float(np.mean(self._ep_lat_dists))  if self._ep_lat_dists  else 0.0
                mean_spd     = float(np.mean(self._ep_speeds))     if self._ep_speeds     else 0.0
                mean_smooth  = float(np.mean(self._ep_smoothness)) if self._ep_smoothness else 0.0
                elapsed      = time.time() - self._training_start

                # ── Log to TensorBoard ─────────────────────────────────────────
                self.logger.record("episode/reward",         self._ep_reward)
                self.logger.record("episode/length",         self._ep_steps)
                self.logger.record("episode/mean_lat_dist",  mean_lat)
                self.logger.record("episode/mean_speed",     mean_spd)
                self.logger.record("episode/mean_smoothness",mean_smooth)
                self.logger.record("episode/count",          self._ep_count)

                # Log termination reason as separate scalars
                # (easier to plot than a string)
                for r, count in self._term_counts.items():
                    self.logger.record(f"termination/{r}", count)

                # ── Log to CSV ─────────────────────────────────────────────────
                with open(self.csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
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

                # ── Print to terminal ──────────────────────────────────────────
                if self.verbose >= 1 or self._ep_count % 10 == 0:
                    print(
                        f"[Ep {self._ep_count:4d}] "
                        f"steps={self._ep_steps:4d}  "
                        f"reward={self._ep_reward:+7.1f}  "
                        f"lat={mean_lat:.2f}m  "
                        f"spd={mean_spd:.1f}km/h  "
                        f"end={reason}"
                    )

                # Reset accumulators for next episode
                self._ep_reward      = 0.0
                self._ep_steps       = 0
                self._ep_lat_dists   = []
                self._ep_speeds      = []
                self._ep_smoothness  = []

        return True   # True = keep training

    def _on_training_end(self) -> None:
        """Print final summary when training completes."""
        elapsed = time.time() - self._training_start
        print(f"\n[Logger] Training complete.")
        print(f"[Logger] Total episodes:  {self._ep_count}")
        print(f"[Logger] Total timesteps: {self.num_timesteps}")
        print(f"[Logger] Elapsed time:    {elapsed/60:.1f} minutes")
        print(f"[Logger] Termination breakdown:")
        for reason, count in self._term_counts.items():
            pct = 100 * count / max(self._ep_count, 1)
            print(f"           {reason:20s}: {count:4d}  ({pct:.1f}%)")
        print(f"[Logger] CSV saved to: {self.csv_path}")
