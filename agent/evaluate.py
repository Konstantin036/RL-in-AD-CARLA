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
