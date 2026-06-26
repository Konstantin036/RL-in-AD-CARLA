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

    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
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

def compute_sleep_duration(elapsed_seconds: float, target_seconds: float) -> float:
    """
    How long to sleep after a step so it takes at least target_seconds of
    real wall-clock time — used for --real-time pacing.

    Pure function: easy to test without CARLA or an actual sleep call.
    """
    return max(0.0, target_seconds - elapsed_seconds)


def run_episode(env, model, episode_num: int, real_time: bool = False) -> EpisodeResult:
    """
    Run exactly one episode with the model's deterministic (mean) action,
    until the environment reports terminated or truncated.

    real_time: if True, pace each step to take at least DELTA_SECONDS
        (carla_env/env.py's fixed simulation timestep, 0.05s) of real
        wall-clock time, so the episode plays out at the same speed you'd
        see in the CARLA spectator view. Default False — without it,
        CARLA ticks as fast as the client/server can process, which is
        faster than real-time on most hardware.
    """
    import time
    from carla_env.env import DELTA_SECONDS

    obs, _info = env.reset()
    total_reward = 0.0
    lateral_distances = []
    step_count = 0
    termination_reason = ""

    terminated = False
    truncated = False
    while not terminated and not truncated:
        step_start = time.time()
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        lateral_distances.append(abs(info["lateral_distance"]))
        step_count += 1
        termination_reason = info["termination_reason"]
        if real_time:
            elapsed = time.time() - step_start
            time.sleep(compute_sleep_duration(elapsed, DELTA_SECONDS))

    mean_lateral = sum(lateral_distances) / len(lateral_distances)

    return EpisodeResult(
        episode_num=episode_num,
        reward=total_reward,
        length=step_count,
        mean_lateral_distance=mean_lateral,
        termination_reason=termination_reason,
    )


def run_evaluation(env, model, n_episodes: int, real_time: bool = False) -> List[EpisodeResult]:
    """
    Run n_episodes evaluation episodes and return the list of results.

    Extension point for future work: a multi-checkpoint comparison script
    can call this once per checkpoint and combine the results, without
    touching run_episode() or compute_summary().
    """
    results = []
    for i in range(1, n_episodes + 1):
        result = run_episode(env, model, episode_num=i, real_time=real_time)
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
    parser.add_argument("--real-time", action="store_true",
                         help="Pace evaluation to match real wall-clock time "
                              "(~0.05s/step), e.g. to watch the car drive at "
                              "normal speed in the CARLA spectator view. "
                              "Default: off (runs as fast as possible).")
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
        results = run_evaluation(env, model, args.episodes, real_time=args.real_time)
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
