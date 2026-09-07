"""
baseline.py
-----------
Baseline controller (planned in CLAUDE.md, not tied to any RL algorithm)

Purpose:
    A simple, hand-coded controller for comparison against the trained
    RL agent, and a quick way to visually confirm the route planner and
    traffic-light detection work end-to-end without waiting on training.

    Not a serious driving policy — no gain tuning beyond "reasonable
    starting point", no lookahead beyond what compute_observation()
    already gives it. Its only job is to center on the route waypoint,
    hold a cruise speed, and brake for red/yellow lights.

Why this is useful beyond the demo:
    It exercises the exact same env.step()/observation/reward pipeline a
    trained agent would, just with a hand-coded "brain" instead of a
    learned one — so it's also the natural baseline to report against
    PPO/SAC/DDPG/TD3 results in the thesis.
"""

import os
import sys
import logging
import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────────
# Add project root to path so all carla_env imports work, same as
# agent/train.py and agent/evaluate.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# logger, not print() — rule 8 in CLAUDE.md ("Use logger (not print) inside
# carla_env/ and agent/"). Configured at module level (not inside
# run_baseline_demo()) so importing this module doesn't silently impose a
# logging config on whatever calls it — same pattern as agent/evaluate.py.
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Controller gains ────────────────────────────────────────────────────────
# Hand-picked, not tuned — this is a baseline, not a competitor.
K_LATERAL         = 1.0   # steer correction per unit of normalized lateral error
K_HEADING         = 1.0   # steer correction per unit of normalized heading error
K_SPEED           = 2.0   # accel correction per unit of normalized speed error
TARGET_SPEED_NORM = 0.4   # ~32 km/h out of the 80 km/h normalization range


def compute_baseline_action(obs: np.ndarray) -> np.ndarray:
    """
    Hand-coded control law over the normalized 5D observation.

    Args:
        obs: [lateral_distance, heading_error, speed, steering, traffic_light],
             see carla_env/observation.py for exact ranges.

    Returns:
        np.ndarray shape (2,): [acceleration, steer], both in [-1, 1]
    """
    lat_norm, heading_norm, speed_norm, _steer_prev, tl_norm = obs

    # Steer back toward the route waypoint: push against both lateral
    # offset and heading misalignment.
    steer = -(K_LATERAL * lat_norm + K_HEADING * heading_norm)
    steer = float(np.clip(steer, -1.0, 1.0))

    must_stop = tl_norm < 0
    if must_stop:
        accel = -1.0  # full brake — traffic light overrides the speed controller
    else:
        accel = K_SPEED * (TARGET_SPEED_NORM - speed_norm)
        accel = float(np.clip(accel, -1.0, 1.0))

    return np.array([accel, steer], dtype=np.float32)


# ── Demo runner ───────────────────────────────────────────────────────────────

def run_baseline_demo(
    map_name: str = "Town10HD_Opt",
    host: str = "localhost",
    port: int = 2000,
    num_episodes: int = 3,
    max_steps_per_episode: int = 600,
):
    """
    Drive CarlaLaneKeepingEnv with the baseline controller and print
    telemetry every 20 steps (or immediately on a violation/destination).

    Purpose: visually confirm route planning and traffic-light compliance
    without a trained model — watch the CARLA spectator window (it
    auto-follows the vehicle) while this prints what the code sees.
    """
    from carla_env.env import CarlaLaneKeepingEnv

    env = None
    try:
        env = CarlaLaneKeepingEnv(
            host=host, port=port, map_name=map_name,
            max_steps=max_steps_per_episode, verbose=False,
        )
        logger.info(f"Environment created on map={map_name}")

        for episode in range(1, num_episodes + 1):
            obs, info = env.reset()
            logger.info(
                f"\n{'-'*60}\n"
                f"  Episode {episode}/{num_episodes} — "
                f"route: {info['route_length']} waypoints\n"
                f"{'-'*60}"
            )

            total_reward = 0.0
            step = 0
            for step in range(max_steps_per_episode):
                action = compute_baseline_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                noteworthy = info.get("red_light_violation") or info.get("destination_reached")
                if step % 20 == 0 or noteworthy:
                    logger.info(
                        f"  step={step:4d}  lat={info['lateral_distance']:+.2f}m  "
                        f"spd={info['speed_kmh']:5.1f}km/h  "
                        f"route={info['route_index']}/{info['route_length']}  "
                        f"remaining={info['remaining_distance']:.1f}m  "
                        f"tl={info['traffic_light_state']}"
                        f"{'[STOP]' if info['traffic_light_must_stop'] else ''}  "
                        f"r={reward:+.3f}  total={total_reward:+.2f}"
                    )

                if info.get("red_light_violation"):
                    logger.info(f"  >>> RED LIGHT VIOLATION at step {step}")

                if terminated or truncated:
                    reason = info.get("termination_reason", "unknown")
                    end_type = "TERMINATED" if terminated else "TRUNCATED"
                    logger.info(f"\n  [{end_type}] after {step+1} steps — {reason}")
                    break

            logger.info(f"  Episode {episode} total reward: {total_reward:+.2f}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        if env is not None:
            env.close()
            logger.info("Environment closed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--map", default="Town10HD_Opt",
        help="CARLA map to drive on (default: Town10HD_Opt — Town03 "
             "currently segfaults this project's CARLA install on load)",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=600)
    args = parser.parse_args()
    run_baseline_demo(
        map_name=args.map,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
    )
