"""
test_env.py
-----------
Phase 6 — Full Environment Integration Test

Purpose:
    Run the complete CarlaLaneKeepingEnv with random actions for a few
    episodes. This confirms all the pieces work together:
        - CARLA connection and sync mode
        - Vehicle spawn and collision sensor
        - Observation computation each step
        - Reward and termination logic
        - Clean reset between episodes

How to run:
    1. Launch CARLA:  ./CarlaUE4.sh -quality-level=Low -fps=20
    2. Run this:      python scripts/test_env.py

What to watch:
    - Terminal prints episode summaries
    - CARLA window shows vehicle spawning and moving randomly
    - Episodes end with collision/off-road (random actions are bad)
    - Each reset spawns a fresh vehicle cleanly
"""

import sys
import os
import logging
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Configure logging so we see INFO messages from env.py
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

from carla_env.env import CarlaLaneKeepingEnv


def run_env_test(num_episodes: int = 3, max_steps_per_episode: int = 200):
    """
    Run the environment for a fixed number of episodes with random actions.

    Args:
        num_episodes:           how many episodes to run
        max_steps_per_episode:  step limit per episode (overrides env default)
    """

    print("=" * 60)
    print("  Phase 6 — Environment Integration Test")
    print("  Random actions, watching episode structure")
    print("=" * 60)

    env = None

    try:
        # ── Create environment ─────────────────────────────────────────────────
        print("\n[SETUP] Creating CarlaLaneKeepingEnv ...")
        env = CarlaLaneKeepingEnv(
            host      = "localhost",
            port      = 2000,
            map_name  = "Town03",
            max_steps = max_steps_per_episode,
            verbose   = False,   # set True to see every step printed
        )
        print("[SETUP] Environment created successfully.")
        print(f"[SETUP] Observation space: {env.observation_space}")
        print(f"[SETUP] Action space:      {env.action_space}")

        # ── Run episodes ───────────────────────────────────────────────────────
        for episode in range(1, num_episodes + 1):
            print(f"\n{'─'*60}")
            print(f"  Episode {episode} / {num_episodes}")
            print(f"{'─'*60}")

            # reset() starts a new episode and returns the first observation
            obs, info = env.reset()

            print(f"  Initial obs:  {obs}")
            print(f"  Initial info: lat={info['lateral_distance']:+.3f}m  "
                  f"hdg={float(np.degrees(info['heading_error'])):.1f}°  "
                  f"spd={info['speed_kmh']:.1f}km/h")

            # Step tracking
            total_reward  = 0.0
            step_rewards  = []
            term_reason   = "running"

            for step in range(max_steps_per_episode):

                # Sample a random action from the action space
                # In training, PPO replaces this with its policy output
                action = env.action_space.sample()

                # Step the environment
                obs, reward, terminated, truncated, info = env.step(action)

                total_reward += reward
                step_rewards.append(reward)

                # Print every 20 steps
                if step % 20 == 0:
                    print(
                        f"  step={step:4d}  "
                        f"lat={info['lateral_distance']:+.3f}m  "
                        f"spd={info['speed_kmh']:5.1f}km/h  "
                        f"r={reward:+.3f}  "
                        f"total={total_reward:+.2f}"
                    )

                # Episode ended
                if terminated or truncated:
                    term_reason = info.get("termination_reason", "unknown")
                    end_type    = "TERMINATED" if terminated else "TRUNCATED"
                    print(f"\n  [{end_type}] after {step+1} steps — {term_reason}")
                    break

            # Episode summary
            print(f"\n  ── Episode {episode} Summary ──")
            print(f"  Steps:        {step+1}")
            print(f"  Total reward: {total_reward:+.2f}")
            print(f"  Mean reward:  {np.mean(step_rewards):+.4f}")
            print(f"  Min reward:   {np.min(step_rewards):+.4f}")
            print(f"  Max reward:   {np.max(step_rewards):+.4f}")
            print(f"  End reason:   {term_reason}")

        print(f"\n{'='*60}")
        print("  Integration test complete.")
        print("  The environment works end-to-end.")
        print("  Next: Phase 7 — Manual driving script.")
        print(f"{'='*60}\n")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Always close the environment — this restores async mode
        if env is not None:
            env.close()
            print("[INFO] Environment closed cleanly.")


if __name__ == "__main__":
    run_env_test(
        num_episodes          = 3,
        max_steps_per_episode = 200,
    )
