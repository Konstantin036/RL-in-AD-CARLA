"""
test_baseline.py
-----------------
Baseline controller unit test (no CARLA required)

compute_baseline_action() is a pure function of the 5D observation array,
so it can be tested offline the same way carla_env/reward.py's functions
are — no CARLA connection needed.

Run from project root:
    python scripts/test_baseline.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.baseline import compute_baseline_action, TARGET_SPEED_NORM


def sep(title=""):
    print(f"\n{'─'*60}")
    if title:
        print(f"  {title}")
        print(f"{'─'*60}")


def test_centered_and_at_speed_drives_straight():
    sep("1. Centered, aligned, at target speed — near-zero steer, near-zero accel")
    obs = np.array([0.0, 0.0, TARGET_SPEED_NORM, 0.0, 1.0], dtype=np.float32)
    action = compute_baseline_action(obs)
    print(f"  action: accel={action[0]:+.3f}  steer={action[1]:+.3f}")
    ok = abs(action[0]) < 0.05 and abs(action[1]) < 0.05
    print(f"  {'✓' if ok else '✗'} PASSED")
    assert ok


def test_off_center_steers_back():
    sep("2. Off to the right of center — steers left (negative)")
    obs = np.array([0.5, 0.0, TARGET_SPEED_NORM, 0.0, 1.0], dtype=np.float32)
    action = compute_baseline_action(obs)
    print(f"  action: accel={action[0]:+.3f}  steer={action[1]:+.3f}")
    ok = action[1] < 0
    print(f"  {'✓' if ok else '✗'} PASSED")
    assert ok


def test_below_target_speed_accelerates():
    sep("3. Below target speed, clear light — positive acceleration")
    obs = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    action = compute_baseline_action(obs)
    print(f"  action: accel={action[0]:+.3f}  steer={action[1]:+.3f}")
    ok = action[0] > 0
    print(f"  {'✓' if ok else '✗'} PASSED")
    assert ok


def test_must_stop_overrides_speed_controller():
    sep("4. Traffic light says must-stop — full brake, even at high speed error")
    # Way below target speed (would normally floor the accelerator), but a
    # red/yellow light applies (obs[4] < 0) — must_stop should win outright.
    obs = np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    action = compute_baseline_action(obs)
    print(f"  action: accel={action[0]:+.3f}  steer={action[1]:+.3f}")
    ok = action[0] == -1.0
    print(f"  {'✓' if ok else '✗'} PASSED")
    assert ok


def test_action_always_in_bounds():
    sep("5. Extreme observations never produce out-of-range actions")
    rng = np.random.RandomState(42)
    ok = True
    for _ in range(1000):
        obs = np.array([
            rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0, 1),
            rng.uniform(-1, 1), rng.choice([-1.0, 1.0]),
        ], dtype=np.float32)
        action = compute_baseline_action(obs)
        if not (-1.0 <= action[0] <= 1.0 and -1.0 <= action[1] <= 1.0):
            ok = False
            break
    print(f"  1000 random observations, all actions in [-1, 1]: {ok}")
    print(f"  {'✓' if ok else '✗'} PASSED")
    assert ok


def main():
    print("=" * 60)
    print("  Baseline Controller Unit Tests")
    print("  (No CARLA connection required)")
    print("=" * 60)

    test_centered_and_at_speed_drives_straight()
    test_off_center_steers_back()
    test_below_target_speed_accelerates()
    test_must_stop_overrides_speed_controller()
    test_action_always_in_bounds()

    print(f"\n{'='*60}")
    print("  All baseline controller tests passed.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
