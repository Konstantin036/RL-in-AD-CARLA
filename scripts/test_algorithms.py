"""
test_algorithms.py
-------------------
Offline unit tests for agent/algorithms.py (no CARLA required).

Run from anywhere:
    python scripts/test_algorithms.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.algorithms import ALGORITHMS, get_run_prefix


def separator(title=""):
    print(f"\n{'─'*55}")
    if title:
        print(f"  {title}")
        print(f"{'─'*55}")


def test_registry_contents():
    separator("1. Registry contains exactly the supported algorithms")
    expected = {"ppo", "sac", "ddpg", "td3"}
    assert set(ALGORITHMS.keys()) == expected, (
        f"Expected {expected}, got {set(ALGORITHMS.keys())}"
    )
    assert "dqn" not in ALGORITHMS, "DQN must not be registered (see spec)"
    print(f"  Registered algorithms: {sorted(ALGORITHMS.keys())}")
    print("  ✓ PASSED")


def test_run_prefix():
    separator("2. get_run_prefix() naming")
    assert get_run_prefix("ppo") == "ppo_lane_keeping"
    assert get_run_prefix("sac") == "sac_lane_keeping"
    assert get_run_prefix("ddpg") == "ddpg_lane_keeping"
    assert get_run_prefix("td3") == "td3_lane_keeping"
    print("  ppo  ->", get_run_prefix("ppo"))
    print("  sac  ->", get_run_prefix("sac"))
    print("  ddpg ->", get_run_prefix("ddpg"))
    print("  td3  ->", get_run_prefix("td3"))
    print("  ✓ PASSED")


def test_run_prefix_unknown_algo():
    separator("3. get_run_prefix() rejects unknown algorithms")
    try:
        get_run_prefix("dqn")
        raise AssertionError("Expected ValueError for unknown algorithm")
    except ValueError as e:
        print(f"  Raised ValueError as expected: {e}")
    print("  ✓ PASSED")


if __name__ == "__main__":
    test_registry_contents()
    test_run_prefix()
    test_run_prefix_unknown_algo()
    print(f"\n{'='*55}")
    print("  ALL TESTS PASSED")
    print(f"{'='*55}\n")
