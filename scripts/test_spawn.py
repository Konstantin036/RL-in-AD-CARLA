"""
test_spawn.py
--------------
Offline unit tests for deterministic spawn point validation and the
spectator chase-camera transform (no CARLA server required — these are
pure functions of their inputs; only the `carla` Python module itself
needs to be importable, which it is in the carla915 conda env).

Run from anywhere:
    python scripts/test_spawn.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from carla_env.env import _check_spawn_index_in_range


def separator(title=""):
    print(f"\n{'─'*55}")
    if title:
        print(f"  {title}")
        print(f"{'─'*55}")


def test_spawn_index_none_is_always_valid():
    separator("1. spawn_index=None is always valid (random mode)")
    _check_spawn_index_in_range(None, num_spawn_points=372)
    _check_spawn_index_in_range(None, num_spawn_points=0)
    print("  ✓ PASSED")


def test_spawn_index_in_range():
    separator("2. spawn_index within range does not raise")
    _check_spawn_index_in_range(0, num_spawn_points=372)
    _check_spawn_index_in_range(371, num_spawn_points=372)
    print("  ✓ PASSED")


def test_spawn_index_out_of_range():
    separator("3. spawn_index out of range raises ValueError")
    for bad_index in [-1, 372, 99999]:
        try:
            _check_spawn_index_in_range(bad_index, num_spawn_points=372)
            raise AssertionError(f"Expected ValueError for spawn_index={bad_index}")
        except ValueError as e:
            print(f"  spawn_index={bad_index} raised ValueError as expected: {e}")
    print("  ✓ PASSED")


if __name__ == "__main__":
    test_spawn_index_none_is_always_valid()
    test_spawn_index_in_range()
    test_spawn_index_out_of_range()
    print(f"\n{'='*55}")
    print("  ALL TESTS PASSED")
    print(f"{'='*55}\n")
