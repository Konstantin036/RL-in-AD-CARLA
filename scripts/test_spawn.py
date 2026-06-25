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

import carla

from carla_env.env import (
    _check_spawn_index_in_range,
    _compute_spectator_transform,
    _compute_effective_spawn_index,
)


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


def test_spectator_transform_is_behind_and_above():
    separator("4. Spectator transform is behind and above the vehicle")
    vehicle_transform = carla.Transform(
        carla.Location(x=0.0, y=0.0, z=0.0),
        carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0),
    )
    cam_transform = _compute_spectator_transform(
        vehicle_transform, distance_m=8.0, height_m=4.0, pitch_deg=-15.0,
    )
    print(f"  vehicle location: {vehicle_transform.location}")
    print(f"  camera location:  {cam_transform.location}")

    assert abs(cam_transform.location.z - 4.0) < 1e-3, \
        "Camera should be 4m above the vehicle"

    forward = vehicle_transform.get_forward_vector()
    displacement_x = cam_transform.location.x - vehicle_transform.location.x
    displacement_y = cam_transform.location.y - vehicle_transform.location.y
    dot = displacement_x * forward.x + displacement_y * forward.y
    assert dot < 0, \
        "Camera should be displaced behind the vehicle (opposite its forward vector)"

    assert cam_transform.rotation.pitch == -15.0
    assert cam_transform.rotation.yaw == vehicle_transform.rotation.yaw
    print("  ✓ PASSED")


def test_spectator_transform_distance():
    separator("5. Spectator transform respects the configured distance")
    vehicle_transform = carla.Transform(
        carla.Location(x=0.0, y=0.0, z=0.0),
        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
    )
    cam_transform = _compute_spectator_transform(
        vehicle_transform, distance_m=10.0, height_m=0.0, pitch_deg=0.0,
    )
    horizontal_distance = (
        cam_transform.location.x ** 2 + cam_transform.location.y ** 2
    ) ** 0.5
    print(f"  horizontal distance from vehicle: {horizontal_distance:.2f}m")
    assert abs(horizontal_distance - 10.0) < 1e-2
    print("  ✓ PASSED")


def test_effective_spawn_index_offset_and_wraparound():
    separator("6. _compute_effective_spawn_index() offset and wraparound")
    # No offset: unchanged
    assert _compute_effective_spawn_index(0, 0, 372) == 0
    assert _compute_effective_spawn_index(5, 0, 372) == 5
    # Offset within range: simple addition
    assert _compute_effective_spawn_index(0, 1, 372) == 1
    assert _compute_effective_spawn_index(5, 3, 372) == 8
    # Offset wraps around at the end of the list
    assert _compute_effective_spawn_index(371, 1, 372) == 0
    print("  train (offset=0) at spawn_index=0  ->", _compute_effective_spawn_index(0, 0, 372))
    print("  eval  (offset=1) at spawn_index=0  ->", _compute_effective_spawn_index(0, 1, 372))
    print("  eval  (offset=1) at spawn_index=371 (wraps) ->", _compute_effective_spawn_index(371, 1, 372))
    print("  ✓ PASSED")


if __name__ == "__main__":
    test_spawn_index_none_is_always_valid()
    test_spawn_index_in_range()
    test_spawn_index_out_of_range()
    test_spectator_transform_is_behind_and_above()
    test_spectator_transform_distance()
    test_effective_spawn_index_offset_and_wraparound()
    print(f"\n{'='*55}")
    print("  ALL TESTS PASSED")
    print(f"{'='*55}\n")
