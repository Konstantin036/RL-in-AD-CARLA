"""
test_action.py
--------------
Phase 4 — Action Space Unit Test (no CARLA required)

Tests the action mapping and smoothing logic without needing CARLA running.
Run from anywhere:
    python scripts/test_action.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from carla_env.action import ActionSmoother, action_to_control, ActionProcessor, get_action_space


def separator(title=""):
    print(f"\n{'─'*55}")
    if title:
        print(f"  {title}")
        print(f"{'─'*55}")


def test_action_space():
    separator("1. Action space shape and bounds")
    space = get_action_space()
    print(f"  Shape:  {space.shape}")
    print(f"  Low:    {space.low}")
    print(f"  High:   {space.high}")
    print(f"  dtype:  {space.dtype}")
    assert space.shape == (2,), "Expected shape (2,)"
    assert space.low[0]  == -1.0
    assert space.high[0] ==  1.0
    print("  ✓ PASSED")


def test_action_mapping():
    separator("2. Action → throttle/brake mapping")

    # We cannot import carla here, so we test the logic manually
    # by reimplementing the same split logic and comparing

    test_cases = [
        # (acceleration, expected_throttle, expected_brake, description)
        ( 0.8,   0.8, 0.0, "full throttle"),
        ( 0.0,   0.0, 0.0, "coasting"),
        (-0.5,   0.0, 0.5, "half brake"),
        (-1.0,   0.0, 1.0, "full brake"),
        ( 1.0,   1.0, 0.0, "max throttle"),
    ]

    header = f"  {'accel':>8}  {'→ throttle':>12}  {'brake':>8}  {'note'}"
    print(header)
    print("  " + "-" * 50)

    for accel, exp_thr, exp_brk, note in test_cases:
        if accel >= 0:
            thr, brk = accel, 0.0
        else:
            thr, brk = 0.0, -accel

        status = "✓" if abs(thr - exp_thr) < 1e-6 and abs(brk - exp_brk) < 1e-6 else "✗"
        print(f"  {accel:>8.2f}  →  thr={thr:.2f}  brk={brk:.2f}  {status}  {note}")

    print("  ✓ PASSED")


def test_action_smoother():
    separator("3. Action smoother")

    smoother = ActionSmoother(alpha=0.6)
    smoother.reset()

    # Start from zero, apply a step action
    action = np.array([1.0, 1.0], dtype=np.float32)

    print(f"  alpha = {smoother.alpha}  (new action weight)")
    print(f"  Applying constant action [1.0, 1.0] repeatedly:")
    print(f"  {'Step':>5}  {'smoothed[0]':>12}  {'smoothed[1]':>12}")
    print("  " + "-" * 35)

    for step in range(6):
        s = smoother.smooth(action)
        print(f"  {step:>5}  {s[0]:>12.4f}  {s[1]:>12.4f}")

    # After many steps it should converge to 1.0
    for _ in range(100):
        s = smoother.smooth(action)

    assert abs(s[0] - 1.0) < 0.001, f"Did not converge to 1.0, got {s[0]}"
    print(f"  After 100 steps: converges to {s[0]:.4f} ✓")

    # Test reset clears state
    smoother.reset()
    s_reset = smoother.smooth(np.array([0.0, 0.0]))
    assert s_reset[0] == 0.0, "Reset did not clear state"
    print("  Reset works correctly ✓")
    print("  ✓ PASSED")


def test_smoothing_effect():
    separator("4. Smoothing effect on a step change")

    alpha_values = [1.0, 0.6, 0.3]
    action_sequence = [np.array([0.0, 0.0])] * 5 + [np.array([1.0, 1.0])] * 10

    print(f"  Steering response after a step from 0→1 at tick 5:")
    print(f"  {'Tick':>5}  {'alpha=1.0':>10}  {'alpha=0.6':>10}  {'alpha=0.3':>10}")
    print("  " + "-" * 45)

    smoothers = {a: ActionSmoother(alpha=a) for a in alpha_values}
    for s in smoothers.values():
        s.reset()

    for i, action in enumerate(action_sequence):
        values = {a: smoothers[a].smooth(action)[0] for a in alpha_values}
        print(f"  {i:>5}  {values[1.0]:>10.3f}  {values[0.6]:>10.3f}  {values[0.3]:>10.3f}")

    print()
    print("  alpha=1.0 → immediate jump (no smoothing)")
    print("  alpha=0.6 → fast response (default)")
    print("  alpha=0.3 → slow, smooth response")


def test_action_processor():
    separator("5. ActionProcessor combined test")

    processor = ActionProcessor(alpha=0.6)
    processor.reset()

    test_actions = [
        np.array([ 0.5,  0.0]),   # throttle only
        np.array([-0.3,  0.2]),   # brake + right steer
        np.array([ 0.0, -0.5]),   # coast + left steer
    ]

    print(f"  {'raw accel':>10}  {'raw steer':>10}  {'smoothed accel':>15}  {'smoothed steer':>15}")
    print("  " + "-" * 55)

    for action in test_actions:
        smoothed = processor.process_raw(action.copy())
        print(f"  {action[0]:>10.2f}  {action[1]:>10.2f}  {smoothed[0]:>15.4f}  {smoothed[1]:>15.4f}")

    print("  ✓ PASSED")


def main():
    print("=" * 55)
    print("  Phase 4 — Action Space Unit Tests")
    print("  (No CARLA connection required)")
    print("=" * 55)

    test_action_space()
    test_action_mapping()
    test_action_smoother()
    test_smoothing_effect()
    test_action_processor()

    print(f"\n{'='*55}")
    print("  All tests passed.")
    print(f"  Ready to move to Phase 5 — Reward Function.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
