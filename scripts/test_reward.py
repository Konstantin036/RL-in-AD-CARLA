"""
test_reward.py
--------------
Phase 5 — Reward Function Unit Test (no CARLA required)

Tests every reward component and termination condition.
Run from project root:
    python scripts/test_reward.py
"""

import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from carla_env.reward import (
    RewardConfig,
    RewardInfo,
    compute_centering_reward,
    compute_speed_reward,
    compute_heading_reward,
    compute_smoothness_reward,
    compute_reward,
    check_termination,
)
from carla_env.observation import ObservationData


def sep(title=""):
    print(f"\n{'─'*60}")
    if title:
        print(f"  {title}")
        print(f"{'─'*60}")


# ── Test 1: Centering reward ───────────────────────────────────────────────────

def test_centering():
    sep("1. Centering reward  r = 1 - |lat| / max_lat")
    cases = [
        (0.0,   1.000, "perfect center"),
        (1.75,  0.500, "half lane off"),
        (3.5,   0.000, "at lane edge"),
        (4.0,   0.000, "beyond lane edge (clipped)"),
        (-1.75, 0.500, "left side symmetry"),
    ]
    print(f"  {'lateral(m)':>12}  {'expected':>10}  {'got':>10}  {'':>6}")
    for lat, expected, note in cases:
        got = compute_centering_reward(lat, max_lateral_m=3.5)
        ok = abs(got - expected) < 0.001
        print(f"  {lat:>12.2f}  {expected:>10.3f}  {got:>10.3f}  {'✓' if ok else '✗'}  {note}")
    print("  ✓ PASSED")


# ── Test 2: Speed reward ───────────────────────────────────────────────────────

def test_speed():
    sep("2. Speed reward  r = exp(-((v-target)^2) / (2*sigma^2))")
    target = 30.0
    sigma  = 10.0
    cases = [
        (30.0, 1.000, "at target speed"),
        (20.0, 0.607, "10 km/h slow"),
        (40.0, 0.607, "10 km/h fast"),
        (0.0,  0.011, "standing still"),
        (60.0, 0.011, "very fast"),
    ]
    print(f"  {'speed(km/h)':>12}  {'expected':>10}  {'got':>10}  {'':>6}")
    for spd, expected, note in cases:
        got = compute_speed_reward(spd, target, sigma)
        ok = abs(got - expected) < 0.005
        print(f"  {spd:>12.1f}  {expected:>10.3f}  {got:>10.3f}  {'✓' if ok else '✗'}  {note}")
    print("  ✓ PASSED")


# ── Test 3: Heading reward ─────────────────────────────────────────────────────

def test_heading():
    sep("3. Heading reward  r = 1 - |heading| / pi")
    cases = [
        (0.0,            1.000, "perfectly aligned"),
        (math.pi / 4,    0.750, "45 degrees off"),
        (math.pi / 2,    0.500, "90 degrees off"),
        (math.pi,        0.000, "pointing backwards"),
        (-math.pi / 4,   0.750, "symmetry check"),
    ]
    print(f"  {'heading(°)':>12}  {'expected':>10}  {'got':>10}  {'':>6}")
    for hdg, expected, note in cases:
        got = compute_heading_reward(hdg)
        ok = abs(got - expected) < 0.005
        hdg_deg = math.degrees(hdg)
        print(f"  {hdg_deg:>12.1f}  {expected:>10.3f}  {got:>10.3f}  {'✓' if ok else '✗'}  {note}")
    print("  ✓ PASSED")


# ── Test: Smoothness reward ────────────────────────────────────────────────────

def test_smoothness():
    sep("3b. Smoothness reward  r = 1 - sum(|action_delta|) / 4.0")
    cases = [
        # (action_delta, expected, description)
        (np.array([0.0,  0.0]), 1.0, "no change — perfectly smooth"),
        (np.array([2.0,  2.0]), 0.0, "both dims flipped full range"),
        (np.array([0.5, -0.3]), 0.8, "moderate change"),
        (np.array([1.0,  0.0]), 0.75, "one dim flipped halfway"),
        (np.array([3.0,  3.0]), 0.0, "beyond max — clipped at 0.0"),
    ]
    for action_delta, expected, desc in cases:
        result = compute_smoothness_reward(action_delta)
        status = "✓" if abs(result - expected) < 1e-6 else "✗"
        print(f"  delta={action_delta}  expected={expected:.3f}  got={result:.3f}  {status}  {desc}")
        assert abs(result - expected) < 1e-6, f"{desc}: expected {expected}, got {result}"
    print("  ✓ PASSED")


# ── Test 4: Full reward at representative states ───────────────────────────────

def test_full_reward():
    sep("4. Full reward at representative states")
    cfg = RewardConfig()

    scenarios = [
        # (lat_m, hdg_deg, spd_kmh, is_terminal, description)
        (0.0,    0.0,   30.0,  False, "perfect — centered, aligned, target speed"),
        (0.0,    0.0,    0.0,  False, "centered + aligned but standing still"),
        (1.0,    5.0,   30.0,  False, "slightly off center, slight heading error"),
        (3.0,   20.0,   15.0,  False, "near edge, heading off, slow"),
        (3.5,   45.0,   10.0,  True,  "off road — terminal"),
    ]

    print(f"  {'lat':>5} {'hdg°':>6} {'spd':>5}  {'total':>7}  "
          f"{'center':>7} {'speed':>7} {'heading':>8}  description")
    print("  " + "-" * 75)

    for lat, hdg_deg, spd, is_term, desc in scenarios:
        obs = ObservationData(
            lateral_distance_m=lat,
            heading_error_rad=math.radians(hdg_deg),
            speed_kmh=spd,
            steering=0.0,
        )
        reward, info = compute_reward(obs, is_terminal=is_term, cfg=cfg)
        term_marker = " ← TERMINAL" if is_term else ""
        print(f"  {lat:>5.1f} {hdg_deg:>6.1f} {spd:>5.1f}  "
              f"{info.total:>+7.3f}  "
              f"{info.r_centering:>7.3f} "
              f"{info.r_speed:>7.3f} "
              f"{info.r_heading:>8.3f}  "
              f"{desc}{term_marker}")

    print("\n  Best possible reward (no step penalty):")
    obs_best = ObservationData(0.0, 0.0, 30.0, 0.0)
    r_best, _ = compute_reward(obs_best, False, cfg)
    print(f"    w_center({cfg.w_center}) * 1.0 "
          f"+ w_speed({cfg.w_speed}) * 1.0 "
          f"+ w_heading({cfg.w_heading}) * 1.0 "
          f"+ step_penalty({cfg.step_penalty})")
    print(f"    = {r_best:+.3f}")
    print("  ✓ PASSED")


# ── Test 5: Termination conditions ────────────────────────────────────────────

def test_termination():
    sep("5. Termination conditions")

    def make_obs(lat=0.0, hdg_deg=0.0, spd=20.0):
        return ObservationData(lat, math.radians(hdg_deg), spd, 0.0)

    cases = [
        # (obs_kwargs, collision, step, max_steps, expected_term, expected_trunc, note)
        (dict(lat=0.0, hdg_deg=0.0),    False, 100,  1000, False, False, "normal step"),
        (dict(lat=0.0, hdg_deg=0.0),    True,  100,  1000, True,  False, "collision"),
        (dict(lat=4.0, hdg_deg=0.0),    False, 100,  1000, True,  False, "off road"),
        (dict(lat=0.0, hdg_deg=95.0),   False, 100,  1000, True,  False, "wrong heading"),
        (dict(lat=0.0, hdg_deg=0.0),    False, 1000, 1000, False, True,  "timeout"),
    ]

    print(f"  {'collision':>10}  {'lat':>5}  {'hdg°':>6}  "
          f"{'step':>5}  {'term':>6}  {'trunc':>6}  note")
    print("  " + "-" * 62)

    for obs_kw, collision, step, max_steps, exp_term, exp_trunc, note in cases:
        obs = make_obs(**obs_kw)
        term, trunc, reason = check_termination(
            obs, collision, step, max_steps
        )
        ok_term  = (term  == exp_term)
        ok_trunc = (trunc == exp_trunc)
        ok = ok_term and ok_trunc
        lat_val = obs_kw.get('lat', 0.0)
        hdg_val = obs_kw.get('hdg_deg', 0.0)
        print(f"  {str(collision):>10}  {lat_val:>5.1f}  {hdg_val:>6.1f}  "
              f"{step:>5}  {str(term):>6}  {str(trunc):>6}  "
              f"{'✓' if ok else '✗'}  {note}"
              f"{' [' + reason + ']' if reason else ''}")

    print("  ✓ PASSED")


# ── Test 6: Reward range sanity check ─────────────────────────────────────────

def test_reward_range():
    sep("6. Reward range sanity check")
    cfg = RewardConfig()
    import random
    random.seed(42)

    min_r, max_r = float('inf'), float('-inf')
    n_samples = 10000

    for _ in range(n_samples):
        obs = ObservationData(
            lateral_distance_m=random.uniform(-4.0,  4.0),
            heading_error_rad =random.uniform(-math.pi, math.pi),
            speed_kmh         =random.uniform(0.0,   80.0),
            steering          =random.uniform(-1.0,   1.0),
        )
        is_terminal = random.random() < 0.05
        r, _ = compute_reward(obs, is_terminal, cfg)
        min_r = min(min_r, r)
        max_r = max(max_r, r)

    print(f"  Sampled {n_samples} random states")
    print(f"  Min reward: {min_r:+.3f}")
    print(f"  Max reward: {max_r:+.3f}")
    print(f"  Expected:   min ≈ {cfg.terminal_penalty + cfg.step_penalty:.1f},  "
          f"max ≈ {cfg.w_center + cfg.w_speed + cfg.w_heading + cfg.step_penalty:.2f}")
    assert min_r >= cfg.terminal_penalty - 0.1, "Min reward unexpectedly low"
    assert max_r <= cfg.w_center + cfg.w_speed + cfg.w_heading + 0.1, "Max reward unexpectedly high"
    print("  ✓ PASSED")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Phase 5 — Reward Function Unit Tests")
    print("  (No CARLA connection required)")
    print("=" * 60)

    test_centering()
    test_speed()
    test_heading()
    test_smoothness()
    test_full_reward()
    test_termination()
    test_reward_range()

    print(f"\n{'='*60}")
    print("  All reward tests passed.")
    print("  Ready to move to Phase 6 — Gym Environment.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
