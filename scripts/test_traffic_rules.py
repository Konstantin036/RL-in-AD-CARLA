"""
test_traffic_rules.py
----------------------
Traffic rules — RedLightViolationDetector unit test (no CARLA required)

get_traffic_light_affordance() needs a live carla.Vehicle and can only be
exercised with CARLA running (see scripts/test_env.py). But the actual
violation logic — RedLightViolationDetector — only depends on the plain
TrafficLightAffordance dataclass and a speed number, so it can be driven
entirely with synthetic data here, the same way scripts/test_reward.py
tests StallDetector offline.

Run from project root:
    python scripts/test_traffic_rules.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from carla_env.reward import RewardConfig
from carla_env.traffic_rules import TrafficLightAffordance, RedLightViolationDetector


def sep(title=""):
    print(f"\n{'─'*60}")
    if title:
        print(f"  {title}")
        print(f"{'─'*60}")


def run_scenario(detector, steps):
    """
    steps: list of (is_at_light, must_stop, speed_kmh) tuples, one per tick.
    Returns the list of violation flags returned by update(), one per tick.
    """
    detector.reset()
    results = []
    for is_at_light, must_stop, speed_kmh in steps:
        affordance = TrafficLightAffordance(
            is_at_light=is_at_light, must_stop=must_stop, state="n/a"
        )
        results.append(detector.update(affordance, speed_kmh))
    return results


# ── Test 1: drives through green — never flagged ──────────────────────────────

def test_green_never_violates():
    sep("1. Driving through a green light never violates")
    cfg = RewardConfig()
    detector = RedLightViolationDetector(cfg)

    # Approaches, enters the (green) trigger zone at speed, exits at speed.
    steps = [
        (False, False, 30.0),
        (True,  False, 25.0),
        (True,  False, 22.0),
        (False, False, 24.0),
    ]
    results = run_scenario(detector, steps)
    ok = not any(results)
    print(f"  violations: {results}")
    print(f"  {'✓' if ok else '✗'} PASSED")
    assert ok


# ── Test 2: drives through red without stopping — violation on exit ───────────

def test_drives_through_red():
    sep("2. Driving through red without stopping — flagged on exit")
    cfg = RewardConfig()
    detector = RedLightViolationDetector(cfg)

    steps = [
        (False, False, 30.0),  # approaching, still green/far
        (True,  True,  28.0),  # enters zone, light already red, doesn't brake
        (True,  True,  26.0),  # still moving fast through the zone
        (False, True,  24.0),  # exits zone — still "must_stop" and still moving fast
    ]
    results = run_scenario(detector, steps)
    print(f"  violations: {results}")
    ok = results == [False, False, False, True]
    print(f"  {'✓' if ok else '✗'} PASSED")
    assert ok


# ── Test 3: stops for red, waits for green, then proceeds — no violation ──────

def test_stops_and_waits():
    sep("3. Stopping for red and waiting for green — no violation")
    cfg = RewardConfig()
    detector = RedLightViolationDetector(cfg)

    steps = [
        (False, False, 30.0),  # approaching
        (True,  True,   3.0),  # enters zone, brakes down below stop threshold
        (True,  True,   0.0),  # stopped, waiting
        (True,  False,  0.0),  # light turns green, still stationary in zone
        (False, False, 20.0),  # accelerates through and exits
    ]
    results = run_scenario(detector, steps)
    ok = not any(results)
    print(f"  violations: {results}")
    print(f"  {'✓' if ok else '✗'} PASSED")
    assert ok


# ── Test 4: disabled via config — never flags anything ─────────────────────────

def test_disabled_via_config():
    sep("4. red_light_enabled=False disables detection entirely")
    cfg = RewardConfig(red_light_enabled=False)
    detector = RedLightViolationDetector(cfg)

    # Same as test 2 (a clear violation), but should never fire when disabled.
    steps = [
        (False, False, 30.0),
        (True,  True,  28.0),
        (True,  True,  26.0),
        (False, True,  24.0),
    ]
    results = run_scenario(detector, steps)
    ok = not any(results)
    print(f"  violations: {results}")
    print(f"  {'✓' if ok else '✗'} PASSED")
    assert ok


# ── Test 5: reset() clears state between episodes ──────────────────────────────

def test_reset_clears_state():
    sep("5. reset() clears state between episodes")
    cfg = RewardConfig()
    detector = RedLightViolationDetector(cfg)

    # Leave the detector mid-violation-setup (in a red zone, moving fast),
    # then reset — the next episode's first update() must not immediately
    # fire just because of leftover state from the previous episode.
    detector.update(TrafficLightAffordance(True, True, "n/a"), 28.0)
    detector.reset()

    affordance = TrafficLightAffordance(is_at_light=False, must_stop=False, state="n/a")
    result = detector.update(affordance, 20.0)
    ok = result is False
    print(f"  post-reset first update violation: {result}")
    print(f"  {'✓' if ok else '✗'} PASSED")
    assert ok


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Traffic Rules — RedLightViolationDetector Unit Tests")
    print("  (No CARLA connection required)")
    print("=" * 60)

    test_green_never_violates()
    test_drives_through_red()
    test_stops_and_waits()
    test_disabled_via_config()
    test_reset_clears_state()

    print(f"\n{'='*60}")
    print("  All traffic rule tests passed.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
