"""
test_evaluate.py
-----------------
Offline unit tests for agent/evaluate.py (no CARLA required).

Run from project root:
    python scripts/test_evaluate.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.evaluate import EpisodeResult, EvaluationSummary, compute_summary


def sep(title=""):
    print(f"\n{'─'*60}")
    if title:
        print(f"  {title}")
        print(f"{'─'*60}")


# ── Test 1: compute_summary() on a known set of episodes ──────────────────────

def test_compute_summary_basic():
    sep("compute_summary() — mixed timeout/collision episodes")
    results = [
        EpisodeResult(episode_num=1, reward=100.0, length=1000,
                      mean_lateral_distance=0.10, termination_reason="timeout"),
        EpisodeResult(episode_num=2, reward=80.0, length=1000,
                      mean_lateral_distance=0.20, termination_reason="timeout"),
        EpisodeResult(episode_num=3, reward=20.0, length=400,
                      mean_lateral_distance=0.50, termination_reason="collision"),
        EpisodeResult(episode_num=4, reward=90.0, length=1000,
                      mean_lateral_distance=0.15, termination_reason="timeout"),
    ]
    summary = compute_summary(results)

    assert summary.n_episodes == 4
    assert summary.mean_reward == 72.5            # (100+80+20+90)/4
    expected_std = 31.12474899             # population std of [100,80,20,90]
    assert abs(summary.std_reward - expected_std) < 1e-6
    expected_mean_lat = (0.10 + 0.20 + 0.50 + 0.15) / 4
    assert abs(summary.mean_lateral_distance - expected_mean_lat) < 1e-9
    assert summary.success_rate == 0.75            # 3 of 4 are "timeout"
    assert summary.mean_length == 850.0            # (1000+1000+400+1000)/4
    assert summary.termination_counts == {"timeout": 3, "collision": 1}
    print("  summary:", summary)
    print("  ✓ PASSED")


# ── Test 2: compute_summary() with no collisions at all ────────────────────────

def test_compute_summary_all_success():
    sep("compute_summary() — all episodes successful")
    results = [
        EpisodeResult(episode_num=1, reward=50.0, length=1000,
                      mean_lateral_distance=0.05, termination_reason="timeout"),
        EpisodeResult(episode_num=2, reward=50.0, length=1000,
                      mean_lateral_distance=0.05, termination_reason="timeout"),
    ]
    summary = compute_summary(results)

    assert summary.success_rate == 1.0
    assert summary.std_reward == 0.0
    assert summary.termination_counts == {"timeout": 2}
    print("  summary:", summary)
    print("  ✓ PASSED")


# ── Test 3: compute_summary() distinguishes all three failure reasons ─────────

def test_compute_summary_all_failure_reasons():
    sep("compute_summary() — collision, off_road, wrong_heading all counted")
    results = [
        EpisodeResult(episode_num=1, reward=10.0, length=100,
                      mean_lateral_distance=1.0, termination_reason="collision"),
        EpisodeResult(episode_num=2, reward=10.0, length=100,
                      mean_lateral_distance=1.0, termination_reason="off_road"),
        EpisodeResult(episode_num=3, reward=10.0, length=100,
                      mean_lateral_distance=1.0, termination_reason="wrong_heading"),
    ]
    summary = compute_summary(results)

    assert summary.success_rate == 0.0
    assert summary.termination_counts == {
        "collision": 1, "off_road": 1, "wrong_heading": 1,
    }
    print("  summary:", summary)
    print("  ✓ PASSED")


def main():
    print("=" * 60)
    print("  EVALUATE.PY OFFLINE TESTS")
    print("  (No CARLA connection required)")
    print("=" * 60)

    test_compute_summary_basic()
    test_compute_summary_all_success()
    test_compute_summary_all_failure_reasons()

    print(f"\n{'='*60}")
    print("  All evaluate.py tests passed.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
