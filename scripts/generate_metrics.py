"""
generate_metrics.py
--------------------
Generate a professional comparison metrics table for all trained algorithms.

For each algorithm, reads data in this priority order:
  1. Evaluation CSV from agent/evaluate.py (deterministic evaluation — preferred)
  2. Last N episodes of the most recent training log (stochastic training policy)

Run from project root — no CARLA required:
    python scripts/generate_metrics.py
    python scripts/generate_metrics.py --window 20   # training log window size
    python scripts/generate_metrics.py --csv out.csv # also write a summary CSV
"""

import os
import sys
import csv
import glob
import argparse
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ALGORITHMS = ["ppo", "sac", "ddpg", "td3"]
RESULTS_ROOT = "results"


# ── Data loading ───────────────────────────────────────────────────────────────

def _read_csv(path):
    """Return list of dicts from a CSV file."""
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def load_eval_csv(algo):
    """
    Load the most comprehensive evaluation CSV produced by agent/evaluate.py
    (the one with the most episodes — gives the most reliable statistics).
    Returns (rows, path) or (None, None) if none exist.

    Columns: episode_num, reward, length, mean_lateral_distance,
             termination_reason
    """
    pattern = os.path.join(
        RESULTS_ROOT, "logs", algo, "eval_runs", "eval_*.csv"
    )
    paths = sorted(glob.glob(pattern))
    if not paths:
        return None, None
    # Pick the file with the most data rows (most evaluation episodes)
    best_path, best_rows = None, []
    for p in paths:
        rows = _read_csv(p)
        if len(rows) > len(best_rows):
            best_rows, best_path = rows, p
    return best_rows, best_path


def load_training_log(algo, window):
    """
    Load the most recent episode_log.csv from training and return the last
    `window` episodes.
    Returns (rows, path, total_episodes, total_steps) or (None, None, 0, 0).

    Columns: episode, timestep, episode_reward, episode_length,
             mean_lateral_dist, mean_speed_kmh, mean_smoothness,
             termination_reason, elapsed_seconds
    """
    pattern = os.path.join(
        RESULTS_ROOT, "logs", algo, "*", "episode_log.csv"
    )
    paths = sorted(glob.glob(pattern))
    if not paths:
        return None, None, 0, 0
    latest = paths[-1]
    all_rows = _read_csv(latest)
    if not all_rows:
        return None, None, 0, 0
    total_eps = len(all_rows)
    total_steps = int(all_rows[-1]["timestep"])
    rows = all_rows[-window:]
    return rows, latest, total_eps, total_steps


# ── Statistics computation ──────────────────────────────────────────────────────

def compute_eval_stats(rows):
    """
    Compute summary statistics from evaluate.py CSV rows.
    Returns a dict of metrics.
    """
    rewards = [float(r["reward"]) for r in rows]
    lengths = [int(r["length"]) for r in rows]
    laterals = [float(r["mean_lateral_distance"]) for r in rows]
    reasons = [r["termination_reason"] for r in rows]

    n = len(rows)
    mean_r = sum(rewards) / n
    std_r = (sum((x - mean_r) ** 2 for x in rewards) / n) ** 0.5
    mean_lat = sum(laterals) / n
    mean_len = sum(lengths) / n
    success = sum(1 for r in reasons if r == "timeout") / n

    counts = {}
    for r in reasons:
        counts[r] = counts.get(r, 0) + 1

    return {
        "n_episodes": n,
        "mean_reward": mean_r,
        "std_reward": std_r,
        "mean_lateral_m": mean_lat,
        "mean_length": mean_len,
        "success_rate": success,
        "termination_counts": counts,
        "mean_speed_kmh": None,   # eval CSV does not log speed
        "source": "evaluation",
    }


def compute_training_stats(rows):
    """
    Compute summary statistics from a training episode_log.csv window.
    Returns a dict of metrics.
    """
    rewards = [float(r["episode_reward"]) for r in rows]
    lengths = [int(r["episode_length"]) for r in rows]
    laterals = [float(r["mean_lateral_dist"]) for r in rows]
    speeds = [float(r["mean_speed_kmh"]) for r in rows]
    reasons = [r["termination_reason"] for r in rows]

    n = len(rows)
    mean_r = sum(rewards) / n
    std_r = (sum((x - mean_r) ** 2 for x in rewards) / n) ** 0.5
    mean_lat = sum(laterals) / n
    mean_len = sum(lengths) / n
    mean_spd = sum(speeds) / n
    success = sum(1 for r in reasons if r == "timeout") / n

    counts = {}
    for r in reasons:
        counts[r] = counts.get(r, 0) + 1

    return {
        "n_episodes": n,
        "mean_reward": mean_r,
        "std_reward": std_r,
        "mean_lateral_m": mean_lat,
        "mean_length": mean_len,
        "success_rate": success,
        "termination_counts": counts,
        "mean_speed_kmh": mean_spd,
        "source": "training",
    }


# ── Algorithm data loader ───────────────────────────────────────────────────────

def load_algo_data(algo, window):
    """
    Load the best available data for one algorithm.
    Returns a dict with keys: algo, stats, training_steps, data_path, note.
    """
    # Try evaluation CSV first (deterministic, gold standard)
    eval_rows, eval_path = load_eval_csv(algo)
    if eval_rows:
        stats = compute_eval_stats(eval_rows)
        # Also look up total training steps from training log for context
        _, _, _, total_steps = load_training_log(algo, 1)
        return {
            "algo": algo,
            "stats": stats,
            "training_steps": total_steps,
            "data_path": eval_path,
            "note": "",
        }

    # Fall back to last N training episodes
    train_rows, train_path, total_eps, total_steps = load_training_log(
        algo, window
    )
    if train_rows:
        stats = compute_training_stats(train_rows)
        note = "(training data, last {} eps — run evaluate.py for clean eval)".format(
            len(train_rows)
        )
        return {
            "algo": algo,
            "stats": stats,
            "training_steps": total_steps,
            "data_path": train_path,
            "note": note,
        }

    return {
        "algo": algo,
        "stats": None,
        "training_steps": 0,
        "data_path": None,
        "note": "not yet trained",
    }


# ── Formatting ──────────────────────────────────────────────────────────────────

def _bar(value, max_value=1.0, width=12):
    """ASCII progress bar."""
    filled = int(round(value / max_value * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def print_full_report(entries, window):
    """Print a professional, human-readable comparison report."""
    W = 70
    print()
    print("=" * W)
    print("  CARLA RL LANE KEEPING — ALGORITHM COMPARISON REPORT")
    print("=" * W)
    print()

    # ── Summary table ─────────────────────────────────────────────────────────
    col_w = 13
    header = (
        "{:<8}  {:>10}  {:>10}  {:>9}  {:>9}  {:>7}  {:>7}".format(
            "Algorithm",
            "Reward",
            "Lateral(m)",
            "Speed",
            "Success%",
            "Steps",
            "Source",
        )
    )
    print(header)
    print("-" * W)

    for entry in entries:
        algo = entry["algo"].upper()
        s = entry["stats"]
        steps_k = entry["training_steps"] // 1000

        if s is None:
            print("{:<8}  {:>10}  {:>10}  {:>9}  {:>9}  {:>7}  {:>7}".format(
                algo, "—", "—", "—", "—",
                "{}k".format(steps_k) if steps_k else "0",
                "none",
            ))
            continue

        speed_str = (
            "{:.1f}".format(s["mean_speed_kmh"])
            if s["mean_speed_kmh"] is not None
            else "n/a"
        )
        print("{:<8}  {:>10}  {:>10}  {:>9}  {:>9}  {:>7}  {:>7}".format(
            algo,
            "{:.1f}±{:.1f}".format(s["mean_reward"], s["std_reward"]),
            "{:.4f}".format(s["mean_lateral_m"]),
            speed_str,
            "{:.0f}%".format(s["success_rate"] * 100),
            "{}k".format(steps_k),
            s["source"][:4],
        ))

    print()

    # ── Per-algorithm detail ──────────────────────────────────────────────────
    for entry in entries:
        algo = entry["algo"].upper()
        s = entry["stats"]
        print("─" * W)
        print("  {}".format(algo))
        print("─" * W)

        if s is None:
            print("  Status: not yet trained")
            print()
            continue

        steps_k = entry["training_steps"] // 1000
        print("  Training steps completed : {:,}  ({:.0f}k)".format(
            entry["training_steps"], steps_k
        ))
        print("  Data source              : {}".format(s["source"]))
        if entry["note"]:
            print("  Note                     : {}".format(entry["note"]))
        print()
        print("  Episodes evaluated       : {}".format(s["n_episodes"]))
        print("  Mean reward              : {:.2f}  (+/- {:.2f})".format(
            s["mean_reward"], s["std_reward"]
        ))
        print("  Mean lateral distance    : {:.4f} m".format(s["mean_lateral_m"]))
        if s["mean_speed_kmh"] is not None:
            print("  Mean speed               : {:.1f} km/h  (target: 30.0)".format(
                s["mean_speed_kmh"]
            ))
        print("  Mean episode length      : {:.0f} steps  ({:.0f} sim-seconds)".format(
            s["mean_length"], s["mean_length"] * 0.05
        ))
        print("  Success rate             : {:.1f}%  {}".format(
            s["success_rate"] * 100,
            _bar(s["success_rate"]),
        ))
        print()
        print("  Termination breakdown:")
        for reason, count in sorted(s["termination_counts"].items()):
            pct = count / s["n_episodes"] * 100
            print("    {:<15s}  {:>3}  ({:.0f}%)".format(reason, count, pct))

        if entry["data_path"]:
            print()
            print("  Data file: {}".format(entry["data_path"]))
        print()

    print("=" * W)
    print("  Metrics note:")
    print("  'evaluation' = deterministic policy, agent/evaluate.py")
    print("  'trai'       = stochastic training policy, last {} eps".format(window))
    print("  For a fair comparison, run evaluate.py on all trained models.")
    print("=" * W)
    print()


def write_summary_csv(entries, path):
    """Write a one-row-per-algorithm summary CSV."""
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    fieldnames = [
        "algorithm", "training_steps", "data_source",
        "n_episodes", "mean_reward", "std_reward",
        "mean_lateral_m", "mean_speed_kmh",
        "success_rate_pct", "mean_episode_length",
        "timeout", "collision", "off_road", "wrong_heading",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            s = entry["stats"]
            tc = s["termination_counts"] if s else {}
            writer.writerow({
                "algorithm": entry["algo"],
                "training_steps": entry["training_steps"],
                "data_source": s["source"] if s else "none",
                "n_episodes": s["n_episodes"] if s else 0,
                "mean_reward": "{:.4f}".format(s["mean_reward"]) if s else "",
                "std_reward": "{:.4f}".format(s["std_reward"]) if s else "",
                "mean_lateral_m": "{:.6f}".format(s["mean_lateral_m"]) if s else "",
                "mean_speed_kmh": (
                    "{:.4f}".format(s["mean_speed_kmh"])
                    if s and s["mean_speed_kmh"] is not None else ""
                ),
                "success_rate_pct": (
                    "{:.2f}".format(s["success_rate"] * 100) if s else ""
                ),
                "mean_episode_length": (
                    "{:.1f}".format(s["mean_length"]) if s else ""
                ),
                "timeout": tc.get("timeout", 0),
                "collision": tc.get("collision", 0),
                "off_road": tc.get("off_road", 0),
                "wrong_heading": tc.get("wrong_heading", 0),
            })
    print("Summary CSV written to: {}".format(path))


# ── Entry point ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate algorithm comparison metrics from available data."
    )
    parser.add_argument(
        "--window", type=int, default=20,
        help="Number of last training episodes to use when no eval CSV exists "
             "(default: 20).",
    )
    parser.add_argument(
        "--csv", default=None,
        help="Optional path to write a one-row-per-algorithm summary CSV.",
    )
    args = parser.parse_args()

    entries = [load_algo_data(algo, args.window) for algo in ALGORITHMS]
    print_full_report(entries, args.window)

    if args.csv:
        write_summary_csv(entries, args.csv)


if __name__ == "__main__":
    main()
