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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ALGORITHMS = ["ppo", "sac", "ddpg", "td3"]
RESULTS_ROOT = "results"
SPEED_TARGET_KMH = 30.0
SAMPLE_EFF_THRESHOLD = 2500   # reward threshold for "capable policy"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _read_csv(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def _rolling_mean(values, window):
    result = []
    n = len(values)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result.append(sum(values[lo:hi]) / (hi - lo))
    return result


def _iqm(values):
    """Interquartile mean — mean of the middle 50% of values."""
    sv = sorted(values)
    n = len(sv)
    lo = n // 4
    hi = 3 * n // 4
    trimmed = sv[lo:hi] if hi > lo else sv
    return sum(trimmed) / len(trimmed)


# ── Training data loader (all runs merged) ─────────────────────────────────────

def _load_all_training_rows(algo):
    """
    Load and merge all episode_log.csv files for the algorithm across all runs,
    sorted by timestep. Returns list of row dicts, or None.
    """
    pattern = os.path.join(RESULTS_ROOT, "logs", algo, "*", "episode_log.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        return None

    all_rows = []
    for p in paths:
        all_rows.extend(_read_csv(p))

    if not all_rows:
        return None

    # Sort by timestep and deduplicate
    all_rows.sort(key=lambda r: int(r["timestep"]))
    seen = {}
    for r in all_rows:
        seen[int(r["timestep"])] = r
    return sorted(seen.values(), key=lambda r: int(r["timestep"]))


# ── Eval CSV loader ────────────────────────────────────────────────────────────

def load_eval_csv(algo):
    """
    Load the most comprehensive evaluation CSV (most episodes; most recent on tie).
    Returns (rows, path) or (None, None).
    """
    pattern = os.path.join(RESULTS_ROOT, "logs", algo, "eval_runs", "eval_*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        return None, None
    best_path, best_rows = None, []
    for p in sorted(paths):
        rows = _read_csv(p)
        if len(rows) >= len(best_rows):
            best_rows, best_path = rows, p
    return best_rows, best_path


def load_training_log(algo, window):
    """Last `window` episodes from merged training data. Returns (rows, total_steps)."""
    rows = _load_all_training_rows(algo)
    if not rows:
        return None, 0
    total_steps = int(rows[-1]["timestep"])
    return rows[-window:], total_steps


# ── Statistics computation ─────────────────────────────────────────────────────

def compute_eval_stats(rows):
    """Stats from evaluate.py CSV (deterministic evaluation)."""
    rewards  = [float(r["reward"]) for r in rows]
    lengths  = [int(r["length"]) for r in rows]
    laterals = [float(r["mean_lateral_distance"]) for r in rows]
    reasons  = [r["termination_reason"] for r in rows]

    n = len(rows)
    mean_r = sum(rewards) / n
    std_r  = (sum((x - mean_r) ** 2 for x in rewards) / n) ** 0.5

    counts = {}
    for r in reasons:
        counts[r] = counts.get(r, 0) + 1

    return {
        "n_episodes":         n,
        "mean_reward":        mean_r,
        "std_reward":         std_r,
        "iqm_reward":         _iqm(rewards),
        "min_reward":         min(rewards),
        "max_reward":         max(rewards),
        "mean_lateral_m":     sum(laterals) / n,
        "max_lateral_m":      max(laterals),
        "mean_length":        sum(lengths) / n,
        "success_rate":       sum(1 for r in reasons if r == "timeout") / n,
        "termination_counts": counts,
        "mean_speed_kmh":     None,
        "mean_smoothness":    None,
        "source":             "evaluation",
    }


def compute_training_stats(rows):
    """Stats from training episode_log.csv window."""
    reward_key  = "episode_reward" if "episode_reward" in rows[0] else "reward"
    lateral_key = "mean_lateral_dist" if "mean_lateral_dist" in rows[0] else "mean_lateral_distance"

    rewards  = [float(r[reward_key]) for r in rows]
    lengths  = [int(r["episode_length"]) if "episode_length" in rows[0] else int(r["length"]) for r in rows]
    laterals = [float(r[lateral_key]) for r in rows]
    reasons  = [r["termination_reason"] for r in rows]

    n = len(rows)
    mean_r = sum(rewards) / n
    std_r  = (sum((x - mean_r) ** 2 for x in rewards) / n) ** 0.5

    speeds = (
        [float(r["mean_speed_kmh"]) for r in rows]
        if "mean_speed_kmh" in rows[0] else None
    )
    smoothness = (
        [float(r["mean_smoothness"]) for r in rows]
        if "mean_smoothness" in rows[0] else None
    )

    counts = {}
    for r in reasons:
        counts[r] = counts.get(r, 0) + 1

    return {
        "n_episodes":         n,
        "mean_reward":        mean_r,
        "std_reward":         std_r,
        "iqm_reward":         _iqm(rewards),
        "min_reward":         min(rewards),
        "max_reward":         max(rewards),
        "mean_lateral_m":     sum(laterals) / n,
        "max_lateral_m":      max(laterals),
        "mean_length":        sum(lengths) / n,
        "success_rate":       sum(1 for r in reasons if r == "timeout") / n,
        "termination_counts": counts,
        "mean_speed_kmh":     sum(speeds) / n if speeds else None,
        "std_speed_kmh":      (sum((s - sum(speeds)/n)**2 for s in speeds)/n)**0.5 if speeds else None,
        "mean_smoothness":    sum(smoothness) / n if smoothness else None,
        "source":             "training",
    }


def compute_extra_training_stats(algo, window=20):
    """
    Extract speed adherence and smoothness from the last `window` training
    episodes across all merged runs. Returns dict or None.
    """
    rows = _load_all_training_rows(algo)
    if not rows:
        return None
    rows = rows[-window:]
    if "mean_speed_kmh" not in rows[0]:
        return None

    speeds = [float(r["mean_speed_kmh"]) for r in rows]
    mean_spd = sum(speeds) / len(speeds)
    std_spd  = (sum((s - mean_spd)**2 for s in speeds) / len(speeds)) ** 0.5

    result = {
        "mean_speed_kmh":    mean_spd,
        "std_speed_kmh":     std_spd,
        "speed_deviation":   abs(mean_spd - SPEED_TARGET_KMH),
        "speed_adherence":   max(0.0, 1.0 - abs(mean_spd - SPEED_TARGET_KMH) / SPEED_TARGET_KMH),
    }

    if "mean_smoothness" in rows[0]:
        smoothness = [float(r["mean_smoothness"]) for r in rows]
        result["mean_smoothness"] = sum(smoothness) / len(smoothness)

    return result


def compute_sample_efficiency(algo, threshold=SAMPLE_EFF_THRESHOLD, window=20):
    """
    Return the training timestep when the rolling-mean reward first crosses
    `threshold`. Returns None if never reached.
    """
    rows = _load_all_training_rows(algo)
    if not rows:
        return None

    reward_key = "episode_reward" if "episode_reward" in rows[0] else "reward"
    rewards    = [float(r[reward_key]) for r in rows]
    timesteps  = [int(r["timestep"]) for r in rows]
    smooth     = _rolling_mean(rewards, window)

    for ts, rm in zip(timesteps, smooth):
        if rm >= threshold:
            return ts
    return None


# ── Algorithm data loader ──────────────────────────────────────────────────────

def load_algo_data(algo, window):
    """Load the best available data for one algorithm."""
    eval_rows, eval_path = load_eval_csv(algo)

    _, total_steps = load_training_log(algo, 1)
    extra = compute_extra_training_stats(algo, window)
    sample_eff = compute_sample_efficiency(algo)

    if eval_rows:
        stats = compute_eval_stats(eval_rows)
        # Augment eval stats with training-derived speed/smoothness
        if extra:
            stats["mean_speed_kmh"]  = extra.get("mean_speed_kmh")
            stats["std_speed_kmh"]   = extra.get("std_speed_kmh")
            stats["speed_deviation"] = extra.get("speed_deviation")
            stats["speed_adherence"] = extra.get("speed_adherence")
            stats["mean_smoothness"] = extra.get("mean_smoothness")
        return {
            "algo":          algo,
            "stats":         stats,
            "training_steps": total_steps,
            "sample_eff_step": sample_eff,
            "data_path":     eval_path,
            "note":          "",
        }

    train_rows, total_steps = load_training_log(algo, window)
    if train_rows:
        stats = compute_training_stats(train_rows)
        if extra:
            stats["speed_deviation"] = extra.get("speed_deviation")
            stats["speed_adherence"] = extra.get("speed_adherence")
        note = "(training data, last {} eps — run evaluate.py for clean eval)".format(len(train_rows))
        return {
            "algo":           algo,
            "stats":          stats,
            "training_steps": total_steps,
            "sample_eff_step": sample_eff,
            "data_path":      None,
            "note":           note,
        }

    return {
        "algo":           algo,
        "stats":          None,
        "training_steps": 0,
        "sample_eff_step": None,
        "data_path":      None,
        "note":           "not yet trained",
    }


# ── Formatting ─────────────────────────────────────────────────────────────────

def _bar(value, max_value=1.0, width=12):
    filled = int(round(value / max_value * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def print_full_report(entries, window):
    W = 70
    print()
    print("=" * W)
    print("  CARLA RL LANE KEEPING — ALGORITHM COMPARISON REPORT")
    print("=" * W)
    print()

    # ── Summary table ─────────────────────────────────────────────────────────
    print("{:<8}  {:>10}  {:>9}  {:>9}  {:>7}  {:>8}  {:>7}  {:>6}".format(
        "Algorithm", "Reward(IQM)", "Lateral(m)", "Success%",
        "Steps", "Speed(km)", "Smooth", "Source",
    ))
    print("-" * W)

    for entry in entries:
        algo   = entry["algo"].upper()
        s      = entry["stats"]
        steps_k = entry["training_steps"] // 1000

        if s is None:
            print("{:<8}  {:>10}  {:>9}  {:>9}  {:>7}  {:>8}  {:>6}  {:>6}".format(
                algo, "—", "—", "—",
                "{}k".format(steps_k) if steps_k else "0",
                "—", "—", "none",
            ))
            continue

        speed_str = (
            "{:.1f}".format(s["mean_speed_kmh"]) if s.get("mean_speed_kmh") else "n/a"
        )
        smooth_str = (
            "{:.3f}".format(s["mean_smoothness"]) if s.get("mean_smoothness") else "n/a"
        )
        print("{:<8}  {:>10}  {:>9}  {:>9}  {:>7}  {:>8}  {:>6}  {:>6}".format(
            algo,
            "{:.1f}".format(s.get("iqm_reward", s["mean_reward"])),
            "{:.4f}".format(s["mean_lateral_m"]),
            "{:.0f}%".format(s["success_rate"] * 100),
            "{}k".format(steps_k),
            speed_str,
            smooth_str,
            s["source"][:4],
        ))

    print()

    # ── Per-algorithm detail ───────────────────────────────────────────────────
    for entry in entries:
        algo = entry["algo"].upper()
        s    = entry["stats"]
        print("─" * W)
        print("  {}".format(algo))
        print("─" * W)

        if s is None:
            print("  Status: not yet trained")
            print()
            continue

        steps_k = entry["training_steps"] // 1000
        print("  Training steps completed   : {:,}  ({:.0f}k)".format(
            entry["training_steps"], steps_k))
        print("  Data source                : {}".format(s["source"]))
        if entry["note"]:
            print("  Note                       : {}".format(entry["note"]))
        print()

        # Performance metrics
        print("  ── Performance ──")
        print("  Episodes evaluated         : {}".format(s["n_episodes"]))
        print("  Mean reward                : {:.2f}  (±{:.2f})".format(
            s["mean_reward"], s["std_reward"]))
        print("  IQM reward                 : {:.2f}".format(
            s.get("iqm_reward", s["mean_reward"])))
        print("  Reward range               : [{:.1f} … {:.1f}]".format(
            s.get("min_reward", 0), s.get("max_reward", 0)))
        print("  Mean lateral distance      : {:.4f} m".format(s["mean_lateral_m"]))
        print("  Max lateral distance       : {:.4f} m  (worst-case episode)".format(
            s.get("max_lateral_m", 0)))
        print("  Mean episode length        : {:.0f} steps  ({:.0f} sim-sec)".format(
            s["mean_length"], s["mean_length"] * 0.05))
        print("  Success rate               : {:.1f}%  {}".format(
            s["success_rate"] * 100, _bar(s["success_rate"])))

        # Speed & smoothness
        if s.get("mean_speed_kmh") is not None:
            print()
            print("  ── Driving Quality (from last {} training eps) ──".format(window))
            print("  Mean speed                 : {:.1f} km/h  (target: {:.0f} km/h)".format(
                s["mean_speed_kmh"], SPEED_TARGET_KMH))
            if s.get("std_speed_kmh") is not None:
                print("  Speed std                  : ±{:.2f} km/h".format(s["std_speed_kmh"]))
            if s.get("speed_deviation") is not None:
                print("  Speed deviation from target: {:.2f} km/h".format(s["speed_deviation"]))
            if s.get("speed_adherence") is not None:
                print("  Speed adherence            : {:.1f}%  {}".format(
                    s["speed_adherence"] * 100, _bar(s["speed_adherence"])))
        if s.get("mean_smoothness") is not None:
            print("  Steering smoothness        : {:.4f}  (1.0 = perfectly smooth)".format(
                s["mean_smoothness"]))

        # Sample efficiency
        se = entry.get("sample_eff_step")
        if se is not None:
            print()
            print("  ── Exploration Efficiency ──")
            print("  Steps to reward ≥ {:4.0f}      : {:,}  ({:.0f}k)".format(
                SAMPLE_EFF_THRESHOLD, se, se / 1000))
        elif entry["training_steps"] > 0:
            print()
            print("  ── Exploration Efficiency ──")
            print("  Steps to reward ≥ {:4.0f}      : threshold not reached".format(
                SAMPLE_EFF_THRESHOLD))

        # Termination breakdown
        print()
        print("  ── Termination Breakdown ──")
        for reason, count in sorted(s["termination_counts"].items()):
            pct = count / s["n_episodes"] * 100
            print("    {:<15s}  {:>3}  ({:.0f}%)".format(reason, count, pct))

        if entry["data_path"]:
            print()
            print("  Data file: {}".format(entry["data_path"]))
        print()

    # ── Exploration strategy table ─────────────────────────────────────────────
    print("=" * W)
    print("  EXPLORATION STRATEGY COMPARISON")
    print("=" * W)
    print()
    exploration = {
        "ppo":  ("On-policy",  "Stochastic", "Entropy bonus (ent_coef=0.05)",
                 "Fixed — requires manual tuning"),
        "sac":  ("Off-policy", "Stochastic", "Auto entropy tuning (target entropy)",
                 "Automatic — self-adjusting"),
        "ddpg": ("Off-policy", "Deterministic", "Gaussian noise on actions (σ=0.1)",
                 "Fixed noise — no entropy"),
        "td3":  ("Off-policy", "Deterministic", "Gaussian noise + target smoothing",
                 "Fixed noise + clipped double Q"),
    }
    print("  {:<6}  {:<12}  {:<13}  {:<30}".format(
        "Algo", "Type", "Policy", "Exploration Mechanism"))
    print("  " + "-" * 64)
    for algo, (t, p, mech, note) in exploration.items():
        print("  {:<6}  {:<12}  {:<13}  {}".format(
            algo.upper(), t, p, mech))
        print("  {:<6}  {:<12}  {:<13}  {}".format("", "", "", note))
        print()

    print("=" * W)
    print("  Metrics legend:")
    print("  IQM           = interquartile mean (mean of middle 50%% — robust to outliers)")
    print("  Speed adherence = 1 - |mean_speed - target| / target")
    print("  Smoothness    = mean action-smoothness reward component (0→1)")
    print("  Sample eff.   = training steps until rolling-mean reward ≥ {:,.0f}".format(
        SAMPLE_EFF_THRESHOLD))
    print("  'eval' source = deterministic policy, 20 episodes via agent/evaluate.py")
    print("=" * W)
    print()


def write_summary_csv(entries, path):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    fieldnames = [
        "algorithm", "training_steps", "data_source",
        "n_episodes", "mean_reward", "std_reward", "iqm_reward",
        "min_reward", "max_reward",
        "mean_lateral_m", "max_lateral_m",
        "mean_speed_kmh", "std_speed_kmh", "speed_deviation", "speed_adherence",
        "mean_smoothness",
        "success_rate_pct", "mean_episode_length",
        "sample_eff_step",
        "timeout", "collision", "off_road", "wrong_heading",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            s  = entry["stats"]
            tc = s["termination_counts"] if s else {}
            writer.writerow({
                "algorithm":          entry["algo"],
                "training_steps":     entry["training_steps"],
                "data_source":        s["source"] if s else "none",
                "n_episodes":         s["n_episodes"] if s else 0,
                "mean_reward":        "{:.4f}".format(s["mean_reward"]) if s else "",
                "std_reward":         "{:.4f}".format(s["std_reward"]) if s else "",
                "iqm_reward":         "{:.4f}".format(s.get("iqm_reward", 0)) if s else "",
                "min_reward":         "{:.4f}".format(s.get("min_reward", 0)) if s else "",
                "max_reward":         "{:.4f}".format(s.get("max_reward", 0)) if s else "",
                "mean_lateral_m":     "{:.6f}".format(s["mean_lateral_m"]) if s else "",
                "max_lateral_m":      "{:.6f}".format(s.get("max_lateral_m", 0)) if s else "",
                "mean_speed_kmh":     "{:.4f}".format(s["mean_speed_kmh"]) if s and s.get("mean_speed_kmh") else "",
                "std_speed_kmh":      "{:.4f}".format(s.get("std_speed_kmh", 0)) if s and s.get("std_speed_kmh") else "",
                "speed_deviation":    "{:.4f}".format(s.get("speed_deviation", 0)) if s and s.get("speed_deviation") is not None else "",
                "speed_adherence":    "{:.4f}".format(s.get("speed_adherence", 0)) if s and s.get("speed_adherence") is not None else "",
                "mean_smoothness":    "{:.4f}".format(s.get("mean_smoothness", 0)) if s and s.get("mean_smoothness") else "",
                "success_rate_pct":   "{:.2f}".format(s["success_rate"] * 100) if s else "",
                "mean_episode_length": "{:.1f}".format(s["mean_length"]) if s else "",
                "sample_eff_step":    entry.get("sample_eff_step", "") or "",
                "timeout":            tc.get("timeout", 0),
                "collision":          tc.get("collision", 0),
                "off_road":           tc.get("off_road", 0),
                "wrong_heading":      tc.get("wrong_heading", 0),
            })
    print("Summary CSV written to: {}".format(path))


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate algorithm comparison metrics from available data."
    )
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    entries = [load_algo_data(algo, args.window) for algo in ALGORITHMS]
    print_full_report(entries, args.window)
    if args.csv:
        write_summary_csv(entries, args.csv)


if __name__ == "__main__":
    main()
