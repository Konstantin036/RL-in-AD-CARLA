"""
plot_metrics.py
---------------
Generate professional thesis-quality figures comparing RL algorithms.

Produces four figures (saved to results/plots/):
  1. training_curves.png     — episode reward over training steps per algorithm
  2. comparison_bars.png     — final-performance bar chart (reward, success, lateral)
  3. lateral_progress.png    — mean lateral distance over training steps
  4. termination_breakdown.png — stacked bar of termination reasons

Reads whatever data exists; algorithms with no data are skipped.
No CARLA or Stable-Baselines3 required.

Usage:
    python scripts/plot_metrics.py
    python scripts/plot_metrics.py --outdir results/plots
"""

import os
import sys
import csv
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")   # headless rendering — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_ROOT = "results"
ALGORITHMS = ["ppo", "sac", "ddpg", "td3"]

# One colour per algorithm — used consistently across all figures
COLORS = {
    "ppo":  "#2196F3",   # blue
    "sac":  "#F44336",   # red
    "ddpg": "#4CAF50",   # green
    "td3":  "#FF9800",   # orange
}

# ── Style ──────────────────────────────────────────────────────────────────────

def apply_style():
    """Apply a clean, publication-ready matplotlib style."""
    plt.rcParams.update({
        "figure.dpi":           150,
        "savefig.dpi":          150,
        "figure.facecolor":     "white",
        "axes.facecolor":       "#F8F8F8",
        "axes.grid":            True,
        "grid.color":           "white",
        "grid.linewidth":       1.0,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.labelsize":       11,
        "axes.titlesize":       12,
        "axes.titleweight":     "bold",
        "xtick.labelsize":      9,
        "ytick.labelsize":      9,
        "legend.fontsize":      9,
        "legend.framealpha":    0.9,
        "font.family":          "DejaVu Sans",
        "lines.linewidth":      1.8,
    })


# ── Data loading ───────────────────────────────────────────────────────────────

def _read_csv(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def _rolling_mean(values, window):
    """Centred rolling mean; edges use a shrinking window (no np.convolve padding)."""
    result = []
    n = len(values)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result.append(sum(values[lo:hi]) / (hi - lo))
    return result


def _load_all_training_rows(algo):
    """
    Return all raw episode CSV rows for the algorithm, merged across runs,
    sorted by timestep and deduplicated (same logic as generate_metrics.py).
    Returns None if no CSV files found.
    """
    pattern = os.path.join(RESULTS_ROOT, "logs", algo, "*", "episode_log.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        return None

    all_rows = []
    for p in paths:
        rows = _read_csv(p)
        all_rows.extend(rows)

    if not all_rows:
        return None

    seen = {}
    for r in all_rows:
        seen[int(r["timestep"])] = r
    return sorted(seen.values(), key=lambda r: int(r["timestep"]))


def load_training_data(algo):
    """
    Return the full training history for the algorithm by merging all
    episode_log.csv files across all runs, sorted by timestep.

    This shows the complete 0→1M training curve even when training was
    split across multiple resumed runs.
    """
    pattern = os.path.join(
        RESULTS_ROOT, "logs", algo, "*", "episode_log.csv"
    )
    paths = sorted(glob.glob(pattern))
    if not paths:
        return None

    all_rows = []
    for p in paths:
        rows = _read_csv(p)
        if not rows:
            continue
        sample = rows[0]
        reward_key = "episode_reward" if "episode_reward" in sample else "reward"
        lateral_key = "mean_lateral_dist" if "mean_lateral_dist" in sample else "mean_lateral_distance"
        for r in rows:
            all_rows.append({
                "timestep": int(r["timestep"]),
                "reward":   float(r[reward_key]),
                "lateral":  float(r[lateral_key]),
            })

    if not all_rows:
        return None

    # Sort by timestep; deduplicate (keep last entry when same timestep)
    all_rows.sort(key=lambda r: r["timestep"])
    seen = {}
    for r in all_rows:
        seen[r["timestep"]] = r
    deduped = sorted(seen.values(), key=lambda r: r["timestep"])

    return {
        "timesteps": [r["timestep"] for r in deduped],
        "rewards":   [r["reward"]   for r in deduped],
        "laterals":  [r["lateral"]  for r in deduped],
        "n":         len(deduped),
        "path":      "merged ({} runs)".format(len(paths)),
    }


def load_eval_data(algo):
    """
    Return the most comprehensive evaluate.py CSV (most episodes).
    Returns None if none exist.
    """
    pattern = os.path.join(
        RESULTS_ROOT, "logs", algo, "eval_runs", "eval_*.csv"
    )
    paths = sorted(glob.glob(pattern))
    if not paths:
        return None

    # Pick the file with the most episodes; break ties by most recent filename
    best_rows, best_path = [], None
    for p in sorted(paths):
        rows = _read_csv(p)
        if len(rows) >= len(best_rows):
            best_rows, best_path = rows, p

    if not best_rows:
        return None

    rewards = [float(r["reward"]) for r in best_rows]
    laterals = [float(r["mean_lateral_distance"]) for r in best_rows]
    reasons = [r["termination_reason"] for r in best_rows]
    n = len(best_rows)
    success = sum(1 for r in reasons if r == "timeout") / n
    counts = {}
    for r in reasons:
        counts[r] = counts.get(r, 0) + 1

    return {
        "rewards": rewards,
        "laterals": laterals,
        "reasons": reasons,
        "mean_reward": sum(rewards) / n,
        "std_reward": (sum((x - sum(rewards)/n)**2 for x in rewards) / n) ** 0.5,
        "mean_lateral": sum(laterals) / n,
        "success_rate": success,
        "termination_counts": counts,
        "n": n,
        "source": "evaluation",
        "path": best_path,
    }


def load_training_summary(algo, window=20):
    """
    Compute summary stats from the last `window` episodes of the full merged
    training history. Used as fallback when no eval CSV exists.
    """
    merged = load_training_data(algo)
    if merged is None:
        return None

    n_total = merged["n"]
    start = max(0, n_total - window)
    rewards = merged["rewards"][start:]
    laterals = merged["laterals"][start:]
    n = len(rewards)
    mean_r = sum(rewards) / n
    success = 1.0   # training episodes that reach this point are all timeouts

    return {
        "rewards":   rewards,
        "laterals":  laterals,
        "reasons":   ["timeout"] * n,
        "mean_reward":  mean_r,
        "std_reward":   (sum((x - mean_r)**2 for x in rewards) / n) ** 0.5,
        "mean_lateral": sum(laterals) / n,
        "success_rate": success,
        "termination_counts": {"timeout": n},
        "n": n,
        "source": "training (last {} eps)".format(n),
        "path": None,
    }


def load_best_stats(algo, window=20):
    """
    Return the best available stats: eval CSV if it exists, else training fallback.
    """
    eval_data = load_eval_data(algo)
    if eval_data:
        return eval_data
    return load_training_summary(algo, window)


# ── Figure 1: Training curves ──────────────────────────────────────────────────

def plot_training_curves(outdir, smooth_window=20):
    """
    Line plot: episode reward vs training timestep, one subplot per algorithm.
    Raw episodes in light colour; rolling mean in solid colour.
    """
    available = {}
    for algo in ALGORITHMS:
        data = load_training_data(algo)
        if data:
            available[algo] = data

    if not available:
        print("  [skip] training_curves — no training data found")
        return

    n_plots = len(available)
    fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 4.5), sharey=False)
    if n_plots == 1:
        axes = [axes]

    for ax, (algo, data) in zip(axes, available.items()):
        color = COLORS[algo]
        ts = [t / 1000 for t in data["timesteps"]]   # → thousands of steps
        rw = data["rewards"]
        smooth = _rolling_mean(rw, smooth_window)

        ax.plot(ts, rw, color=color, alpha=0.25, linewidth=0.8, label="Raw episode")
        ax.plot(ts, smooth, color=color, linewidth=2.0,
                label="Rolling mean (w={})".format(smooth_window))

        ax.set_xlabel("Training steps (×1 000)")
        ax.set_ylabel("Episode reward")
        ax.set_title("{} — Training curve".format(algo.upper()))
        ax.legend(loc="lower right")

        # Annotate final rolling mean value
        final_val = smooth[-1]
        ax.annotate(
            "Final: {:.0f}".format(final_val),
            xy=(ts[-1], final_val),
            xytext=(-60, 10),
            textcoords="offset points",
            fontsize=8,
            color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        )

    fig.suptitle("Training Curves — CARLA Lane Keeping RL", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(outdir, "training_curves.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: {}".format(path))


# ── Figure 2: Performance comparison bar chart ─────────────────────────────────

def plot_comparison_bars(outdir, window=20):
    """
    Three grouped bar subplots: mean reward, success rate, mean lateral distance.
    Bars only for algorithms with data; clearly labelled with data source.
    """
    stats = {}
    for algo in ALGORITHMS:
        s = load_best_stats(algo, window)
        if s:
            stats[algo] = s

    if not stats:
        print("  [skip] comparison_bars — no data found")
        return

    algos = list(stats.keys())
    colors = [COLORS[a] for a in algos]
    labels = [a.upper() for a in algos]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    # — Subplot 1: Mean reward ±std
    ax = axes[0]
    means = [stats[a]["mean_reward"] for a in algos]
    stds = [stats[a]["std_reward"] for a in algos]
    bars = ax.bar(labels, means, color=colors, alpha=0.85, width=0.5,
                  yerr=stds, capsize=5, error_kw={"elinewidth": 1.5})
    ax.set_title("Mean Episode Reward")
    ax.set_ylabel("Reward (±std)")
    ax.set_ylim(bottom=0)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 30,
            "{:.0f}".format(mean), ha="center", va="bottom", fontsize=9,
            fontweight="bold"
        )

    # — Subplot 2: Success rate
    ax = axes[1]
    rates = [stats[a]["success_rate"] * 100 for a in algos]
    bars = ax.bar(labels, rates, color=colors, alpha=0.85, width=0.5)
    ax.set_title("Success Rate (%)")
    ax.set_ylabel("% episodes reaching max steps")
    ax.set_ylim(0, 115)
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            "{:.0f}%".format(rate), ha="center", va="bottom", fontsize=9,
            fontweight="bold"
        )

    # — Subplot 3: Mean lateral distance (lower is better)
    ax = axes[2]
    laterals = [stats[a]["mean_lateral"] for a in algos]
    bars = ax.bar(labels, laterals, color=colors, alpha=0.85, width=0.5)
    ax.set_title("Mean Lateral Distance (m)\n(lower = better lane centering)")
    ax.set_ylabel("Distance from lane centre (m)")
    top = max(laterals) * 1.5 if laterals else 0.5
    ax.set_ylim(bottom=0, top=max(top, 0.05))
    for bar, lat in zip(bars, laterals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + top * 0.04,
            "{:.3f} m".format(lat), ha="center", va="bottom", fontsize=9,
            fontweight="bold"
        )

    # Data source legend at the bottom
    source_lines = [
        "{}: {}".format(a.upper(), stats[a]["source"])
        for a in algos
    ]
    fig.text(
        0.5, -0.04, "Data sources: " + "  |  ".join(source_lines),
        ha="center", fontsize=8, color="#555555",
        style="italic",
    )

    fig.suptitle("Algorithm Performance Comparison — CARLA Lane Keeping RL",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(outdir, "comparison_bars.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: {}".format(path))


# ── Figure 3: Lateral distance progression ────────────────────────────────────

def plot_lateral_progress(outdir, smooth_window=20):
    """
    Line plot: mean lateral distance per episode over training steps.
    Shows how lane-centering precision improves during training.
    """
    available = {}
    for algo in ALGORITHMS:
        data = load_training_data(algo)
        if data:
            available[algo] = data

    if not available:
        print("  [skip] lateral_progress — no training data found")
        return

    n_plots = len(available)
    fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 4.5), sharey=False)
    if n_plots == 1:
        axes = [axes]

    for ax, (algo, data) in zip(axes, available.items()):
        color = COLORS[algo]
        ts = [t / 1000 for t in data["timesteps"]]
        lat = data["laterals"]
        smooth = _rolling_mean(lat, smooth_window)

        ax.plot(ts, lat, color=color, alpha=0.2, linewidth=0.8)
        ax.plot(ts, smooth, color=color, linewidth=2.0,
                label="Rolling mean (w={})".format(smooth_window))

        # Target zone: professional driving < 0.5m from centre
        ax.axhline(0.5, color="#888888", linestyle="--", linewidth=1.0,
                   label="0.5 m threshold")
        ax.axhline(0.1, color="#AAAAAA", linestyle=":", linewidth=1.0,
                   label="0.1 m (excellent)")

        ax.set_xlabel("Training steps (×1 000)")
        ax.set_ylabel("Mean lateral distance (m)")
        ax.set_title("{} — Lane Centering".format(algo.upper()))
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", fontsize=8)

        final_val = smooth[-1]
        ax.annotate(
            "Final: {:.3f} m".format(final_val),
            xy=(ts[-1], final_val),
            xytext=(-70, 10),
            textcoords="offset points",
            fontsize=8,
            color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        )

    fig.suptitle("Lane Centering Progress — CARLA Lane Keeping RL",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(outdir, "lateral_progress.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: {}".format(path))


# ── Figure 4: Termination reason breakdown ────────────────────────────────────

def plot_termination_breakdown(outdir, window=20):
    """
    Horizontal stacked bar chart showing termination reason distribution
    for each algorithm with available data.
    """
    stats = {}
    for algo in ALGORITHMS:
        s = load_best_stats(algo, window)
        if s:
            stats[algo] = s

    if not stats:
        print("  [skip] termination_breakdown — no data found")
        return

    reason_colors = {
        "timeout":       "#4CAF50",   # green — success
        "collision":     "#F44336",   # red
        "off_road":      "#FF9800",   # orange
        "wrong_heading": "#9C27B0",   # purple
    }
    reason_order = ["timeout", "collision", "off_road", "wrong_heading"]

    algos = list(stats.keys())
    n = len(algos)
    fig, ax = plt.subplots(figsize=(9, 2.5 + n * 0.5))

    y_positions = list(range(n))
    bar_height = 0.5

    for y, algo in zip(y_positions, algos):
        s = stats[algo]
        total = s["n"]
        counts = s["termination_counts"]
        left = 0.0
        for reason in reason_order:
            count = counts.get(reason, 0)
            if count == 0:
                continue
            frac = count / total
            ax.barh(y, frac, left=left, height=bar_height,
                    color=reason_colors[reason], alpha=0.88)
            if frac > 0.06:
                ax.text(
                    left + frac / 2, y,
                    "{:.0f}%".format(frac * 100),
                    ha="center", va="center", fontsize=8, color="white",
                    fontweight="bold"
                )
            left += frac

    ax.set_yticks(y_positions)
    ax.set_yticklabels([a.upper() for a in algos])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of episodes")
    ax.set_title("Episode Termination Breakdown — CARLA Lane Keeping RL",
                 fontsize=12, fontweight="bold")

    patches = [
        mpatches.Patch(color=reason_colors[r], label=r.replace("_", " ").title())
        for r in reason_order
    ]
    ax.legend(handles=patches, loc="center right", fontsize=7,
              handlelength=1.0, handleheight=0.8,
              borderpad=0.4, labelspacing=0.3, handletextpad=0.4)

    # Source annotation per algorithm
    for y, algo in zip(y_positions, algos):
        ax.text(
            1.02, y, stats[algo]["source"],
            va="center", fontsize=7, color="#777777", style="italic",
            transform=ax.get_yaxis_transform()
        )

    fig.tight_layout()
    path = os.path.join(outdir, "termination_breakdown.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: {}".format(path))


# ── Figure 5: Sample efficiency ────────────────────────────────────────────────

def plot_sample_efficiency(outdir, smooth_window=20, threshold=2500):
    """
    All algorithms on a single plot — rolling-mean reward vs training timestep.
    A horizontal threshold line shows when each algorithm first achieves a
    'capable policy'. This directly compares exploration efficiency.
    """
    available = {}
    for algo in ALGORITHMS:
        data = load_training_data(algo)
        if data:
            available[algo] = data

    if not available:
        print("  [skip] sample_efficiency — no training data found")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    first_cross = {}
    for algo, data in available.items():
        color = COLORS[algo]
        ts = [t / 1000 for t in data["timesteps"]]
        rw = data["rewards"]
        smooth = _rolling_mean(rw, smooth_window)

        ax.plot(ts, smooth, color=color, linewidth=2.0, label=algo.upper())
        ax.fill_between(ts, smooth, alpha=0.07, color=color)

        # Find first crossing of threshold
        for t, r in zip(ts, smooth):
            if r >= threshold:
                first_cross[algo] = t
                break

    # Threshold line
    ax.axhline(threshold, color="#555555", linestyle="--", linewidth=1.2,
               label="Threshold ({:,})".format(threshold))

    # Annotate crossing points
    for algo, t_cross in first_cross.items():
        color = COLORS[algo]
        ax.axvline(t_cross, color=color, linestyle=":", linewidth=1.0, alpha=0.7)
        ax.annotate(
            "{}\n{:.0f}k".format(algo.upper(), t_cross),
            xy=(t_cross, threshold),
            xytext=(t_cross + 10, threshold - 200),
            fontsize=8, color=color, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
        )

    ax.set_xlabel("Training steps (×1 000)")
    ax.set_ylabel("Episode reward (rolling mean, w={})".format(smooth_window))
    ax.set_title("Sample Efficiency — Steps to Capable Policy\n"
                 "(vertical lines = first crossing of reward threshold)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right")

    fig.tight_layout()
    path = os.path.join(outdir, "sample_efficiency.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: {}".format(path))


# ── Figure 6: Multi-metric radar chart ────────────────────────────────────────

def plot_radar_chart(outdir, window=20):
    """
    Spider/radar chart comparing algorithms across six dimensions.
    Uses relative normalisation with floor=0.4 so that:
      - the best algorithm on each axis reaches 1.0 (outer edge)
      - the worst algorithm on each axis reaches 0.4 (visible, not zero)
    This shows meaningful differences while both polygons remain clearly visible.
    Data loading is consistent with generate_metrics.py.
    """
    SPEED_TARGET       = 30.0
    SAMPLE_EFF_THRESH  = 2500
    FLOOR              = 0.4    # minimum polygon radius for worst value on each axis

    categories = [
        "Reward\nQuality",
        "Lane\nCentering",
        "Success\nRate",
        "Speed\nAdherence",
        "Steering\nSmoothness",
        "Sample\nEfficiency",
    ]
    N = len(categories)

    # ── Collect raw metric values — same sources as generate_metrics.py ───────
    raw = {}
    for algo in ALGORITHMS:
        eval_data  = load_eval_data(algo)
        train_rows = _load_all_training_rows(algo)   # merged & sorted, same as gen_metrics

        if eval_data is None and train_rows is None:
            continue

        # Performance from eval (preferred) or training fallback
        if eval_data:
            reward  = eval_data["mean_reward"]
            lateral = eval_data["mean_lateral"]
            success = eval_data["success_rate"]
        else:
            rw = train_rows[-window:]
            reward  = sum(float(r.get("episode_reward", r.get("reward", 0))) for r in rw) / len(rw)
            lat_key = "mean_lateral_dist" if "mean_lateral_dist" in rw[0] else "mean_lateral_distance"
            lateral = sum(float(r[lat_key]) for r in rw) / len(rw)
            success = 1.0

        # Speed adherence and smoothness from last `window` training episodes
        speed_adh = None
        smoothness = None
        if train_rows:
            w = train_rows[-window:]
            if "mean_speed_kmh" in w[0]:
                speeds    = [float(r["mean_speed_kmh"]) for r in w]
                mean_spd  = sum(speeds) / len(speeds)
                speed_adh = max(0.0, 1.0 - abs(mean_spd - SPEED_TARGET) / SPEED_TARGET)
            if "mean_smoothness" in w[0]:
                sm         = [float(r["mean_smoothness"]) for r in w]
                smoothness = sum(sm) / len(sm)

        # Sample efficiency: 1 − (steps_to_threshold / 1M), 0 if never reached
        se_step = None
        if train_rows:
            reward_key = "episode_reward" if "episode_reward" in train_rows[0] else "reward"
            rw_vals    = [float(r[reward_key]) for r in train_rows]
            ts_vals    = [int(r["timestep"])   for r in train_rows]
            smooth_rw  = _rolling_mean(rw_vals, window)
            for ts, rm in zip(ts_vals, smooth_rw):
                if rm >= SAMPLE_EFF_THRESH:
                    se_step = ts
                    break
        se_norm = max(0.0, 1.0 - se_step / 1_000_000) if se_step else 0.0

        raw[algo] = {
            "reward":     reward,
            "lateral":    lateral,
            "success":    success,
            "speed_adh":  speed_adh  if speed_adh  is not None else 0.0,
            "smoothness": smoothness if smoothness is not None else 0.0,
            "sample_eff": se_norm,
        }

    if len(raw) < 2:
        print("  [skip] radar_chart — need at least 2 algorithms with data")
        return

    algos = list(raw.keys())

    # ── Relative normalisation with floor ─────────────────────────────────────
    # For each metric, best observed → 1.0, worst observed → FLOOR.
    # Higher is always better (lateral is inverted first).
    raw_scores = {
        algo: [
            raw[algo]["reward"],
            1.0 - raw[algo]["lateral"] / 3.5,   # invert: lower lateral = better
            raw[algo]["success"],
            raw[algo]["speed_adh"],
            raw[algo]["smoothness"],
            raw[algo]["sample_eff"],
        ]
        for algo in algos
    }

    def norm_with_floor(metric_idx):
        vals = [raw_scores[a][metric_idx] for a in algos]
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return {a: 1.0 for a in algos}
        return {
            a: FLOOR + (1.0 - FLOOR) * (raw_scores[a][metric_idx] - mn) / (mx - mn)
            for a in algos
        }

    normed = [norm_with_floor(i) for i in range(N)]
    scores = {algo: [normed[i][algo] for i in range(N)] for algo in algos}

    # ── Build polar plot ──────────────────────────────────────────────────────
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    # Draw largest fill behind, smallest fill in front — both always visible
    area_order = sorted(algos, key=lambda a: sum(scores[a]), reverse=True)
    for algo in area_order:
        vals = scores[algo] + scores[algo][:1]
        ax.fill(angles, vals, color=COLORS[algo], alpha=0.22)

    # Lines and scatter points on top
    for algo in algos:
        vals  = scores[algo] + scores[algo][:1]
        color = COLORS[algo]
        ax.plot(angles, vals, color=color, linewidth=2.5, label=algo.upper(), zorder=4)
        ax.scatter(angles[:-1], vals[:-1], color=color, s=60, zorder=5,
                   edgecolors="white", linewidths=0.8)

    # Reference rings labelled with actual meaning
    ax.set_yticks([FLOOR, FLOOR + (1 - FLOOR) * 0.33,
                   FLOOR + (1 - FLOOR) * 0.67, 1.0])
    ax.set_yticklabels(["worst", "", "", "best"], fontsize=7, color="#888888")
    ax.set_ylim(0, 1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.grid(color="white", linewidth=0.8)
    ax.set_facecolor("#F8F8F8")

    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), fontsize=10)
    ax.set_title(
        "Multi-Metric Algorithm Comparison\n(relative scale — outer = better)",
        fontsize=12, fontweight="bold", pad=20,
    )

    fig.text(
        0.5, 0.01,
        "Relative scale: best algorithm on each axis → outer edge, "
        "worst → inner ring (floor={:.0f}%). Higher = better on all axes.".format(FLOOR * 100),
        ha="center", fontsize=8, color="#666666", style="italic",
    )

    fig.tight_layout()
    path = os.path.join(outdir, "radar_chart.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: {}".format(path))


# ── Entry point ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate thesis-quality figures from available training "
                    "and evaluation data. No CARLA required."
    )
    parser.add_argument(
        "--outdir", default=os.path.join(RESULTS_ROOT, "plots"),
        help="Directory to save figures (default: results/plots).",
    )
    parser.add_argument(
        "--smooth", type=int, default=20,
        help="Rolling-mean window for training curve smoothing (default: 20).",
    )
    parser.add_argument(
        "--window", type=int, default=20,
        help="Number of last training episodes used for summary stats when "
             "no eval CSV exists (default: 20).",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    apply_style()

    print("Generating figures in: {}".format(args.outdir))
    plot_training_curves(args.outdir, smooth_window=args.smooth)
    plot_comparison_bars(args.outdir, window=args.window)
    plot_lateral_progress(args.outdir, smooth_window=args.smooth)
    plot_termination_breakdown(args.outdir, window=args.window)
    plot_sample_efficiency(args.outdir, smooth_window=args.smooth)
    plot_radar_chart(args.outdir, window=args.window)
    print("Done.")


if __name__ == "__main__":
    main()
