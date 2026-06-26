#!/usr/bin/env python3
"""Generate PEBBLE experiment analysis plots.

Produces:
  - Group A: Learning curve plots (true_episode_reward vs steps) from train.csv files
  - Group B: Timing/behavior charts (3 styles) from human_labels CSVs
"""

from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Import helpers from existing analysis script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_offline_correlations import (
    compute_correctness,
    compute_replay_count,
    get_confidence_value,
    get_deliberation_value,
    parse_float,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
COMPARE_ROOT = Path("/home/marzii/scratch/compare_runs/pebble/cmput654")
BASELINE_TRAIN_CSV = COMPARE_ROOT / "online/4484717/seed_12345/pebble/train.csv"
OFFLINE_ROOT = COMPARE_ROOT / "offline"

CSV_S1_DIR = Path("/home/marzii/cmput654/PEBBLE/logs/human_queries/4484717")
CSV_1K_PATH = Path(
    "/home/marzii/cmput654/PEBBLE/logs/human_queries/4568316/human_labels_1k_s1_no_skip.csv"
)

OUTPUT_ROOT = Path("/home/marzii/cmput654/PEBBLE/logs/pebble_analysis_plots")
LC_DIR = OUTPUT_ROOT / "learning_curves"
BC_DIR = OUTPUT_ROOT / "bar_charts"

# ---------------------------------------------------------------------------
# Run configurations
# ---------------------------------------------------------------------------
PARTICIPANTS = {
    "Person 1": {
        "s1": "Gary_s1_4484717",
        "s1x": "Gary_s1_500_4484717",
        "s1x_label": "S1 (500 labels)",
        "csv": "human_labels_Gary_s1.csv",
    },
    "Person 2": {
        "s1": "Yang_s1_4521401",
        "s1x": "Yang_s1_500_4524197",
        "s1x_label": "S1 (500 labels)",
        "csv": "human_labels_Yang_s1.csv",
    },
    "Person 3": {
        "s1": "kiarash_s1_4521211",
        "s1x": "kiarash_s1_500_4557631",
        "s1x_label": "S1 (500 labels)",
        "csv": "human_labels_kiarash_s1.csv",
    },
    "Person 4": {
        "s1": "marzieh_s1_4520920",
        "s1x": "marzieh_s1500_4484717",
        "s1x_label": "S1 (500 labels)",
        "csv": "human_labels_marzieh_s1.csv",
    },
}

# Colors
C_BASELINE = "#888888"
C_S1 = "#4c72b0"
C_S1X = "#2a9d8f"
C_CORRECT = "#4c72b0"
C_INCORRECT = "#c44e52"

SMOOTH_WINDOW = 20


# ===================================================================
# Utility functions
# ===================================================================

def load_train_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load step and true_episode_reward from a train.csv file."""
    steps, rewards = [], []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = parse_float(row.get("step"))
            r = parse_float(row.get("true_episode_reward"))
            if not (math.isnan(s) or math.isnan(r)):
                steps.append(s)
                rewards.append(r)
    return np.array(steps), np.array(rewards)


def smooth(values: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Simple rolling average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="same")
    # Fix boundary effects
    for i in range(window // 2):
        smoothed[i] = np.mean(values[: i + window // 2 + 1])
    for i in range(len(values) - window // 2, len(values)):
        smoothed[i] = np.mean(values[i - window // 2 :])
    return smoothed


def load_human_labels(path: Path) -> Dict[str, np.ndarray]:
    """Load human labels CSV, compute correctness and extract metrics."""
    correctness_list = []
    decision_time_list = []
    confidence_list = []
    replay_count_list = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            c = compute_correctness(row)
            dt = get_deliberation_value(row)  # falls back to decision_time
            conf = get_confidence_value(row)
            rc = compute_replay_count(row)
            correctness_list.append(c)
            decision_time_list.append(dt)
            confidence_list.append(conf)
            replay_count_list.append(rc)

    return {
        "correctness": np.array(correctness_list, dtype=float),
        "decision_time": np.array(decision_time_list, dtype=float),
        "confidence": np.array(confidence_list, dtype=float),
        "replay_count": np.array(replay_count_list, dtype=float),
    }


def split_by_correctness(
    metric: np.ndarray, correctness: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Split metric into correct and incorrect arrays, dropping NaN pairs."""
    mask = ~np.isnan(metric) & ~np.isnan(correctness)
    m = metric[mask]
    c = correctness[mask]
    return m[c >= 0.5], m[c < 0.5]


# ===================================================================
# Group A: Learning Curve Plots
# ===================================================================

def _add_peak_line(ax: plt.Axes, steps: np.ndarray, rewards: np.ndarray,
                   color: str, label: str) -> None:
    """Add a dotted horizontal line at the peak of the smoothed curve."""
    smoothed = smooth(rewards)
    peak_val = float(np.max(smoothed))
    peak_idx = int(np.argmax(smoothed))
    peak_step = steps[peak_idx]
    ax.axhline(peak_val, color=color, linestyle=":", linewidth=1.2, alpha=0.7)
    ax.annotate(f"{label} peak: {peak_val:.0f}",
                xy=(peak_step, peak_val), xytext=(5, 5),
                textcoords="offset points", fontsize=8, color=color,
                fontstyle="italic")


def plot_learning_curve_2lines(
    baseline_steps: np.ndarray,
    baseline_rewards: np.ndarray,
    run_steps: np.ndarray,
    run_rewards: np.ndarray,
    participant: str,
    run_label: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    # Raw data (faint)
    ax.plot(baseline_steps, baseline_rewards, color=C_BASELINE, alpha=0.12, linewidth=0.8)
    ax.plot(run_steps, run_rewards, color=C_S1, alpha=0.12, linewidth=0.8)

    # Smoothed
    ax.plot(baseline_steps, smooth(baseline_rewards), color=C_BASELINE,
            linewidth=2, label="Scripted Teacher")
    ax.plot(run_steps, smooth(run_rewards), color=C_S1,
            linewidth=2, label=f"{participant} S1")

    # Peak dotted lines
    _add_peak_line(ax, baseline_steps, baseline_rewards, C_BASELINE, "Scripted Teacher")
    _add_peak_line(ax, run_steps, run_rewards, C_S1, f"{participant} S1")

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("True Episode Reward", fontsize=12)
    ax.set_title(f"Learning Curve: Scripted Teacher vs {participant} S1", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_learning_curve_3lines(
    baseline_steps: np.ndarray,
    baseline_rewards: np.ndarray,
    s1_steps: np.ndarray,
    s1_rewards: np.ndarray,
    s1x_steps: np.ndarray,
    s1x_rewards: np.ndarray,
    participant: str,
    s1_label: str,
    s1x_label: str,
    s1x_run_label: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    # Raw (faint)
    ax.plot(baseline_steps, baseline_rewards, color=C_BASELINE, alpha=0.12, linewidth=0.8)
    ax.plot(s1_steps, s1_rewards, color=C_S1, alpha=0.12, linewidth=0.8)
    ax.plot(s1x_steps, s1x_rewards, color=C_S1X, alpha=0.12, linewidth=0.8)

    # Smoothed
    ax.plot(baseline_steps, smooth(baseline_rewards), color=C_BASELINE,
            linewidth=2, label="Scripted Teacher")
    ax.plot(s1_steps, smooth(s1_rewards), color=C_S1,
            linewidth=2, label=f"{participant} S1")
    ax.plot(s1x_steps, smooth(s1x_rewards), color=C_S1X,
            linewidth=2, label=f"{participant} {s1x_label}")

    # Peak dotted lines
    _add_peak_line(ax, baseline_steps, baseline_rewards, C_BASELINE, "Scripted Teacher")
    _add_peak_line(ax, s1_steps, s1_rewards, C_S1, f"{participant} S1")
    _add_peak_line(ax, s1x_steps, s1x_rewards, C_S1X, f"{participant} {s1x_label}")

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("True Episode Reward", fontsize=12)
    ax.set_title(f"Learning Curve: {participant} — S1 vs {s1x_label} vs Baseline", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def generate_learning_curves() -> None:
    print("\n=== Group A: Learning Curves ===")
    LC_DIR.mkdir(parents=True, exist_ok=True)

    # Load baseline
    print("Loading baseline train.csv ...")
    bl_steps, bl_rewards = load_train_csv(BASELINE_TRAIN_CSV)
    print(f"  Baseline: {len(bl_steps)} points, steps [{bl_steps[0]:.0f} .. {bl_steps[-1]:.0f}]")

    # Set 1: S1 vs Baseline
    print("\n--- Set 1: S1 vs Baseline ---")
    for name, cfg in PARTICIPANTS.items():
        s1_path = OFFLINE_ROOT / cfg["s1"] / "pebble" / "train.csv"
        s1_steps, s1_rewards = load_train_csv(s1_path)
        print(f"  {name} S1: {len(s1_steps)} points")
        plot_learning_curve_2lines(
            bl_steps, bl_rewards, s1_steps, s1_rewards,
            participant=name,
            run_label=cfg["s1"],
            output_path=LC_DIR / f"s1_vs_baseline_{name}.png",
        )

    # Set 2: S1 + S1_500 + Baseline
    print("\n--- Set 2: S1 + S1_500 + Baseline ---")
    for name, cfg in PARTICIPANTS.items():
        s1_path = OFFLINE_ROOT / cfg["s1"] / "pebble" / "train.csv"
        s1x_path = OFFLINE_ROOT / cfg["s1x"] / "pebble" / "train.csv"
        s1_steps, s1_rewards = load_train_csv(s1_path)
        s1x_steps, s1x_rewards = load_train_csv(s1x_path)
        print(f"  {name} S1: {len(s1_steps)} pts, S1x: {len(s1x_steps)} pts")
        plot_learning_curve_3lines(
            bl_steps, bl_rewards,
            s1_steps, s1_rewards,
            s1x_steps, s1x_rewards,
            participant=name,
            s1_label=cfg["s1"],
            s1x_label=cfg["s1x_label"],
            s1x_run_label=cfg["s1x"],
            output_path=LC_DIR / f"s1_s1x_baseline_{name}.png",
        )


# ===================================================================
# Group B: Timing / Behavior Charts (3 styles)
# ===================================================================

METRICS = [
    ("decision_time", "Decision Time (s)"),
    ("confidence", "Confidence (1-5)"),
    ("replay_count", "Replay Count"),
]


# --- Style 1: Violin + Box ---------------------------------------------------

def _violin_subplot(ax: plt.Axes, correct: np.ndarray, incorrect: np.ndarray,
                    ylabel: str, title: str) -> None:
    data_to_plot = []
    labels = []
    colors = []
    for vals, label, color in [
        (correct, "Correct", C_CORRECT),
        (incorrect, "Incorrect", C_INCORRECT),
    ]:
        if len(vals) > 0:
            data_to_plot.append(vals)
            labels.append(f"{label}\n(n={len(vals)})")
            colors.append(color)

    if len(data_to_plot) == 0:
        ax.set_title(f"{title}\n(no data)")
        return

    positions = list(range(len(data_to_plot)))
    parts = ax.violinplot(data_to_plot, positions=positions, showmeans=False,
                          showmedians=False, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.35)

    # Box plot overlay
    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.15,
                    showfliers=False, patch_artist=True, zorder=3)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.7)
    for element in ["whiskers", "caps"]:
        for line in bp[element]:
            line.set_color("black")
            line.set_linewidth(1)
    for line in bp["medians"]:
        line.set_color("black")
        line.set_linewidth(2)

    # Mean markers
    for i, vals in enumerate(data_to_plot):
        mean_val = np.mean(vals)
        med_val = np.median(vals)
        ax.plot(i, mean_val, marker="D", color="black", markersize=6, zorder=4)
        ax.annotate(f"mean={mean_val:.2f}\nmed={med_val:.2f}",
                    xy=(i, max(mean_val, med_val)),
                    xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=8, color="black")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")


def plot_violin_box(data: Dict[str, np.ndarray], title: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for ax, (key, ylabel) in zip(axes, METRICS):
        correct, incorrect = split_by_correctness(data[key], data["correctness"])
        _violin_subplot(ax, correct, incorrect, ylabel, ylabel.split("(")[0].strip())
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# --- Style 2: Raincloud -------------------------------------------------------

def _raincloud_subplot(ax: plt.Axes, correct: np.ndarray, incorrect: np.ndarray,
                       ylabel: str, title: str) -> None:
    groups = [
        (incorrect, "Incorrect", C_INCORRECT, 0),
        (correct, "Correct", C_CORRECT, 1),
    ]
    rng = np.random.default_rng(42)

    for vals, label, color, pos in groups:
        if len(vals) == 0:
            continue

        # Half-violin via histogram-based density
        if len(vals) > 1 and np.std(vals) > 0:
            n_bins = min(30, max(10, len(vals) // 5))
            counts, bin_edges = np.histogram(vals, bins=n_bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            density = counts.astype(float)
            if density.max() > 0:
                density = density / density.max() * 0.35
            ax.fill_betweenx(bin_centers, pos - density, pos, alpha=0.35,
                             color=color, step="mid")
            ax.step(pos - density, bin_centers, color=color, linewidth=1, where="mid")

        # Box plot
        bp = ax.boxplot([vals], positions=[pos + 0.15], widths=0.08,
                        vert=True, showfliers=False, patch_artist=True, zorder=3)
        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.7)
        for line in bp["medians"]:
            line.set_color("black")
            line.set_linewidth(2)

        # Jittered scatter
        jitter = rng.normal(0, 0.02, size=len(vals))
        ax.scatter(pos + 0.28 + jitter, vals, alpha=0.4, s=12, color=color,
                   edgecolors="none", zorder=2)

        # Mean marker
        mean_val = np.mean(vals)
        ax.plot(pos + 0.15, mean_val, marker="D", color="black", markersize=5, zorder=4)

    ax.set_xticks([0, 1])
    n_inc = len(incorrect)
    n_cor = len(correct)
    ax.set_xticklabels([f"Incorrect\n(n={n_inc})", f"Correct\n(n={n_cor})"], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")


def plot_raincloud(data: Dict[str, np.ndarray], title: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    for ax, (key, ylabel) in zip(axes, METRICS):
        correct, incorrect = split_by_correctness(data[key], data["correctness"])
        _raincloud_subplot(ax, correct, incorrect, ylabel, ylabel.split("(")[0].strip())
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# --- Style 3: Grouped Bar Charts ----------------------------------------------

def plot_combined_bar_chart(
    all_data: Dict[str, Dict[str, np.ndarray]],
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """Bar chart with all participants on one plot for a single metric.

    Each participant gets a Correct and Incorrect bar.
    Bar height = mean (with std error bar), horizontal line = median.
    """
    names = list(all_data.keys())
    n_groups = len(names)
    x = np.arange(n_groups)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, n_groups * 2.5), 6))

    for side, (label, color, offset) in enumerate([
        ("Correct", C_CORRECT, -width / 2),
        ("Incorrect", C_INCORRECT, width / 2),
    ]):
        means, stds, medians, counts = [], [], [], []
        for name in names:
            data = all_data[name]
            correct_vals, incorrect_vals = split_by_correctness(
                data[metric_key], data["correctness"]
            )
            vals = correct_vals if side == 0 else incorrect_vals
            if len(vals) > 0:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
                medians.append(np.median(vals))
                counts.append(len(vals))
            else:
                means.append(0)
                stds.append(0)
                medians.append(0)
                counts.append(0)

        bars = ax.bar(x + offset, means, width * 0.9, color=color, alpha=0.75,
                      yerr=stds, capsize=4, error_kw={"linewidth": 1.2},
                      label=label)

        # Median line + annotations on each bar
        for i, bar in enumerate(bars):
            if counts[i] == 0:
                continue
            bx = bar.get_x()
            bw = bar.get_width()
            med = medians[i]
            # Median horizontal line
            ax.hlines(med, bx + bw * 0.1, bx + bw * 0.9,
                      color="black", linewidth=2, zorder=4)
            # Annotation: mean and median values
            top = means[i] + stds[i]
            ax.annotate(f"x\u0304={means[i]:.1f}\nm={med:.1f}\nn={counts[i]}",
                        xy=(bx + bw / 2, top),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=7, linespacing=1.1)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_single_bar_chart(
    data: Dict[str, np.ndarray],
    title: str,
    output_path: Path,
) -> None:
    """Single-participant bar chart with 3 subplots (one per metric)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    width = 0.35

    for ax, (key, ylabel) in zip(axes, METRICS):
        correct_vals, incorrect_vals = split_by_correctness(
            data[key], data["correctness"]
        )
        x = np.arange(1)
        for side, (label, color, offset, vals) in enumerate([
            ("Correct", C_CORRECT, -width / 2, correct_vals),
            ("Incorrect", C_INCORRECT, width / 2, incorrect_vals),
        ]):
            if len(vals) == 0:
                continue
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            med_val = np.median(vals)
            n = len(vals)

            bar = ax.bar(x + offset, [mean_val], width * 0.9, color=color,
                         alpha=0.75, yerr=[std_val], capsize=5,
                         error_kw={"linewidth": 1.2}, label=f"{label} (n={n})")
            # Median line
            bx = bar[0].get_x()
            bw = bar[0].get_width()
            ax.hlines(med_val, bx + bw * 0.1, bx + bw * 0.9,
                      color="black", linewidth=2, zorder=4)
            # Annotation
            top = mean_val + std_val
            ax.annotate(f"x\u0304={mean_val:.1f}\nm={med_val:.1f}",
                        xy=(bx + bw / 2, top),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=8, linespacing=1.1)

        ax.set_xticks([])
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel.split("(")[0].strip(), fontsize=11)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ===================================================================
# Generate all timing charts for a dataset
# ===================================================================

def generate_timing_charts() -> None:
    print("\n=== Group B: Timing Charts ===")
    BC_DIR.mkdir(parents=True, exist_ok=True)

    # Load all S1 data
    all_data: Dict[str, Dict[str, np.ndarray]] = {}
    for name, cfg in PARTICIPANTS.items():
        csv_path = CSV_S1_DIR / cfg["csv"]
        print(f"Loading {csv_path.name} ...")
        data = load_human_labels(csv_path)
        n_total = len(data["correctness"])
        n_correct = int(np.nansum(data["correctness"] >= 0.5))
        print(f"  {name}: {n_total} rows, {n_correct} correct, "
              f"{n_total - n_correct} incorrect")
        all_data[name] = data

    # 3 combined charts: one per metric, all 4 participants
    print("\n--- Combined S1 Charts (all participants per metric) ---")
    for key, ylabel in METRICS:
        metric_title = ylabel.split("(")[0].strip()
        plot_combined_bar_chart(
            all_data,
            metric_key=key,
            ylabel=ylabel,
            title=f"{metric_title} — All Participants S1",
            output_path=BC_DIR / f"s1_all_{key}.png",
        )

    # 1k data — single figure with 3 subplots
    print("\n--- 1k Charts ---")
    print(f"Loading {CSV_1K_PATH.name} ...")
    data_1k = load_human_labels(CSV_1K_PATH)
    n_total = len(data_1k["correctness"])
    n_correct = int(np.nansum(data_1k["correctness"] >= 0.5))
    print(f"  1k: {n_total} rows, {n_correct} correct, {n_total - n_correct} incorrect")
    plot_single_bar_chart(data_1k, "1k S1 No-Skip", BC_DIR / "1k_bars.png")


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    print("PEBBLE Experiment Analysis Plots")
    print("=" * 50)
    generate_learning_curves()
    generate_timing_charts()
    print(f"\nAll plots saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
