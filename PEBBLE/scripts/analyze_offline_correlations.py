#!/usr/bin/env python3
"""Analyze offline human-label CSVs and plot correlations per run.

Computes correlations between correctness (binary 0/1) and:
1) deliberation/decision time
2) confidence
3) replay count

The script groups rows by run ID extracted from query_id by stripping the
trailing "_qXXXX" question suffix.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.stats import pearsonr, spearmanr

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


RUN_RE = re.compile(r"^(?P<run>.+)_q\d+$")
STEP_RE = re.compile(r"step(?P<step>\d+)", re.IGNORECASE)


@dataclass
class CorrResult:
    n: int
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-run correlations from offline human label CSV files."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="CSV files and/or directories containing offline label/comparison CSV files.",
    )
    parser.add_argument(
        "--name-filter",
        default="",
        help="Optional substring filter for CSV filenames (e.g., 'comparisons' or 'human_labels').",
    )
    parser.add_argument(
        "--output-dir",
        default="PEBBLE/logs/offline_correlation_analysis",
        help="Directory to save summary CSV and plots.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="If set, recursively search input directories for CSV files.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="Minimum paired samples required to compute a correlation.",
    )
    parser.add_argument(
        "--combine-all",
        action="store_true",
        help="If set, pool all rows into one batch instead of splitting by run.",
    )
    return parser.parse_args()


def find_csv_files(inputs: Sequence[str], recursive: bool, name_filter: str = "") -> List[Path]:
    files: List[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_file() and path.suffix.lower() == ".csv":
            files.append(path)
            continue
        if path.is_dir():
            pattern = "**/*.csv" if recursive else "*.csv"
            files.extend(sorted(path.glob(pattern)))
    if name_filter:
        needle = name_filter.lower()
        files = [p for p in files if needle in p.name.lower()]
    if not files:
        if name_filter:
            raise FileNotFoundError(
                f"No matching CSV files found in --inputs with --name-filter '{name_filter}'."
            )
        raise FileNotFoundError("No matching CSV files found in --inputs.")
    return sorted(set(files))


def parse_float(value: object) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if text == "":
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def normalize_choice(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in {"A", "B"}:
        return text
    if text in {"LEFT", "L"}:
        return "A"
    if text in {"RIGHT", "R"}:
        return "B"
    return None


def extract_run_id(query_id: object) -> str:
    qid = str(query_id).strip()
    match = RUN_RE.match(qid)
    if match:
        return match.group("run")
    return qid


def parse_bool_like(value: object) -> float:
    if value is None:
        return math.nan
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "correct"}:
        return 1.0
    if text in {"0", "false", "f", "no", "n", "incorrect", "wrong"}:
        return 0.0
    return math.nan


def compute_correctness(row: Dict[str, object]) -> float:
    # Preferred when available in comparisons exports.
    acc = parse_bool_like(row.get("choice_accuracy"))
    if not math.isnan(acc):
        return acc

    choice = normalize_choice(row.get("choice"))
    if choice is None:
        choice = normalize_choice(row.get("choice_made"))

    correct_choice = normalize_choice(
        row.get("higher_reward_trajectory", row.get("correct_choice"))
    )
    if choice is None or correct_choice is None:
        return math.nan
    return 1.0 if choice == correct_choice else 0.0


def infer_run_id(row: Dict[str, object]) -> str:
    query_id = row.get("query_id")
    if query_id is not None and str(query_id).strip():
        return extract_run_id(query_id)

    source = str(row.get("_source_csv", "")).strip()
    if source:
        return Path(source).stem

    participant = str(row.get("participant_id", "")).strip()
    session = str(row.get("session_id", "")).strip()
    if participant or session:
        return f"{participant}_session{session}".strip("_")

    return "unknown_run"


def get_deliberation_value(row: Dict[str, object]) -> float:
    # Use deliberation_time first when present; otherwise fallback to decision_time.
    value = parse_float(row.get("deliberation_time"))
    if not math.isnan(value):
        return value
    return parse_float(row.get("decision_time"))


def get_confidence_value(row: Dict[str, object]) -> float:
    value = parse_float(row.get("confidence"))
    if not math.isnan(value):
        return value
    return parse_float(row.get("confidence_score"))


def compute_replay_count(row: Dict[str, object]) -> float:
    total = parse_float(row.get("replay_count"))
    if not math.isnan(total):
        return total
    replay_a = parse_float(row.get("replay_trajectory_a"))
    replay_b = parse_float(row.get("replay_trajectory_b"))
    if math.isnan(replay_a) and math.isnan(replay_b):
        return math.nan
    replay_a = 0.0 if math.isnan(replay_a) else replay_a
    replay_b = 0.0 if math.isnan(replay_b) else replay_b
    return replay_a + replay_b


def drop_nan_pairs(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = ~np.isnan(x) & ~np.isnan(y)
    return x[mask], y[mask]


def corr_with_correctness(x: np.ndarray, y: np.ndarray, min_samples: int) -> CorrResult:
    x, y = drop_nan_pairs(x, y)
    if len(x) < min_samples:
        return CorrResult(len(x), math.nan, math.nan, math.nan, math.nan)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return CorrResult(len(x), math.nan, math.nan, math.nan, math.nan)

    if HAS_SCIPY:
        pr, pp = pearsonr(x, y)
        sr, sp = spearmanr(x, y)
        return CorrResult(len(x), float(pr), float(pp), float(sr), float(sp))

    # Fallback without scipy: compute Pearson only.
    pr = float(np.corrcoef(x, y)[0, 1])
    return CorrResult(len(x), pr, math.nan, math.nan, math.nan)


def fmt(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def step_sort_key(run_id: str) -> Tuple[int, str]:
    match = STEP_RE.search(run_id)
    if match:
        return int(match.group("step")), run_id
    return 10**12, run_id


def plot_metric_vs_correct(
    ax: plt.Axes,
    metric: np.ndarray,
    correct: np.ndarray,
    title: str,
    xlabel: str,
    corr: CorrResult,
) -> None:
    x, y = drop_nan_pairs(metric, correct)
    if len(x) == 0:
        ax.set_title(f"{title}\n(no valid samples)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Choice accuracy group")
        return

    # Binary outcome is better visualized by showing metric distributions per class.
    incorrect_vals = x[y < 0.5]
    correct_vals = x[y >= 0.5]

    groups = [
        (0, "Incorrect", incorrect_vals, "#c44e52"),
        (1, "Correct", correct_vals, "#4c72b0"),
    ]
    rng = np.random.default_rng(0)

    for pos, _label, vals, color in groups:
        if len(vals) == 0:
            continue

        y_jitter = rng.normal(pos, 0.06, size=len(vals))
        ax.scatter(vals, y_jitter, alpha=0.55, s=24, color=color, edgecolors="none")

        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        mean = float(np.mean(vals))
        ax.hlines(pos, q1, q3, color=color, linewidth=7, alpha=0.45)
        ax.vlines(med, pos - 0.15, pos + 0.15, color="black", linewidth=2)
        ax.plot(mean, pos, marker="D", markersize=5, color="black")

    ax.set_ylim(-0.4, 1.4)
    ax.set_yticks([0, 1], labels=["Incorrect", "Correct"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Choice accuracy group")
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title(
        (
            f"{title}\n"
            f"Pearson r={fmt(corr.pearson_r)} (p={fmt(corr.pearson_p)}), "
            f"Spearman rho={fmt(corr.spearman_rho)} (p={fmt(corr.spearman_p)})"
        ),
        fontsize=10,
    )


def plot_all_runs_overview(summary_rows: List[Dict[str, object]], output_dir: Path) -> None:
    ordered = sorted(summary_rows, key=lambda row: step_sort_key(str(row["run_id"])))
    run_ids = [str(r["run_id"]) for r in ordered]

    # Shorten tick labels for readability while preserving uniqueness where possible.
    tick_labels = []
    for rid in run_ids:
        step_match = STEP_RE.search(rid)
        if step_match:
            tick_labels.append(f"step{step_match.group('step')}")
        else:
            tick_labels.append(rid[:24] + ("..." if len(rid) > 24 else ""))

    x = np.arange(len(run_ids))

    metrics = [
        (
            "deliberation_pearson_r",
            "Deliberation vs correctness",
            "#2a9d8f",
        ),
        (
            "confidence_pearson_r",
            "Confidence vs correctness",
            "#457b9d",
        ),
        (
            "replay_pearson_r",
            "Replay count vs correctness",
            "#e76f51",
        ),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, constrained_layout=True)

    for ax, (key, title, color) in zip(axes, metrics):
        values = np.array([parse_float(r.get(key)) for r in ordered], dtype=float)
        valid = ~np.isnan(values)

        if np.any(valid):
            ax.plot(x[valid], values[valid], marker="o", linewidth=2, color=color)
            ax.scatter(x[valid], values[valid], s=35, color=color)

        ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("Pearson r")
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xticks(x, labels=tick_labels, rotation=35, ha="right")
    axes[-1].set_xlabel("Run")
    fig.suptitle("All-step correlation overview (one point per run)", fontsize=13)
    fig.savefig(output_dir / "all_runs_correlation_overview.png", dpi=150)
    plt.close(fig)


def load_rows(csv_files: Iterable[Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in csv_files:
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["_source_csv"] = str(path)
                rows.append(row)
    if not rows:
        raise ValueError("No rows found in input CSV files.")
    return rows


def main() -> None:
    args = parse_args()
    csv_files = find_csv_files(args.inputs, recursive=args.recursive, name_filter=args.name_filter)
    rows = load_rows(csv_files)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_run: Dict[str, List[Dict[str, object]]] = {}
    if args.combine_all:
        by_run["all_pooled_rows"] = rows
    else:
        for row in rows:
            run_id = infer_run_id(row)
            by_run.setdefault(run_id, []).append(row)

    summary_rows: List[Dict[str, object]] = []

    for run_id, run_rows in sorted(by_run.items()):
        correct = np.array([compute_correctness(r) for r in run_rows], dtype=float)
        deliberation = np.array(
            [get_deliberation_value(r) for r in run_rows],
            dtype=float,
        )
        confidence = np.array([get_confidence_value(r) for r in run_rows], dtype=float)
        replay_count = np.array([compute_replay_count(r) for r in run_rows], dtype=float)

        corr_delib = corr_with_correctness(deliberation, correct, args.min_samples)
        corr_conf = corr_with_correctness(confidence, correct, args.min_samples)
        corr_replay = corr_with_correctness(replay_count, correct, args.min_samples)

        summary_rows.append(
            {
                "run_id": run_id,
                "n_rows": len(run_rows),
                "accuracy": float(np.nanmean(correct)) if np.any(~np.isnan(correct)) else math.nan,
                "deliberation_n": corr_delib.n,
                "deliberation_pearson_r": corr_delib.pearson_r,
                "deliberation_pearson_p": corr_delib.pearson_p,
                "deliberation_spearman_rho": corr_delib.spearman_rho,
                "deliberation_spearman_p": corr_delib.spearman_p,
                "confidence_n": corr_conf.n,
                "confidence_pearson_r": corr_conf.pearson_r,
                "confidence_pearson_p": corr_conf.pearson_p,
                "confidence_spearman_rho": corr_conf.spearman_rho,
                "confidence_spearman_p": corr_conf.spearman_p,
                "replay_n": corr_replay.n,
                "replay_pearson_r": corr_replay.pearson_r,
                "replay_pearson_p": corr_replay.pearson_p,
                "replay_spearman_rho": corr_replay.spearman_rho,
                "replay_spearman_p": corr_replay.spearman_p,
            }
        )

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
        plot_metric_vs_correct(
            axes[0],
            deliberation,
            correct,
            title="Deliberation time vs correctness",
            xlabel="Decision time (s)",
            corr=corr_delib,
        )
        plot_metric_vs_correct(
            axes[1],
            confidence,
            correct,
            title="Confidence vs correctness",
            xlabel="Confidence",
            corr=corr_conf,
        )
        plot_metric_vs_correct(
            axes[2],
            replay_count,
            correct,
            title="Replay count vs correctness",
            xlabel="Replay count",
            corr=corr_replay,
        )
        if args.combine_all:
            fig.suptitle(f"All pooled rows | n={len(run_rows)}", fontsize=12)
        else:
            fig.suptitle(f"Run: {run_id} | n={len(run_rows)}", fontsize=12)

        run_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id)
        fig.savefig(output_dir / f"{run_safe}_correlations.png", dpi=150)
        plt.close(fig)

    summary_csv = output_dir / "correlations_summary.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with summary_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

        # Save an extra figure that compares correlation values across all runs.
        if not args.combine_all and len(summary_rows) > 1:
            plot_all_runs_overview(summary_rows, output_dir)

    print(f"Processed {len(summary_rows)} runs from {len(csv_files)} CSV files.")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved run plots in: {output_dir}")
    if not HAS_SCIPY:
        print("scipy not found: Spearman and p-values are returned as nan.")


if __name__ == "__main__":
    main()
