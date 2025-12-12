"""
RND / posterior analysis pipeline.

Loads per-step/episode logs from runs/*/events.csv and episodes.csv,
computes summary statistics, and writes figures to figs/final/.

Run from repo root:
    python -m analysis.final_analysis_rnd
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = [
    PROJECT_ROOT / "runs" / "big_run_demo",
    PROJECT_ROOT / "runs" / "slim_run_demo",
]


def _load_events(run_dir: Path) -> pd.DataFrame:
    events_path = run_dir / "events.csv"
    if not events_path.exists():
        raise FileNotFoundError(f"events.csv not found in {run_dir}")
    return pd.read_csv(events_path)


def _ensure_out(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _plot_series(df: pd.DataFrame, y: str, out: Path, title: str, window: int = 5000) -> None:
    _ensure_out(out.parent)
    plt.figure(figsize=(8, 4))
    df_sorted = df.sort_values("step_global")
    if window > 1:
        series = df_sorted[y].rolling(window=window, min_periods=max(10, window // 10)).mean()
    else:
        series = df_sorted[y]
    plt.plot(df_sorted["step_global"], series, label=y)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(y.replace("_", " "))
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def _plot_histograms(df: pd.DataFrame, column: str, out: Path) -> None:
    _ensure_out(out.parent)
    plt.figure(figsize=(9, 3))
    steps = df["step_global"]
    max_step = steps.max()
    bins = [
        (steps <= 0.33 * max_step, "early"),
        ((steps > 0.33 * max_step) & (steps <= 0.66 * max_step), "mid"),
        (steps > 0.66 * max_step, "late"),
    ]
    for idx, (mask, label) in enumerate(bins, start=1):
        plt.subplot(1, 3, idx)
        plt.hist(df.loc[mask, column].clip(-5, np.inf), bins=40, color="steelblue")
        plt.title(f"{column} ({label})")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def analyze_run(run_dir: Path, figs_dir: Path) -> List[Path]:
    df = _load_events(run_dir)
    outputs: List[Path] = []

    # time series
    for col, title in [
        ("posterior_mean", "RND posterior mean over time"),
        ("r_rnd", "RND reward over time (smoothed)"),
        ("r_bayes", "Bayes reward over time (smoothed)"),
        ("r_total", "Total reward over time (smoothed)"),
    ]:
        out = figs_dir / f"{run_dir.name}_{col}_over_time.png"
        _plot_series(df, col, out, title)
        outputs.append(out)

    # histograms early/mid/late
    for col in ["r_rnd", "r_bayes", "posterior_mean", "rnd_raw"]:
        out = figs_dir / f"{run_dir.name}_{col}_hist_early_mid_late.png"
        _plot_histograms(df, col, out)
        outputs.append(out)

    # aggregate stats saved as CSV
    stats = {
        "mean_r_rnd": df["r_rnd"].mean(),
        "std_r_rnd": df["r_rnd"].std(),
        "mean_posterior": df["posterior_mean"].mean(),
        "std_posterior": df["posterior_mean"].std(),
        "mean_r_bayes": df["r_bayes"].mean(),
        "std_r_bayes": df["r_bayes"].std(),
    }
    stats_path = figs_dir / f"{run_dir.name}_rnd_stats.csv"
    pd.DataFrame([stats]).to_csv(stats_path, index=False)
    outputs.append(stats_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        default=[str(p) for p in DEFAULT_RUNS],
        help="Run directories to analyze",
    )
    parser.add_argument(
        "--figs-dir",
        type=str,
        default=str(PROJECT_ROOT / "report" / "figs" / "final"),
        help="Output directory for figures",
    )
    args = parser.parse_args()

    figs_dir = Path(args.figs_dir)
    outputs: List[Path] = []
    for rd in args.run_dirs:
        run_dir = Path(rd)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        if not (run_dir / "events.csv").exists():
            continue
        outputs.extend(analyze_run(run_dir, figs_dir))
    print("Saved:", *[str(o) for o in outputs], sep="\n- ")


if __name__ == "__main__":
    main()
