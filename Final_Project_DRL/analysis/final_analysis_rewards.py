"""
Reward decomposition + Bayesian/posterior analysis.

Run from repo root:
    python -m analysis.final_analysis_rewards
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = [
    PROJECT_ROOT / "runs" / "big_run_demo",
    PROJECT_ROOT / "runs" / "slim_run_demo",
]


def _ensure_out(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_logs(run_dir: Path):
    episodes = pd.read_csv(run_dir / "episodes.csv") if (run_dir / "episodes.csv").exists() else None
    events = pd.read_csv(run_dir / "events.csv") if (run_dir / "events.csv").exists() else None
    progress = pd.read_csv(run_dir / "progress.csv") if (run_dir / "progress.csv").exists() else None
    return episodes, events, progress


def plot_episode_rewards(episodes: pd.DataFrame, out: Path) -> Path:
    _ensure_out(out)
    plt.figure(figsize=(8, 4))
    x = episodes["episode_id"]
    for col in ["return_env", "return_rnd", "return_novel", "return_bayes", "return_total_with_intrinsic"]:
        if col in episodes:
            plt.plot(x, episodes[col], label=col)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Episode returns by component")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def plot_posterior(progress: pd.DataFrame, out: Path) -> Path:
    _ensure_out(out)
    plt.figure(figsize=(7, 4))
    for k in progress["milestone"].unique():
        subset = progress[progress["milestone"] == k]
        plt.plot(subset.index, subset["alpha"] / (subset["alpha"] + subset["beta"] + 1e-8), label=k)
    plt.xlabel("Update index")
    plt.ylabel("Posterior mean (alpha/(alpha+beta))")
    plt.title("Bayesian posterior by milestone")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def aggregate_tables(episodes: pd.DataFrame, out_csv: Path) -> Path:
    _ensure_out(out_csv)
    total_eps = len(episodes)
    thirds = [0, int(0.33 * total_eps), int(0.66 * total_eps), total_eps]
    rows = []
    labels = ["early", "mid", "late"]
    for i, label in enumerate(labels):
        subset = episodes.iloc[thirds[i] : thirds[i + 1]]
        if subset.empty:
            continue
        row = {"phase": label}
        for col in ["return_env", "return_rnd", "return_novel", "return_bayes", "return_total_with_intrinsic"]:
            if col in subset:
                row[f"mean_{col}"] = subset[col].mean()
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def analyze(run_dir: Path, figs_dir: Path, tables_dir: Path) -> List[Path]:
    episodes, events, progress = load_logs(run_dir)
    outputs: List[Path] = []
    if episodes is not None:
        outputs.append(plot_episode_rewards(episodes, figs_dir / f"{run_dir.name}_episode_rewards.png"))
        outputs.append(aggregate_tables(episodes, tables_dir / f"{run_dir.name}_rewards_summary.csv"))
    if progress is not None and not progress.empty:
        outputs.append(plot_posterior(progress, figs_dir / f"{run_dir.name}_posterior.png"))
    # per-step smoothing for components
    if events is not None:
        for col in ["r_env", "r_rnd", "r_novel", "r_bayes", "r_total"]:
            if col not in events:
                continue
            series = events.sort_values("step_global")[col].rolling(window=5000, min_periods=200).mean()
            plt.figure(figsize=(8, 3))
            plt.plot(events.sort_values("step_global")["step_global"], series, label=col)
            plt.title(f"{col} smoothed over steps")
            plt.xlabel("Step")
            plt.ylabel(col)
            plt.tight_layout()
            out = figs_dir / f"{run_dir.name}_{col}_smoothed.png"
            plt.savefig(out, dpi=200)
            plt.close()
            outputs.append(out)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        default=[str(p) for p in DEFAULT_RUNS],
    )
    parser.add_argument("--figs-dir", type=str, default=str(PROJECT_ROOT / "report" / "figs" / "final"))
    parser.add_argument("--tables-dir", type=str, default=str(PROJECT_ROOT / "analysis" / "tables"))
    args = parser.parse_args()
    outputs: List[Path] = []
    for rd in args.run_dirs:
        run_dir = Path(rd)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        if not (run_dir / "events.csv").exists() and not (run_dir / "episodes.csv").exists():
            continue
        outputs.extend(analyze(run_dir, Path(args.figs_dir), Path(args.tables_dir)))
    print("Saved:", *[str(o) for o in outputs], sep="\n- ")


if __name__ == "__main__":
    main()
