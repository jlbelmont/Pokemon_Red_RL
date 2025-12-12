"""
Metrics and ablation summaries.

Run from repo root:
    python -m analysis.metrics
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = [
    PROJECT_ROOT / "runs" / "big_run_demo",
    PROJECT_ROOT / "runs" / "slim_run_demo",
]

def load_logs(run_dir: Path) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for name in ["events", "episodes", "positions", "progress"]:
        p = run_dir / f"{name}.csv"
        if p.exists():
            out[name] = pd.read_csv(p)
    return out


def episode_summary(df: pd.DataFrame) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for col in ["return_env", "return_rnd", "return_novel", "return_bayes", "return_total_with_intrinsic"]:
        if col in df:
            summary[f"mean_{col}"] = df[col].mean()
    summary["num_episodes"] = len(df)
    return summary


def coverage_metrics(pos: pd.DataFrame) -> Dict[str, float]:
    return {"unique_maps": pos["map_id"].nunique(), "total_steps": len(pos)}


def milestone_flags(events: pd.DataFrame) -> Dict[str, float]:
    # count unique milestone flag patterns
    if "milestone_flags" not in events:
        return {}
    return {"unique_milestone_patterns": events["milestone_flags"].nunique()}


def summarize_run(run_dir: Path) -> Dict[str, float]:
    logs = load_logs(run_dir)
    summary = {"run": run_dir.name}
    if "episodes" in logs:
        summary.update(episode_summary(logs["episodes"]))
    if "positions" in logs:
        summary.update(coverage_metrics(logs["positions"]))
    if "events" in logs:
        summary.update(milestone_flags(logs["events"]))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        default=[str(p) for p in DEFAULT_RUNS],
    )
    parser.add_argument("--out-csv", type=str, default=str(PROJECT_ROOT / "analysis" / "tables" / "ablation_summary.csv"))
    args = parser.parse_args()

    summaries: List[Dict[str, float]] = []
    for rd in args.run_dirs:
        run_dir = Path(rd)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        if not run_dir.exists():
            continue
        summaries.append(summarize_run(run_dir))
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
