"""
Map visitation analysis.

Loads positions.csv for a run and produces map-level visitation plots.

Run from repo root:
    python -m analysis.final_map_visitation
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = [
    PROJECT_ROOT / "runs" / "big_run_demo",
    PROJECT_ROOT / "runs" / "slim_run_demo",
]


def _ensure_out(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _map_lookup() -> Dict[int, str]:
    return {}


def load_positions(run_dir: Path) -> pd.DataFrame:
    pos_path = run_dir / "positions.csv"
    if not pos_path.exists():
        raise FileNotFoundError(f"positions.csv not found in {run_dir}")
    return pd.read_csv(pos_path)


def plot_map_counts(df: pd.DataFrame, lookup: Dict[int, str], out: Path) -> Path:
    _ensure_out(out)
    counts = df.groupby("map_id").size().reset_index(name="visits")
    counts = counts.sort_values("visits", ascending=False)
    labels = [lookup.get(int(mid), str(mid)) for mid in counts["map_id"]]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(counts)), counts["visits"])
    plt.xticks(range(len(counts)), labels, rotation=90, fontsize=7)
    plt.ylabel("Step visits")
    plt.title("Map visitation counts")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def plot_map_steps(df: pd.DataFrame, out: Path) -> Path:
    _ensure_out(out)
    plt.figure(figsize=(8, 4))
    plt.plot(df["step_global"], df["map_steps"])
    plt.xlabel("Step")
    plt.ylabel("Steps on current map")
    plt.title("Map dwell time over training")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def analyze(run_dir: Path, figs_dir: Path) -> List[Path]:
    df = load_positions(run_dir)
    lookup = _map_lookup()
    outputs: List[Path] = []
    outputs.append(plot_map_counts(df, lookup, figs_dir / f"{run_dir.name}_map_visits.png"))
    if "map_steps" in df.columns:
        outputs.append(plot_map_steps(df, figs_dir / f"{run_dir.name}_map_dwell.png"))
    # simple coverage metric
    coverage = {"unique_maps": df["map_id"].nunique(), "total_steps": len(df)}
    cov_path = figs_dir / f"{run_dir.name}_map_coverage.csv"
    pd.DataFrame([coverage]).to_csv(cov_path, index=False)
    outputs.append(cov_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        default=[str(p) for p in DEFAULT_RUNS],
    )
    parser.add_argument("--figs-dir", type=str, default=str(PROJECT_ROOT / "report" / "figs" / "final"))
    args = parser.parse_args()
    outputs: List[Path] = []
    for rd in args.run_dirs:
        run_dir = Path(rd)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        if not (run_dir / "positions.csv").exists():
            continue
        outputs.extend(analyze(run_dir, Path(args.figs_dir)))
    print("Saved:", *[str(o) for o in outputs], sep="\n- ")


if __name__ == "__main__":
    main()
