"""Parse reward-success blocks from training logs and emit summary artifacts."""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


_TRAIN_STEP_RE = re.compile(r"\[train\]\s+step\s+(\d+)")
_ENTRY_RE = re.compile(
    r"^-\s+(?P<label>[^(:]+?)(?:\s+\((?P<family>[^)]+)\))?:\s*(?:≥|>=)\s*"
    r"(?P<threshold>[0-9.]+)\s+mean=(?P<mean>[0-9.]+)\s+\(95% CI "
    r"\[(?P<lo>[0-9.]+),\s*(?P<hi>[0-9.]+)\]\)\s+\|\s+successes\s+"
    r"(?P<success>\d+)/(?P<trials>\d+)"
)


def _parse_lines(lines: List[str], source_log: str) -> pd.DataFrame:
    rows: List[dict] = []
    current_step: int | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        step_match = _TRAIN_STEP_RE.search(line)
        if step_match:
            current_step = int(step_match.group(1))
        if line.strip().startswith("Reward success probabilities"):
            block_step = current_step
            i += 1
            while i < len(lines):
                entry_line = lines[i].strip()
                if not entry_line.startswith("-"):
                    break
                if block_step is None:
                    i += 1
                    continue
                entry_match = _ENTRY_RE.match(entry_line)
                if not entry_match:
                    print(f"[reward_success] Skipping unparsable line: {entry_line}")
                    i += 1
                    continue
                label = entry_match.group("label").strip()
                rows.append(
                    {
                        "step": block_step,
                        "metric": label,
                        "reward_family": (entry_match.group("family") or "").strip(),
                        "threshold": float(entry_match.group("threshold")),
                        "posterior_mean": float(entry_match.group("mean")),
                        "ci_lower": float(entry_match.group("lo")),
                        "ci_upper": float(entry_match.group("hi")),
                        "successes": int(entry_match.group("success")),
                        "trials": int(entry_match.group("trials")),
                        "source_log": source_log,
                    }
                )
                i += 1
            continue
        i += 1
    return pd.DataFrame(rows)


def parse_reward_success(log_path: Path) -> pd.DataFrame:
    """Extract reward-success blocks from a training log."""
    if not log_path.exists():
        raise FileNotFoundError(log_path)
    lines = log_path.read_text().splitlines()
    df = _parse_lines(lines, log_path.name)
    if df.empty:
        return df
    df["success_rate"] = df["successes"] / df["trials"].where(df["trials"] != 0, 1)
    return df.sort_values(["metric", "step"]).reset_index(drop=True)


def plot_reward_success(df: pd.DataFrame, output_path: Path) -> None:
    """Create a multi-panel plot of posterior means over steps."""
    if df.empty:
        print("[reward_success] No data available for plotting.")
        return
    metrics = sorted(df["metric"].unique())
    n_metrics = len(metrics)
    ncols = min(3, n_metrics)
    nrows = math.ceil(n_metrics / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 2.8 * nrows),
        sharex=False,
        sharey=True,
    )
    if not isinstance(axes, (list, tuple)):
        axes = axes if hasattr(axes, "flat") else [axes]
    flat_axes_iter = axes.flat if hasattr(axes, "flat") else axes
    flat_axes = list(flat_axes_iter)
    for ax in flat_axes:
        ax.set_ylim(0.0, 1.05)
    for metric, ax in zip(metrics, flat_axes):
        subset = df[df["metric"] == metric].sort_values("step")
        ax.plot(
            subset["step"],
            subset["posterior_mean"],
            marker="o",
            color="#2f4b7c",
            label="Posterior mean",
        )
        ax.fill_between(
            subset["step"],
            subset["ci_lower"],
            subset["ci_upper"],
            color="#2f4b7c",
            alpha=0.2,
            label="95% CI",
        )
        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel("Step")
        ax.set_ylabel("Success prob.")
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    for ax in list(flat_axes)[n_metrics:]:
        ax.axis("off")
    fig.suptitle("Reward-success posteriors over training")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    if output_path.suffix.lower() != ".png":
        fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="Path to training log.")
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("artifacts/reward_success_timeseries.csv"),
        help="Destination CSV path.",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("figures/reward_success_timeseries.pdf"),
        help="Output plot path.",
    )
    args = parser.parse_args()

    df = parse_reward_success(args.log)
    if df.empty:
        print("[reward_success] No reward-success blocks found.")
        return
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv_output, index=False)
    plot_reward_success(df, args.figure_output)
    print(
        f"[reward_success] Parsed {len(df)} entries across "
        f"{df['metric'].nunique()} metrics."
    )


if __name__ == "__main__":
    main()
