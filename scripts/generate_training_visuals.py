#!/usr/bin/env python3
"""
Generate quick-look training visuals (episode rewards, coverage, quests, performance).

Example:
    python scripts/generate_training_visuals.py \
        --summary checkpoints/slideshow_viz_small/logs/slideshow_viz_small/train_summary.csv \
        --perf checkpoints/slideshow_viz_small/logs/slideshow_viz_small/perf_log.csv \
        --output-dir artifacts/slideshow_viz_small
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _parse_quest_hits(values: Iterable[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for raw in values:
        if not isinstance(raw, str):
            continue
        entry = raw.strip()
        if not entry or entry.lower() in {"none", "nan"}:
            continue
        parts = re.split(r"[;|]", entry)
        for part in parts:
            label = part.strip()
            if not label:
                continue
            amount = 1.0
            if ":" in label:
                name, val = label.split(":", 1)
                label = name.strip()
                try:
                    amount = float(val.strip())
                except ValueError:
                    amount = 1.0
            counts[label] += amount
    return counts


def plot_episode_metrics(df: pd.DataFrame, output_dir: Path, title: str | None = None) -> list[Path]:
    files: list[Path] = []
    df = df.copy()
    df["episode"] = _coerce_numeric(df["episode"])
    df["mean_reward"] = _coerce_numeric(df["mean_reward"])
    df["total_env_steps"] = _coerce_numeric(df["total_env_steps"])
    df["coverage_pct"] = _coerce_numeric(df.get("coverage_pct", pd.Series(dtype=float))) * 100.0
    df["epsilon"] = _coerce_numeric(df.get("epsilon", pd.Series(dtype=float)))
    df.sort_values("episode", inplace=True)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(df["episode"], df["mean_reward"], marker="o", color="#4c78a8", label="Mean reward")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Mean reward")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax2 = ax1.twinx()
    ax2.plot(df["episode"], df["total_env_steps"], marker="s", color="#f58518", label="Episode steps")
    ax2.set_ylabel("Episode steps")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    if title:
        ax1.set_title(f"{title} – Reward & Steps")
    reward_path = output_dir / "episode_reward_steps.png"
    fig.tight_layout()
    fig.savefig(reward_path, dpi=200)
    plt.close(fig)
    files.append(reward_path)

    if "coverage_pct" in df and df["coverage_pct"].notna().any():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(df["episode"].astype(int), df["coverage_pct"], color="#54a24b")
        ax.set_xlabel("Episode")
        ax.set_ylabel("World coverage (%)")
        ax.set_title(f"{title} – Coverage" if title else "Coverage per episode")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        coverage_path = output_dir / "coverage_per_episode.png"
        fig.tight_layout()
        fig.savefig(coverage_path, dpi=200)
        plt.close(fig)
        files.append(coverage_path)

    if df["epsilon"].notna().any():
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(df["episode"], df["epsilon"], marker="o", color="#b279a2")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Epsilon")
        ax.set_title(f"{title} – Exploration schedule" if title else "Epsilon per episode")
        ax.grid(True, linestyle="--", alpha=0.4)
        epsilon_path = output_dir / "epsilon_schedule.png"
        fig.tight_layout()
        fig.savefig(epsilon_path, dpi=200)
        plt.close(fig)
        files.append(epsilon_path)

    return files


def plot_perf(perf_df: pd.DataFrame, output_dir: Path, title: str | None = None) -> Path:
    perf_df = perf_df.copy()
    perf_df["sample"] = range(1, len(perf_df) + 1)
    perf_df["steps_per_sec"] = _coerce_numeric(perf_df["steps_per_sec"])
    perf_df["global_step"] = _coerce_numeric(perf_df["global_step"])
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(perf_df["global_step"], perf_df["steps_per_sec"], color="#e45756")
    ax.set_xlabel("Global step")
    ax.set_ylabel("Steps / second")
    ax.set_title(f"{title} – Throughput" if title else "Steps per second")
    ax.grid(True, linestyle="--", alpha=0.4)
    path = output_dir / "steps_per_second.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_quest_counts(quests: Counter[str], output_dir: Path, title: str | None = None) -> Path | None:
    if not quests:
        return None
    labels, values = zip(*quests.most_common())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(labels, values, color="#72b7b2")
    ax.set_xlabel("Hits")
    ax.set_title(f"{title} – Quest hits" if title else "Quest hits")
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    path = output_dir / "quest_hits.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_trace(trace_path: Path | None, output_dir: Path, title: str | None = None) -> Path | None:
    if not trace_path or not trace_path.exists():
        return None
    try:
        with open(trace_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except json.JSONDecodeError:
        return None
    coords = payload.get("frames") or []
    xs, ys = [], []
    for frame in coords:
        x = frame.get("x")
        y = frame.get("y")
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    if not xs:
        return None
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(xs, ys, marker=".", linestyle="-", color="#ff9da6", alpha=0.8)
    ax.set_xlabel("X tile")
    ax.set_ylabel("Y tile")
    ax.set_title(f"{title} – Best episode trace" if title else "Best episode trace")
    ax.grid(True, linestyle="--", alpha=0.4)
    path = output_dir / "best_episode_trace.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate quick training visuals.")
    parser.add_argument("--summary", required=True, help="Path to train_summary.csv")
    parser.add_argument("--perf", required=True, help="Path to perf_log.csv")
    parser.add_argument("--quest", default=None, help="Optional curriculum_events.csv for quest hits")
    parser.add_argument("--trace", default=None, help="Optional best_episode_trace.json for trajectory plot")
    parser.add_argument(
        "--output-dir",
        default="artifacts/training_visuals",
        help="Directory for generated PNGs",
    )
    parser.add_argument("--title", default=None, help="Optional title prefix for charts")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(args.summary)
    perf_df = pd.read_csv(args.perf)
    generated: list[Path] = []
    generated.extend(plot_episode_metrics(summary_df, output_dir, args.title))
    generated.append(plot_perf(perf_df, output_dir, args.title))
    quests = Counter()
    if args.quest and os.path.exists(args.quest):
        quest_df = pd.read_csv(args.quest, names=["wall_time", "episode", "env", "step", "reason", "label", "path"])
        quests = _parse_quest_hits(quest_df["label"])
    else:
        quests = _parse_quest_hits(summary_df.get("quest_hits", []))
    quest_path = plot_quest_counts(quests, output_dir, args.title)
    if quest_path:
        generated.append(quest_path)
    trace_path = plot_trace(Path(args.trace) if args.trace else None, output_dir, args.title)
    if trace_path:
        generated.append(trace_path)

    print("Generated visuals:")
    for path in generated:
        print(f" - {path}")


if __name__ == "__main__":
    main()
