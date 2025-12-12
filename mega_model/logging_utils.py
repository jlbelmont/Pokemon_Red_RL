"""
Logging utilities for reward decomposition and progress monitoring.

Matches the schema described in REWARD_LOGGING_SPEC.md and BAYES_PROGRESS_MONITORING.md.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch


def _to_float(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    if isinstance(x, np.ndarray):
        return float(x.item())
    return float(x)


class PerformanceTracker:
    """Tracks throughput statistics (SPS, episodes/hour)."""

    def __init__(self) -> None:
        self.start_time = time.time()
        self.last_time = self.start_time
        self.steps = 0
        self.episodes = 0

    def step(self, n: int = 1) -> None:
        self.steps += n

    def episode(self, n: int = 1) -> None:
        self.episodes += n

    def stats(self) -> dict:
        now = time.time()
        elapsed = max(now - self.start_time, 1e-6)
        delta = max(now - self.last_time, 1e-6)
        sps = self.steps / elapsed
        eps_per_hour = 3600.0 * self.episodes / elapsed
        self.last_time = now
        return {"sps": sps, "episodes_per_hour": eps_per_hour, "since_start": elapsed, "delta": delta}


class RewardLogger:
    """
    Buffered CSV logger for step and episode summaries.

    Keeps file I/O minimal while producing machine-readable logs.
    """

    def __init__(
        self,
        log_dir: str | Path,
        flush_every: int = 100,
        step_filename: str = "events.csv",
        episode_filename: str = "episodes.csv",
        progress_filename: str = "progress.csv",
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self.step_path = self.log_dir / step_filename
        self.episode_path = self.log_dir / episode_filename
        self.progress_path = self.log_dir / progress_filename
        self._step_buffer: list[dict] = []
        self._episode_buffer: list[dict] = []
        self._progress_buffer: list[dict] = []

    def log_step(
        self,
        step_global: int,
        env_id: int,
        episode_id: int,
        rewards: Dict[str, float],
        weights: Optional[Dict[str, float]] = None,
        milestone_flags: Optional[Iterable[int]] = None,
        map_id: Optional[int] = None,
        city_id: Optional[int] = None,
        quest_id: Optional[int] = None,
        posterior_mean: Optional[float] = None,
        alarm_score: Optional[float] = None,
        rnd_scale: Optional[float] = None,
        rnd_raw: Optional[float] = None,
    ) -> None:
        record = {
            "step_global": step_global,
            "env_id": env_id,
            "episode_id": episode_id,
            "r_env": _to_float(rewards.get("env", 0.0)),
            "r_rnd": _to_float(rewards.get("rnd", 0.0)),
            "r_novel": _to_float(rewards.get("novel", 0.0)),
            "r_bayes": _to_float(rewards.get("bayes", 0.0)),
            "r_total": _to_float(rewards.get("total", 0.0)),
            "w_rnd": _to_float(weights.get("rnd", 1.0)) if weights else 1.0,
            "w_novel": _to_float(weights.get("novel", 1.0)) if weights else 1.0,
            "w_bayes": _to_float(weights.get("bayes", 1.0)) if weights else 1.0,
            "map_id": map_id if map_id is not None else -1,
            "city_id": city_id if city_id is not None else -1,
            "quest_id": quest_id if quest_id is not None else -1,
            "posterior_mean": posterior_mean if posterior_mean is not None else -1.0,
            "alarm_score": alarm_score if alarm_score is not None else -1.0,
            "rnd_scale": rnd_scale if rnd_scale is not None else 1.0,
            "rnd_raw": rnd_raw if rnd_raw is not None else -1.0,
        }
        if milestone_flags is not None:
            record["milestone_flags"] = ";".join(map(str, milestone_flags))
        self._step_buffer.append(record)
        if len(self._step_buffer) >= self.flush_every:
            self.flush_steps()

    def log_episode(self, summary: Dict[str, float | int | str]) -> None:
        self._episode_buffer.append(summary)
        if len(self._episode_buffer) >= self.flush_every:
            self.flush_episodes()

    def log_progress(self, milestone: str, alpha: float, beta: float, alarm: float) -> None:
        self._progress_buffer.append(
            {"milestone": milestone, "alpha": alpha, "beta": beta, "alarm_score": alarm}
        )
        if len(self._progress_buffer) >= self.flush_every:
            self.flush_progress()

    def flush_steps(self) -> None:
        if not self._step_buffer:
            return
        write_header = not self.step_path.exists()
        with self.step_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self._step_buffer[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(self._step_buffer)
        self._step_buffer.clear()

    def flush_episodes(self) -> None:
        if not self._episode_buffer:
            return
        write_header = not self.episode_path.exists()
        with self.episode_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self._episode_buffer[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(self._episode_buffer)
        self._episode_buffer.clear()

    def flush_progress(self) -> None:
        if not self._progress_buffer:
            return
        write_header = not self.progress_path.exists()
        with self.progress_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self._progress_buffer[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(self._progress_buffer)
        self._progress_buffer.clear()

    def close(self) -> None:
        self.flush_steps()
        self.flush_episodes()
        self.flush_progress()
