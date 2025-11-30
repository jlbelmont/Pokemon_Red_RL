"""Static configuration for progress-analysis pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MilestoneConfig:
    name: str
    step_budget: int
    alert_threshold: float
    reward_scale_bounds: tuple[float, float]


MILESTONES: list[MilestoneConfig] = [
    MilestoneConfig("oak_parcel_assigned", 80_000, 0.70, (0.3, 0.8)),
    MilestoneConfig("oak_parcel_received", 120_000, 0.70, (0.3, 0.8)),
    MilestoneConfig("boulder_badge", 220_000, 0.60, (0.4, 0.9)),
    MilestoneConfig("cascade_badge", 420_000, 0.55, (0.4, 0.9)),
]

DATA_ROOT = Path("artifacts/runs")
SUMMARY_CSV = Path("logs/train_summary_2env.csv")
INTRINSIC_COLUMNS = ["run_id", "episode", "intrinsic_reward"]
