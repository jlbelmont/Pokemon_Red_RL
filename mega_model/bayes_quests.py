"""
Bayesian quest and milestone posterior tracking.

Implements Beta-Bernoulli updates per BAYES_PROGRESS_MONITORING.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional


@dataclass
class MilestonePosterior:
    name: str
    alpha: float = 1.0
    beta: float = 1.0
    target: float = 0.5  # target success probability for alarm calculations

    def update(self, success: bool) -> None:
        if success:
            self.alpha += 1.0
        else:
            self.beta += 1.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def alarm_score(self) -> float:
        # Simple alarm: probability of being below target
        return float(self.target - self.mean)

    def as_dict(self) -> Dict[str, float]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "mean": self.mean,
            "alarm_score": self.alarm_score(),
        }


class BayesianQuestMonitor:
    """Maintains Beta posteriors over multiple milestones."""

    def __init__(
        self,
        milestones: Iterable[str],
        target: float = 0.5,
        shaping_weight: float = 0.0,
    ) -> None:
        self.milestones: Dict[str, MilestonePosterior] = {
            name: MilestonePosterior(name=name, target=target) for name in milestones
        }
        self.shaping_weight = shaping_weight

    def update_episode(self, achieved: Dict[str, bool]) -> Dict[str, Dict[str, float]]:
        summaries: Dict[str, Dict[str, float]] = {}
        for name, success in achieved.items():
            if name not in self.milestones:
                self.milestones[name] = MilestonePosterior(name=name, target=0.5)
            self.milestones[name].update(success)
            summaries[name] = self.milestones[name].as_dict()
        return summaries

    def shaping_reward(self, achieved: Dict[str, bool]) -> float:
        reward = 0.0
        for name, success in achieved.items():
            posterior = self.milestones.get(name)
            if posterior is None:
                continue
            reward += posterior.mean if success else -posterior.alarm_score()
        return self.shaping_weight * reward

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        return {name: m.as_dict() for name, m in self.milestones.items()}
