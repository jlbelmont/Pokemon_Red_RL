from __future__ import annotations

from typing import Dict


class ExplorationProgressReward:
    """Reward unique exploration and penalise long stagnation regardless of map."""

    def __init__(
        self,
        *,
        reward_scale: float = 1.0,
        stagnation_timeout: int = 800,
        stagnation_penalty: float = -8.0,
        penalty_interval: int = 120,
    ) -> None:
        self.reward_scale = float(reward_scale)
        self.stagnation_timeout = max(1, int(stagnation_timeout))
        self.stagnation_penalty = float(stagnation_penalty)
        self.penalty_interval = max(1, int(penalty_interval))
        self._last_unique = 0.0
        self._steps_since_progress = 0

    def reset(self) -> None:
        self._last_unique = 0.0
        self._steps_since_progress = 0

    def compute(self, obs, info: Dict) -> float:  # noqa: D401 - simple hook
        unique_tiles = float(info.get("episode_unique_tiles") or 0.0)
        reward = 0.0
        if unique_tiles > self._last_unique:
            reward += (unique_tiles - self._last_unique) * self.reward_scale
            self._last_unique = unique_tiles
            self._steps_since_progress = 0
        else:
            self._steps_since_progress += 1

        if (
            self.stagnation_penalty != 0.0
            and self._steps_since_progress >= self.stagnation_timeout
        ):
            overdue = self._steps_since_progress - self.stagnation_timeout
            if overdue % self.penalty_interval == 0:
                reward += self.stagnation_penalty

        return reward
