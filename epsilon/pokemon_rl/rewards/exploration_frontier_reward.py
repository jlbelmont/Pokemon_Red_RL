from __future__ import annotations

from typing import Dict, Optional, Tuple


class ExplorationFrontierReward:
    """Rewards expanding the frontier (maximum Manhattan distance) from the episode start."""

    def __init__(
        self,
        *,
        distance_reward: float = 1.0,
        min_gain: int = 1,
    ) -> None:
        self.distance_reward = float(distance_reward)
        self.min_gain = max(1, int(min_gain))
        self._origin: Optional[Tuple[int, int]] = None
        self._max_distance: int = 0

    def reset(self) -> None:
        self._origin = None
        self._max_distance = 0

    def compute(self, obs, info: Dict) -> float:
        coords = info.get("agent_coords")
        if not coords:
            return 0.0
        pos = (int(coords[0]), int(coords[1]))
        if self._origin is None:
            self._origin = pos
            self._max_distance = 0
            return 0.0

        distance = abs(pos[0] - self._origin[0]) + abs(pos[1] - self._origin[1])
        if distance >= self._max_distance + self.min_gain:
            gain = distance - self._max_distance
            self._max_distance = distance
            return self.distance_reward * gain
        return 0.0
