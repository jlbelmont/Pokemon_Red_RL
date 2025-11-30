from __future__ import annotations

from typing import Dict, Set


class MapVisitReward:
    """Rewards visiting new map IDs to encourage broader exploration."""

    def __init__(self, *, map_reward: float = 5.0) -> None:
        self.map_reward = float(map_reward)
        self._visited_maps: Set[int] = set()

    def reset(self) -> None:
        # Persist visited maps across episodes to encourage global coverage.
        pass

    def compute(self, obs, info: Dict) -> float:
        map_id = info.get("map_id")
        if map_id is None:
            return 0.0
        map_id = int(map_id)
        if map_id in self._visited_maps:
            return 0.0
        self._visited_maps.add(map_id)
        return self.map_reward
