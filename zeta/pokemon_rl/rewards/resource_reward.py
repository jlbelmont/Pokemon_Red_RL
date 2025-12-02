from __future__ import annotations

from typing import Dict, Iterable, Set


class ResourceManagementReward:
    """Rewards visiting support locations (e.g., Poké Centers) or holding utility items."""

    def __init__(
        self,
        *,
        map_keywords: Iterable[str] | None = None,
        map_reward: float = 5.0,
        utility_item_ids: Iterable[int] | None = None,
        item_reward: float = 2.0,
    ) -> None:
        self.map_keywords = {kw.lower() for kw in (map_keywords or ["pokemon center"])}
        self.map_reward = float(map_reward)
        self.utility_item_ids: Set[int] = {int(x) for x in (utility_item_ids or [])}
        self.item_reward = float(item_reward)
        self._visited_maps: Set[str] = set()
        self._seen_items: Set[int] = set()

    def reset(self) -> None:
        # Persist progress across episodes to encourage long-term planning.
        pass

    def compute(self, obs, info: Dict) -> float:
        reward = 0.0
        map_name = (info.get("map_name") or "").lower()
        for keyword in self.map_keywords:
            if keyword and keyword in map_name and map_name not in self._visited_maps:
                reward += self.map_reward
                self._visited_maps.add(map_name)
                break

        bag_items = info.get("bag_items")
        if bag_items and self.utility_item_ids:
            for item_tuple in bag_items:
                if not isinstance(item_tuple, (tuple, list)) or len(item_tuple) != 2:
                    continue
                item_id = int(item_tuple[0])
                if item_id in self.utility_item_ids and item_id not in self._seen_items:
                    reward += self.item_reward
                    self._seen_items.add(item_id)
        return reward
