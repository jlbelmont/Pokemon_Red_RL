from __future__ import annotations

from typing import Iterable, Set


class ItemCollectionReward:
    """Rewards the agent for collecting new items and key items."""

    def __init__(
        self,
        *,
        item_reward: float = 1.0,
        key_item_reward: float = 5.0,
        key_item_ids: Iterable[int] | None = None,
        reward_once_per_item: bool = True,
    ) -> None:
        self.item_reward = float(item_reward)
        self.key_item_reward = float(key_item_reward)
        self.key_item_ids: Set[int] = set(int(x) for x in key_item_ids or [])
        self.reward_once_per_item = bool(reward_once_per_item)
        self._seen_items: Set[int] = set()
        self._seen_key_items: Set[int] = set()

    def reset(self) -> None:
        if not self.reward_once_per_item:
            self._seen_items.clear()
            self._seen_key_items.clear()

    def compute(self, obs, info) -> float:
        bag_items = info.get("bag_items")
        if not bag_items:
            return 0.0
        reward = 0.0
        for item_tuple in bag_items:
            if not isinstance(item_tuple, (tuple, list)) or len(item_tuple) != 2:
                continue
            item_id, quantity = int(item_tuple[0]), int(item_tuple[1])
            if self.reward_once_per_item and item_id in self._seen_items:
                pass
            else:
                if item_id not in self._seen_items:
                    reward += self.item_reward
                    self._seen_items.add(item_id)
            is_key_item = (
                item_id in self.key_item_ids or quantity == 1
            )
            if is_key_item:
                if not self.reward_once_per_item or item_id not in self._seen_key_items:
                    reward += self.key_item_reward
                    self._seen_key_items.add(item_id)
        return reward
