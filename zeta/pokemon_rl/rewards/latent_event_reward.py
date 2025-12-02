from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Dict, Iterable

from ..envs.map_features import region_one_hot


class LatentEventReward:
    """
    Provides reward when the agent visits novel high-level configurations
    (map region + badge/story/inventory state). Revisits receive exponentially
    decayed bonuses.
    """

    def __init__(
        self,
        *,
        base_reward: float = 25.0,
        revisit_decay: float = 0.5,
    ) -> None:
        self.base_reward = float(base_reward)
        self.revisit_decay = float(revisit_decay)
        self._state_counts: defaultdict[str, int] = defaultdict(int)
        self._last_signature: str | None = None

    def reset(self) -> None:
        self._last_signature = None

    def compute(self, obs, info: Dict) -> float:
        signature = self._build_signature(info)
        if signature is None:
            return 0.0

        # Avoid rewarding unchanged state every step.
        if signature == self._last_signature:
            return 0.0

        self._last_signature = signature
        self._state_counts[signature] += 1
        count = self._state_counts[signature]
        if count <= 1:
            return self.base_reward
        return self.base_reward * (self.revisit_decay ** (count - 1))

    def _build_signature(self, info: Dict) -> str | None:
        map_id = int(info.get("map_id") or 0)
        badge_bits = int(info.get("badge_bits") or 0)
        champion = 1 if info.get("champion_defeated") else 0
        pokedex_owned = int(info.get("pokedex_owned_count") or 0)

        key_items = tuple(sorted(int(item) for item in info.get("key_item_ids") or []))
        story_flags = info.get("story_flags") or {}
        if isinstance(story_flags, dict):
            active_flags = tuple(sorted(name for name, value in story_flags.items() if value))
        else:
            active_flags = ()

        region_vector = region_one_hot(map_id)
        region_tuple = tuple(int(v) for v in region_vector.tolist())

        payload = (
            map_id,
            region_tuple,
            badge_bits,
            champion,
            pokedex_owned // 5,  # bucketized by increments of five
            key_items,
            active_flags,
        )
        digest = hashlib.sha1(str(payload).encode("utf-8")).hexdigest()
        return digest
