class LearnedMapEmbeddingReward:
    """Decaying reward based on previously seen coordinate embeddings.

    When `persist_across_episodes` is True the visit counts are preserved
    between resets so coordinates remain non-novel over the whole run.
    """

    def __init__(
        self,
        base_reward: float = 1.0,
        decay: float = 0.9,
        min_reward: float = 0.0,
        include_map_id: bool = True,
        *,
        persist_across_episodes: bool = False,
    ) -> None:
        self.base_reward = float(base_reward)
        self.decay = float(decay)
        self.min_reward = float(min_reward)
        self.include_map_id = bool(include_map_id)
        self.persist_across_episodes = bool(persist_across_episodes)
        self._counts: dict[tuple[int, int, int], int] = {}

    def _key(self, info) -> tuple[int, int, int]:
        x, y = info.get("agent_coords", (0, 0))
        map_id = info.get("map_id", 0) if self.include_map_id else 0
        return int(map_id), int(x), int(y)

    def compute(self, obs, info) -> float:
        key = self._key(info)
        count = self._counts.get(key, 0)
        reward = self.base_reward * (self.decay ** count)
        reward = max(self.min_reward, reward)
        self._counts[key] = count + 1
        return float(reward)

    def reset(self) -> None:
        if not self.persist_across_episodes:
            self._counts.clear()
