from __future__ import annotations

from typing import Dict, Iterable


class StoryFlagReward:
    """Rewards the agent the first time configured story flags become true."""

    def __init__(
        self,
        flag_definitions: Dict[str, Dict[str, float | int]] | None,
        default_reward: float = 150.0,
    ) -> None:
        self.flag_definitions: Dict[str, Dict[str, float | int]] = flag_definitions or {}
        self.default_reward = float(default_reward)
        self._awarded: set[str] = set()

    def _reward_for(self, name: str) -> float:
        data = self.flag_definitions.get(name, {})
        reward = data.get("reward") if isinstance(data, dict) else None
        return float(reward) if reward is not None else self.default_reward

    def compute(self, obs, info) -> float:
        story_flags = info.get("story_flags", {}) or {}
        reward = 0.0
        for name, active in story_flags.items():
            if active and name not in self._awarded:
                reward += self._reward_for(name)
                self._awarded.add(name)
        return reward

    def reset(self) -> None:
        # Keep awarded flags across resets so long-term progress is respected.
        pass
