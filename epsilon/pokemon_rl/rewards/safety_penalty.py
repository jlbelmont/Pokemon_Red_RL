from __future__ import annotations

from typing import Dict


class SafetyPenalty:
    """Penalizes unsafe outcomes such as blackouts, defeats, or critically low HP."""

    def __init__(
        self,
        *,
        loss_penalty: float = -25.0,
        blackout_penalty: float = -50.0,
        low_hp_threshold: float = 0.1,
        low_hp_penalty: float = -2.0,
    ) -> None:
        self.loss_penalty = float(loss_penalty)
        self.blackout_penalty = float(blackout_penalty)
        self.low_hp_threshold = max(0.0, min(1.0, float(low_hp_threshold)))
        self.low_hp_penalty = float(low_hp_penalty)
        self._last_penalized_step: int | None = None

    def reset(self) -> None:
        self._last_penalized_step = None

    def compute(self, obs, info: Dict) -> float:
        reward = 0.0
        step = int(info.get("steps") or 0)
        result = info.get("last_battle_result")
        if result in {"lost", "blackout"} and self._last_penalized_step != step:
            reward += self.blackout_penalty if result == "blackout" else self.loss_penalty
            self._last_penalized_step = step

        hp_info = info.get("first_pokemon_hp")
        if isinstance(hp_info, dict):
            current = int(hp_info.get("current") or 0)
            maximum = int(hp_info.get("max") or 0)
            if maximum > 0 and current / maximum <= self.low_hp_threshold:
                reward += self.low_hp_penalty
        return reward
