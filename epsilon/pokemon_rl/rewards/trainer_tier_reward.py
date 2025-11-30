from __future__ import annotations

from typing import Dict, Iterable, Set


class TrainerBattleReward:
    """Rewards trainer battle victories with tiered bonuses (wild, trainer, gym, elite)."""

    def __init__(
        self,
        *,
        wild_reward: float = 5.0,
        trainer_reward: float = 20.0,
        gym_reward: float = 100.0,
        elite_reward: float = 250.0,
        gym_map_ids: Iterable[int] | None = None,
        elite_map_ids: Iterable[int] | None = None,
    ) -> None:
        self.wild_reward = float(wild_reward)
        self.trainer_reward = float(trainer_reward)
        self.gym_reward = float(gym_reward)
        self.elite_reward = float(elite_reward)
        self.gym_map_ids: Set[int] = {int(x) for x in (gym_map_ids or [])}
        self.elite_map_ids: Set[int] = {int(x) for x in (elite_map_ids or [])}
        self._last_reward_step: int | None = None

    def reset(self) -> None:
        self._last_reward_step = None

    def compute(self, obs, info: Dict) -> float:
        last_result = info.get("last_battle_result")
        if last_result != "won":
            return 0.0

        step = int(info.get("steps") or 0)
        if self._last_reward_step == step:
            return 0.0

        self._last_reward_step = step
        map_id = int(info.get("map_id") or 0)
        battle_type = int(info.get("battle_type") or 0)

        if map_id in self.elite_map_ids:
            return self.elite_reward
        if map_id in self.gym_map_ids:
            return self.gym_reward

        # Heuristic: treat non-zero battle_type >= 2 as trainer encounters.
        if battle_type >= 2:
            return self.trainer_reward
        return self.wild_reward
