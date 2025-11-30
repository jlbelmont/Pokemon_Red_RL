from __future__ import annotations

from typing import Dict, Optional


class BattleDamageReward:
    """Reward shaping based on in-battle damage and escape outcomes.

    Positive reward is granted when the opponent's HP decreases relative to the
    previous step. Escaping from a battle incurs a penalty to discourage running
    away. All state is reset at the start of each episode.
    """

    def __init__(
        self,
        *,
        damage_scale: float = 5.0,
        escape_penalty: float = -10.0,
    ) -> None:
        self.damage_scale = float(damage_scale)
        self.escape_penalty = float(escape_penalty)
        self._prev_enemy_hp: Optional[int] = None
        self._prev_enemy_max: Optional[int] = None
        self._was_in_battle = False

    def _extract_enemy_hp(self, info: Dict) -> tuple[int, int]:
        hp = info.get("enemy_hp") or {}
        current = int(hp.get("current") or 0)
        maximum = int(hp.get("max") or 0)
        return current, maximum

    def compute(self, obs, info: Dict) -> float:
        reward = 0.0
        in_battle = bool(info.get("in_battle"))
        enemy_hp, enemy_max = self._extract_enemy_hp(info)

        if in_battle:
            if self._prev_enemy_hp is None or not self._was_in_battle:
                self._prev_enemy_hp = enemy_hp
                self._prev_enemy_max = enemy_max if enemy_max > 0 else None
            else:
                prev_hp = self._prev_enemy_hp
                if prev_hp is not None and enemy_hp >= 0:
                    delta = max(0, prev_hp - enemy_hp)
                    if delta > 0:
                        denom = self._prev_enemy_max or enemy_max or 1
                        reward += self.damage_scale * (delta / max(1, denom))
                self._prev_enemy_hp = enemy_hp
                if enemy_max > 0:
                    self._prev_enemy_max = enemy_max
            self._was_in_battle = True
            return reward

        # Not currently in battle
        if info.get("battle_result") == "escaped" and self._was_in_battle:
            reward += self.escape_penalty

        self._prev_enemy_hp = None
        self._prev_enemy_max = None
        self._was_in_battle = False
        return reward

    def reset(self) -> None:
        self._prev_enemy_hp = None
        self._prev_enemy_max = None
        self._was_in_battle = False

