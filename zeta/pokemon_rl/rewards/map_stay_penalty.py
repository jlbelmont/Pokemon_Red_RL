from __future__ import annotations

from typing import Dict, Iterable, Set


class MapStayPenalty:
    """Penalty for spending too long inside a set of maps without leaving."""

    def __init__(
        self,
        *,
        map_id: int | None = 0,
        map_ids: Iterable[int] | None = None,
        map_names: Iterable[str] | None = None,
        interval: int = 200,
        penalty: float = -5.0,
        name: str | None = None,
        escalate_mode: str | None = None,
        escalate_rate: float = 1.0,
        max_penalty: float | None = None,
    ) -> None:
        target_maps: Set[int] = set(int(mid) for mid in map_ids or [])
        if map_id is not None:
            target_maps.add(int(map_id))
        target_names: Set[str] = set()
        for entry in map_names or []:
            if isinstance(entry, str):
                norm = entry.strip().lower()
                if norm:
                    target_names.add(norm)
        if not target_maps and not target_names:
            raise ValueError("MapStayPenalty requires map_id, map_ids, or map_names.")
        self._target_maps: Set[int] = target_maps
        self._target_map_names: Set[str] = target_names
        self.name = name or "map_stay"
        self.interval = max(1, int(interval))
        self.penalty = float(penalty)
        self.escalate_mode = (escalate_mode or "none").lower()
        self.escalate_rate = float(escalate_rate)
        self.max_penalty = float(max_penalty) if max_penalty is not None else None
        self._ticks = 0
        self._applications = 0

    def reset(self) -> None:
        self._ticks = 0
        self._applications = 0

    def _scaled_penalty(self) -> float:
        if self.escalate_mode == "linear":
            penalty = self.penalty * (1.0 + self.escalate_rate * self._applications)
        elif self.escalate_mode == "exponential":
            penalty = self.penalty * (self.escalate_rate ** self._applications)
        elif self.escalate_mode == "quadratic":
            penalty = self.penalty * (1.0 + self.escalate_rate * (self._applications**2))
        else:
            penalty = self.penalty
        if self.max_penalty is not None:
            penalty = max(self.max_penalty, penalty) if penalty < 0 else min(self.max_penalty, penalty)
        return penalty

    def compute(self, obs, info: Dict) -> float:
        current_map = int(info.get("map_id", -1))
        current_name = str(info.get("map_name") or "").strip().lower()
        in_target = current_map in self._target_maps or (
            self._target_map_names and current_name in self._target_map_names
        )
        if not in_target:
            self._ticks = 0
            self._applications = 0
            return 0.0
        self._ticks += 1
        if self._ticks % self.interval == 0:
            penalty = self._scaled_penalty()
            self._applications += 1
            return penalty
        return 0.0
