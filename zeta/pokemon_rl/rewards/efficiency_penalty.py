from __future__ import annotations

from typing import Dict, Optional, Tuple


class EfficiencyPenalty:
    """Applies a small per-step penalty and discourages idle loops."""

    def __init__(
        self,
        *,
        step_penalty: float = -0.001,
        idle_penalty: float = -0.1,
        idle_threshold: int = 20,
    ) -> None:
        self.step_penalty = float(step_penalty)
        self.idle_penalty = float(idle_penalty)
        self.idle_threshold = max(1, int(idle_threshold))
        self._last_coords: Optional[Tuple[int, int]] = None
        self._idle_streak: int = 0

    def reset(self) -> None:
        self._last_coords = None
        self._idle_streak = 0

    def compute(self, obs, info: Dict) -> float:
        reward = self.step_penalty
        coords = info.get("agent_coords")
        in_battle = bool(info.get("in_battle"))
        if coords:
            current = (int(coords[0]), int(coords[1]))
            if self._last_coords == current and not in_battle:
                self._idle_streak += 1
                if self._idle_streak >= self.idle_threshold:
                    reward += self.idle_penalty
                    self._idle_streak = 0
            else:
                self._idle_streak = 0
            self._last_coords = current
        return reward
