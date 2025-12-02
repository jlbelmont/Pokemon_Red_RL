from __future__ import annotations

from typing import Dict


class RevisitPenalty:
    """
    Applies a light penalty whenever the agent returns to an already visited tile.

    - base_penalty: magnitude applied on the second visit (converted to negative)
    - excess_scale: extra penalty multiplier for each additional revisit beyond the second
    - ratio_scale: additional penalty scaled by the per-episode revisit ratio
    """

    def __init__(
        self,
        *,
        base_penalty: float = 0.02,
        excess_scale: float = 0.01,
        ratio_scale: float = 0.015,
    ) -> None:
        self.base_penalty = -abs(float(base_penalty))
        self.excess_scale = max(0.0, float(excess_scale))
        self.ratio_scale = -abs(float(ratio_scale))

    def compute(self, obs, info: Dict) -> float:
        penalty = 0.0
        visit_count = int(info.get("tile_visit_count") or 0)
        if visit_count > 1:
            excess = max(0, visit_count - 2)
            penalty += self.base_penalty * (1.0 + self.excess_scale * excess)

        revisit_ratio = float(info.get("episode_revisit_ratio") or 0.0)
        if revisit_ratio > 0.0:
            penalty += self.ratio_scale * revisit_ratio
        return float(penalty)

    def reset(self) -> None:
        # No internal state to reset.
        return None
