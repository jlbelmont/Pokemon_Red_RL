from math import hypot
from typing import Set, Tuple


class MapExplorationReward:
    """
    Rewards new coordinates, but less if surrounded by visited tiles, and more if
    farther from the start. This encourages spreading out rather than crawling.

    - base_reward: reward for a brand new coordinate before adjustments
    - neighbor_radius: Chebyshev radius to check around the new tile for visited neighbors
    - neighbor_weight: penalty per visited neighbor found within the radius
    - distance_weight: bonus proportional to normalized distance from the start tile
    - min_reward: lower clamp for new-tile reward after adjustments
    - persist_across_episodes: keep the visited set across resets when True
    """

    def __init__(
        self,
        base_reward: float = 1.0,
        neighbor_radius: int = 1,
        neighbor_weight: float = 0.15,
        distance_weight: float = 0.5,
        min_reward: float = 0.05,
        *,
        persist_across_episodes: bool = False,
    ):
        self.base_reward = base_reward
        self.neighbor_radius = max(0, int(neighbor_radius))
        self.neighbor_weight = float(neighbor_weight)
        self.distance_weight = float(distance_weight)
        self.min_reward = float(min_reward)
        self.persist_across_episodes = bool(persist_across_episodes)
        self.visited: Set[Tuple[int, int]] = set()
        self.start: Tuple[int, int] | None = None
        self.recent_positions = []
        self.stagnation_radius = 2
        self.stagnation_window = 12
        self.stagnation_penalty = 0.02

    def _neighbor_count(self, x: int, y: int) -> int:
        if self.neighbor_radius <= 0 or not self.visited:
            return 0
        r = self.neighbor_radius
        cnt = 0
        for nx in range(x - r, x + r + 1):
            for ny in range(y - r, y + r + 1):
                if (nx, ny) == (x, y):
                    continue
                if (nx, ny) in self.visited:
                    cnt += 1
        return cnt

    def _distance_bonus(self, x: int, y: int) -> float:
        if not self.start:
            return 0.0
        sx, sy = self.start
        d = hypot(float(x - sx), float(y - sy))
        # normalize by max possible (~sqrt(2)*255) to keep in [0,1]
        norm = d / (255.0 * 2**0.5)
        return self.distance_weight * float(max(0.0, min(1.0, norm)))

    def compute(self, obs, info):
        x, y = info.get("agent_coords", (0, 0))
        coord = (int(x), int(y))

        # Initialize the starting tile on first call per episode
        if self.start is None:
            self.start = coord

        if coord in self.visited:
            return 0.0

        # New tile: add to visited and compute shaped reward
        self.visited.add(coord)
        self.recent_positions.append(coord)
        if len(self.recent_positions) > self.stagnation_window:
            self.recent_positions.pop(0)
        shaped_penalty = 0.0
        if len(self.recent_positions) == self.stagnation_window:
            recent_unique = {
                c
                for c in self.recent_positions
                if abs(c[0] - coord[0]) <= self.stagnation_radius
                and abs(c[1] - coord[1]) <= self.stagnation_radius
            }
            if len(recent_unique) <= 3:
                shaped_penalty = self.stagnation_penalty
        neighbors = self._neighbor_count(*coord)
        shaped = (
            self.base_reward
            - self.neighbor_weight * neighbors
            + self._distance_bonus(*coord)
            - shaped_penalty
        )
        return float(max(self.min_reward, shaped))

    def reset(self):
        if not self.persist_across_episodes:
            self.visited.clear()
        self.start = None
        self.recent_positions.clear()
