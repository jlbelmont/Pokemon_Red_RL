from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


class CurriculumReward:
    """Rewards reaching curriculum goals (map + optional coordinate regions)."""

    def __init__(self, goals: Iterable[Dict] | None = None) -> None:
        self.goals: List[Dict] = []
        for goal in goals or []:
            if not isinstance(goal, dict):
                continue
            processed = {
                "name": goal.get("name", "goal"),
                "map_id": goal.get("map_id"),
                "coords": goal.get("coords"),
                "radius": goal.get("radius", 0),
                "reward": float(goal.get("reward", 50.0)),
                "repeatable": bool(goal.get("repeatable", False)),
            }
            if processed["map_id"] is None:
                continue
            self.goals.append(processed)
        self._completed: set[str] = set()

    def reset(self) -> None:
        # Keep curriculum progress unless goal marked repeatable.
        pass

    def compute(self, obs, info: Dict) -> float:
        coords = info.get("agent_coords")
        if not self.goals or not coords:
            return 0.0
        cx, cy = int(coords[0]), int(coords[1])
        map_id = int(info.get("map_id") or 0)
        reward = 0.0

        for goal in self.goals:
            if goal["map_id"] != map_id:
                continue
            name = goal["name"]
            if not goal["repeatable"] and name in self._completed:
                continue
            target = goal.get("coords")
            radius = int(goal.get("radius") or 0)
            if target:
                tx, ty = int(target[0]), int(target[1])
                if abs(cx - tx) + abs(cy - ty) > radius:
                    continue
            reward += goal["reward"]
            if not goal["repeatable"]:
                self._completed.add(name)
        return reward
