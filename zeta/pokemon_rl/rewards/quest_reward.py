from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple


class QuestReward:
    """Rewards quest chains via ordered stages and prerequisite tracking."""

    def __init__(self, quests: Iterable[Dict] | None = None) -> None:
        self.quests: List[Dict] = []
        for quest in quests or []:
            processed = self._normalise_quest(quest)
            if processed:
                self.quests.append(processed)
        self._quest_progress: Dict[str, int] = {quest["name"]: 0 for quest in self.quests}
        self._completed_quests: set[str] = set()

    def _normalise_stage(self, stage: Dict, default_reward: float) -> Optional[Dict]:
        if not isinstance(stage, dict):
            return None
        map_id = stage.get("map_id")
        story_flag = stage.get("story_flag")
        coords = stage.get("coords")
        map_name = stage.get("map_name")
        town = stage.get("town") or stage.get("town_label")
        badge = stage.get("badge")
        if (
            map_id is None
            and story_flag is None
            and not coords
            and not map_name
            and not town
            and not badge
        ):
            return None
        reward = float(stage.get("reward", default_reward))
        requires = stage.get("requires") or []
        if isinstance(requires, str):
            requires = [requires]
        requires = [str(req).strip().lower() for req in requires if str(req).strip()]
        label = stage.get("label")
        map_name_value = map_name.strip().lower() if isinstance(map_name, str) else None
        town_value = town.strip().lower() if isinstance(town, str) else None
        badge_value = badge.strip().lower() if isinstance(badge, str) else None
        return {
            "map_id": map_id,
            "map_name": map_name_value,
            "coords": tuple(coords) if coords else None,
            "story_flag": story_flag,
            "town": town_value,
            "badge": badge_value,
            "reward": reward,
            "label": label,
            "requires": set(requires),
        }

    def _normalise_quest(self, quest: Dict) -> Optional[Dict]:
        if not isinstance(quest, dict):
            return None
        name = str(quest.get("name") or f"quest_{len(self.quests) + 1}")
        repeatable = bool(quest.get("repeatable", False))
        default_reward = float(quest.get("reward", 50.0))
        requires = quest.get("requires") or []
        if isinstance(requires, str):
            requires = [requires]
        requires = [str(req).strip().lower() for req in requires if str(req).strip()]
        stages_raw = quest.get("stages")
        stages: List[Dict] = []
        if isinstance(stages_raw, list) and stages_raw:
            for stage_def in stages_raw:
                processed = self._normalise_stage(stage_def, default_reward)
                if processed:
                    stages.append(processed)
        else:
            processed = self._normalise_stage(quest, default_reward)
            if processed:
                stages.append(processed)
        if not stages:
            return None
        for idx, stage in enumerate(stages):
            if not stage.get("label"):
                stage["label"] = f"{name}_stage_{idx + 1}"
        return {
            "name": name,
            "repeatable": repeatable,
            "requires": set(requires),
            "stages": stages,
        }

    def reset(self) -> None:
        # Quest progress is persistent across episodes to reflect long-term advancement.
        pass

    def _requirements_met(self, requirements: set[str], completed: set[str]) -> bool:
        if not requirements:
            return True
        return requirements.issubset(completed)

    def _match_stage(self, stage: Dict, map_id: int, coords, story_flags: Dict, info: Dict) -> bool:
        map_match = True
        if stage["map_id"] is not None:
            map_match = stage["map_id"] == map_id
        if map_match and stage["map_name"]:
            current_name = str(info.get("map_name") or "").strip().lower()
            map_match = current_name == stage["map_name"]
        if map_match and stage["town"]:
            town_flag = info.get("new_town_visited")
            town_candidates = [
                str(town_flag).strip().lower() if town_flag else "",
                str(info.get("map_name") or "").strip().lower(),
            ]
            map_match = stage["town"] in town_candidates
        coord_match = True
        if stage["coords"] and coords:
            coord_match = tuple(coords) == tuple(stage["coords"])
        flag_match = True
        if stage["story_flag"]:
            flag_match = bool(story_flags.get(stage["story_flag"], False))
        badge_match = True
        if stage["badge"]:
            badges = info.get("badges") or {}
            badge_match = bool(badges.get(stage["badge"], False))
        return map_match and coord_match and flag_match and badge_match

    def compute(self, obs, info: Dict) -> float:
        if not self.quests:
            return 0.0
        reward = 0.0
        triggered_labels: List[str] = []
        coords = info.get("agent_coords")
        map_id = int(info.get("map_id", -1))
        story_flags = info.get("story_flags") or {}
        for quest in self.quests:
            quest_name = quest["name"]
            if quest_name in self._completed_quests and not quest["repeatable"]:
                continue
            if not self._requirements_met(quest["requires"], self._completed_quests):
                continue
            stage_index = self._quest_progress.get(quest_name, 0)
            if stage_index >= len(quest["stages"]):
                if quest["repeatable"]:
                    stage_index = 0
                    self._quest_progress[quest_name] = 0
                else:
                    continue
            stage = quest["stages"][stage_index]
            if not self._requirements_met(stage["requires"], self._completed_quests):
                continue
            if self._match_stage(stage, map_id, coords, story_flags, info):
                reward += stage["reward"]
                stage_index += 1
                self._quest_progress[quest_name] = stage_index
                triggered_labels.append(f"{quest_name}.{stage['label']}")
                if stage_index >= len(quest["stages"]):
                    self._completed_quests.add(quest_name)
        if triggered_labels and isinstance(info, dict):
            quest_events = info.setdefault("quest_events", [])
            quest_events.extend(triggered_labels)
        return reward
