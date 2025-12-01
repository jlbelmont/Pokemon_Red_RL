from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

# Region categories
REGION_UNKNOWN = 0
REGION_TOWN = 1
REGION_ROUTE = 2
REGION_INTERIOR = 3
REGION_GYM = 4
REGION_DUNGEON = 5

_REGION_LOOKUP: Dict[int, int] = {
    0x00: REGION_TOWN,  # Pallet Town
    0x01: REGION_INTERIOR,
    0x02: REGION_INTERIOR,
    0x03: REGION_INTERIOR,
    0x04: REGION_INTERIOR,
    0x05: REGION_TOWN,  # Viridian City
    0x06: REGION_GYM,
    0x07: REGION_INTERIOR,
    0x08: REGION_INTERIOR,
    0x09: REGION_INTERIOR,
    0x0A: REGION_ROUTE,
    0x0B: REGION_ROUTE,
    0x0C: REGION_DUNGEON,  # Viridian Forest
    0x0D: REGION_INTERIOR,
    0x0E: REGION_INTERIOR,
    0x10: REGION_ROUTE,
    0x11: REGION_GYM,
    0x14: REGION_GYM,
    0x18: REGION_GYM,
    0x1C: REGION_ROUTE,
    0x1D: REGION_ROUTE,
    0x1E: REGION_DUNGEON,
    0x20: REGION_TOWN,
    0x21: REGION_INTERIOR,
    0x22: REGION_INTERIOR,
    0x24: REGION_GYM,
    0x30: REGION_ROUTE,
    0x33: REGION_GYM,
    0x34: REGION_GYM,
    0x63: REGION_DUNGEON,  # Elite Four chambers
    0x64: REGION_DUNGEON,
    0x65: REGION_DUNGEON,
    0x66: REGION_DUNGEON,
    0x67: REGION_DUNGEON,
}

_REGION_COUNT = 6  # unknown + five classes


def region_one_hot(map_id: int) -> np.ndarray:
    vec = np.zeros(_REGION_COUNT, dtype=np.float32)
    region = _REGION_LOOKUP.get(map_id, REGION_UNKNOWN)
    vec[region] = 1.0
    return vec


def _normalize_coords(coords: Tuple[int, int]) -> Tuple[float, float]:
    x, y = coords
    return float(x) / 255.0, float(y) / 255.0


def extract_map_features(info: Dict) -> np.ndarray:
    """Encode high-level context (region, progress, inventory) for the policy."""
    coords = info.get("agent_coords") or (0, 0)
    norm_x, norm_y = _normalize_coords(coords)

    map_id = int(info.get("map_id") or 0)
    region_vec = region_one_hot(map_id)
    outdoor_flag = 1.0 if _REGION_LOOKUP.get(map_id, REGION_UNKNOWN) in {REGION_TOWN, REGION_ROUTE} else 0.0

    badge_count = float(info.get("badge_count", 0)) / 8.0
    champion_flag = 1.0 if info.get("champion_defeated") else 0.0

    story_flags = info.get("story_flags") or {}
    story_progress = 0.0
    if isinstance(story_flags, dict) and story_flags:
        on_flags = sum(1 for value in story_flags.values() if value)
        story_progress = on_flags / float(len(story_flags))

    key_item_ids = info.get("key_item_ids") or []
    key_item_count = min(len(key_item_ids), 20) / 20.0

    pokedex_owned = float(info.get("pokedex_owned_count") or 0) / 151.0

    in_battle = 1.0 if info.get("in_battle") else 0.0
    battle_type = float(info.get("battle_type") or 0) / 10.0
    recent_catch = 1.0 if info.get("last_battle_result") == "caught" else 0.0

    hp_info = info.get("first_pokemon_hp") or {}
    hp_current = float(hp_info.get("current") or 0)
    hp_max = float(hp_info.get("max") or 1)
    hp_ratio = hp_current / hp_max if hp_max > 0 else 0.0

    enemy_hp_info = info.get("enemy_hp") or {}
    enemy_hp_current = float(enemy_hp_info.get("current") or 0)
    enemy_hp_max = float(enemy_hp_info.get("max") or 1)
    enemy_hp_ratio = enemy_hp_current / enemy_hp_max if enemy_hp_max > 0 else 0.0

    episode_unique = float(info.get("episode_unique_tiles") or 0.0)
    total_unique = float(info.get("total_unique_tiles") or 0.0)
    tile_visit_count = float(info.get("tile_visit_count") or 0.0)
    episode_revisit_ratio = float(info.get("episode_revisit_ratio") or 0.0)

    episode_unique_norm = min(episode_unique / 1024.0, 1.0)
    total_unique_norm = min(total_unique / 4096.0, 1.0)
    tile_visit_norm = min(tile_visit_count / 10.0, 1.0)
    episode_revisit_ratio = min(max(episode_revisit_ratio, 0.0), 1.0)

    feature_vec = np.concatenate(
        [
            np.array(
                [
                    norm_x,
                    norm_y,
                    float(map_id) / 255.0,
                    outdoor_flag,
                    badge_count,
                    champion_flag,
                    story_progress,
                    key_item_count,
                    pokedex_owned,
                    in_battle,
                    battle_type,
                    recent_catch,
                    hp_ratio,
                    enemy_hp_ratio,
                    episode_unique_norm,
                    total_unique_norm,
                    tile_visit_norm,
                    episode_revisit_ratio,
                ],
                dtype=np.float32,
            ),
            region_vec,
        ]
    )
    return feature_vec
