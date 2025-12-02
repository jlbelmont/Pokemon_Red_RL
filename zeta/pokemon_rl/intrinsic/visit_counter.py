from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, Optional, Tuple

import numpy as np


CellId = Tuple[int, int, int, Tuple[str, ...]]
TransitionKey = Tuple[int, int]


@dataclass
class VisitStats:
    cell_id: CellId
    n_global: int
    n_episode: int
    transition_key: Optional[TransitionKey]
    transition_visits: int


class VisitCounter:
    """Tracks coarse cell visitation counts and map transitions."""

    def __init__(
        self,
        num_envs: int,
        *,
        bin_size: int = 4,
        include_story: bool = False,
        story_allowlist: Optional[Iterable[str]] = None,
    ) -> None:
        self.num_envs = max(1, int(num_envs))
        self.bin_size = max(1, int(bin_size))
        self.include_story = bool(include_story)
        self.story_allowlist = None
        if story_allowlist:
            self.story_allowlist = {name.lower() for name in story_allowlist}
        self.global_counts: Dict[CellId, int] = defaultdict(int)
        self.episode_counts: list[Dict[CellId, int]] = [defaultdict(int) for _ in range(self.num_envs)]
        self.transition_counts: Dict[TransitionKey, int] = defaultdict(int)
        self.prev_map_ids: list[Optional[int]] = [None for _ in range(self.num_envs)]

    def _story_hash(self, info: dict) -> Tuple[str, ...]:
        if not self.include_story:
            return tuple()
        flags = info.get("story_flags") or {}
        if not isinstance(flags, dict):
            return tuple()
        active: list[str] = []
        for name, active_flag in flags.items():
            if not active_flag:
                continue
            key = str(name).lower()
            if self.story_allowlist and key not in self.story_allowlist:
                continue
            active.append(key)
        if not active:
            return tuple()
        active.sort()
        # Keep a short prefix to avoid ballooning the cell id.
        return tuple(active[:16])

    def _cell_id_from_info(self, info: dict) -> CellId:
        map_id = int(info.get("map_id") or 0)
        coords = info.get("agent_coords") or (0, 0)
        x = int(coords[0]) // self.bin_size
        y = int(coords[1]) // self.bin_size
        return (map_id, x, y, self._story_hash(info))

    def start_episode(self, env_id: int) -> None:
        idx = env_id % self.num_envs
        self.episode_counts[idx].clear()
        self.prev_map_ids[idx] = None

    def observe(self, env_id: int, info: dict) -> VisitStats:
        idx = env_id % self.num_envs
        cell_id = self._cell_id_from_info(info)
        self.global_counts[cell_id] += 1
        n_global = self.global_counts[cell_id]
        episode_counter = self.episode_counts[idx]
        episode_counter[cell_id] += 1
        n_episode = episode_counter[cell_id]

        transition_key: Optional[TransitionKey] = None
        transition_visits = 0
        current_map = cell_id[0]
        prev_map = self.prev_map_ids[idx]
        if prev_map is None:
            self.prev_map_ids[idx] = current_map
        elif prev_map != current_map:
            transition_key = (prev_map, current_map)
            self.transition_counts[transition_key] += 1
            transition_visits = self.transition_counts[transition_key]
            self.prev_map_ids[idx] = current_map
        else:
            transition_visits = self.transition_counts.get((prev_map, current_map), 0)

        return VisitStats(
            cell_id=cell_id,
            n_global=n_global,
            n_episode=n_episode,
            transition_key=transition_key,
            transition_visits=transition_visits,
        )

    @staticmethod
    def intrinsic_bonus(n_global: int, n_episode: int, alpha: float, beta: float, epsilon: float = 1.0) -> float:
        inv = alpha / max(1.0, float(n_global) + epsilon)
        first_visit = beta if n_episode == 1 else 0.0
        return inv + first_visit

    @staticmethod
    def transition_bonus(
        transition_visits: int,
        base_bonus: float,
        max_visits: int,
    ) -> float:
        if transition_visits <= 0 or base_bonus <= 0.0:
            return 0.0
        if max_visits > 0 and transition_visits > max_visits:
            return 0.0
        return base_bonus / float(transition_visits)

    def state_dict(self) -> Dict[str, object]:
        return {
            "global_counts": list(self.global_counts.items()),
            "transition_counts": list(self.transition_counts.items()),
            "prev_map_ids": list(self.prev_map_ids),
            "bin_size": self.bin_size,
            "include_story": self.include_story,
            "story_allowlist": sorted(self.story_allowlist) if self.story_allowlist else None,
        }

    def load_state_dict(self, state: Optional[Dict[str, object]]) -> None:
        if not state:
            return
        global_counts = state.get("global_counts") or []
        self.global_counts = defaultdict(int)
        for cell_id, count in global_counts:
            try:
                self.global_counts[tuple(cell_id)] = int(count)
            except Exception:
                continue
        transition_counts = state.get("transition_counts") or []
        self.transition_counts = defaultdict(int)
        for key, count in transition_counts:
            try:
                self.transition_counts[tuple(key)] = int(count)
            except Exception:
                continue
        prev_ids = state.get("prev_map_ids") or []
        for idx in range(min(len(prev_ids), len(self.prev_map_ids))):
            value = prev_ids[idx]
            self.prev_map_ids[idx] = None if value is None else int(value)


class EpisodicLatentMemory:
    """Per-env episodic memory that emits a bonus for latents far from prior states."""

    def __init__(self, num_envs: int, max_items: int = 256, distance_threshold: float = 0.5) -> None:
        self.num_envs = max(1, int(num_envs))
        self.max_items = max(1, int(max_items))
        self.distance_threshold = max(1e-6, float(distance_threshold))
        self._buffers: list[list[np.ndarray]] = [[] for _ in range(self.num_envs)]

    def reset(self, env_id: int) -> None:
        self._buffers[env_id % self.num_envs].clear()

    def bonus(self, env_id: int, latent: np.ndarray) -> float:
        idx = env_id % self.num_envs
        buffer = self._buffers[idx]
        vec = np.asarray(latent, dtype=np.float32).ravel()
        if vec.size == 0:
            return 0.0
        if not buffer:
            buffer.append(vec)
            return 1.0
        distances = [float(np.linalg.norm(vec - other)) for other in buffer]
        min_dist = min(distances) if distances else float("inf")
        if min_dist >= self.distance_threshold:
            buffer.append(vec)
            if len(buffer) > self.max_items:
                buffer.pop(0)
            return min(2.0, min_dist / self.distance_threshold)
        return 0.0

    def state_dict(self) -> Dict[str, object]:
        return {
            "buffers": [
                [vec.tolist() for vec in buffer]
                for buffer in self._buffers
            ]
        }

    def load_state_dict(self, state: Optional[Dict[str, object]]) -> None:
        if not state:
            return
        buffers = state.get("buffers") or []
        restored: list[list[np.ndarray]] = []
        for env_buffer in buffers:
            env_list: list[np.ndarray] = []
            for vec in env_buffer or []:
                try:
                    arr = np.asarray(vec, dtype=np.float32)
                except Exception:
                    continue
                env_list.append(arr)
                if len(env_list) >= self.max_items:
                    break
            restored.append(env_list)
        while len(restored) < self.num_envs:
            restored.append([])
        self._buffers = restored[: self.num_envs]
