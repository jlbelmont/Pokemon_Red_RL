from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Hashable, Optional, Tuple


@dataclass
class ArchiveEntry:
    cell_id: Hashable
    path: str
    visits: int
    score: float
    last_seen: int
    label: str


class StateArchive:
    """Disk-backed archive of savestates keyed by coarse cells."""

    def __init__(
        self,
        root_dir: str,
        *,
        max_cells: int = 50000,
        capture_min_visits: int = 3,
    ) -> None:
        self.root_dir = os.path.normpath(root_dir)
        self.max_cells = max(1, int(max_cells))
        self.capture_min_visits = max(1, int(capture_min_visits))
        self.entries: Dict[Hashable, ArchiveEntry] = {}
        self.rng = random.Random(0)
        os.makedirs(self.root_dir, exist_ok=True)

    def _evict(self) -> None:
        if not self.entries:
            return
        # Evict the stalest entry (highest visits, then oldest timestamp).
        victim = max(self.entries.values(), key=lambda e: (e.visits, time.time() - e.last_seen))
        try:
            os.remove(victim.path)
        except OSError:
            pass
        self.entries.pop(victim.cell_id, None)

    def maybe_add(
        self,
        cell_id: Hashable,
        env,
        *,
        reward_estimate: float,
        global_step: int,
        label: Optional[str] = None,
        force: bool = False,
        global_visits: int = 0,
    ) -> Optional[ArchiveEntry]:
        entry = self.entries.get(cell_id)
        if entry:
            entry.visits += 1
            if reward_estimate > entry.score:
                entry.score = reward_estimate
            entry.last_seen = global_step
            return entry
        if not force and global_visits > self.capture_min_visits:
            return None
        if len(self.entries) >= self.max_cells:
            self._evict()
        saver = getattr(getattr(env, "env", env), "save_state", None)
        if not callable(saver):
            return None
        filename = f"cell_{len(self.entries):05d}_{int(time.time()*1000)}.state"
        path = os.path.join(self.root_dir, filename)
        if not saver(path):
            return None
        entry = ArchiveEntry(
            cell_id=cell_id,
            path=path,
            visits=1,
            score=reward_estimate,
            last_seen=global_step,
            label=label or os.path.basename(path),
        )
        self.entries[cell_id] = entry
        return entry

    def sample_frontier(self) -> Optional[ArchiveEntry]:
        if not self.entries:
            return None
        # Prefer entries with few visits; break ties randomly.
        min_visits = min(entry.visits for entry in self.entries.values())
        frontier = [entry for entry in self.entries.values() if entry.visits == min_visits]
        return self.rng.choice(frontier)

    def touch(self, entry: ArchiveEntry, *, global_step: int) -> None:
        entry.visits += 1
        entry.last_seen = global_step
        self.entries[entry.cell_id] = entry
