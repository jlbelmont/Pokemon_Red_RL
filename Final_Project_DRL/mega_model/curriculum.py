"""
Curriculum manager for savestate-based training and swarming.

Mirrors the savestate behavior in pokemonred_puffer by saving/loading the same
.state format produced by PyBoy/RedGymEnv, while layering registry + sampling
logic for curriculum tiers (early/mid/late).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple


@dataclass
class SavestateEntry:
    tag: str
    path: str
    tier: str = "mid"
    score: float = 0.0
    uses: int = 0


class CurriculumManager:
    """
    Tracks savestates, records milestone-triggered saves, and samples states for swarming.
    """

    def __init__(
        self,
        base_dir: Path,
        run_name: str,
        tier_weights: Optional[Dict[str, float]] = None,
        window: int = 50,
    ) -> None:
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.run_name = run_name
        self.run_dir = self.base_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.run_dir / "registry.json"
        self.entries: Dict[str, SavestateEntry] = {}
        self.history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self.tier_weights = tier_weights or {"early": 0.5, "mid": 0.3, "late": 0.2}
        self._load_registry()

    # ---------- registry helpers ----------
    def _load_registry(self) -> None:
        if self.registry_path.exists():
            data = json.loads(self.registry_path.read_text())
            for tag, meta in data.items():
                self.entries[tag] = SavestateEntry(**meta)

    def _save_registry(self) -> None:
        payload = {k: asdict(v) for k, v in self.entries.items()}
        self.registry_path.write_text(json.dumps(payload, indent=2))

    # ---------- tier heuristics ----------
    @staticmethod
    def _tier_from_tag(tag: str) -> str:
        if "pallet" in tag or "map_00" in tag or "badge_0" in tag:
            return "early"
        if "badge_" in tag:
            try:
                badge = int(tag.split("_")[-1])
                if badge >= 6:
                    return "late"
                return "mid"
            except Exception:
                return "mid"
        if "elite" in tag or "champion" in tag:
            return "late"
        return "mid"

    # ---------- public API ----------
    def list_states(self) -> List[Path]:
        return [Path(e.path) for e in self.entries.values() if Path(e.path).exists()]

    def register_state(self, tag: str, path: Path, tier: Optional[str] = None) -> None:
        tier = tier or self._tier_from_tag(tag)
        entry = SavestateEntry(tag=tag, path=str(path), tier=tier, score=0.0, uses=0)
        self.entries[tag] = entry
        self._save_registry()

    def record_outcome(self, tag: Optional[str], reward: float) -> None:
        if tag is None or tag not in self.entries:
            return
        self.history[tag].append(reward)
        hist = self.history[tag]
        self.entries[tag].score = sum(hist) / max(len(hist), 1)
        self._save_registry()

    def maybe_save_milestone(self, env, tag: str, step: int) -> Optional[Path]:
        """
        Save a savestate using the native env API and register it.
        """
        filename = f"{tag}_{step:07d}.state"
        path = self.run_dir / filename
        try:
            env._save_emulator_state(path)
        except Exception:
            return None
        self.register_state(tag, path, tier=self._tier_from_tag(tag))
        return path

    def sample_state(self) -> Optional[Tuple[str, Path]]:
        if not self.entries:
            return None
        # group by tier
        tier_to_entries: Dict[str, List[SavestateEntry]] = defaultdict(list)
        for e in self.entries.values():
            if Path(e.path).exists():
                tier_to_entries[e.tier].append(e)
        if not tier_to_entries:
            return None
        # choose tier then state
        tiers = list(tier_to_entries.keys())
        weights = [self.tier_weights.get(t, 0.1) for t in tiers]
        total = sum(weights)
        weights = [w / total for w in weights]
        tier_choice = random.choices(tiers, weights=weights, k=1)[0]
        candidates = tier_to_entries[tier_choice]
        scores = []
        for e in candidates:
            base = max(0.05, 1.0 + e.score)
            # Bias toward maps further from spawn (higher map ids) and later badges.
            map_boost = 1.0
            badge_boost = 1.0
            if e.tag.startswith("map_"):
                try:
                    map_id = int(e.tag.split("_")[1])
                    # gently boost farther maps
                    map_boost = 1.0 + (map_id / 200.0)
                except Exception:
                    pass
            if e.tag.startswith("badge_"):
                try:
                    badge_id = int(e.tag.split("_")[1])
                    badge_boost = 1.0 + (badge_id * 0.1)
                except Exception:
                    pass
            scores.append(base * map_boost * badge_boost)
        total_score = sum(scores)
        probs = [s / total_score for s in scores]
        choice = random.choices(candidates, weights=probs, k=1)[0]
        choice.uses += 1
        self._save_registry()
        return choice.tag, Path(choice.path)

    def summary(self) -> str:
        return f"savestates={len(self.entries)}"
