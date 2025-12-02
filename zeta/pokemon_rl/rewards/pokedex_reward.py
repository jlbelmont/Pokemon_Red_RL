from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple


class PokedexReward:
    """Rewards the agent for catching new Pokémon species and reaching collection milestones."""

    def __init__(
        self,
        *,
        new_species_reward: float = 10.0,
        milestone_rewards: Iterable[Tuple[int, float]] | None = None,
    ) -> None:
        self.new_species_reward = float(new_species_reward)
        milestones = list(milestone_rewards or [])
        self.milestone_rewards: List[Tuple[int, float]] = sorted(
            ((int(count), float(value)) for count, value in milestones),
            key=lambda pair: pair[0],
        )
        self._caught_species: Set[int] = set()
        self._milestones_hit: Set[int] = set()
        self._last_owned_count: int = 0

    def reset(self) -> None:
        # We keep global progress to encourage continued expansion across episodes.
        pass

    def compute(self, obs, info: Dict) -> float:
        reward = 0.0
        species_id = info.get("encounter_species")
        if info.get("caught_pokemon") and species_id:
            species_id = int(species_id)
            if species_id not in self._caught_species:
                self._caught_species.add(species_id)
                reward += self.new_species_reward

        owned_count = int(info.get("pokedex_owned_count") or 0)
        if owned_count != self._last_owned_count:
            for threshold, bonus in self.milestone_rewards:
                if owned_count >= threshold and threshold not in self._milestones_hit:
                    reward += bonus
                    self._milestones_hit.add(threshold)
            self._last_owned_count = owned_count
        return reward
