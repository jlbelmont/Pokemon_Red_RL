class ChampionReward:
    """Reward issued once when the champion is defeated."""

    def __init__(self, reward: float = 1000.0) -> None:
        self.reward = float(reward)
        self._awarded = False

    def compute(self, obs, info) -> float:
        if info.get("champion_defeated") and not self._awarded:
            self._awarded = True
            return self.reward
        return 0.0

    def reset(self) -> None:
        # Do not reset _awarded so long runs keep the reward only once.
        pass
