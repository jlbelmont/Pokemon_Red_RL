class BadgeReward:
    """Rewards the agent for obtaining new gym badges."""

    BADGE_NAMES = (
        "boulder",
        "cascade",
        "thunder",
        "rainbow",
        "soul",
        "marsh",
        "volcano",
        "earth",
    )

    def __init__(self, reward_per_badge: float = 200.0) -> None:
        self.reward_per_badge = float(reward_per_badge)
        self._awarded_mask = 0

    def compute(self, obs, info) -> float:
        badge_bits = int(info.get("badge_bits", 0))
        new_mask = badge_bits & ~self._awarded_mask
        if new_mask:
            newly_obtained = bin(new_mask).count("1")
            self._awarded_mask |= new_mask
            return float(newly_obtained) * self.reward_per_badge
        return 0.0

    def reset(self) -> None:
        # Do not clear _awarded_mask; badges persist across episodes in long runs.
        pass
