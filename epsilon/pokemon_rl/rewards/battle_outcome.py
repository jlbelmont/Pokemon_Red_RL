class BattleOutcomeReward:
    """Reward shaping for trainer battles based on win/loss outcomes."""

    def __init__(
        self,
        win_reward: float = 20.0,
        loss_penalty: float = -15.0,
    ) -> None:
        self.win_reward = float(win_reward)
        self.loss_penalty = float(loss_penalty)
        self._last_outcome = None

    def compute(self, obs, info) -> float:
        battle_result = info.get("battle_result")
        reward = 0.0
        if battle_result == "won" and self._last_outcome != "won":
            reward += self.win_reward
        elif battle_result in {"lost", "blackout"} and self._last_outcome not in {"lost", "blackout"}:
            reward += self.loss_penalty
        self._last_outcome = battle_result
        return reward

    def reset(self) -> None:
        self._last_outcome = None
