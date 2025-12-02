from __future__ import annotations

import numpy as np


class NoveltyReward:
    """Soft novelty reward using downsampled/quantised observations with decay.

    Optionally persists the visitation counts across episodes when
    `persist_across_episodes` is enabled.
    """

    def __init__(
        self,
        base_reward: float = 1.0,
        decay: float = 0.9,
        min_reward: float = 0.0,
        sample_stride: int = 4,
        quantisation: int = 32,
        *,
        persist_across_episodes: bool = False,
    ) -> None:
        self.base_reward = float(base_reward)
        self.decay = float(decay)
        self.min_reward = float(min_reward)
        self.sample_stride = max(1, int(sample_stride))
        self.quantisation = max(1, int(quantisation))
        self.persist_across_episodes = bool(persist_across_episodes)
        self._counts: dict[bytes, int] = {}

    def _tokenise(self, obs: np.ndarray) -> bytes:
        arr = obs
        if arr.ndim == 3 and arr.shape[-1] > 1:
            arr = arr.mean(axis=-1)
        elif arr.ndim == 3:
            arr = arr[..., 0]
        arr = arr.astype(np.float32)[:: self.sample_stride, :: self.sample_stride]
        arr = np.floor(arr / self.quantisation)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr.tobytes()

    def compute(self, obs, info) -> float:
        token = self._tokenise(obs)
        count = self._counts.get(token, 0)
        reward = self.base_reward * (self.decay ** count)
        reward = max(self.min_reward, reward)
        self._counts[token] = count + 1
        return float(reward)

    def reset(self) -> None:
        if not self.persist_across_episodes:
            self._counts.clear()
