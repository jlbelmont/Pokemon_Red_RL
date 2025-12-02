from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class RunningRewardNormalizer:
    """Keeps a running mean/std of intrinsic rewards for stabilisation."""

    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        batch_mean = float(values.mean().item())
        batch_var = float(values.var(unbiased=False).item())
        batch_count = float(values.numel())
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.mean = new_mean
        self.var = M2 / max(total, 1e-8)
        self.count = total

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        std = math.sqrt(max(self.var, 1e-6))
        return (values - self.mean) / std

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: Optional[dict]) -> None:
        if not state:
            return
        self.mean = float(state.get("mean", self.mean))
        self.var = float(state.get("var", self.var))
        self.count = float(state.get("count", self.count))


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, output_dim),
    )


class RNDModule(nn.Module):
    """Random Network Distillation on top of the policy latent."""

    def __init__(self, latent_dim: int, hidden_dim: int = 256, output_dim: Optional[int] = None):
        super().__init__()
        output_dim = output_dim or hidden_dim
        self.target = _build_mlp(latent_dim, hidden_dim, output_dim)
        for param in self.target.parameters():
            param.requires_grad_(False)
        self.predictor = _build_mlp(latent_dim, hidden_dim, output_dim)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        target = self.target(latents)
        pred = self.predictor(latents)
        return ((pred - target) ** 2).mean(dim=1)
