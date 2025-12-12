"""
Random Network Distillation for intrinsic motivation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class RunningStats:
    mean: torch.Tensor
    var: torch.Tensor
    count: torch.Tensor

    @classmethod
    def create(cls, device: torch.device) -> "RunningStats":
        return cls(
            mean=torch.zeros(1, device=device),
            var=torch.ones(1, device=device),
            count=torch.tensor(1e-4, device=device),
        )

    def update(self, x: torch.Tensor) -> None:
        batch_mean = x.mean()
        batch_var = x.var(unbiased=False)
        batch_count = torch.tensor(x.numel(), device=x.device, dtype=torch.float32)

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count

        self.mean = new_mean
        self.var = m2 / tot_count
        self.count = tot_count

    def normalize(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + eps)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, output_dim),
    )


class RNDModel(nn.Module):
    """
    Lightweight RND module.

    Expects latent features from the policy encoder (e.g., SSM output).
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 256,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.target = _mlp(feature_dim, hidden_dim, output_dim)
        self.predictor = _mlp(feature_dim, hidden_dim, output_dim)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.normalize = normalize
        self.running_stats: Optional[RunningStats] = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        target = self.target(features).detach()
        pred = self.predictor(features)
        return F.mse_loss(pred, target, reduction="none").mean(dim=-1)

    def intrinsic_reward(self, features: torch.Tensor) -> torch.Tensor:
        rewards = self.forward(features)
        if not self.normalize:
            return rewards
        if self.running_stats is None:
            self.running_stats = RunningStats.create(features.device)
        self.running_stats.update(rewards.detach())
        return self.running_stats.normalize(rewards)

    def loss(self, features: torch.Tensor) -> torch.Tensor:
        # Predictor learns to match a fixed target.
        return self.forward(features)
