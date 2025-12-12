"""
WRAM flag decoding and embedding utilities.

These helpers turn the structured dictionary observations from pokemonred_puffer
into compact feature vectors for the hierarchical network.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn


def _ensure_batch(x: torch.Tensor) -> torch.Tensor:
    return x if x.dim() > 1 else x.unsqueeze(0)


class FlagEncoder(nn.Module):
    """
    Encodes WRAM flags and lightweight scalar features.

    Inputs are dict observations from RedGymEnv or a batch of such dicts.
    """

    def __init__(
        self,
        event_dim: int = 320,
        milestone_indices: Sequence[int] | None = None,
        embed_dim: int = 64,
        extra_scalar_keys: Sequence[str] = (
            "direction",
            "blackout_map_id",
            "battle_type",
            "map_id",
            "rival_3",
            "game_corner_rocket",
            "saffron_guard",
            "lapras",
        ),
    ) -> None:
        super().__init__()
        self.event_dim = event_dim
        self.milestone_indices = (
            list(milestone_indices) if milestone_indices is not None else list(range(min(16, event_dim)))
        )
        self.extra_scalar_keys = list(extra_scalar_keys)
        self.event_embed = nn.Linear(event_dim, embed_dim)
        self.output_dim = embed_dim + len(self.extra_scalar_keys)

    def forward(
        self, obs: Dict[str, torch.Tensor] | List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = self._batchify(obs)
        events = batch.get("events", torch.zeros(len(batch), self.event_dim, device=self._device(batch)))
        events = _ensure_batch(events.float())
        if events.shape[1] != self.event_dim:
            raise ValueError(f"Expected events dim {self.event_dim}, got {events.shape}")

        event_feat = self.event_embed(events)
        scalars = []
        for key in self.extra_scalar_keys:
            value = batch.get(key)
            if value is None:
                scalars.append(torch.zeros(len(batch), 1, device=event_feat.device))
            else:
                scalars.append(_ensure_batch(value.float()))
        scalar_feat = torch.cat(scalars, dim=-1) if scalars else torch.zeros(
            len(batch), 0, device=event_feat.device
        )
        features = torch.cat([event_feat, scalar_feat], dim=-1)
        milestone_flags = events[:, self.milestone_indices]
        return features, milestone_flags

    def _batchify(
        self, obs: Dict[str, torch.Tensor] | List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        if isinstance(obs, list):
            keys = obs[0].keys()
            batch: Dict[str, torch.Tensor] = {}
            for key in keys:
                vals = [torch.as_tensor(o[key]) for o in obs if key in o]
                batch[key] = torch.stack([_ensure_batch(v)[0] for v in vals])
            return batch

        batch_obs: Dict[str, torch.Tensor] = {}
        for key, value in obs.items():
            tensor = torch.as_tensor(value)
            batch_obs[key] = _ensure_batch(tensor)
        return batch_obs

    def _device(self, batch: Dict[str, torch.Tensor]) -> torch.device:
        for v in batch.values():
            return v.device
        return torch.device("cpu")
