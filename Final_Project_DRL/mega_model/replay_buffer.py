"""
Replay buffer for DQN with optional recurrent state storage.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from .networks import HierarchicalState


@dataclass
class Transition:
    frames: torch.Tensor
    structured: Optional[torch.Tensor]
    action: int
    reward: float
    reward_components: Dict[str, float]
    done: bool
    next_frames: torch.Tensor
    next_structured: Optional[torch.Tensor]
    state: Optional[HierarchicalState]


class ReplayBuffer:
    def __init__(self, capacity: int, device: torch.device) -> None:
        self.capacity = capacity
        self.device = device
        self.buffer: List[Transition] = []
        self.pos = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def add(
        self,
        frames: torch.Tensor,
        structured: Optional[torch.Tensor],
        action: int,
        reward: float,
        reward_components: Dict[str, float],
        done: bool,
        next_frames: torch.Tensor,
        next_structured: Optional[torch.Tensor],
        state: Optional[HierarchicalState] = None,
    ) -> None:
        if state is not None:
            state = state.detach()
        transition = Transition(
            frames=frames.detach().cpu(),
            structured=structured.detach().cpu() if structured is not None else None,
            action=action,
            reward=reward,
            reward_components=reward_components,
            done=done,
            next_frames=next_frames.detach().cpu(),
            next_structured=next_structured.detach().cpu() if next_structured is not None else None,
            state=state,
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor | List[Optional[HierarchicalState]]]:
        batch = random.sample(self.buffer, batch_size)
        frames = torch.stack([t.frames for t in batch]).to(self.device).squeeze(1)
        next_frames = torch.stack([t.next_frames for t in batch]).to(self.device).squeeze(1)
        structured = (
            torch.stack([t.structured for t in batch if t.structured is not None]).to(self.device)
            if batch[0].structured is not None
            else None
        )
        if structured is not None and structured.dim() == 3:
            structured = structured.squeeze(1)
        next_structured = (
            torch.stack([t.next_structured for t in batch if t.next_structured is not None]).to(
                self.device
            )
            if batch[0].next_structured is not None
            else None
        )
        if next_structured is not None and next_structured.dim() == 3:
            next_structured = next_structured.squeeze(1)
        actions = torch.tensor([t.action for t in batch], device=self.device, dtype=torch.long)
        rewards = torch.tensor([t.reward for t in batch], device=self.device, dtype=torch.float32)
        dones = torch.tensor([t.done for t in batch], device=self.device, dtype=torch.float32)
        states = [t.state for t in batch]
        reward_components = {
            "env": torch.tensor([t.reward_components.get("env", 0.0) for t in batch], device=self.device),
            "rnd": torch.tensor([t.reward_components.get("rnd", 0.0) for t in batch], device=self.device),
            "novel": torch.tensor(
                [t.reward_components.get("novel", 0.0) for t in batch], device=self.device
            ),
            "bayes": torch.tensor(
                [t.reward_components.get("bayes", 0.0) for t in batch], device=self.device
            ),
        }
        return {
            "frames": frames,
            "structured": structured,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "next_frames": next_frames,
            "next_structured": next_structured,
            "states": states,
            "reward_components": reward_components,
        }
