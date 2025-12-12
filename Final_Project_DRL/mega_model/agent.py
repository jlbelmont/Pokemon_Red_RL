"""
SlimHierarchicalDQN agent that wraps the hierarchical encoder, replay buffer,
intrinsic rewards, and Bayesian monitoring utilities.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .bayes_quests import BayesianQuestMonitor
from .flags import FlagEncoder
from .networks import HierarchicalState, SlimHierarchicalQNetwork
from .replay_buffer import ReplayBuffer
from .rnd import RNDModel


@dataclass
class AgentConfig:
    num_actions: int
    frame_stack: int
    structured_dim: int
    replay_capacity: int = 100_000
    gamma: float = 0.99
    lr: float = 3e-4
    lr_rnd: float = 1e-4
    grad_clip: float = 5.0
    target_update_interval: int = 1000
    rnd_weight: float = 0.1
    novelty_weight: float = 0.05
    bayes_weight: float = 0.05
    milestones: Sequence[str] = ()


class SlimHierarchicalDQN:
    def __init__(
        self,
        config: AgentConfig,
        device: torch.device,
        cnn_channels=None,
        gru_size: int = 128,
        lstm_size: int = 128,
        ssm_size: int = 128,
        structured_hidden: int = 128,
    ) -> None:
        self.device = device
        self.config = config
        self.network = SlimHierarchicalQNetwork(
            frame_channels=config.frame_stack,
            num_actions=config.num_actions,
            structured_dim=config.structured_dim,
            cnn_channels=cnn_channels or (32, 64, 64),
            gru_size=gru_size,
            lstm_size=lstm_size,
            ssm_size=ssm_size,
            structured_hidden=structured_hidden,
        ).to(device)
        self.target_network = SlimHierarchicalQNetwork(
            frame_channels=config.frame_stack,
            num_actions=config.num_actions,
            structured_dim=config.structured_dim,
            cnn_channels=cnn_channels or (32, 64, 64),
            gru_size=gru_size,
            lstm_size=lstm_size,
            ssm_size=ssm_size,
            structured_hidden=structured_hidden,
        ).to(device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.lr)
        self.rnd = RNDModel(feature_dim=self.network.ssm_size).to(device)  # type: ignore[attr-defined]
        self.rnd_opt = torch.optim.Adam(self.rnd.predictor.parameters(), lr=config.lr_rnd)

        self.flag_encoder = FlagEncoder()
        self.bayes_monitor = BayesianQuestMonitor(
            milestones=config.milestones, target=0.5, shaping_weight=config.bayes_weight
        )

        self.replay = ReplayBuffer(capacity=config.replay_capacity, device=device)
        self.global_step = 0
        self.episodes = defaultdict(int)
        self.episode_returns = defaultdict(float)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 1e-6
        self.novelty_counts: Dict[int, Counter] = defaultdict(Counter)

    def reset_hidden(self, num_envs: int) -> List[HierarchicalState]:
        return [self.network.initial_state(batch_size=1, device=self.device) for _ in range(num_envs)]

    def act(
        self,
        frames: torch.Tensor,
        structured: Optional[torch.Tensor],
        states: List[HierarchicalState],
        done_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[HierarchicalState], Dict[str, torch.Tensor]]:
        self.network.eval()
        batch_state = self._stack_states(states)
        q_values, new_state, aux = self.network(frames, structured, batch_state, done_mask)
        greedy_actions = q_values.argmax(dim=-1)
        if self.epsilon > 0:
            random_mask = torch.rand_like(greedy_actions.float()) < self.epsilon
            random_actions = torch.randint(
                0, self.config.num_actions, greedy_actions.shape, device=self.device
            )
            actions = torch.where(random_mask, random_actions, greedy_actions)
        else:
            actions = greedy_actions
        new_states = self._unstack_state(new_state)
        self.network.train()
        return actions, new_states, aux

    def intrinsic_rewards(
        self, aux: Dict[str, torch.Tensor], milestone_flags: torch.Tensor, env_ids: Sequence[int]
    ) -> Dict[str, torch.Tensor]:
        rnd_raw = self.rnd.intrinsic_reward(aux["ssm"].detach())
        novelty_values = []
        posterior_means = []
        for idx, env_id in enumerate(env_ids):
            key = tuple(milestone_flags[idx].long().cpu().tolist())
            self.novelty_counts[env_id][key] += 1
            novelty_values.append(1.0 / (self.novelty_counts[env_id][key] ** 0.5))
            # bayesian shaping: more weight when posteriors are confident
            posterior_mean = milestone_flags[idx].float().mean().item()
            posterior_means.append(posterior_mean)
        novelty_reward = torch.tensor(novelty_values, device=self.device, dtype=torch.float32)
        posterior_tensor = torch.tensor(posterior_means, device=self.device, dtype=torch.float32)
        bayes_reward = self.config.bayes_weight * posterior_tensor
        # influence RND by Bayes confidence (elementwise scale)
        bayes_scale = 1.0 + posterior_tensor
        rnd_reward = rnd_raw * bayes_scale
        return {
            "rnd": rnd_reward,
            "rnd_raw": rnd_raw,
            "bayes_scale": bayes_scale,
            "posterior_mean": posterior_tensor,
            "novel": novelty_reward,
            "bayes": bayes_reward,
        }

    def update_bayes_from_milestones(self, milestones: torch.Tensor) -> Dict[str, Dict[str, float]]:
        milestone_dict = {f"flag_{i}": bool(m.item()) for i, m in enumerate(milestones)}
        return self.bayes_monitor.update_episode(milestone_dict)

    def add_transition(
        self,
        frames: torch.Tensor,
        structured: Optional[torch.Tensor],
        action: torch.Tensor,
        reward_total: torch.Tensor,
        reward_components: Dict[str, torch.Tensor],
        done: torch.Tensor,
        next_frames: torch.Tensor,
        next_structured: Optional[torch.Tensor],
        states: List[HierarchicalState],
    ) -> None:
        batch_size = frames.shape[0]
        for i in range(batch_size):
            rc = {k: float(v[i].detach().cpu().item()) for k, v in reward_components.items()}
            self.replay.add(
                frames=frames[i].unsqueeze(0),
                structured=structured[i].unsqueeze(0) if structured is not None else None,
                action=int(action[i].item()),
                reward=float(reward_total[i].item()),
                reward_components=rc,
                done=bool(done[i].item()),
                next_frames=next_frames[i].unsqueeze(0),
                next_structured=next_structured[i].unsqueeze(0)
                if next_structured is not None
                else None,
                state=states[i],
            )

    def learn(self, batch_size: int) -> Dict[str, float]:
        if len(self.replay) < batch_size:
            return {}
        batch = self.replay.sample(batch_size)
        state_batch = self._stack_states(batch["states"])
        q_values, _, aux = self.network(
            batch["frames"], batch["structured"], state_batch, done=None  # type: ignore[arg-type]
        )
        q_sa = q_values.gather(1, batch["actions"].unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target_q, _, _ = self.target_network(
                batch["next_frames"], batch["next_structured"], state_batch, done=None  # type: ignore[arg-type]
            )
            max_next = target_q.max(dim=1).values
            target = batch["rewards"] + (1.0 - batch["dones"]) * self.config.gamma * max_next
        loss = F.smooth_l1_loss(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.grad_clip)
        self.optimizer.step()

        rnd_loss = self.rnd.loss(aux["ssm"].detach())
        self.rnd_opt.zero_grad()
        rnd_loss.mean().backward()
        self.rnd_opt.step()

        self.global_step += 1
        if self.global_step % self.config.target_update_interval == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        return {"loss": float(loss.item()), "rnd_loss": float(rnd_loss.mean().item())}

    def reset_episode(self, env_id: int) -> None:
        self.novelty_counts[env_id].clear()
        self.episodes[env_id] += 1
        self.episode_returns[env_id] = 0.0

    def _stack_states(self, states: List[Optional[HierarchicalState]]) -> HierarchicalState:
        if any(s is None for s in states):
            return self.network.initial_state(batch_size=len(states), device=self.device)
        return HierarchicalState(
            gru=torch.cat([s.gru for s in states if s is not None], dim=0),
            lstm_h=torch.cat([s.lstm_h for s in states if s is not None], dim=0),
            lstm_c=torch.cat([s.lstm_c for s in states if s is not None], dim=0),
            ssm=torch.cat([s.ssm for s in states if s is not None], dim=0),
        )

    def _unstack_state(self, state: HierarchicalState) -> List[HierarchicalState]:
        batch = state.gru.shape[0]
        return [
            HierarchicalState(
                gru=state.gru[i : i + 1],
                lstm_h=state.lstm_h[i : i + 1],
                lstm_c=state.lstm_c[i : i + 1],
                ssm=state.ssm[i : i + 1],
            )
            for i in range(batch)
        ]
