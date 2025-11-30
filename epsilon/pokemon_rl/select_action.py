import numpy as np
import torch
from typing import Dict, Optional


HiddenState = Dict[str, object]


def _move_hidden_to_device(hidden: Optional[HiddenState], device: torch.device) -> Optional[HiddenState]:
    if hidden is None:
        return None
    gru = hidden.get("gru")
    lstm = hidden.get("lstm")
    if gru is None or lstm is None:
        return None
    lstm_h, lstm_c = lstm
    return {
        "gru": gru.to(device),
        "lstm": (lstm_h.to(device), lstm_c.to(device)),
    }


def _detach_hidden(hidden: Optional[HiddenState]) -> Optional[HiddenState]:
    if hidden is None:
        return None
    gru = hidden.get("gru")
    lstm = hidden.get("lstm")
    if gru is None or lstm is None:
        return None
    lstm_h, lstm_c = lstm
    return {
        "gru": gru.detach(),
        "lstm": (lstm_h.detach(), lstm_c.detach()),
    }


def select_action(
    model,
    obs,
    map_feat,
    goal_feat,
    epsilon,
    action_space,
    device=None,
    hidden_state: Optional[HiddenState] = None,
):
    """Epsilon-greedy policy helper for manual scripts."""
    model.reset_noise()
    if np.random.rand() < epsilon:
        return action_space.sample(), hidden_state
    dev = device or next(model.parameters()).device
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
    map_tensor = torch.tensor(map_feat, dtype=torch.float32, device=dev).unsqueeze(0)
    goal_tensor = torch.tensor(goal_feat, dtype=torch.float32, device=dev).unsqueeze(0)
    hidden = _move_hidden_to_device(hidden_state, dev)
    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        quantiles, next_hidden, _, _, _ = model(obs_tensor, map_tensor, goal_tensor, hidden)
    if model_was_training:
        model.train()
    q_mean = quantiles.mean(dim=2)
    action = int(torch.argmax(q_mean, dim=1).item())
    return action, _detach_hidden(next_hidden)
