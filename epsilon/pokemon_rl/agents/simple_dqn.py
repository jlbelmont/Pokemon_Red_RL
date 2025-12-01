import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        scale = self.pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1)
        return x * scale


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, reduction: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcite(out_channels, reduction=reduction)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = F.relu(out + identity, inplace=True)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.view(b, c, h * w).transpose(1, 2)
        tokens = self.norm(tokens)
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        attn_out = self.dropout(attn_out)
        attn_out = attn_out.transpose(1, 2).reshape(b, c, h, w)
        return x + attn_out


class TinySSMLayer(nn.Module):
    """Minimal selective SSM-style update: s_t = gate * s_{t-1} + B(x); y = C(s)."""

    def __init__(self, in_dim: int, state_dim: int, head_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.state_dim = state_dim
        self.head_dim = head_dim
        self.in_norm = nn.LayerNorm(in_dim)
        self.gate = nn.Linear(in_dim, num_heads)
        self.input_proj = nn.Linear(in_dim, num_heads * state_dim)
        self.out_proj = nn.Linear(num_heads * state_dim, num_heads * head_dim)

    def forward(self, x: torch.Tensor, prev_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.size(0)
        if prev_state is None:
            prev_state = x.new_zeros(batch, self.num_heads, self.state_dim)
        x_norm = self.in_norm(x)
        decay = torch.sigmoid(self.gate(x_norm)).unsqueeze(-1)
        b = self.input_proj(x_norm).view(batch, self.num_heads, self.state_dim)
        new_state = decay * prev_state + b
        out = self.out_proj(new_state.view(batch, -1))
        return out, new_state


class LightweightSSM(nn.Module):
    """Stacked lightweight SSM arm that produces a small contextual summary."""

    def __init__(
        self,
        input_dim: int,
        state_dim: int = 32,
        head_dim: int = 32,
        num_heads: int = 2,
        num_layers: int = 1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        layer_input_dims = [input_dim] + [num_heads * head_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            TinySSMLayer(in_dim, state_dim, head_dim, num_heads) for in_dim in layer_input_dims
        )

    @property
    def output_dim(self) -> int:
        return self.num_heads * self.head_dim

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(self.num_layers, batch, self.num_heads, self.state_dim, device=device, dtype=dtype)

    def forward(
        self, x: torch.Tensor, prev_state: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.size(0)
        if prev_state is None or prev_state.shape != (self.num_layers, batch, self.num_heads, self.state_dim):
            prev_state = self.init_state(batch, x.device, x.dtype)
        next_states = []
        out = x
        for idx, layer in enumerate(self.layers):
            layer_state = prev_state[idx]
            out, state = layer(out, layer_state)
            next_states.append(state)
        stacked = torch.stack(next_states, dim=0)
        return out, stacked


class NoisyLinear(nn.Module):
    """Factorised Gaussian noisy linear layer."""

    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_eps", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_eps.copy_(eps_out.ger(eps_in))
        self.bias_eps.copy_(eps_out)

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_eps
            bias = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(input, weight, bias)


class SimpleDQN(nn.Module):
    """
    CNN encoder with dual recurrent heads (GRU + LSTM) feeding a dueling
    distributional (quantile) head that uses NoisyNets for exploration.
    """

    def __init__(
        self,
        obs_shape,
        map_feat_dim,
        goal_dim,
        n_actions,
        num_quantiles: int = 51,
        ssm_cfg: Optional[dict] = None,
        gru_hidden_size: int = 144,
        lstm_hidden_size: int = 144,
    ):
        super().__init__()
        c, h, w = obs_shape
        self.n_actions = n_actions
        self.num_quantiles = num_quantiles
        self.input_channels = int(c)
        self.obs_height = int(h)
        self.obs_width = int(w)
        self.map_feat_dim = int(map_feat_dim)
        self.goal_dim = int(goal_dim)

        # Moderate-size encoder (keeps all components but targets ~1M params).
        stem_channels = 96
        self.encoder_dim = stem_channels
        self.stem = nn.Sequential(
            nn.Conv2d(c, 48, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )
        self.residual = nn.Sequential(
            ResidualBlock(stem_channels, stem_channels),
            ResidualBlock(stem_channels, stem_channels),
            ResidualBlock(stem_channels, stem_channels),
        )
        self.spatial_attn = SpatialAttention(stem_channels, num_heads=4, dropout=0.1)
        self.post_attn = nn.Sequential(
            nn.Conv2d(stem_channels, stem_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dropout = nn.Dropout(p=0.1)

        self.map_net = nn.Sequential(
            nn.Linear(map_feat_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 96),
            nn.ReLU(inplace=True),
        )
        self.goal_encoder = nn.Sequential(
            nn.Linear(goal_dim, 48),
            nn.LayerNorm(48),
            nn.ReLU(inplace=True),
        )
        self.goal_gamma = nn.Linear(48, 96)
        self.goal_beta = nn.Linear(48, 96)

        self.use_ssm = bool(ssm_cfg.get("use_ssm_encoder", False)) if ssm_cfg else False
        self.ssm_encoder: Optional[LightweightSSM] = None
        self.ssm_out_dim = 0
        map_embed_dim = 96
        fused_dim = stem_channels + map_embed_dim
        if self.use_ssm:
            self.ssm_encoder = LightweightSSM(
                input_dim=fused_dim,
                state_dim=int(ssm_cfg.get("ssm_state_dim", 32)),
                head_dim=int(ssm_cfg.get("ssm_head_dim", 32)),
                num_heads=int(ssm_cfg.get("ssm_heads", 2)),
                num_layers=int(ssm_cfg.get("ssm_layers", 1)),
            )
            self.ssm_out_dim = self.ssm_encoder.output_dim
            fused_dim += self.ssm_out_dim

        self.gru_hidden_size = int(gru_hidden_size)
        self.lstm_hidden_size = int(lstm_hidden_size)
        self.hidden_size = self.gru_hidden_size + self.lstm_hidden_size
        self.gru = nn.GRU(fused_dim, self.gru_hidden_size, batch_first=True)
        self.lstm = nn.LSTM(fused_dim, self.lstm_hidden_size, batch_first=True)

        duel_hidden = 96
        self.advantage = nn.Sequential(
            NoisyLinear(self.hidden_size, duel_hidden),
            nn.ReLU(),
            nn.Linear(duel_hidden, n_actions * num_quantiles),
        )
        self.value = nn.Sequential(
            NoisyLinear(self.hidden_size, duel_hidden),
            nn.ReLU(),
            nn.Linear(duel_hidden, num_quantiles),
        )
        self.high_value_head = nn.Sequential(
            NoisyLinear(self.hidden_size, duel_hidden),
            nn.ReLU(),
            nn.Linear(duel_hidden, n_actions),
        )
        self.aux_head = nn.Sequential(
            nn.Linear(self.hidden_size, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, map_feat_dim),
        )
        self.novelty_head = nn.Linear(self.hidden_size, goal_dim)
        self._validate_encoder_shape()

    def init_hidden(self, batch_size: int, device: torch.device):
        gru_state = torch.zeros(1, batch_size, self.gru_hidden_size, device=device)
        lstm_h = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
        lstm_c = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
        hidden = {"gru": gru_state, "lstm": (lstm_h, lstm_c)}
        if self.use_ssm and self.ssm_encoder is not None:
            hidden["ssm"] = self.ssm_encoder.init_state(batch_size, device, gru_state.dtype)
        return hidden

    def forward(self, obs, map_feat, goal_feat, hidden=None):
        if obs.ndim != 4:
            raise ValueError(f"Expected observations with shape (batch, channels, height, width), got {obs.shape}.")
        if obs.size(1) != self.input_channels:
            raise ValueError(
                f"Observation channel mismatch: model expects {self.input_channels} channels but received {obs.size(1)}."
            )
        batch = obs.size(0)
        if hidden is None:
            hidden = self.init_hidden(batch, obs.device)

        if map_feat.ndim != 2 or map_feat.size(1) != self.map_feat_dim:
            raise ValueError(
                f"Map feature dimension mismatch: expected {(batch, self.map_feat_dim)} but received {tuple(map_feat.shape)}."
            )
        if goal_feat.ndim != 2 or goal_feat.size(1) != self.goal_dim:
            raise ValueError(
                f"Goal feature dimension mismatch: expected {(batch, self.goal_dim)} but received {tuple(goal_feat.shape)}."
            )

        features = self.stem(obs)
        features = self.residual(features)
        features = self.spatial_attn(features)
        features = self.post_attn(features)
        features = self.global_pool(features).view(batch, -1)
        features = self.feature_dropout(features)

        map_embed = self.map_net(map_feat)
        goal_encoded = self.goal_encoder(goal_feat)
        gamma = torch.tanh(self.goal_gamma(goal_encoded))
        beta = self.goal_beta(goal_encoded)
        map_embed = map_embed * (1 + gamma) + beta
        fused_vec = torch.cat([features, map_embed], dim=1)

        next_ssm_state = None
        if self.use_ssm and self.ssm_encoder is not None:
            ssm_state = None
            if isinstance(hidden, dict):
                ssm_state = hidden.get("ssm")
            if ssm_state is not None:
                ssm_state = ssm_state.to(fused_vec.device)
            ssm_out, next_ssm_state = self.ssm_encoder(fused_vec, ssm_state)
            fused_vec = torch.cat([fused_vec, ssm_out], dim=1)

        fused = fused_vec.unsqueeze(1)

        if isinstance(hidden, dict):
            gru_hidden = hidden.get("gru")
            lstm_hidden = hidden.get("lstm")
            if gru_hidden is None or lstm_hidden is None:
                init = self.init_hidden(batch, obs.device)
                gru_hidden = init["gru"]
                lstm_hidden = init["lstm"]
        else:
            gru_hidden = hidden
            lstm_hidden = (
                torch.zeros(1, batch, self.lstm_hidden_size, device=obs.device, dtype=obs.dtype),
                torch.zeros(1, batch, self.lstm_hidden_size, device=obs.device, dtype=obs.dtype),
            )

        gru_hidden = gru_hidden.to(fused.device)
        lstm_hidden = (
            lstm_hidden[0].to(fused.device),
            lstm_hidden[1].to(fused.device),
        )

        gru_output, next_gru = self.gru(fused, gru_hidden)
        lstm_output, (next_lstm_h, next_lstm_c) = self.lstm(fused, lstm_hidden)
        output = torch.cat([gru_output, lstm_output], dim=2).squeeze(1)
        next_hidden = {
            "gru": next_gru,
            "lstm": (next_lstm_h, next_lstm_c),
        }
        if self.use_ssm and next_ssm_state is not None:
            next_hidden["ssm"] = next_ssm_state

        adv = self.advantage(output).view(batch, self.n_actions, self.num_quantiles)
        val = self.value(output).view(batch, 1, self.num_quantiles)
        q_quantiles = val + adv - adv.mean(dim=1, keepdim=True)
        high_adv = self.high_value_head(output).unsqueeze(-1)
        q_quantiles = q_quantiles + high_adv
        aux_pred = self.aux_head(output)
        novelty_pred = self.novelty_head(output)
        return q_quantiles, next_hidden, aux_pred, novelty_pred, output

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def _validate_encoder_shape(self) -> None:
        """Run a dry forward-pass on zeros to surface shape mismatches early."""
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_channels, self.obs_height, self.obs_width)
            encoded = self.stem(dummy)
            encoded = self.residual(encoded)
            encoded = self.spatial_attn(encoded)
            encoded = self.post_attn(encoded)
            pooled = self.global_pool(encoded).view(1, -1)
            if pooled.shape[-1] != self.encoder_dim:
                raise ValueError(
                    f"Incompatible input resolution {(self.obs_height, self.obs_width)} for encoder; "
                    f"expected pooled dim {self.encoder_dim} but received {pooled.shape[-1]}."
                )


class LowLevelDQNPolicy(nn.Module):
    """Wrapper that exposes the current SimpleDQN as a goal-ready low-level controller."""

    def __init__(self, backbone: SimpleDQN, goal_dim: int = 0):
        super().__init__()
        self.backbone = backbone
        self.goal_dim = goal_dim

    def forward(self, obs, map_feat, goal_feat, goal: Optional[torch.Tensor] = None, hidden=None):
        # Future HIRO controllers can blend `goal` into goal_feat; for now we ignore it.
        return self.backbone(obs, map_feat, goal_feat, hidden)

    def init_hidden(self, batch_size: int, device: torch.device):
        return self.backbone.init_hidden(batch_size, device)

    def reset_noise(self):
        self.backbone.reset_noise()
