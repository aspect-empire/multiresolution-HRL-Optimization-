"""Low-level policy for continuous process parameter control."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


def mlp(input_dim: int, output_dim: int, hidden_dim: int = 128) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class LowLevelActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = mlp(obs_dim, 2 * action_dim, hidden_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        out = self.net(obs)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -5, 2)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        return dist, mean


class LowLevelCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = mlp(obs_dim + action_dim, 1, hidden_dim)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x).squeeze(-1)
