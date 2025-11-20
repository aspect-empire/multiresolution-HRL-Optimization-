"""High-level policy for weld point allocation and ordering."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


class HighLevelPolicy(nn.Module):
    """Graph-aware policy that assigns the next weld point to a station."""

    def __init__(self, num_points: int, num_stations: int, hidden_dim: int = 128):
        super().__init__()
        self.num_points = num_points
        self.num_stations = num_stations
        self.encoder = nn.Sequential(
            nn.Linear(num_points + num_stations + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, num_stations)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs) -> Tuple[Categorical, torch.Tensor]:
        remaining = obs["remaining"]
        last_station = obs["last_station"]
        adjacency = obs["adjacency"]

        remaining_tensor = torch.tensor(remaining, dtype=torch.float32)
        station_one_hot = torch.nn.functional.one_hot(torch.tensor(last_station), num_classes=self.num_stations + 1).float()
        adjacency_flat = torch.tensor(adjacency, dtype=torch.float32).flatten()
        x = torch.cat([remaining_tensor, station_one_hot, adjacency_flat], dim=0)
        features = self.encoder(x)
        logits = self.actor(features)
        dist = Categorical(logits=logits)
        value = self.value_head(features)
        return dist, value.squeeze(-1)
