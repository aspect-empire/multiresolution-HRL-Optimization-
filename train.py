"""Training entry point for the hierarchical welding optimizer prototype."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from welding_hrl import (
    CostWeights,
    HighLevelPolicy,
    LineDesign,
    LowLevelActor,
    LowLevelCritic,
    ProcessWindow,
    WeldPoint,
    WeldingEnv,
    Workstation,
)


def make_demo_design() -> LineDesign:
    weld_points = [
        WeldPoint(id=0, material="steel", predecessors=[], weight=1.0),
        WeldPoint(id=1, material="aluminum", predecessors=[0], weight=1.2),
        WeldPoint(id=2, material="steel", predecessors=[0], weight=0.8),
        WeldPoint(id=3, material="aluminum", predecessors=[1, 2], weight=1.5, process_window=ProcessWindow((350, 750), (2, 6), (0.06, 0.18), 18.0)),
    ]
    workstations = [
        Workstation(id=0, reachable_points=[0, 1], changeover_time=2.0, load_time=1.0, unload_time=1.0),
        Workstation(id=1, reachable_points=[2, 3], changeover_time=1.5, load_time=1.0, unload_time=1.0),
        Workstation(id=2, reachable_points=[0, 2, 3], changeover_time=2.5, load_time=1.0, unload_time=1.0),
    ]
    return LineDesign(workstations=workstations, weld_points=weld_points)


def flatten_obs(obs: dict, num_stations: int) -> torch.Tensor:
    remaining = torch.tensor(obs["remaining"], dtype=torch.float32)
    adjacency = torch.tensor(obs["adjacency"], dtype=torch.float32).flatten()
    last_station = torch.nn.functional.one_hot(torch.tensor(obs["last_station"]), num_classes=num_stations + 1).float()
    return torch.cat([remaining, adjacency, last_station], dim=0)


@dataclass
class Trajectory:
    logp_high: List[torch.Tensor]
    value_high: List[torch.Tensor]
    logp_low: List[torch.Tensor]
    q_low: List[torch.Tensor]
    rewards: List[float]


class Trainer:
    def __init__(self, env: WeldingEnv, gamma: float = 0.99):
        self.env = env
        obs_size = len(env.design.weld_points) + env.design.adjacency_mask().size + (env.num_workstations + 1)
        self.high_policy = HighLevelPolicy(env.num_points, env.num_workstations)
        self.low_actor = LowLevelActor(obs_dim=obs_size, action_dim=3)
        self.low_critic = LowLevelCritic(obs_dim=obs_size, action_dim=3)
        self.gamma = gamma
        self.optim_high = Adam(self.high_policy.parameters(), lr=3e-4)
        self.optim_low_actor = Adam(self.low_actor.parameters(), lr=3e-4)
        self.optim_low_critic = Adam(self.low_critic.parameters(), lr=5e-4)

    def run_episode(self) -> Trajectory:
        obs, _ = self.env.reset()
        traj = Trajectory(logp_high=[], value_high=[], logp_low=[], q_low=[], rewards=[])
        done = False
        while not done:
            dist_high, value_high = self.high_policy(obs)
            station = dist_high.sample()
            obs_vec = flatten_obs(obs, self.env.num_workstations)
            dist_low, low_mean = self.low_actor(obs_vec)
            low_action = dist_low.rsample()
            q_value = self.low_critic(obs_vec, low_action)
            next_obs, reward, done, _, _ = self.env.step((station.item(), low_action.detach().numpy()))

            traj.logp_high.append(dist_high.log_prob(station))
            traj.value_high.append(value_high)
            traj.logp_low.append(dist_low.log_prob(low_action).sum())
            traj.q_low.append(q_value)
            traj.rewards.append(reward)
            obs = next_obs
        return traj

    def _discounted_returns(self, rewards: List[float]) -> List[float]:
        returns: List[float] = []
        running = 0.0
        for r in reversed(rewards):
            running = r + self.gamma * running
            returns.append(running)
        returns.reverse()
        return returns

    def train_step(self) -> dict:
        traj = self.run_episode()
        returns = torch.tensor(self._discounted_returns(traj.rewards), dtype=torch.float32)

        # High-level actor-critic update
        values = torch.stack(traj.value_high)
        advantages = returns - values.detach()
        loss_high_policy = -(torch.stack(traj.logp_high) * advantages).mean()
        loss_high_value = F.mse_loss(values, returns)
        self.optim_high.zero_grad()
        (loss_high_policy + loss_high_value).backward()
        self.optim_high.step()

        # Low-level SAC-like update with simple TD(0)
        obs_q = torch.stack(traj.q_low)
        logp_low = torch.stack(traj.logp_low)
        target_q = returns - 0.1 * logp_low.detach()
        loss_q = F.mse_loss(obs_q, target_q)
        self.optim_low_critic.zero_grad()
        loss_q.backward()
        self.optim_low_critic.step()

        loss_low_actor = (0.1 * logp_low - obs_q.detach()).mean()
        self.optim_low_actor.zero_grad()
        loss_low_actor.backward()
        self.optim_low_actor.step()

        return {
            "episode_reward": float(sum(traj.rewards)),
            "loss_high_policy": float(loss_high_policy.item()),
            "loss_high_value": float(loss_high_value.item()),
            "loss_low_actor": float(loss_low_actor.item()),
            "loss_low_critic": float(loss_q.item()),
        }


def main():
    design = make_demo_design()
    env = WeldingEnv(design, weights=CostWeights(takt=1.0, energy=0.2, precision=0.6))
    trainer = Trainer(env)
    for epoch in range(5):
        metrics = trainer.train_step()
        print(f"Epoch {epoch}: {metrics}")


if __name__ == "__main__":
    main()
