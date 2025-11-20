"""Gym environment for hierarchical welding optimization."""
from __future__ import annotations

from typing import Dict, Tuple

import gym
import numpy as np
from gym import spaces

from welding_hrl.math_model import (
    CostWeights,
    LineDesign,
    ProcessPlan,
    compute_heat_input,
    evaluate_plan,
)


class WeldingEnv(gym.Env):
    """A lightweight environment that exposes a two-level decision process.

    High-level actions: assign a weld point to a workstation.
    Low-level actions: set continuous process parameters for that weld point.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, design: LineDesign, weights: CostWeights | None = None):
        super().__init__()
        self.design = design
        self.weights = weights or CostWeights()
        self.num_points = len(design.weld_points)
        self.num_workstations = len(design.workstations)

        self.high_action_space = spaces.Discrete(self.num_workstations)
        # Low-level continuous action: [temperature, power, speed]
        self.low_action_space = spaces.Box(low=np.array([300, 2.0, 0.05]), high=np.array([800, 7.0, 0.2]), dtype=np.float32)

        # Observation packs remaining mask, adjacency info, and placeholder station state
        self.observation_space = spaces.Dict(
            {
                "remaining": spaces.MultiBinary(self.num_points),
                "adjacency": spaces.Box(low=0, high=1, shape=(self.num_points, self.num_points), dtype=np.float32),
                "last_station": spaces.Discrete(self.num_workstations + 1),
            }
        )
        self._reset_buffers()

    def _reset_buffers(self) -> None:
        self.remaining_mask = np.ones(self.num_points, dtype=np.int32)
        self.allocation: Dict[int, int] = {}
        self.parameters: Dict[int, Tuple[float, float, float]] = {}
        self.current_station = self.num_workstations  # sentinel: none selected yet

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self._reset_buffers()
        adjacency = self.design.adjacency_mask()
        obs = {
            "remaining": self.remaining_mask.copy(),
            "adjacency": adjacency,
            "last_station": np.array(self.current_station, dtype=np.int64),
        }
        return obs, {}

    def step(self, action: Tuple[int, np.ndarray]):  # type: ignore[override]
        station_id, params = action
        temperature, power, speed = params.tolist()
        done = False
        reward = 0.0
        info = {}

        next_point = self._next_schedulable_point()
        if next_point is None:
            done = True
            obs = self._build_obs()
            return obs, 0.0, done, False, {"reason": "no-points"}

        if not self._check_station_reachability(next_point, station_id):
            reward -= 5.0
        else:
            self.allocation[next_point] = station_id
            self.parameters[next_point] = (temperature, power, speed)
            self.remaining_mask[next_point] = 0
            self.current_station = station_id
            reward += self._local_reward(next_point, station_id, temperature, power, speed)

        if self.remaining_mask.sum() == 0:
            done = True
            total_reward = -self._terminal_cost()
            reward += total_reward
            info.update({"terminal_cost": total_reward})

        obs = self._build_obs()
        return obs, float(reward), done, False, info

    def _local_reward(self, point_id: int, station_id: int, temperature: float, power: float, speed: float) -> float:
        point = self.design.weld_points[point_id]
        window = point.process_window
        reward = 0.0
        heat_input = compute_heat_input(power, speed)
        if not (window.temperature[0] <= temperature <= window.temperature[1]):
            reward -= 2.0
        if not (window.power[0] <= power <= window.power[1]):
            reward -= 2.0
        if not (window.speed[0] <= speed <= window.speed[1]):
            reward -= 1.0
        if heat_input > window.max_heat_input:
            reward -= 5.0
        reward -= point.weight * abs(temperature - 0.8 * window.temperature[1]) * 1e-3
        reward += 0.1 * (1.0 - station_id / max(1, self.num_workstations - 1))
        return reward

    def _next_schedulable_point(self) -> int | None:
        adjacency = self.design.adjacency_mask()
        for i, remaining in enumerate(self.remaining_mask):
            if remaining == 0:
                continue
            preds = np.where(adjacency[:, i] > 0)[0]
            if all(self.remaining_mask[pred] == 0 for pred in preds):
                return i
        return None

    def _check_station_reachability(self, point_id: int, station_id: int) -> bool:
        ws = self.design.workstations[station_id]
        return point_id in ws.reachable_points

    def _terminal_cost(self) -> float:
        plan = self._build_plan()
        cost, _ = evaluate_plan(plan, self.design, self.weights)
        return cost

    def _build_plan(self) -> ProcessPlan:
        ordering: Dict[int, list[int]] = {ws.id: [] for ws in self.design.workstations}
        for point_id, station_id in self.allocation.items():
            ordering[station_id].append(point_id)
        for seq in ordering.values():
            seq.sort()
        temperature = {pid: params[0] for pid, params in self.parameters.items()}
        power = {pid: params[1] for pid, params in self.parameters.items()}
        speed = {pid: params[2] for pid, params in self.parameters.items()}
        return ProcessPlan(
            allocation=self.allocation.copy(),
            ordering=ordering,
            temperature=temperature,
            power=power,
            speed=speed,
        )

    def _build_obs(self):
        return {
            "remaining": self.remaining_mask.copy(),
            "adjacency": self.design.adjacency_mask(),
            "last_station": np.array(self.current_station, dtype=np.int64),
        }

    def render(self, mode="human"):
        if mode != "human":
            raise NotImplementedError
        print(f"Remaining: {self.remaining_mask}, allocation: {self.allocation}")
