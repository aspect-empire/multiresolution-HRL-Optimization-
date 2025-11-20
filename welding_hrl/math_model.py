"""Mathematical abstractions for the welding optimization problem."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class ProcessWindow:
    """Safe ranges for welding process parameters."""

    temperature: Tuple[float, float]
    power: Tuple[float, float]
    speed: Tuple[float, float]
    max_heat_input: float


@dataclass
class WeldPoint:
    """Represents a weld or locating point on the body-in-white."""

    id: int
    material: str
    predecessors: List[int] = field(default_factory=list)
    process_window: ProcessWindow = field(default_factory=lambda: ProcessWindow((300, 800), (2, 7), (0.05, 0.2), 20.0))
    weight: float = 1.0  # influence on geometry/cost


@dataclass
class Workstation:
    """Represents a station that can execute a subset of weld points."""

    id: int
    reachable_points: List[int]
    changeover_time: float
    load_time: float
    unload_time: float


@dataclass
class LineDesign:
    """Static description of the welding line."""

    workstations: List[Workstation]
    weld_points: List[WeldPoint]

    def adjacency_mask(self) -> np.ndarray:
        """Adjacency matrix that encodes precedence constraints."""

        n = len(self.weld_points)
        mask = np.zeros((n, n), dtype=np.float32)
        for wp in self.weld_points:
            for pred in wp.predecessors:
                mask[pred, wp.id] = 1.0
        return mask


@dataclass
class ProcessPlan:
    """Decision variables for one episode of planning."""

    allocation: Dict[int, int]  # weld_point -> workstation
    ordering: Dict[int, List[int]]  # workstation -> list of weld_point ids
    temperature: Dict[int, float]
    power: Dict[int, float]
    speed: Dict[int, float]


@dataclass
class CostWeights:
    """Multi-objective scalarization weights."""

    takt: float = 1.0
    energy: float = 0.3
    precision: float = 0.5


def compute_heat_input(power: float, speed: float) -> float:
    return power / max(speed, 1e-6)


def weld_time(speed: float, path_length: float = 0.15) -> float:
    return path_length / max(speed, 1e-6)


def evaluate_plan(plan: ProcessPlan, design: LineDesign, weights: CostWeights) -> Tuple[float, Dict[str, float]]:
    """Proxy objective for a process plan.

    Returns a tuple of (scalarized_cost, diagnostics).
    """

    takt_times: Dict[int, float] = {}
    energy_cost = 0.0
    precision_penalty = 0.0

    for ws in design.workstations:
        sequence = plan.ordering.get(ws.id, [])
        time_accum = ws.load_time + ws.unload_time + ws.changeover_time
        for point_id in sequence:
            power = plan.power.get(point_id, 4.0)
            speed = plan.speed.get(point_id, 0.12)
            temperature = plan.temperature.get(point_id, 600.0)
            point = design.weld_points[point_id]
            window = point.process_window
            heat = compute_heat_input(power, speed)
            time_accum += weld_time(speed)
            energy_cost += power * time_accum * 0.001

            # barrier-like penalties for exceeding process windows
            if not (window.temperature[0] <= temperature <= window.temperature[1]):
                precision_penalty += 2.0
            if not (window.power[0] <= power <= window.power[1]):
                precision_penalty += 2.0
            if not (window.speed[0] <= speed <= window.speed[1]):
                precision_penalty += 1.0
            if heat > window.max_heat_input:
                precision_penalty += 5.0

            precision_penalty += point.weight * abs(temperature - 0.8 * window.temperature[1]) * 1e-3

        takt_times[ws.id] = time_accum

    bottleneck = max(takt_times.values()) if takt_times else 0.0
    score = (
        weights.takt * bottleneck
        + weights.energy * energy_cost
        + weights.precision * precision_penalty
    )
    diagnostics = {
        "bottleneck": bottleneck,
        "energy_cost": energy_cost,
        "precision_penalty": precision_penalty,
    }
    return score, diagnostics
