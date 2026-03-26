from __future__ import annotations

from typing import List, Literal, TypedDict

Action2 = Literal[0, 1]
Action4 = Literal[0, 1, 2, 3]


class TrafficSample(TypedDict, total=False):
    local: List[float]
    neighbor_mean: List[float]
    neighbor_directional: List[float]
    interaction: List[float]
    temporal: List[float]
    # History-aware cooperative features (v2)
    neighbor_delta: List[float]       # change in neighbor mean vs last timestep
    neighbor_rolling: List[float]     # rolling mean of neighbor states (window=3)
    cross_temporal: List[float]       # local_delta dot neighbor_delta trend signal
    phase: float
    elapsed: float
    source: str
    intersection_id: str
    timestep: int


class SumoRow(TypedDict):
    intersection_id: str
    timestep: int
    local: List[float]
    phase: float
    elapsed: float
