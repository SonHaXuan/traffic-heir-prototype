from __future__ import annotations

from typing import List, Literal, TypedDict

Action2 = Literal[0, 1]
Action4 = Literal[0, 1, 2, 3]


class TrafficSample(TypedDict, total=False):
    local: List[float]
    neighbor_mean: List[float]
    interaction: List[float]
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
