from __future__ import annotations

from typing import List

from .types import TrafficSample


def _scale(values: List[float], factor: float = 20.0) -> List[float]:
    return [v / factor for v in values]


def _local_self_interaction(local: List[float]) -> List[float]:
    # compact self-interaction proxy that avoids leaking neighbor context
    return [v * v / 20.0 for v in local]


def local_features(sample: TrafficSample) -> List[float]:
    local = list(sample["local"])
    return _scale(local) + [float(sample["elapsed"]) / 60.0, float(sample["phase"])]


def cooperative_features(sample: TrafficSample) -> List[float]:
    local = _scale(list(sample["local"]))
    neighbor_mean = _scale(list(sample["neighbor_mean"]))
    interaction = _scale(list(sample["interaction"]), factor=10.0)
    elapsed = [float(sample["elapsed"]) / 60.0]
    phase = [float(sample["phase"])]
    return local + neighbor_mean + interaction + elapsed + phase


def cooperative_no_interaction_features(sample: TrafficSample) -> List[float]:
    local = _scale(list(sample["local"]))
    neighbor_mean = _scale(list(sample["neighbor_mean"]))
    elapsed = [float(sample["elapsed"]) / 60.0]
    phase = [float(sample["phase"])]
    return local + neighbor_mean + elapsed + phase


def cooperative_no_neighbor_features(sample: TrafficSample) -> List[float]:
    local_raw = list(sample["local"])
    local = _scale(local_raw)
    self_interaction = _scale(_local_self_interaction(local_raw), factor=10.0)
    elapsed = [float(sample["elapsed"]) / 60.0]
    phase = [float(sample["phase"])]
    return local + self_interaction + elapsed + phase
