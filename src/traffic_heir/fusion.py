from __future__ import annotations

from typing import List

from .types import TrafficSample


def _scale(values: List[float], factor: float = 20.0) -> List[float]:
    return [v / factor for v in values]


def _local_self_interaction(local: List[float]) -> List[float]:
    return [v * v / 20.0 for v in local]


def local_features(sample: TrafficSample) -> List[float]:
    local = list(sample["local"])
    return _scale(local) + [float(sample["elapsed"]) / 60.0, float(sample["phase"])]


def cooperative_features(sample: TrafficSample) -> List[float]:
    local = _scale(list(sample["local"]))
    neighbor_mean = _scale(list(sample["neighbor_mean"]))
    directional = _scale(list(sample.get("neighbor_directional", [])))
    interaction = _scale(list(sample["interaction"]), factor=10.0)
    temporal = _scale(list(sample.get("temporal", [])), factor=20.0)
    elapsed = [float(sample["elapsed"]) / 60.0]
    phase = [float(sample["phase"])]
    return local + neighbor_mean + directional + interaction + temporal + elapsed + phase


def cooperative_no_interaction_features(sample: TrafficSample) -> List[float]:
    local = _scale(list(sample["local"]))
    neighbor_mean = _scale(list(sample["neighbor_mean"]))
    directional = _scale(list(sample.get("neighbor_directional", [])))
    temporal = _scale(list(sample.get("temporal", [])), factor=20.0)
    elapsed = [float(sample["elapsed"]) / 60.0]
    phase = [float(sample["phase"])]
    return local + neighbor_mean + directional + temporal + elapsed + phase


def cooperative_no_neighbor_features(sample: TrafficSample) -> List[float]:
    local_raw = list(sample["local"])
    local = _scale(local_raw)
    self_interaction = _scale(_local_self_interaction(local_raw), factor=10.0)
    temporal = _scale(list(sample.get("temporal", [])), factor=20.0)
    elapsed = [float(sample["elapsed"]) / 60.0]
    phase = [float(sample["phase"])]
    return local + self_interaction + temporal + elapsed + phase


def cooperative_temporal_features(sample: TrafficSample) -> List[float]:
    """History-aware cooperative features (v2).

    Extends the base cooperative feature set with:
    - neighbor_delta:   change in neighbor state vs previous timestep
    - neighbor_rolling: rolling mean of neighbor state (temporal smoothing)
    - cross_temporal:   local × neighbor trend alignment signal

    These are designed to help the model generalise across timesteps under
    a temporally-correct train/val split.
    """
    local = _scale(list(sample["local"]))
    neighbor_mean = _scale(list(sample["neighbor_mean"]))
    directional = _scale(list(sample.get("neighbor_directional", [])))
    interaction = _scale(list(sample["interaction"]), factor=10.0)
    temporal = _scale(list(sample.get("temporal", [])), factor=20.0)
    neighbor_delta = _scale(list(sample.get("neighbor_delta", [0.0] * len(sample["local"]))), factor=10.0)
    neighbor_rolling = _scale(list(sample.get("neighbor_rolling", sample["neighbor_mean"])))
    cross_temporal = _scale(list(sample.get("cross_temporal", [0.0] * len(sample["local"]))), factor=10.0)
    elapsed = [float(sample["elapsed"]) / 60.0]
    phase = [float(sample["phase"])]
    return (local + neighbor_mean + directional + interaction + temporal
            + neighbor_delta + neighbor_rolling + cross_temporal
            + elapsed + phase)


def simple_fusion_features(sample: TrafficSample) -> List[float]:
    local = _scale(list(sample["local"]))
    neighbor_mean = _scale(list(sample["neighbor_mean"]))
    elapsed = [float(sample["elapsed"]) / 60.0]
    phase = [float(sample["phase"])]
    return local + neighbor_mean + elapsed + phase


def graph_lite_features(sample: TrafficSample) -> List[float]:
    local = _scale(list(sample["local"]))
    neighbor_mean = _scale(list(sample["neighbor_mean"]))
    directional = _scale(list(sample.get("neighbor_directional", [])))
    elapsed = [float(sample["elapsed"]) / 60.0]
    phase = [float(sample["phase"])]
    return local + neighbor_mean + directional + elapsed + phase


def cooperative_no_direction_features(sample: TrafficSample) -> List[float]:
    local = _scale(list(sample["local"]))
    neighbor_mean = _scale(list(sample["neighbor_mean"]))
    interaction = _scale(list(sample["interaction"]), factor=10.0)
    temporal = _scale(list(sample.get("temporal", [])), factor=20.0)
    elapsed = [float(sample["elapsed"]) / 60.0]
    phase = [float(sample["phase"])]
    return local + neighbor_mean + interaction + temporal + elapsed + phase
