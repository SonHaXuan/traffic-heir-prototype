from __future__ import annotations

from typing import Dict, List


def _scale(values: List[float], factor: float = 20.0) -> List[float]:
    return [v / factor for v in values]


def local_features(sample: Dict[str, object]) -> List[float]:
    local = list(sample["local"])  # type: ignore[arg-type]
    return _scale(local) + [float(sample["elapsed"]) / 60.0, float(sample["phase"])]


def cooperative_features(sample: Dict[str, object]) -> List[float]:
    local = _scale(list(sample["local"]))  # type: ignore[arg-type]
    neighbor_mean = _scale(list(sample["neighbor_mean"]))  # type: ignore[arg-type]
    interaction = _scale(list(sample["interaction"]), factor=10.0)  # type: ignore[arg-type]
    elapsed = [float(sample["elapsed"]) / 60.0]
    phase = [float(sample["phase"])]
    return local + neighbor_mean + interaction + elapsed + phase


def cooperative_no_interaction_features(sample: Dict[str, object]) -> List[float]:
    local = _scale(list(sample["local"]))  # type: ignore[arg-type]
    neighbor_mean = _scale(list(sample["neighbor_mean"]))  # type: ignore[arg-type]
    elapsed = [float(sample["elapsed"]) / 60.0]
    phase = [float(sample["phase"])]
    return local + neighbor_mean + elapsed + phase


def cooperative_no_neighbor_features(sample: Dict[str, object]) -> List[float]:
    local = _scale(list(sample["local"]))  # type: ignore[arg-type]
    interaction = _scale(list(sample["interaction"]), factor=10.0)  # type: ignore[arg-type]
    elapsed = [float(sample["elapsed"]) / 60.0]
    phase = [float(sample["phase"])]
    return local + interaction + elapsed + phase
