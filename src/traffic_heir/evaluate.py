from __future__ import annotations

import random
from typing import List, Sequence, Tuple

from .fusion import (
    cooperative_features,
    cooperative_no_direction_features,
    cooperative_no_interaction_features,
    cooperative_no_neighbor_features,
    cooperative_temporal_features,
    graph_lite_features,
    local_features,
    simple_fusion_features,
)
from .labels import decision_label, local_heuristic_label
from .types import TrafficSample


def build_splits(
    dataset: Sequence[TrafficSample], train_ratio: float, seed: int = 7
) -> Tuple[List[TrafficSample], List[TrafficSample]]:
    shuffled = list(dataset)
    random.Random(seed).shuffle(shuffled)
    split = int(len(shuffled) * train_ratio)
    return shuffled[:split], shuffled[split:]


def build_xy(samples: Sequence[TrafficSample], mode: str) -> Tuple[List[List[float]], List[int]]:
    xs: List[List[float]] = []
    ys: List[int] = []
    for sample in samples:
        if mode == "local":
            xs.append(local_features(sample))
        elif mode == "simple_fusion":
            xs.append(simple_fusion_features(sample))
        elif mode == "graph_lite":
            xs.append(graph_lite_features(sample))
        elif mode == "coop_no_interaction":
            xs.append(cooperative_no_interaction_features(sample))
        elif mode == "coop_no_neighbor":
            xs.append(cooperative_no_neighbor_features(sample))
        elif mode == "coop_no_direction":
            xs.append(cooperative_no_direction_features(sample))
        elif mode == "coop_temporal":
            xs.append(cooperative_temporal_features(sample))
        else:
            xs.append(cooperative_features(sample))
        ys.append(decision_label(sample))
    return xs, ys


def heuristic_predict(samples: Sequence[TrafficSample]) -> List[int]:
    return [local_heuristic_label(sample) for sample in samples]


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return sum(int(a == b) for a, b in zip(y_true, y_pred)) / max(1, len(y_true))
