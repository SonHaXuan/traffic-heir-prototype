from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

from .fusion import (
    cooperative_features,
    cooperative_no_interaction_features,
    cooperative_no_neighbor_features,
    local_features,
)
from .labels import decision_label, local_heuristic_label


def build_splits(
    dataset: Sequence[Dict[str, object]], train_ratio: float, seed: int = 7
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    shuffled = list(dataset)
    random.Random(seed).shuffle(shuffled)
    split = int(len(shuffled) * train_ratio)
    return shuffled[:split], shuffled[split:]


def build_xy(samples: Sequence[Dict[str, object]], mode: str) -> Tuple[List[List[float]], List[int]]:
    xs: List[List[float]] = []
    ys: List[int] = []
    for sample in samples:
        if mode == "local":
            xs.append(local_features(sample))
        elif mode == "coop_no_interaction":
            xs.append(cooperative_no_interaction_features(sample))
        elif mode == "coop_no_neighbor":
            xs.append(cooperative_no_neighbor_features(sample))
        else:
            xs.append(cooperative_features(sample))
        ys.append(decision_label(sample))
    return xs, ys


def heuristic_predict(samples: Sequence[Dict[str, object]]) -> List[int]:
    return [local_heuristic_label(sample) for sample in samples]


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return sum(int(a == b) for a, b in zip(y_true, y_pred)) / max(1, len(y_true))
