from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .fusion import cooperative_features, local_features
from .labels import decision_label


def build_splits(dataset: Sequence[Dict[str, object]], train_ratio: float) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    split = int(len(dataset) * train_ratio)
    return list(dataset[:split]), list(dataset[split:])


def build_xy(samples: Sequence[Dict[str, object]], mode: str) -> Tuple[List[List[float]], List[int]]:
    xs: List[List[float]] = []
    ys: List[int] = []
    for sample in samples:
        xs.append(local_features(sample) if mode == "local" else cooperative_features(sample))
        ys.append(decision_label(sample))
    return xs, ys


def heuristic_predict(samples: Sequence[Dict[str, object]]) -> List[int]:
    return [decision_label(sample) for sample in samples]


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return sum(int(a == b) for a, b in zip(y_true, y_pred)) / max(1, len(y_true))
