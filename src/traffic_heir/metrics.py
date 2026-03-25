from __future__ import annotations

from collections import Counter
from typing import Dict, Sequence


def confusion_counts(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for truth, pred in zip(y_true, y_pred):
        key = f"{truth}->{pred}"
        out[key] = out.get(key, 0) + 1
    return out


def distribution(values: Sequence[int]) -> Dict[int, int]:
    return dict(sorted(Counter(values).items()))


def per_class_precision_recall_f1(y_true: Sequence[int], y_pred: Sequence[int], classes: Sequence[int]) -> Dict[int, Dict[str, float]]:
    metrics: Dict[int, Dict[str, float]] = {}
    for cls in classes:
        tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == cls and pred == cls)
        fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth != cls and pred == cls)
        fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == cls and pred != cls)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        metrics[int(cls)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(sum(1 for truth in y_true if truth == cls)),
        }
    return metrics


def macro_f1_from_per_class(per_class: Dict[int, Dict[str, float]]) -> float:
    if not per_class:
        return 0.0
    return sum(values["f1"] for values in per_class.values()) / len(per_class)
