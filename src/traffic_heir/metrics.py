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
