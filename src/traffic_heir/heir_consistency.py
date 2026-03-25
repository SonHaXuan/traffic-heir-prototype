from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .fusion import cooperative_features
from .models import TrainResult, poly_activation
from .types import TrafficSample


def manual_forward(result: TrainResult, features: Sequence[float]) -> float:
    hidden = []
    for row, bias in zip(result.weights1, result.bias1):
        acc = bias + sum(w * x for w, x in zip(row, features))
        hidden.append(poly_activation(acc))
    return result.bias2[0] + sum(w * x for w, x in zip(result.weights2[0], hidden))


def manual_forward_sample(result: TrainResult, sample: TrafficSample) -> float:
    return manual_forward(result, cooperative_features(sample))


def check_export_shape(result: TrainResult, exported_path: str | Path) -> bool:
    text = Path(exported_path).read_text(encoding="utf-8")
    return (
        text.count("[") > 3
        and "@compile(scheme='ckks')" in text
        and f"B2 = {result.bias2[0]:.6f}" in text
    )
