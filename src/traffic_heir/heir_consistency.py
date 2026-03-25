from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Sequence

from .fusion import cooperative_features
from .models import MultiClassTrainResult, TrainResult, forward_multiclass_logits_batch, poly_activation
from .types import TrafficSample


def manual_forward(result: TrainResult, features: Sequence[float]) -> float:
    hidden = []
    for row, bias in zip(result.weights1, result.bias1):
        acc = bias + sum(w * x for w, x in zip(row, features))
        hidden.append(poly_activation(acc))
    return result.bias2[0] + sum(w * x for w, x in zip(result.weights2[0], hidden))


def manual_forward_sample(result: TrainResult, sample: TrafficSample) -> float:
    return manual_forward(result, cooperative_features(sample))


def manual_multiclass_forward(result: MultiClassTrainResult, features: Sequence[float]) -> list[float]:
    return forward_multiclass_logits_batch(
        [list(features)],
        result.weights1,
        result.bias1,
        result.weights2,
        result.bias2,
        he_friendly=True,
    )[0]


def manual_multiclass_forward_sample(result: MultiClassTrainResult, sample: TrafficSample) -> list[float]:
    return manual_multiclass_forward(result, cooperative_features(sample))


def check_export_shape(result: TrainResult, exported_path: str | Path) -> bool:
    text = Path(exported_path).read_text(encoding="utf-8")
    return (
        text.count("[") > 3
        and "@compile(scheme='ckks')" in text
        and f"B2 = {result.bias2[0]:.6f}" in text
    )


def load_export_metadata(exported_path: str | Path) -> Dict[str, Any]:
    text = Path(exported_path).read_text(encoding="utf-8")
    match = re.search(r"MODEL_METADATA = (\{.*?\})\n\n", text, re.S)
    if not match:
        raise ValueError("MODEL_METADATA block not found in exported HEIR stub")
    return json.loads(match.group(1))


def exported_binary_forward(exported_path: str | Path, features: Sequence[float]) -> float:
    metadata = load_export_metadata(exported_path)
    hidden = []
    for row, bias in zip(metadata["weights1"], metadata["bias1"]):
        acc = bias + sum(w * x for w, x in zip(row, features))
        hidden.append(poly_activation(acc))
    return metadata["bias2"][0] + sum(w * x for w, x in zip(metadata["weights2"][0], hidden))


def exported_matches_result(result: TrainResult, exported_path: str | Path, sample: TrafficSample, *, tolerance: float = 1e-6) -> bool:
    features = cooperative_features(sample)
    manual = manual_forward(result, features)
    exported = exported_binary_forward(exported_path, features)
    return abs(manual - exported) <= tolerance
