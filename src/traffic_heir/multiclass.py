from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .fusion import cooperative_features, local_features
from .models import (
    MultiClassTrainResult,
    TrainResult,
    forward_logits_batch,
    predict_multiclass_batch,
    train_two_layer_multiclass_network,
    train_two_layer_network,
)
from .types import TrafficSample


@dataclass
class MultiClassOVRResult:
    models: List[TrainResult]
    classes: List[int]
    val_accuracy: float


def _build_features(samples: Sequence[TrafficSample], mode: str) -> List[List[float]]:
    if mode == "local":
        return [local_features(sample) for sample in samples]
    return [cooperative_features(sample) for sample in samples]


def train_one_vs_rest(
    samples_train: Sequence[TrafficSample],
    y_train: Sequence[int],
    samples_val: Sequence[TrafficSample],
    y_val: Sequence[int],
    *,
    mode: str,
    classes: Sequence[int],
    hidden_dim: int,
    epochs: int,
    lr: float,
    seed: int,
    he_friendly: bool,
) -> MultiClassOVRResult:
    x_train = _build_features(samples_train, mode)
    x_val = _build_features(samples_val, mode)
    models: List[TrainResult] = []
    for idx, cls in enumerate(classes):
        binary_train = [1 if y == cls else 0 for y in y_train]
        binary_val = [1 if y == cls else 0 for y in y_val]
        models.append(
            train_two_layer_network(
                x_train,
                binary_train,
                x_val,
                binary_val,
                hidden_dim=hidden_dim,
                epochs=epochs,
                lr=lr,
                seed=seed + idx,
                he_friendly=he_friendly,
            )
        )
    preds = predict_one_vs_rest(models, classes, x_val, he_friendly=he_friendly)
    correct = sum(int(a == b) for a, b in zip(y_val, preds))
    return MultiClassOVRResult(models=models, classes=list(classes), val_accuracy=correct / max(1, len(y_val)))


def train_multiclass(
    samples_train: Sequence[TrafficSample],
    y_train: Sequence[int],
    samples_val: Sequence[TrafficSample],
    y_val: Sequence[int],
    *,
    mode: str,
    classes: Sequence[int],
    hidden_dim: int,
    epochs: int,
    lr: float,
    seed: int,
    he_friendly: bool,
    class_weighting: bool = True,
) -> MultiClassTrainResult:
    x_train = _build_features(samples_train, mode)
    x_val = _build_features(samples_val, mode)
    return train_two_layer_multiclass_network(
        x_train,
        y_train,
        x_val,
        y_val,
        classes=classes,
        hidden_dim=hidden_dim,
        epochs=epochs,
        lr=lr,
        seed=seed,
        he_friendly=he_friendly,
        class_weighting=class_weighting,
    )


def predict_one_vs_rest(
    models: Sequence[TrainResult],
    classes: Sequence[int],
    xs: Sequence[Sequence[float]],
    *,
    he_friendly: bool,
) -> List[int]:
    all_logits = [forward_logits_batch(xs, m.weights1, m.bias1, m.weights2, m.bias2, he_friendly) for m in models]
    outputs: List[int] = []
    for row_idx in range(len(xs)):
        scores = [all_logits[m_idx][row_idx] for m_idx in range(len(models))]
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        outputs.append(classes[best_idx])
    return outputs


def predict_multiclass(
    model: MultiClassTrainResult,
    xs: Sequence[Sequence[float]],
    *,
    he_friendly: bool,
) -> List[int]:
    return predict_multiclass_batch(
        xs,
        model.weights1,
        model.bias1,
        model.weights2,
        model.bias2,
        model.classes,
        he_friendly=he_friendly,
    )
