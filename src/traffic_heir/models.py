from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class TrainResult:
    weights1: List[List[float]]
    bias1: List[float]
    weights2: List[List[float]]
    bias2: List[float]
    train_accuracy: float
    val_accuracy: float


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 20), -20)))


def poly_activation(x: float) -> float:
    return 0.125 * x * x + 0.5 * x + 0.25


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    return correct / max(1, len(y_true))


def train_two_layer_network(
    x_train: Sequence[Sequence[float]],
    y_train: Sequence[int],
    x_val: Sequence[Sequence[float]],
    y_val: Sequence[int],
    hidden_dim: int,
    epochs: int,
    lr: float,
    seed: int,
    he_friendly: bool,
) -> TrainResult:
    rng = random.Random(seed)
    input_dim = len(x_train[0])

    w1 = [[rng.uniform(-0.1, 0.1) for _ in range(input_dim)] for _ in range(hidden_dim)]
    b1 = [0.0 for _ in range(hidden_dim)]
    w2 = [[rng.uniform(-0.1, 0.1) for _ in range(hidden_dim)]]
    b2 = [0.0]

    for _ in range(epochs):
        for x, y in zip(x_train, y_train):
            hidden_pre = [_dot(row, x) + b for row, b in zip(w1, b1)]
            if he_friendly:
                hidden = [poly_activation(v) for v in hidden_pre]
                hidden_grad = [0.25 * v + 0.5 for v in hidden_pre]
            else:
                hidden = [max(0.0, v) for v in hidden_pre]
                hidden_grad = [1.0 if v > 0 else 0.0 for v in hidden_pre]

            logit = _dot(w2[0], hidden) + b2[0]
            pred = _sigmoid(logit)
            error = pred - y

            old_w2 = list(w2[0])
            for j in range(hidden_dim):
                w2[0][j] -= lr * error * hidden[j]
            b2[0] -= lr * error

            for j in range(hidden_dim):
                delta = error * old_w2[j] * hidden_grad[j]
                for i in range(input_dim):
                    w1[j][i] -= lr * delta * x[i]
                b1[j] -= lr * delta

    train_pred = predict_batch(x_train, w1, b1, w2, b2, he_friendly)
    val_pred = predict_batch(x_val, w1, b1, w2, b2, he_friendly)
    return TrainResult(
        weights1=w1,
        bias1=b1,
        weights2=w2,
        bias2=b2,
        train_accuracy=_accuracy(y_train, train_pred),
        val_accuracy=_accuracy(y_val, val_pred),
    )


def forward_logits_batch(
    xs: Sequence[Sequence[float]],
    w1: Sequence[Sequence[float]],
    b1: Sequence[float],
    w2: Sequence[Sequence[float]],
    b2: Sequence[float],
    he_friendly: bool,
) -> List[float]:
    outputs: List[float] = []
    for x in xs:
        hidden_pre = [_dot(row, x) + b for row, b in zip(w1, b1)]
        if he_friendly:
            hidden = [poly_activation(v) for v in hidden_pre]
        else:
            hidden = [max(0.0, v) for v in hidden_pre]
        outputs.append(_dot(w2[0], hidden) + b2[0])
    return outputs


def predict_batch(
    xs: Sequence[Sequence[float]],
    w1: Sequence[Sequence[float]],
    b1: Sequence[float],
    w2: Sequence[Sequence[float]],
    b2: Sequence[float],
    he_friendly: bool,
) -> List[int]:
    logits = forward_logits_batch(xs, w1, b1, w2, b2, he_friendly)
    return [1 if _sigmoid(logit) >= 0.5 else 0 for logit in logits]
