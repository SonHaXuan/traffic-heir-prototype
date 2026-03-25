from __future__ import annotations

from typing import Dict

from .config import PrototypeConfig
from .evaluate import accuracy, build_splits, build_xy, heuristic_predict
from .models import TrainResult, train_two_layer_network
from .synthetic import generate_dataset


def run_experiment(config: PrototypeConfig) -> Dict[str, object]:
    dataset = generate_dataset(config)
    train_samples, val_samples = build_splits(dataset, config.train_ratio)

    x_train_local, y_train = build_xy(train_samples, mode="local")
    x_val_local, y_val = build_xy(val_samples, mode="local")
    x_train_coop, _ = build_xy(train_samples, mode="coop")
    x_val_coop, _ = build_xy(val_samples, mode="coop")

    heuristic_acc = accuracy(y_val, heuristic_predict(val_samples))

    local_result: TrainResult = train_two_layer_network(
        x_train_local,
        y_train,
        x_val_local,
        y_val,
        hidden_dim=config.local_hidden_dim,
        epochs=config.epochs,
        lr=config.learning_rate,
        seed=config.seed,
        he_friendly=False,
    )

    coop_result: TrainResult = train_two_layer_network(
        x_train_coop,
        y_train,
        x_val_coop,
        y_val,
        hidden_dim=config.coop_hidden_dim,
        epochs=config.epochs,
        lr=config.learning_rate,
        seed=config.seed + 1,
        he_friendly=True,
    )

    return {
        "dataset_size": len(dataset),
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "heuristic_val_accuracy": heuristic_acc,
        "local_result": local_result,
        "coop_result": coop_result,
    }
