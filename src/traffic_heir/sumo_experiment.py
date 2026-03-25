from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from .config import PrototypeConfig
from .evaluate import accuracy, build_splits, build_xy
from .labels import decision_label
from .models import TrainResult, train_two_layer_network
from .sumo_data import build_samples_from_grouped, group_by_timestep, load_sumo_csv


def run_sumo_binary_experiment(
    csv_path: str | Path,
    adjacency_path: str | Path | None = None,
    config: PrototypeConfig | None = None,
) -> Dict[str, object]:
    cfg = config or PrototypeConfig(num_samples=6, epochs=80)
    rows = load_sumo_csv(csv_path)
    grouped = group_by_timestep(rows)
    adjacency = None
    if adjacency_path is not None:
        adjacency = json.loads(Path(adjacency_path).read_text(encoding="utf-8"))
    samples = build_samples_from_grouped(grouped, adjacency=adjacency)
    train_samples, val_samples = build_splits(samples, cfg.train_ratio, seed=cfg.seed)
    _, y_train = build_xy(train_samples, mode="local")
    _, y_val = build_xy(val_samples, mode="local")

    x_train, _ = build_xy(train_samples, mode="coop")
    x_val, _ = build_xy(val_samples, mode="coop")
    result: TrainResult = train_two_layer_network(
        x_train,
        y_train,
        x_val,
        y_val,
        hidden_dim=cfg.coop_hidden_dim,
        epochs=cfg.epochs,
        lr=cfg.learning_rate,
        seed=cfg.seed,
        he_friendly=True,
    )
    label_distribution = {0: sum(1 for s in samples if decision_label(s, cfg) == 0), 1: sum(1 for s in samples if decision_label(s, cfg) == 1)}
    return {
        "rows": len(rows),
        "samples": len(samples),
        "train": len(train_samples),
        "val": len(val_samples),
        "val_accuracy": result.val_accuracy,
        "label_distribution": label_distribution,
    }
