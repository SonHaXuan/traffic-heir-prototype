from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from .config import PrototypeConfig
from .evaluate import build_splits, build_xy
from .labels import decision_label
from .metrics import confusion_counts, distribution
from .models import TrainResult, predict_batch, train_two_layer_network
from .reporting import write_metrics_report
from .sumo_data import build_samples_from_grouped, group_by_timestep, load_sumo_csv


def run_sumo_binary_experiment(
    csv_path: str | Path,
    adjacency_path: str | Path | None = None,
    config: PrototypeConfig | None = None,
    report_path: str | Path | None = None,
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

    modes = {
        "local": (cfg.local_hidden_dim, False),
        "coop": (cfg.coop_hidden_dim, True),
        "coop_no_direction": (cfg.coop_hidden_dim, True),
        "coop_no_interaction": (cfg.coop_hidden_dim, True),
    }
    all_labels = [decision_label(s, cfg) for s in samples]
    train_labels = [decision_label(s, cfg) for s in train_samples]
    metrics: Dict[str, object] = {
        "rows": len(rows),
        "timesteps": len(grouped),
        "samples": len(samples),
        "train": len(train_samples),
        "val": len(val_samples),
        "label_distribution": distribution(all_labels),
        "train_distribution": distribution(train_labels),
        "val_distribution": distribution(y_val),
        "source_csv": str(csv_path),
        "uses_adjacency": adjacency_path is not None,
        "adjacency_nodes": len(adjacency) if adjacency is not None else 0,
    }

    for mode, (hidden_dim, he_friendly) in modes.items():
        x_train, _ = build_xy(train_samples, mode=mode)
        x_val, _ = build_xy(val_samples, mode=mode)
        result: TrainResult = train_two_layer_network(
            x_train,
            y_train,
            x_val,
            y_val,
            hidden_dim=hidden_dim,
            epochs=cfg.epochs,
            lr=cfg.learning_rate,
            seed=cfg.seed + len(mode),
            he_friendly=he_friendly,
        )
        preds = predict_batch(x_val, result.weights1, result.bias1, result.weights2, result.bias2, he_friendly)
        metrics[f"{mode}_val_accuracy"] = result.val_accuracy
        metrics[f"{mode}_pred_distribution"] = distribution(preds)
        metrics[f"{mode}_confusion"] = confusion_counts(y_val, preds)

    metrics["val_accuracy"] = metrics["coop_val_accuracy"]
    metrics["pred_distribution"] = metrics["coop_pred_distribution"]
    metrics["confusion"] = metrics["coop_confusion"]
    metrics["eval_story"] = {
        "cooperative_gain_over_local": round(float(metrics["coop_val_accuracy"]) - float(metrics["local_val_accuracy"]), 6),
        "directional_gain_within_coop": round(float(metrics["coop_val_accuracy"]) - float(metrics["coop_no_direction_val_accuracy"]), 6),
        "interaction_gain_within_coop": round(float(metrics["coop_val_accuracy"]) - float(metrics["coop_no_interaction_val_accuracy"]), 6),
        "label_balance_gap_train_val": round(abs(len(train_labels) - len(y_val)) / max(1, len(samples)), 6),
        "sample_per_timestep": round(len(samples) / max(1, len(grouped)), 6),
    }
    if report_path:
        write_metrics_report(metrics, report_path)
    return metrics
