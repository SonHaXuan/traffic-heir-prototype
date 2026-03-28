#!/usr/bin/env python3
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from traffic_heir.action_space import decision_label_4
from traffic_heir.config import PrototypeConfig
from traffic_heir.fusion import cooperative_features
from traffic_heir.metrics import confusion_counts, macro_f1_from_per_class, per_class_precision_recall_f1
from traffic_heir.multiclass import predict_multiclass, train_multiclass, train_one_vs_rest, predict_one_vs_rest
from traffic_heir.reporting import write_metrics_report
from traffic_heir.synthetic import generate_dataset
from traffic_heir.evaluate import build_splits


CLASSES = [0, 1, 2, 3]


def main() -> None:
    config = PrototypeConfig(num_samples=720, epochs=140, coop_hidden_dim=14, learning_rate=0.008)
    samples = generate_dataset(config)
    train_samples, val_samples = build_splits(samples, config.train_ratio, seed=config.seed)
    y_train = [decision_label_4(s, config) for s in train_samples]
    y_val = [decision_label_4(s, config) for s in val_samples]

    ovr = train_one_vs_rest(
        train_samples,
        y_train,
        val_samples,
        y_val,
        mode="coop",
        classes=CLASSES,
        hidden_dim=config.coop_hidden_dim,
        epochs=config.epochs,
        lr=config.learning_rate,
        seed=config.seed,
        he_friendly=True,
    )
    x_val = [cooperative_features(s) for s in val_samples]
    ovr_preds = predict_one_vs_rest(ovr.models, ovr.classes, x_val, he_friendly=True)
    ovr_per_class = per_class_precision_recall_f1(y_val, ovr_preds, CLASSES)

    result = train_multiclass(
        train_samples,
        y_train,
        val_samples,
        y_val,
        mode="coop",
        classes=CLASSES,
        hidden_dim=config.coop_hidden_dim,
        epochs=config.epochs,
        lr=config.learning_rate,
        seed=config.seed,
        he_friendly=True,
        class_weighting=True,
    )
    preds = predict_multiclass(result, x_val, he_friendly=True)
    per_class = per_class_precision_recall_f1(y_val, preds, CLASSES)
    report = ROOT / "reports" / "action4_metrics.json"
    payload = {
        "val_accuracy": result.val_accuracy,
        "macro_f1": macro_f1_from_per_class(per_class),
        "label_distribution": dict(sorted(Counter(y_val).items())),
        "prediction_distribution": dict(sorted(Counter(preds).items())),
        "confusion": confusion_counts(y_val, preds),
        "per_class": per_class,
        "ovr_val_accuracy": ovr.val_accuracy,
        "ovr_macro_f1": macro_f1_from_per_class(ovr_per_class),
        "ovr_prediction_distribution": dict(sorted(Counter(ovr_preds).items())),
        "ovr_per_class": ovr_per_class,
    }
    write_metrics_report(payload, report)
    print("action4 val accuracy:", round(result.val_accuracy, 4))
    print("action4 macro-F1:", round(payload["macro_f1"], 4))
    print("ovr val accuracy:", round(ovr.val_accuracy, 4))
    print("ovr macro-F1:", round(payload["ovr_macro_f1"], 4))
    print("val label distribution:", dict(sorted(Counter(y_val).items())))
    print("prediction distribution:", dict(sorted(Counter(preds).items())))
    print("report:", report)


if __name__ == "__main__":
    main()
