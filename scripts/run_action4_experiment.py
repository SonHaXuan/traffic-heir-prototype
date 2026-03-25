#!/usr/bin/env python3
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from traffic_heir.action_space import decision_label_4
from traffic_heir.config import PrototypeConfig
from traffic_heir.multiclass import train_one_vs_rest
from traffic_heir.synthetic import generate_dataset
from traffic_heir.evaluate import build_splits


def main() -> None:
    config = PrototypeConfig(num_samples=300, epochs=80)
    samples = generate_dataset(config)
    train_samples, val_samples = build_splits(samples, config.train_ratio, seed=config.seed)
    y_train = [decision_label_4(s, config) for s in train_samples]
    y_val = [decision_label_4(s, config) for s in val_samples]
    result = train_one_vs_rest(
        train_samples,
        y_train,
        val_samples,
        y_val,
        mode="coop",
        classes=[0, 1, 2, 3],
        hidden_dim=config.coop_hidden_dim,
        epochs=config.epochs,
        lr=config.learning_rate,
        seed=config.seed,
        he_friendly=True,
    )
    print("action4 val accuracy:", round(result.val_accuracy, 4))
    print("val label distribution:", dict(sorted(Counter(y_val).items())))


if __name__ == "__main__":
    main()
