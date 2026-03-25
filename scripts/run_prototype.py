#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from traffic_heir.config import PrototypeConfig
from traffic_heir.train import run_experiment


def main() -> None:
    config = PrototypeConfig()
    results = run_experiment(config)
    local = results["local_result"]
    coop = results["coop_result"]

    print("=== traffic-heir prototype ===")
    print(f"dataset size: {results['dataset_size']}")
    print(f"train/val: {results['train_size']}/{results['val_size']}")
    print(f"heuristic val acc: {results['heuristic_val_accuracy']:.4f}")
    print(f"local plaintext model val acc: {local.val_accuracy:.4f}")
    print(f"cooperative HE-friendly model val acc: {coop.val_accuracy:.4f}")
    print(f"cooperation gain over local: {coop.val_accuracy - local.val_accuracy:+.4f}")


if __name__ == "__main__":
    main()
