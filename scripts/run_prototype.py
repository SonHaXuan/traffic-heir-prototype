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
    coop_plain = results["coop_plaintext_result"]
    coop_he = results["coop_result"]
    no_interaction = results["ablation_no_interaction"]
    no_neighbor = results["ablation_no_neighbor"]

    print("=== traffic-heir prototype ===")
    print(f"dataset size: {results['dataset_size']}")
    print(f"train/val: {results['train_size']}/{results['val_size']}")
    print(f"fixed-time val acc: {results['fixed_time_val_accuracy']:.4f}")
    print(f"local heuristic val acc: {results['heuristic_val_accuracy']:.4f}")
    print(f"local max-pressure val acc: {results['max_pressure_val_accuracy']:.4f}")
    print(f"local plaintext model val acc: {local.val_accuracy:.4f}")
    print(f"cooperative plaintext model val acc: {coop_plain.val_accuracy:.4f}")
    print(f"cooperative HE-friendly model val acc: {coop_he.val_accuracy:.4f}")
    print(f"ablation no-interaction val acc: {no_interaction.val_accuracy:.4f}")
    print(f"ablation no-neighbor val acc: {no_neighbor.val_accuracy:.4f}")
    print(f"cooperation gain over local: {coop_he.val_accuracy - local.val_accuracy:+.4f}")


if __name__ == "__main__":
    main()
