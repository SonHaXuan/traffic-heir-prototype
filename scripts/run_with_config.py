#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from traffic_heir.config_io import load_config
from traffic_heir.train import run_experiment


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python3 scripts/run_with_config.py <config.json>")
        raise SystemExit(1)
    config = load_config(sys.argv[1])
    results = run_experiment(config)
    print({
        "dataset_size": results["dataset_size"],
        "local": round(results["local_result"].val_accuracy, 4),
        "coop_plain": round(results["coop_plaintext_result"].val_accuracy, 4),
        "coop_he": round(results["coop_result"].val_accuracy, 4),
    })


if __name__ == "__main__":
    main()
