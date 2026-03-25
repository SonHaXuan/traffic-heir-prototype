#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from traffic_heir.config import PrototypeConfig
from traffic_heir.sumo_experiment import run_sumo_binary_experiment


def main() -> None:
    if len(sys.argv) not in (2, 3):
        print("usage: python3 scripts/run_sumo_experiment.py <sumo.csv> [adjacency.json]")
        raise SystemExit(1)
    csv_path = sys.argv[1]
    adjacency_path = sys.argv[2] if len(sys.argv) == 3 else None
    report = ROOT / "reports" / "sumo_binary_metrics.json"
    result = run_sumo_binary_experiment(csv_path, adjacency_path, PrototypeConfig(num_samples=6, epochs=80), report_path=report)
    print(result)
    print("report:", report)


if __name__ == "__main__":
    main()
