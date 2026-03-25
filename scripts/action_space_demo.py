#!/usr/bin/env python3
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from traffic_heir.action_space import decision_label_4, local_heuristic_label_4
from traffic_heir.config import PrototypeConfig
from traffic_heir.synthetic import generate_dataset


def main() -> None:
    samples = generate_dataset(PrototypeConfig(num_samples=200))
    coop_counts = Counter(decision_label_4(sample) for sample in samples)
    local_counts = Counter(local_heuristic_label_4(sample) for sample in samples)
    print("4-action cooperative distribution:", dict(sorted(coop_counts.items())))
    print("4-action local distribution:", dict(sorted(local_counts.items())))


if __name__ == "__main__":
    main()
