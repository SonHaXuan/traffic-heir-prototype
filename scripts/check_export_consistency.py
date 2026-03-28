#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from traffic_heir.config import PrototypeConfig
from traffic_heir.heir_consistency import check_export_shape, manual_forward_sample
from traffic_heir.heir_export import export_heir_stub
from traffic_heir.synthetic import generate_dataset
from traffic_heir.train import run_experiment


def main() -> None:
    config = PrototypeConfig(num_samples=120, epochs=40)
    results = run_experiment(config)
    result = results["coop_result"]
    sample = generate_dataset(config)[0]
    score = manual_forward_sample(result, sample)
    out = ROOT / "generated" / "consistency_heir_stub.py"
    export_heir_stub(result, out)
    assert check_export_shape(result, out)
    print(f"manual cooperative score: {score:.6f}")
    print(f"export check passed: {out}")


if __name__ == "__main__":
    main()
