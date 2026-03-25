#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from traffic_heir.config import PrototypeConfig
from traffic_heir.heir_export import export_heir_stub
from traffic_heir.train import run_experiment


def main() -> None:
    config = PrototypeConfig(num_samples=120, epochs=40)
    results = run_experiment(config)
    local = results["local_result"]
    coop = results["coop_result"]

    assert 0.0 <= results["fixed_time_val_accuracy"] <= 1.0
    assert 0.0 <= results["heuristic_val_accuracy"] <= 1.0
    assert 0.0 <= results["max_pressure_val_accuracy"] <= 1.0
    assert 0.0 <= local.val_accuracy <= 1.0
    assert 0.0 <= coop.val_accuracy <= 1.0
    assert 0.0 <= results["ablation_no_interaction"].val_accuracy <= 1.0
    assert 0.0 <= results["ablation_no_neighbor"].val_accuracy <= 1.0

    out = ROOT / "generated" / "smoke_heir_stub.py"
    export_heir_stub(coop, out)
    assert out.exists()
    assert "@compile(scheme='ckks')" in out.read_text(encoding="utf-8")
    print("smoke test passed")


if __name__ == "__main__":
    main()
