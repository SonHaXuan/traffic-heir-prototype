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
    config = PrototypeConfig()
    results = run_experiment(config)
    coop = results["coop_result"]
    out = ROOT / "generated" / "traffic_policy_heir_stub.py"
    export_heir_stub(coop, out)
    print(f"generated: {out}")


if __name__ == "__main__":
    main()
