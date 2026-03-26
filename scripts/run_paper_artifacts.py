#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run_step(label: str, args: list[str]) -> None:
    print(f"\n=== {label} ===")
    cmd = [PYTHON, *args]
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    run_step("prototype_default", ["scripts/run_with_config.py", "configs/experiment/prototype_default.json"])
    run_step("seed_sweep", ["scripts/run_seed_sweep.py"])
    run_step("action4", ["scripts/run_action4_experiment.py"])
    run_step("heir_report", ["scripts/build_heir_report.py"])
    run_step(
        "sumo_binary",
        [
            "scripts/run_sumo_experiment.py",
            "data/sumo/raw/sample_states.csv",
            "configs/sumo/sample_adjacency.json",
        ],
    )
    run_step("paper_tables", ["scripts/build_paper_tables.py"])
    run_step("summary_report", ["scripts/build_summary_report.py"])
    print("\nAll paper artifacts generated under reports/.")


if __name__ == "__main__":
    main()
