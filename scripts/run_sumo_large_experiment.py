"""
run_sumo_large_experiment.py
----------------------------
Runs the SUMO binary experiment on the expanded dataset
(500 samples, 5 intersections, 100 timesteps).

Steps:
  1. Generate synthetic large dataset if not already present
  2. Run sumo binary experiment
  3. Save to reports/sumo_large_metrics.json

Usage:
  python3 scripts/run_sumo_large_experiment.py [--regenerate]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

CSV_PATH = ROOT / "data" / "sumo" / "raw" / "large_states.csv"
ADJ_PATH = ROOT / "configs" / "sumo" / "large_adjacency.json"
REPORT_PATH = ROOT / "reports" / "sumo_large_metrics.json"


def main():
    regenerate = "--regenerate" in sys.argv

    # Step 1 – generate if missing or forced
    if not CSV_PATH.exists() or regenerate:
        print("=== Generating large SUMO dataset ===")
        subprocess.run([PYTHON, "scripts/generate_sumo_large.py"], cwd=ROOT, check=True)
    else:
        print(f"Using existing dataset: {CSV_PATH}")

    # Step 2 – run experiment, save to dedicated report
    print("=== Running SUMO large experiment ===")
    import json
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    from traffic_heir.config import PrototypeConfig
    from traffic_heir.sumo_experiment import run_sumo_binary_experiment

    # Larger hidden dim + more epochs for meaningful signal at 500-sample scale
    cfg = PrototypeConfig(num_samples=500, epochs=200, coop_hidden_dim=32)
    result = run_sumo_binary_experiment(
        csv_path=CSV_PATH,
        adjacency_path=ADJ_PATH,
        config=cfg,
        report_path=REPORT_PATH,
        split_mode="temporal",  # avoid temporal leakage
    )

    # Print summary
    ev = result.get("eval_story", {})
    print()
    print("=== SUMO Large Experiment Summary ===")
    print(f"  Samples:       {result['samples']} ({result['train']} train / {result['val']} val)")
    print(f"  Intersections: {result['adjacency_nodes']}")
    print(f"  Timesteps:     {result['timesteps']}")
    print(f"  Label balance: {result['label_distribution']}")
    print()
    print(f"  Coop HE val acc:  {result['coop_val_accuracy']:.4f}")
    print(f"  Local val acc:    {result['local_val_accuracy']:.4f}")
    print(f"  Coop gain:        +{ev.get('cooperative_gain_over_local', 0)*100:.1f} pp")
    print(f"  Directional gain: +{ev.get('directional_gain_within_coop', 0)*100:.1f} pp")
    print()
    print(f"  Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
