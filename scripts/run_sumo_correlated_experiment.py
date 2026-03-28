"""
run_sumo_correlated_experiment.py
----------------------------------
Runs the SUMO binary experiment on the correlated corridor dataset
(7 intersections × 300 timesteps = 2100 rows).

The correlated dataset includes upstream spillback between adjacent
intersections, making neighbor features genuinely informative (Pearson r ≈ 0.97
between adjacent intersections vs ~0.3 for the independent large dataset).

Steps:
  1. Generate correlated data if not already present
  2. Run sumo binary experiment with temporal split + full ablation
  3. Compute bootstrap CI for each model variant
  4. Save to reports/sumo_correlated_metrics.json

Usage:
  python3 scripts/run_sumo_correlated_experiment.py [--regenerate]
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
PYTHON = sys.executable

CSV_PATH = ROOT / "data" / "sumo" / "raw" / "correlated_states.csv"
ADJ_PATH = ROOT / "configs" / "sumo" / "correlated_adjacency.json"
REPORT_PATH = ROOT / "reports" / "sumo_correlated_metrics.json"


def main() -> None:
    regenerate = "--regenerate" in sys.argv

    # Step 1 – generate correlated data if missing or forced
    if not CSV_PATH.exists() or regenerate:
        print("=== Generating correlated SUMO dataset ===")
        subprocess.run(
            [PYTHON, "scripts/generate_sumo_correlated.py"],
            cwd=ROOT,
            check=True,
        )
    else:
        print(f"Using existing correlated dataset: {CSV_PATH}")

    # Step 2 – run experiment
    print("=== Running SUMO correlated experiment ===")
    from traffic_heir.config import PrototypeConfig
    from traffic_heir.evaluate import build_xy
    from traffic_heir.models import predict_batch
    from traffic_heir.reporting import write_metrics_report
    from traffic_heir.stats import paired_ttest, effect_size_cohens_d
    from traffic_heir.sumo_experiment import run_sumo_binary_experiment

    cfg = PrototypeConfig(
        epochs=200,
        coop_hidden_dim=32,
        local_hidden_dim=16,
        learning_rate=0.008,
    )

    result = run_sumo_binary_experiment(
        csv_path=CSV_PATH,
        adjacency_path=ADJ_PATH,
        config=cfg,
        report_path=None,          # we'll write enhanced report below
        split_mode="temporal",
    )

    # Step 3 – Wilson score 95% CI for each model accuracy
    # Wilson interval is correct for a single-run proportion estimate.
    # (Bootstrap CI requires raw predictions which are not returned by run_sumo_binary_experiment.)
    import math as _math

    def _wilson_ci(acc: float, n: int, z: float = 1.96) -> tuple:
        """Wilson score interval for a proportion."""
        if n == 0:
            return (0.0, 0.0)
        p = acc
        denom = 1.0 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        margin = (z * _math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
        return (max(0.0, round(centre - margin, 4)),
                min(1.0, round(centre + margin, 4)))

    n_val = int(result.get("val", 420))
    ci_results: dict = {}
    for mode in ("local", "simple_fusion", "graph_lite", "coop", "coop_temporal"):
        acc_key = f"{mode}_val_accuracy"
        if acc_key not in result:
            continue
        acc = float(result[acc_key])
        lo, hi = _wilson_ci(acc, n_val)
        ci_results[mode] = {
            "val_accuracy": acc,
            "ci_lower_95": lo,
            "ci_upper_95": hi,
            "ci_method": "Wilson score interval",
            "n_val": n_val,
        }

    result["wilson_ci"] = ci_results

    # Step 4 – write report
    write_metrics_report(result, REPORT_PATH)

    # Print summary
    ev = result.get("eval_story", {})
    print()
    print("=== Correlated SUMO Experiment Summary ===")
    print(f"  Dataset: {result['samples']} samples "
          f"({result['train']} train / {result['val']} val)")
    print(f"  Intersections: {result['adjacency_nodes']}, "
          f"Timesteps: {result['timesteps']}, Split: {result['split_mode']}")
    print()
    print("  Model accuracies:")
    for mode in ("local", "simple_fusion", "graph_lite",
                 "coop", "coop_temporal", "coop_no_neighbor",
                 "coop_no_direction", "coop_no_interaction"):
        key = f"{mode}_val_accuracy"
        if key in result:
            ci = ci_results.get(mode, {})
            ci_str = (f"  [95% Wilson CI: {ci.get('ci_lower_95', '?'):.4f}–"
                      f"{ci.get('ci_upper_95', '?'):.4f}]") if mode in ci_results else ""
            print(f"    {mode:<22}: {float(result[key]):.4f}{ci_str}")

    print()
    print("  Progressive fusion gains (eval_story):")
    for k in ("simple_fusion_gain_over_local", "graph_lite_gain_over_local",
              "cooperative_gain_over_local", "temporal_coop_gain_over_coop",
              "ml_coop_gain_over_heuristic_coop"):
        if k in ev:
            print(f"    {k:<40}: {ev[k]:+.4f}")

    print()
    print(f"  Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
