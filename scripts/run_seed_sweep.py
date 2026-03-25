#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from traffic_heir.config import PrototypeConfig
from traffic_heir.reporting import summarize_runs, write_markdown_table, write_metrics_report
from traffic_heir.train import run_experiment


def main() -> None:
    seeds = [7, 13, 23]
    rows = []
    local_vals = []
    coop_vals = []
    nodir_vals = []
    noneigh_vals = []
    for seed in seeds:
        cfg = PrototypeConfig(seed=seed)
        result = run_experiment(cfg)
        local = result["local_result"].val_accuracy
        coop = result["coop_result"].val_accuracy
        nodir = result["ablation_no_direction"].val_accuracy
        noneigh = result["ablation_no_neighbor"].val_accuracy
        rows.append({
            "seed": seed,
            "local": round(local, 4),
            "coop_he": round(coop, 4),
            "no_direction": round(nodir, 4),
            "no_neighbor": round(noneigh, 4),
        })
        local_vals.append(local)
        coop_vals.append(coop)
        nodir_vals.append(nodir)
        noneigh_vals.append(noneigh)
    summary = {
        "local": summarize_runs(local_vals),
        "coop_he": summarize_runs(coop_vals),
        "no_direction": summarize_runs(nodir_vals),
        "no_neighbor": summarize_runs(noneigh_vals),
    }
    metrics_path = ROOT / "reports" / "seed_sweep_metrics.json"
    table_path = ROOT / "reports" / "seed_sweep_table.md"
    write_metrics_report({"rows": rows, "summary": summary}, metrics_path)
    write_markdown_table(rows, table_path)
    print({"metrics": str(metrics_path), "table": str(table_path), "summary": summary})


if __name__ == "__main__":
    main()
