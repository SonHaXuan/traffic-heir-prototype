#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from traffic_heir.config import PrototypeConfig
from traffic_heir.reporting import summarize_runs, summarize_with_ci, write_markdown_table, write_metrics_report
from traffic_heir.stats import effect_size_cohens_d, paired_ttest
from traffic_heir.train import run_experiment

SEEDS = [7, 13, 23, 31, 41, 53, 67]


def main() -> None:
    rows = []
    local_vals = []
    coop_vals = []
    nodir_vals = []
    noneigh_vals = []

    for seed in SEEDS:
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

    # Summary statistics + 95% CI
    summary = {
        "local": summarize_with_ci(local_vals),
        "coop_he": summarize_with_ci(coop_vals),
        "no_direction": summarize_with_ci(nodir_vals),
        "no_neighbor": summarize_with_ci(noneigh_vals),
    }

    # Statistical significance: coop vs local (paired t-test + effect size)
    t_stat, p_value = paired_ttest(coop_vals, local_vals)
    d = effect_size_cohens_d(coop_vals, local_vals)
    significance = {
        "coop_vs_local_t_stat": round(t_stat, 4),
        "coop_vs_local_p_value": round(p_value, 4),
        "coop_vs_local_cohens_d": round(d, 4),
        "n_seeds": len(SEEDS),
        "significant_at_0_05": p_value < 0.05,
        "significant_at_0_10": p_value < 0.10,
    }

    metrics_path = ROOT / "reports" / "seed_sweep_metrics.json"
    table_path = ROOT / "reports" / "seed_sweep_table.md"
    write_metrics_report({"rows": rows, "summary": summary, "significance": significance},
                         metrics_path)
    write_markdown_table(rows, table_path)

    print("\n=== Seed Sweep Summary ===")
    s = summary
    print(f"  coop_he:  {s['coop_he']['mean']:.4f} ± {s['coop_he']['stdev']:.4f} "
          f"  [95% CI: {s['coop_he']['ci_lower_95']:.4f} – {s['coop_he']['ci_upper_95']:.4f}]")
    print(f"  local:    {s['local']['mean']:.4f} ± {s['local']['stdev']:.4f} "
          f"  [95% CI: {s['local']['ci_lower_95']:.4f} – {s['local']['ci_upper_95']:.4f}]")
    print(f"\n  Paired t-test (coop vs local): t={t_stat:.3f}, p={p_value:.4f}, d={d:.3f}")
    sig = "YES" if p_value < 0.05 else ("marginal" if p_value < 0.10 else "NO")
    print(f"  Significant at α=0.05: {sig}")
    print(f"\n  Reports: {metrics_path}, {table_path}")


if __name__ == "__main__":
    main()
