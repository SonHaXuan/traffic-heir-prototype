#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

REQUIRED = {
    "prototype_default_metrics.json": ["local_model", "coop_plaintext", "coop_he_friendly", "robustness"],
    "seed_sweep_metrics.json": ["rows", "summary"],
    "action4_metrics.json": ["val_accuracy", "macro_f1", "ovr_macro_f1"],
    "heir_export_report.json": ["shape_check_passed", "consistency_check_passed", "export_path"],
    "sumo_binary_metrics.json": ["val_accuracy", "samples", "timesteps", "eval_story"],
    "sumo_large_metrics.json": ["val_accuracy", "samples", "timesteps", "eval_story", "adjacency_nodes"],
    "sumo_correlated_metrics.json": ["val_accuracy", "samples", "timesteps", "eval_story", "adjacency_nodes", "wilson_ci"],
    "heir_overhead_report.json": ["accuracy_gap", "latency_estimates_ms", "communication_cost_7_intersections"],
    "summary_report.json": ["prototype", "seed_sweep", "action4", "heir_export", "sumo_binary"],
    "key_numbers.json": ["main_result", "seed_sweep", "robustness", "ablation", "sumo_correlated", "heir_overhead"],
    "paper_results_table.md": "table",
    "summary_report.md": "table",
    "results_narrative.md": "narrative",
    "key_numbers.md": "table",
    "latex_tables.tex": "latex",
    "latex_tables.md": "table",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    checked: list[str] = []
    for name, spec in REQUIRED.items():
        path = REPORTS / name
        require(path.exists(), f"Missing report artifact: {name}")
        require(path.stat().st_size > 0, f"Empty report artifact: {name}")
        if path.suffix == ".json":
            payload = load_json(path)
            require(isinstance(payload, dict), f"JSON artifact must be an object: {name}")
            for key in spec or []:
                require(key in payload, f"Missing key '{key}' in {name}")
        else:
            text = path.read_text(encoding="utf-8")
            require(len(text.strip()) > 0, f"Markdown/tex artifact is blank: {name}")
            if spec == "table":
                require("|" in text, f"Markdown artifact does not look tabular: {name}")
            elif spec == "narrative":
                require(text.lstrip().startswith("#"), f"Narrative markdown should start with a heading: {name}")
                require("Key findings" in text, f"Narrative markdown missing key findings section: {name}")
            elif spec == "latex":
                require(r"\begin{table}" in text, f"LaTeX artifact missing \\begin{{table}}: {name}")
                require(r"\toprule" in text, f"LaTeX artifact missing \\toprule (booktabs): {name}")
        checked.append(name)

    sumo_large = load_json(REPORTS / "sumo_large_metrics.json")
    require(sumo_large["samples"] >= 400, "SUMO large dataset should have >=400 samples")
    require(sumo_large["adjacency_nodes"] >= 5, "SUMO large should use >=5 intersection nodes")
    require(sumo_large["val_accuracy"] > 0.7, "SUMO large val accuracy should be >70%")
    require(sumo_large.get("split_mode") == "temporal", "SUMO large must use temporal split to avoid leakage")

    sumo_corr = load_json(REPORTS / "sumo_correlated_metrics.json")
    require(sumo_corr["samples"] >= 2000, "Correlated SUMO dataset should have >=2000 samples")
    require(sumo_corr["adjacency_nodes"] >= 7, "Correlated SUMO should use >=7 intersection nodes")
    require(sumo_corr["val_accuracy"] > 0.85, "Correlated SUMO coop val accuracy should be >85%")
    require(sumo_corr.get("split_mode") == "temporal", "Correlated SUMO must use temporal split")
    require(sumo_corr["eval_story"]["cooperative_gain_over_local"] > 0, "Correlated SUMO must show positive cooperative gain")
    require("wilson_ci" in sumo_corr, "Correlated SUMO report must include Wilson CI")

    overhead = load_json(REPORTS / "heir_overhead_report.json")
    require("accuracy_gap" in overhead, "HEIR overhead must report accuracy gap")
    require(overhead["accuracy_gap"]["gap_pp"] < 5.0, "HEIR accuracy gap should be <5 pp")
    require("latency_estimates_ms" in overhead, "HEIR overhead must report latency estimates")

    summary = load_json(REPORTS / "summary_report.json")
    require(summary["heir_export"]["shape_check_passed"] is True, "HEIR shape check must pass")
    require(summary["heir_export"]["consistency_check_passed"] is True, "HEIR consistency check must pass")
    require(summary["sumo_binary"]["samples"] >= summary["sumo_binary"]["timesteps"], "SUMO samples should be >= timesteps")

    print({"validated": checked, "count": len(checked)})


if __name__ == "__main__":
    main()
