#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from traffic_heir.reporting import write_markdown_table, write_metrics_report


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def safe_round(value: float | int | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def main() -> None:
    reports = ROOT / "reports"
    proto = load_json(reports / "prototype_default_metrics.json")
    seed = load_json(reports / "seed_sweep_metrics.json")
    action4 = load_json(reports / "action4_metrics.json")
    sumo = load_json(reports / "sumo_binary_metrics.json")
    heir = load_json(reports / "heir_export_report.json")

    summary = {
        "prototype": {
            "local_model": safe_round(proto.get("local_model")),
            "coop_plaintext": safe_round(proto.get("coop_plaintext")),
            "coop_he_friendly": safe_round(proto.get("coop_he_friendly")),
            "coop_minus_local": safe_round((proto.get("coop_he_friendly", 0.0) - proto.get("local_model", 0.0))),
            "coop_minus_plaintext": safe_round((proto.get("coop_he_friendly", 0.0) - proto.get("coop_plaintext", 0.0))),
            "robust_coop_he": safe_round(proto.get("robustness", {}).get("coop_he_friendly")),
            "robust_local": safe_round(proto.get("robustness", {}).get("local_model")),
        },
        "seed_sweep": {
            "local_mean": safe_round(seed.get("summary", {}).get("local", {}).get("mean")),
            "local_stdev": safe_round(seed.get("summary", {}).get("local", {}).get("stdev")),
            "coop_he_mean": safe_round(seed.get("summary", {}).get("coop_he", {}).get("mean")),
            "coop_he_stdev": safe_round(seed.get("summary", {}).get("coop_he", {}).get("stdev")),
            "coop_gain_over_local_mean": safe_round(
                seed.get("summary", {}).get("coop_he", {}).get("mean", 0.0)
                - seed.get("summary", {}).get("local", {}).get("mean", 0.0)
            ),
        },
        "action4": {
            "val_accuracy": safe_round(action4.get("val_accuracy")),
            "macro_f1": safe_round(action4.get("macro_f1")),
            "ovr_val_accuracy": safe_round(action4.get("ovr_val_accuracy")),
            "ovr_macro_f1": safe_round(action4.get("ovr_macro_f1")),
        },
        "heir_export": {
            "shape_check_passed": bool(heir.get("shape_check_passed", False)),
            "consistency_check_passed": bool(heir.get("consistency_check_passed", False)),
            "metadata_val_accuracy": safe_round(heir.get("metadata_val_accuracy")),
            "hidden_dim": int(heir.get("hidden_dim", 0)) if heir else 0,
            "input_dim": int(heir.get("input_dim", 0)) if heir else 0,
        },
        "sumo_binary": {
            "val_accuracy": safe_round(sumo.get("val_accuracy")),
            "cooperative_gain_over_local": safe_round(sumo.get("eval_story", {}).get("cooperative_gain_over_local")),
            "directional_gain_within_coop": safe_round(sumo.get("eval_story", {}).get("directional_gain_within_coop")),
            "interaction_gain_within_coop": safe_round(sumo.get("eval_story", {}).get("interaction_gain_within_coop")),
            "sample_per_timestep": safe_round(sumo.get("eval_story", {}).get("sample_per_timestep")),
            "label_balance_gap_train_val": safe_round(sumo.get("eval_story", {}).get("label_balance_gap_train_val")),
            "uses_adjacency": bool(sumo.get("uses_adjacency", False)),
            "samples": int(sumo.get("samples", 0)) if sumo else 0,
            "timesteps": int(sumo.get("timesteps", 0)) if sumo else 0,
            "adjacency_nodes": int(sumo.get("adjacency_nodes", 0)) if sumo else 0,
        },
    }

    rows = [
        {"section": "prototype", "metric": "local_model", "value": summary["prototype"]["local_model"]},
        {"section": "prototype", "metric": "coop_he_friendly", "value": summary["prototype"]["coop_he_friendly"]},
        {"section": "prototype", "metric": "coop_minus_local", "value": summary["prototype"]["coop_minus_local"]},
        {"section": "seed_sweep", "metric": "local_mean", "value": summary["seed_sweep"]["local_mean"]},
        {"section": "seed_sweep", "metric": "coop_he_mean", "value": summary["seed_sweep"]["coop_he_mean"]},
        {"section": "seed_sweep", "metric": "coop_gain_over_local_mean", "value": summary["seed_sweep"]["coop_gain_over_local_mean"]},
        {"section": "action4", "metric": "val_accuracy", "value": summary["action4"]["val_accuracy"]},
        {"section": "action4", "metric": "macro_f1", "value": summary["action4"]["macro_f1"]},
        {"section": "heir_export", "metric": "shape_check_passed", "value": summary["heir_export"]["shape_check_passed"]},
        {"section": "heir_export", "metric": "consistency_check_passed", "value": summary["heir_export"]["consistency_check_passed"]},
        {"section": "heir_export", "metric": "metadata_val_accuracy", "value": summary["heir_export"]["metadata_val_accuracy"]},
        {"section": "sumo_binary", "metric": "val_accuracy", "value": summary["sumo_binary"]["val_accuracy"]},
        {"section": "sumo_binary", "metric": "samples", "value": summary["sumo_binary"]["samples"]},
        {"section": "sumo_binary", "metric": "timesteps", "value": summary["sumo_binary"]["timesteps"]},
        {"section": "sumo_binary", "metric": "interaction_gain_within_coop", "value": summary["sumo_binary"]["interaction_gain_within_coop"]},
        {"section": "sumo_binary", "metric": "sample_per_timestep", "value": summary["sumo_binary"]["sample_per_timestep"]},
    ]

    summary_json = reports / "summary_report.json"
    summary_md = reports / "summary_report.md"
    write_metrics_report(summary, summary_json)
    write_markdown_table(rows, summary_md)
    print({"summary_json": str(summary_json), "summary_md": str(summary_md)})


if __name__ == "__main__":
    main()
