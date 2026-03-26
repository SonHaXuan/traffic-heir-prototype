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
    "summary_report.json": ["prototype", "seed_sweep", "action4", "heir_export", "sumo_binary"],
    "paper_results_table.md": None,
    "summary_report.md": None,
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    checked: list[str] = []
    for name, keys in REQUIRED.items():
        path = REPORTS / name
        require(path.exists(), f"Missing report artifact: {name}")
        require(path.stat().st_size > 0, f"Empty report artifact: {name}")
        if path.suffix == ".json":
            payload = load_json(path)
            require(isinstance(payload, dict), f"JSON artifact must be an object: {name}")
            for key in keys or []:
                require(key in payload, f"Missing key '{key}' in {name}")
        else:
            text = path.read_text(encoding="utf-8")
            require(len(text.strip()) > 0, f"Markdown artifact is blank: {name}")
            require("|" in text, f"Markdown artifact does not look tabular: {name}")
        checked.append(name)

    summary = load_json(REPORTS / "summary_report.json")
    require(summary["heir_export"]["shape_check_passed"] is True, "HEIR shape check must pass")
    require(summary["heir_export"]["consistency_check_passed"] is True, "HEIR consistency check must pass")
    require(summary["sumo_binary"]["samples"] >= summary["sumo_binary"]["timesteps"], "SUMO samples should be >= timesteps")

    print({"validated": checked, "count": len(checked)})


if __name__ == "__main__":
    main()
