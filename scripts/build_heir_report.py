#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from traffic_heir.config import PrototypeConfig
from traffic_heir.heir_consistency import (
    check_export_shape,
    exported_matches_result,
    load_export_metadata,
    manual_forward_sample,
)
from traffic_heir.heir_export import export_heir_stub
from traffic_heir.reporting import write_metrics_report
from traffic_heir.synthetic import generate_dataset
from traffic_heir.train import run_experiment


def main() -> None:
    config = PrototypeConfig(num_samples=120, epochs=40)
    results = run_experiment(config)
    result = results["coop_result"]
    sample = generate_dataset(config)[0]

    out = ROOT / "generated" / "heir_report_stub.py"
    export_heir_stub(result, out)
    metadata = load_export_metadata(out)
    manual_score = manual_forward_sample(result, sample)
    shape_ok = check_export_shape(result, out)
    consistency_ok = exported_matches_result(result, out, sample)

    payload = {
        "export_path": str(out),
        "shape_check_passed": bool(shape_ok),
        "consistency_check_passed": bool(consistency_ok),
        "manual_sample_score": round(float(manual_score), 6),
        "metadata_val_accuracy": round(float(metadata.get("val_accuracy", 0.0)), 6),
        "metadata_train_accuracy": round(float(metadata.get("train_accuracy", 0.0)), 6),
        "hidden_dim": len(metadata.get("bias1", [])),
        "input_dim": len(metadata.get("weights1", [[]])[0]) if metadata.get("weights1") else 0,
        "export_kind": "heir_stub_ckks_compile_decorated",
        "notes": [
            "This report verifies export structure and metadata consistency.",
            "It does not claim end-to-end encrypted execution through the HEIR runtime.",
        ],
    }
    report = ROOT / "reports" / "heir_export_report.json"
    write_metrics_report(payload, report)
    print({"report": str(report), "export_path": str(out), "shape_ok": shape_ok, "consistency_ok": consistency_ok})


if __name__ == "__main__":
    main()
