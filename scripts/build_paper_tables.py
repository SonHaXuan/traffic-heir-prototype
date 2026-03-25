#!/usr/bin/env python3
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from traffic_heir.reporting import write_markdown_table


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    reports = ROOT / "reports"
    proto = load_json(reports / "prototype_default_metrics.json") if (reports / "prototype_default_metrics.json").exists() else {}
    sumo = load_json(reports / "sumo_binary_metrics.json") if (reports / "sumo_binary_metrics.json").exists() else {}
    action4 = load_json(reports / "action4_metrics.json") if (reports / "action4_metrics.json").exists() else {}
    rows = []
    if proto:
        rows.append({"experiment": "prototype_binary", "metric": "coop_he", "value": round(proto.get("coop_he_friendly", 0.0), 4)})
        rows.append({"experiment": "prototype_binary", "metric": "no_direction", "value": round(proto.get("ablation_no_direction", 0.0), 4)})
        rows.append({"experiment": "prototype_binary", "metric": "no_neighbor", "value": round(proto.get("ablation_no_neighbor", 0.0), 4)})
    if sumo:
        rows.append({"experiment": "sumo_binary", "metric": "val_accuracy", "value": round(sumo.get("val_accuracy", 0.0), 4)})
    if action4:
        rows.append({"experiment": "action4", "metric": "val_accuracy", "value": round(action4.get("val_accuracy", 0.0), 4)})
    out = reports / "paper_results_table.md"
    write_markdown_table(rows, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
