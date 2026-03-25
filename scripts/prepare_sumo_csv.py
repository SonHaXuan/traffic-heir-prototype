#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import json

from traffic_heir.sumo_data import build_samples_from_grouped, group_by_timestep, load_sumo_csv


def main() -> None:
    if len(sys.argv) not in (2, 3):
        print("usage: python3 scripts/prepare_sumo_csv.py <path-to-sumo-csv> [path-to-adjacency-json]")
        raise SystemExit(1)

    csv_path = Path(sys.argv[1])
    adjacency = None
    if len(sys.argv) == 3:
        adjacency = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    rows = load_sumo_csv(csv_path)
    grouped = group_by_timestep(rows)
    samples = build_samples_from_grouped(grouped, adjacency=adjacency)
    print(f"loaded rows: {len(rows)}")
    print(f"timesteps: {len(grouped)}")
    print(f"constructed cooperative samples: {len(samples)}")
    if samples:
        print(f"sample keys: {sorted(samples[0].keys())}")


if __name__ == "__main__":
    main()
