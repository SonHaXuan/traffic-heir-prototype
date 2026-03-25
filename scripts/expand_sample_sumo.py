#!/usr/bin/env python3
import csv
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: python3 scripts/expand_sample_sumo.py <input.csv> <output.csv>")
        raise SystemExit(1)

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    rng = random.Random(7)

    with src.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = rows[0].keys()

    expanded = []
    for repeat in range(20):
        timestep_offset = repeat * 10
        for row in rows:
            new_row = dict(row)
            new_row["timestep"] = str(int(float(row["timestep"])) + timestep_offset)
            for key in ["q_n", "q_s", "q_e", "q_w", "w_n", "w_s", "w_e", "w_w"]:
                value = float(row[key])
                new_row[key] = f"{max(0.0, value * (1.0 + rng.uniform(-0.18, 0.18))):.4f}"
            new_row["elapsed"] = f"{min(60.0, max(0.0, float(row['elapsed']) + rng.uniform(-6.0, 6.0))):.4f}"
            expanded.append(new_row)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(expanded)
    print(f"wrote {len(expanded)} rows to {dst}")


if __name__ == "__main__":
    main()
