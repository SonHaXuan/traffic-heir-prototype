from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List

REQUIRED_COLUMNS = {
    "intersection_id",
    "timestep",
    "q_n",
    "q_s",
    "q_e",
    "q_w",
    "w_n",
    "w_s",
    "w_e",
    "w_w",
    "phase",
    "elapsed",
}


def load_sumo_csv(path: str | Path) -> List[Dict[str, object]]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing SUMO columns: {sorted(missing)}")
        rows: List[Dict[str, object]] = []
        for row in reader:
            rows.append(
                {
                    "intersection_id": row["intersection_id"],
                    "timestep": int(float(row["timestep"])),
                    "local": [
                        float(row["q_n"]),
                        float(row["q_s"]),
                        float(row["q_e"]),
                        float(row["q_w"]),
                        float(row["w_n"]),
                        float(row["w_s"]),
                        float(row["w_e"]),
                        float(row["w_w"]),
                    ],
                    "phase": float(row["phase"]),
                    "elapsed": float(row["elapsed"]),
                }
            )
    return rows


def group_by_timestep(rows: Iterable[Dict[str, object]]) -> Dict[int, List[Dict[str, object]]]:
    grouped: Dict[int, List[Dict[str, object]]] = {}
    for row in rows:
        timestep = int(row["timestep"])
        grouped.setdefault(timestep, []).append(dict(row))
    return grouped


def build_samples_from_grouped(grouped: Dict[int, List[Dict[str, object]]]) -> List[Dict[str, object]]:
    samples: List[Dict[str, object]] = []
    for timestep in sorted(grouped):
        rows = grouped[timestep]
        for idx, row in enumerate(rows):
            neighbors = [r for j, r in enumerate(rows) if j != idx]
            if not neighbors:
                continue
            local = list(row["local"])  # type: ignore[arg-type]
            neighbor_mean = [sum(values) / len(neighbors) for values in zip(*[n["local"] for n in neighbors])]  # type: ignore[index]
            interaction = [a * b / 20.0 for a, b in zip(local, neighbor_mean)]
            samples.append(
                {
                    "local": local,
                    "neighbor_mean": neighbor_mean,
                    "interaction": interaction,
                    "phase": row["phase"],
                    "elapsed": row["elapsed"],
                    "source": "sumo",
                }
            )
    return samples
