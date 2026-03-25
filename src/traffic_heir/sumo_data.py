from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from .types import SumoRow, TrafficSample

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

AdjacencyMap = Mapping[str, Sequence[str]]


def load_sumo_csv(path: str | Path) -> List[SumoRow]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing SUMO columns: {sorted(missing)}")
        rows: List[SumoRow] = []
        seen = set()
        for row in reader:
            key = (row["intersection_id"], int(float(row["timestep"])))
            if key in seen:
                raise ValueError(f"Duplicate SUMO row for intersection/timestep: {key}")
            seen.add(key)
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


def group_by_timestep(rows: Iterable[SumoRow]) -> Dict[int, List[SumoRow]]:
    grouped: Dict[int, List[SumoRow]] = {}
    for row in rows:
        grouped.setdefault(row["timestep"], []).append(dict(row))
    return grouped


def _directional_summary(row: SumoRow, row_by_id: Dict[str, SumoRow], neighbor_ids: Sequence[str]) -> List[float]:
    if not neighbor_ids:
        return [0.0] * 8
    first = row_by_id[neighbor_ids[0]] if neighbor_ids[0] in row_by_id else row
    last = row_by_id[neighbor_ids[-1]] if neighbor_ids[-1] in row_by_id else row
    up = first["local"]
    down = last["local"]
    return [
        up[0] + up[1],
        up[2] + up[3],
        down[0] + down[1],
        down[2] + down[3],
        up[4] + up[5],
        up[6] + up[7],
        down[4] + down[5],
        down[6] + down[7],
    ]


def build_samples_from_grouped(
    grouped: Dict[int, List[SumoRow]],
    adjacency: AdjacencyMap | None = None,
) -> List[TrafficSample]:
    samples: List[TrafficSample] = []
    for timestep in sorted(grouped):
        rows = grouped[timestep]
        row_by_id = {row["intersection_id"]: row for row in rows}
        for row in rows:
            if adjacency is None:
                neighbor_ids = [r["intersection_id"] for r in rows if r["intersection_id"] != row["intersection_id"]]
                neighbors = [r for r in rows if r["intersection_id"] != row["intersection_id"]]
            else:
                neighbor_ids = list(adjacency.get(row["intersection_id"], []))
                neighbors = [row_by_id[nid] for nid in neighbor_ids if nid in row_by_id]
            if not neighbors:
                continue
            local = list(row["local"])
            neighbor_mean = [sum(values) / len(neighbors) for values in zip(*[n["local"] for n in neighbors])]
            neighbor_directional = _directional_summary(row, row_by_id, neighbor_ids)
            interaction = [a * b / 20.0 for a, b in zip(local, neighbor_mean)]
            samples.append(
                {
                    "intersection_id": row["intersection_id"],
                    "timestep": row["timestep"],
                    "local": local,
                    "neighbor_mean": neighbor_mean,
                    "neighbor_directional": neighbor_directional,
                    "interaction": interaction,
                    "phase": row["phase"],
                    "elapsed": row["elapsed"],
                    "source": "sumo",
                }
            )
    return samples
