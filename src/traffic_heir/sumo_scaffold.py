from __future__ import annotations

from pathlib import Path
from typing import Dict, List


EXPECTED_STATE_FIELDS = [
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
]


def expected_sumo_layout() -> Dict[str, object]:
    return {
        "description": "Scaffold for future SUMO integration",
        "expected_csv_columns": EXPECTED_STATE_FIELDS,
        "topologies": ["two_intersections", "corridor", "grid_3x3"],
    }


def ensure_sumo_dirs(root: str | Path) -> List[Path]:
    root = Path(root)
    paths = [
        root / "data" / "sumo" / "raw",
        root / "data" / "sumo" / "processed",
        root / "configs" / "sumo",
    ]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return paths
