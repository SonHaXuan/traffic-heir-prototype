from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import PrototypeConfig


def load_config(path: str | Path) -> PrototypeConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return PrototypeConfig(**data)


def save_config(config: PrototypeConfig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8")
    return path


def apply_overrides(config: PrototypeConfig, **overrides: Any) -> PrototypeConfig:
    data = asdict(config)
    for key, value in overrides.items():
        if value is None:
            continue
        if key not in data:
            raise KeyError(f"Unknown config field: {key}")
        data[key] = value
    return PrototypeConfig(**data)
