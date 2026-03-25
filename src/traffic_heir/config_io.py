from __future__ import annotations

import json
from pathlib import Path

from .config import PrototypeConfig


def load_config(path: str | Path) -> PrototypeConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return PrototypeConfig(**data)
