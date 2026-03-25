#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from traffic_heir.sumo_scaffold import ensure_sumo_dirs, expected_sumo_layout


def main() -> None:
    created = ensure_sumo_dirs(ROOT)
    print("created dirs:")
    for path in created:
        print(f"- {path}")
    print("expected layout:")
    print(expected_sumo_layout())


if __name__ == "__main__":
    main()
