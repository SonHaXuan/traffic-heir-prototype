from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .models import TrainResult


def export_heir_stub(result: TrainResult, out_path: str | Path) -> Path:
    out_path = Path(out_path)

    def fmt_vector(values: Iterable[float]) -> str:
        return ", ".join(f"{v:.6f}" for v in values)

    lines = [
        "# Auto-generated HEIR stub for a low-depth cooperative traffic decision model",
        "# This is a prototype export and should be adapted to the exact HEIR frontend API.",
        "",
        "from heir import compile",
        "from heir.mlir import Secret, F64",
        "",
        "W1 = [",
    ]
    for row in result.weights1:
        lines.append(f"    [{fmt_vector(row)}],")
    lines.extend([
        "]",
        f"B1 = [{fmt_vector(result.bias1)}]",
        "W2 = [" + fmt_vector(result.weights2[0]) + "]",
        f"B2 = {result.bias2[0]:.6f}",
        "",
        "def poly_act(x):",
        "    return 0.125 * x * x + 0.5 * x + 0.25",
        "",
        "@compile(scheme='ckks')",
        "def traffic_policy(*features):",
        "    h = []",
        "    for row, b in zip(W1, B1):",
        "        acc = b",
        "        for w, x in zip(row, features):",
        "            acc = acc + w * x",
        "        h.append(poly_act(acc))",
        "    out = B2",
        "    for w, x in zip(W2, h):",
        "        out = out + w * x",
        "    return out",
        "",
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
