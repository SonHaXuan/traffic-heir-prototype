from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, Iterable, List

from .stats import bootstrap_ci


def write_metrics_report(metrics: Dict[str, object], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return path


def summarize_runs(values: Iterable[float]) -> Dict[str, float]:
    vals = list(values)
    if not vals:
        return {"mean": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": statistics.fmean(vals),
        "stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def summarize_with_ci(
    values: Iterable[float],
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Like summarize_runs but also includes 95% bootstrap CI.
    Treats each value in `values` as a single accuracy observation (e.g.,
    per-seed val accuracy), so we bootstrap over the list itself.
    """
    vals = list(values)
    if not vals:
        return {"mean": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0,
                "ci_lower_95": 0.0, "ci_upper_95": 0.0}
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    # Bootstrap over the accuracy values themselves
    y_true = [1] * len(vals)      # dummy — we only need accuracy = mean
    y_pred_proxy = [1] * len(vals)
    # Use bootstrap_ci on a proxy: resample accuracy values directly
    import random as _rng
    rng = _rng.Random(seed)
    boot_means: List[float] = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(len(vals))] for _ in range(len(vals))]
        boot_means.append(sum(sample) / len(sample))
    boot_means.sort()
    lo = boot_means[max(0, int(0.025 * n_boot))]
    hi = boot_means[min(n_boot - 1, int(0.975 * n_boot))]
    return {
        "mean": mean,
        "stdev": std,
        "min": min(vals),
        "max": max(vals),
        "ci_lower_95": lo,
        "ci_upper_95": hi,
    }


def write_markdown_table(rows: List[Dict[str, object]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
