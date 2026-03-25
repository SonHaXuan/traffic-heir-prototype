from __future__ import annotations

from typing import Dict, List, Sequence

from .labels import local_heuristic_label


def fixed_time_predict(samples: Sequence[Dict[str, object]], action: int = 0) -> List[int]:
    return [action for _ in samples]


def max_pressure_predict(samples: Sequence[Dict[str, object]]) -> List[int]:
    outputs: List[int] = []
    for sample in samples:
        local: List[float] = sample["local"]  # type: ignore[assignment]
        q_n, q_s, q_e, q_w, w_n, w_s, w_e, w_w = local
        ns_pressure = q_n + q_s + 0.4 * (w_n + w_s)
        ew_pressure = q_e + q_w + 0.4 * (w_e + w_w)
        outputs.append(0 if ns_pressure >= ew_pressure else 1)
    return outputs


def local_heuristic_predict(samples: Sequence[Dict[str, object]]) -> List[int]:
    return [local_heuristic_label(sample) for sample in samples]
