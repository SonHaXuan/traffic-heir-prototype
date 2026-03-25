from __future__ import annotations

from typing import Dict, List

from .config import PrototypeConfig


# Binary action space for prototype:
# 0 -> favor north-south phase
# 1 -> favor east-west phase


def local_heuristic_label(sample: Dict[str, object]) -> int:
    local: List[float] = sample["local"]  # type: ignore[assignment]
    elapsed: float = float(sample["elapsed"])
    q_n, q_s, q_e, q_w, w_n, w_s, w_e, w_w = local

    ns_pressure = q_n + q_s + 0.4 * (w_n + w_s)
    ew_pressure = q_e + q_w + 0.4 * (w_e + w_w)

    elapsed_bias = 0.02 * elapsed
    ns_score = ns_pressure + elapsed_bias
    ew_score = ew_pressure - elapsed_bias
    return 0 if ns_score >= ew_score else 1



def decision_label(sample: Dict[str, object], config: PrototypeConfig | None = None) -> int:
    local: List[float] = sample["local"]  # type: ignore[assignment]
    neighbor_mean: List[float] = sample["neighbor_mean"]  # type: ignore[assignment]
    elapsed: float = float(sample["elapsed"])

    q_n, q_s, q_e, q_w, w_n, w_s, w_e, w_w = local
    nq_n, nq_s, nq_e, nq_w, nw_n, nw_s, nw_e, nw_w = neighbor_mean

    ns_pressure = q_n + q_s + 0.4 * (w_n + w_s)
    ew_pressure = q_e + q_w + 0.4 * (w_e + w_w)

    neighbor_ns = nq_n + nq_s + 0.3 * (nw_n + nw_s)
    neighbor_ew = nq_e + nq_w + 0.3 * (nw_e + nw_w)

    cfg = config or PrototypeConfig()
    local_margin = ns_pressure - ew_pressure
    neighbor_margin = neighbor_ns - neighbor_ew

    # If the local state is ambiguous, neighboring intersections can flip the
    # recommendation to reduce spillback along the corridor.
    if abs(local_margin) < cfg.ambiguity_threshold:
        decision_value = local_margin - 0.8 * neighbor_margin + 0.04 * elapsed
        return 0 if decision_value >= 0 else 1

    elapsed_bias = 0.02 * elapsed
    ns_score = ns_pressure + 0.10 * neighbor_ew + elapsed_bias
    ew_score = ew_pressure + 0.10 * neighbor_ns - elapsed_bias
    return 0 if ns_score >= ew_score else 1
