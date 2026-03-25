from __future__ import annotations

from typing import List

from .config import PrototypeConfig
from .types import Action2, TrafficSample


# Binary action space for prototype:
# 0 -> favor north-south phase
# 1 -> favor east-west phase


def local_heuristic_label(sample: TrafficSample) -> Action2:
    local: List[float] = sample["local"]
    elapsed: float = float(sample["elapsed"])
    q_n, q_s, q_e, q_w, w_n, w_s, w_e, w_w = local

    ns_pressure = q_n + q_s + 0.4 * (w_n + w_s)
    ew_pressure = q_e + q_w + 0.4 * (w_e + w_w)

    elapsed_bias = 0.02 * elapsed
    ns_score = ns_pressure + elapsed_bias
    ew_score = ew_pressure - elapsed_bias
    return 0 if ns_score >= ew_score else 1



def decision_label(sample: TrafficSample, config: PrototypeConfig | None = None) -> Action2:
    local: List[float] = sample["local"]
    neighbor_mean: List[float] = sample["neighbor_mean"]
    neighbor_directional: List[float] = sample.get("neighbor_directional", [0.0] * 8)
    elapsed: float = float(sample["elapsed"])

    q_n, q_s, q_e, q_w, w_n, w_s, w_e, w_w = local
    nq_n, nq_s, nq_e, nq_w, nw_n, nw_s, nw_e, nw_w = neighbor_mean
    up_ns, up_ew, down_ns, down_ew, up_wait_ns, up_wait_ew, down_wait_ns, down_wait_ew = neighbor_directional

    ns_pressure = q_n + q_s + 0.4 * (w_n + w_s)
    ew_pressure = q_e + q_w + 0.4 * (w_e + w_w)

    neighbor_ns = nq_n + nq_s + 0.3 * (nw_n + nw_s)
    neighbor_ew = nq_e + nq_w + 0.3 * (nw_e + nw_w)

    cfg = config or PrototypeConfig()
    local_margin = ns_pressure - ew_pressure
    directional_margin = (up_ns + 0.5 * down_ns + 0.2 * up_wait_ns) - (up_ew + 0.5 * down_ew + 0.2 * up_wait_ew)
    spillback_penalty = cfg.spillback_penalty_weight * ((down_ns + down_wait_ns) - (down_ew + down_wait_ew))

    if abs(local_margin) < cfg.ambiguity_threshold:
        decision_value = (
            0.85 * local_margin
            - cfg.neighbor_margin_weight * (neighbor_ns - neighbor_ew)
            + cfg.directional_weight * directional_margin
            - 0.22 * spillback_penalty
            + cfg.elapsed_weight * elapsed
        )
        return 0 if decision_value >= 0 else 1

    elapsed_bias = 0.02 * elapsed
    ns_score = ns_pressure + cfg.neighbor_ew_bonus * neighbor_ew + 1.25 * cfg.directional_weight * (up_ns - down_ns) - 0.18 * spillback_penalty + elapsed_bias
    ew_score = ew_pressure + cfg.neighbor_ns_bonus * neighbor_ns + 1.25 * cfg.directional_weight * (up_ew - down_ew) + 0.18 * spillback_penalty - elapsed_bias
    return 0 if ns_score >= ew_score else 1
