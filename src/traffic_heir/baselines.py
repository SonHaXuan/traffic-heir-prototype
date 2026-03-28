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


def cooperative_max_pressure_predict(samples: Sequence[Dict[str, object]]) -> List[int]:
    """
    Cooperative max-pressure baseline: combines local and neighbor queue pressure
    using a simple weighted vote.

    This establishes a 'rule-based cooperative' baseline so we can show that
    ML-based cooperation outperforms naive rule-based cooperation.

    Decision rule:
      ns_score = local NS pressure + 0.3 × neighbor NS pressure
      ew_score = local EW pressure + 0.3 × neighbor EW pressure
      predict 0 (NS) if ns_score >= ew_score, else 1 (EW)
    """
    outputs: List[int] = []
    for sample in samples:
        local: List[float] = sample["local"]  # type: ignore[assignment]
        q_n, q_s, q_e, q_w, w_n, w_s, w_e, w_w = local
        local_ns = q_n + q_s + 0.4 * (w_n + w_s)
        local_ew = q_e + q_w + 0.4 * (w_e + w_w)

        nbr: List[float] = sample.get("neighbor_mean", [0.0] * 8)  # type: ignore
        nbr_ns = nbr[0] + nbr[1] + 0.3 * (nbr[4] + nbr[5])
        nbr_ew = nbr[2] + nbr[3] + 0.3 * (nbr[6] + nbr[7])

        ns_score = local_ns + 0.3 * nbr_ns
        ew_score = local_ew + 0.3 * nbr_ew
        outputs.append(0 if ns_score >= ew_score else 1)
    return outputs
