from __future__ import annotations

from .config import PrototypeConfig
from .labels import decision_label, local_heuristic_label
from .types import Action4, TrafficSample


# 4-action scaffold:
# 0 -> keep current phase
# 1 -> switch phase
# 2 -> extend current phase
# 3 -> reduce current phase / prepare transition

def _urgency(sample: TrafficSample) -> float:
    q_n, q_s, q_e, q_w, w_n, w_s, w_e, w_w = sample["local"]
    return (q_n + q_s + q_e + q_w + 0.2 * (w_n + w_s + w_e + w_w)) / 8.0


def decision_label_4(sample: TrafficSample, config: PrototypeConfig | None = None) -> Action4:
    cfg = config or PrototypeConfig()
    preferred_binary = decision_label(sample, config=cfg)
    current_phase = int(float(sample["phase"]))
    elapsed = float(sample["elapsed"])
    urgency = _urgency(sample)

    if preferred_binary == current_phase:
        if urgency > 11.0 and elapsed < 24.0:
            return 2
        return 0

    if elapsed >= 22.0 or urgency > 12.0:
        return 1
    return 3



def local_heuristic_label_4(sample: TrafficSample, config: PrototypeConfig | None = None) -> Action4:
    preferred_binary = local_heuristic_label(sample)
    current_phase = int(float(sample["phase"]))
    elapsed = float(sample["elapsed"])
    urgency = _urgency(sample)
    if preferred_binary == current_phase:
        if urgency > 11.5 and elapsed < 24.0:
            return 2
        return 0
    if elapsed >= 22.0 or urgency > 12.5:
        return 1
    return 3
