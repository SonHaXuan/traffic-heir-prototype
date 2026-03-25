from __future__ import annotations

from .config import PrototypeConfig
from .labels import decision_label, local_heuristic_label
from .types import Action4, TrafficSample


# 4-action scaffold:
# 0 -> keep current phase
# 1 -> switch phase
# 2 -> extend current phase
# 3 -> reduce current phase / prepare transition

def decision_label_4(sample: TrafficSample, config: PrototypeConfig | None = None) -> Action4:
    cfg = config or PrototypeConfig()
    preferred_binary = decision_label(sample, config=cfg)
    current_phase = int(float(sample["phase"]))
    elapsed = float(sample["elapsed"])

    if preferred_binary == current_phase:
        return 2 if elapsed < 20.0 else 0
    return 1 if elapsed >= 10.0 else 3



def local_heuristic_label_4(sample: TrafficSample, config: PrototypeConfig | None = None) -> Action4:
    preferred_binary = local_heuristic_label(sample)
    current_phase = int(float(sample["phase"]))
    elapsed = float(sample["elapsed"])
    if preferred_binary == current_phase:
        return 2 if elapsed < 20.0 else 0
    return 1 if elapsed >= 10.0 else 3
