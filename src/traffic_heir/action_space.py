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
    return (q_n + q_s + q_e + q_w + 0.25 * (w_n + w_s + w_e + w_w)) / 8.0


def _phase_margin(sample: TrafficSample) -> float:
    q_n, q_s, q_e, q_w, w_n, w_s, w_e, w_w = sample["local"]
    current_phase = int(float(sample["phase"]))
    ns_pressure = q_n + q_s + 0.35 * (w_n + w_s)
    ew_pressure = q_e + q_w + 0.35 * (w_e + w_w)
    margin = ns_pressure - ew_pressure
    return margin if current_phase == 0 else -margin


def _directional_pressure(sample: TrafficSample) -> float:
    directional = sample.get("neighbor_directional", [0.0] * 8)
    up_ns, up_ew, down_ns, down_ew, up_wait_ns, up_wait_ew, down_wait_ns, down_wait_ew = directional
    return (up_ns + 0.55 * down_ns + 0.18 * up_wait_ns + 0.08 * down_wait_ns) - (
        up_ew + 0.55 * down_ew + 0.18 * up_wait_ew + 0.08 * down_wait_ew
    )


def decision_label_4(sample: TrafficSample, config: PrototypeConfig | None = None) -> Action4:
    cfg = config or PrototypeConfig()
    preferred_binary = decision_label(sample, config=cfg)
    current_phase = int(float(sample["phase"]))
    elapsed = float(sample["elapsed"])
    urgency = _urgency(sample)
    margin_current = _phase_margin(sample)
    directional = _directional_pressure(sample)
    directional_for_current = directional if current_phase == 0 else -directional

    if preferred_binary == current_phase:
        strong_current = margin_current >= 1.5 or directional_for_current >= 2.0
        weak_current = margin_current <= 1.25 and directional_for_current <= 1.5
        if strong_current and urgency >= 7.0 and elapsed <= 34.0:
            return 2
        if weak_current and (elapsed >= 22.0 or urgency <= 7.0):
            return 3
        return 0

    if elapsed >= 16.0 or margin_current <= -3.5 or directional_for_current <= -4.5 or urgency >= 11.0:
        return 1
    return 3


def local_heuristic_label_4(sample: TrafficSample, config: PrototypeConfig | None = None) -> Action4:
    preferred_binary = local_heuristic_label(sample)
    current_phase = int(float(sample["phase"]))
    elapsed = float(sample["elapsed"])
    urgency = _urgency(sample)
    margin_current = _phase_margin(sample)
    if preferred_binary == current_phase:
        if margin_current >= 1.5 and urgency >= 7.0 and elapsed <= 32.0:
            return 2
        if margin_current <= 1.0 and (urgency <= 6.5 or elapsed >= 24.0):
            return 3
        return 0
    if elapsed >= 18.0 or margin_current <= -4.0 or urgency >= 11.5:
        return 1
    return 3
