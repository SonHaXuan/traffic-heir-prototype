from __future__ import annotations

import math
import random
from typing import List

from .config import PrototypeConfig
from .types import TrafficSample


def _intersection_state(
    rng: random.Random,
    idx: int,
    timestep: int,
    corridor_wave: float,
    cross_wave: float,
    spillback: float,
    prev_state: dict[str, float] | None,
) -> dict[str, float]:
    offset = idx - 1
    local_ns_bias = corridor_wave + 0.18 * math.sin(0.35 * timestep - 0.7 * offset)
    local_ew_bias = cross_wave + 0.18 * math.cos(0.31 * timestep + 0.5 * offset)
    base = 8.0 + 2.4 * math.sin(0.15 * timestep + idx)

    q_n = max(0.0, base + rng.uniform(-1.8, 5.8) + 7.5 * max(local_ns_bias, -0.25))
    q_s = max(0.0, base + rng.uniform(-1.8, 5.8) + 7.0 * max(local_ns_bias + 0.05, -0.25))
    q_e = max(0.0, base + rng.uniform(-1.8, 5.8) + 7.3 * max(local_ew_bias, -0.25) + 3.0 * spillback)
    q_w = max(0.0, base + rng.uniform(-1.8, 5.8) + 6.8 * max(local_ew_bias - 0.05, -0.25) + 2.4 * spillback)

    if prev_state is not None:
        q_n = 0.65 * prev_state["q_n"] + 0.35 * q_n
        q_s = 0.65 * prev_state["q_s"] + 0.35 * q_s
        q_e = 0.60 * prev_state["q_e"] + 0.40 * q_e
        q_w = 0.60 * prev_state["q_w"] + 0.40 * q_w

    w_n = q_n * (0.65 + rng.uniform(0.0, 0.75))
    w_s = q_s * (0.65 + rng.uniform(0.0, 0.75))
    w_e = q_e * (0.65 + rng.uniform(0.0, 0.75))
    w_w = q_w * (0.65 + rng.uniform(0.0, 0.75))

    phase_score = (q_n + q_s + 0.25 * (w_n + w_s)) - (q_e + q_w + 0.25 * (w_e + w_w))
    phase = 0.0 if phase_score >= 0.0 else 1.0
    if prev_state is not None and rng.random() < 0.42:
        phase = prev_state["phase"]
    elapsed = rng.uniform(0, 15) if prev_state is None or phase != prev_state["phase"] else min(60.0, prev_state["elapsed"] + rng.uniform(4, 10))
    return {
        "q_n": q_n,
        "q_s": q_s,
        "q_e": q_e,
        "q_w": q_w,
        "w_n": w_n,
        "w_s": w_s,
        "w_e": w_e,
        "w_w": w_w,
        "elapsed": elapsed,
        "phase": phase,
    }


def _as_vector(state: dict[str, float]) -> List[float]:
    return [
        state["q_n"],
        state["q_s"],
        state["q_e"],
        state["q_w"],
        state["w_n"],
        state["w_s"],
        state["w_e"],
        state["w_w"],
    ]


def _directional_summary(local_idx: int, states: List[dict[str, float]]) -> List[float]:
    upstream = states[max(0, local_idx - 1)]
    downstream = states[min(len(states) - 1, local_idx + 1)]
    return [
        upstream["q_n"] + upstream["q_s"],
        upstream["q_e"] + upstream["q_w"],
        downstream["q_n"] + downstream["q_s"],
        downstream["q_e"] + downstream["q_w"],
        upstream["w_n"] + upstream["w_s"],
        upstream["w_e"] + upstream["w_w"],
        downstream["w_n"] + downstream["w_s"],
        downstream["w_e"] + downstream["w_w"],
    ]


def generate_dataset(config: PrototypeConfig) -> List[TrafficSample]:
    rng = random.Random(config.seed)
    dataset: List[TrafficSample] = []
    prev_states: List[dict[str, float] | None] = [None for _ in range(config.num_intersections)]
    history_by_idx: List[List[List[float]]] = [[] for _ in range(config.num_intersections)]

    for timestep in range(config.num_samples):
        corridor_wave = rng.uniform(-0.18, 0.24) + 0.24 * math.sin(timestep / 9.0)
        cross_wave = rng.uniform(-0.18, 0.24) + 0.20 * math.cos(timestep / 11.0)
        spillback = max(0.0, 0.65 * math.sin(timestep / 7.0) + rng.uniform(-0.08, 0.22))
        states = []
        for i in range(config.num_intersections):
            state = _intersection_state(rng, i, timestep, corridor_wave, cross_wave, spillback, prev_states[i])
            states.append(state)
            prev_states[i] = state

        target_idx = rng.randrange(config.num_intersections)
        target = states[target_idx]
        neighbors = [s for i, s in enumerate(states) if i != target_idx]
        local = _as_vector(target)
        neighbor_mean = [sum(values) / len(neighbors) for values in zip(*[_as_vector(n) for n in neighbors])]
        neighbor_directional = _directional_summary(target_idx, states)
        interaction = [a * b / 20.0 for a, b in zip(local, neighbor_mean)]
        prev_local = history_by_idx[target_idx][-1] if history_by_idx[target_idx] else None
        local_delta = [0.0 for _ in local] if prev_local is None else [a - b for a, b in zip(local, prev_local)]
        rolling_local = [sum(values) / len(values) for values in zip(*((history_by_idx[target_idx][-3:] + [local]) if history_by_idx[target_idx] else [local]))]
        local_vs_roll = [a - b for a, b in zip(local, rolling_local)]

        dataset.append(
            {
                "local": local,
                "neighbor_mean": neighbor_mean,
                "neighbor_directional": neighbor_directional,
                "interaction": interaction,
                "temporal": local_delta + local_vs_roll,
                "phase": target["phase"],
                "elapsed": target["elapsed"],
            }
        )
        for idx, state in enumerate(states):
            history_by_idx[idx].append(_as_vector(state))

    return dataset
