from __future__ import annotations

import math
import random
from typing import List

from .config import PrototypeConfig
from .types import TrafficSample


def _intersection_state(rng: random.Random, idx: int, corridor_bias: float, cross_bias: float) -> dict[str, float]:
    base = 0.6 + 0.15 * math.sin(idx)
    q_n = rng.uniform(4, 20) * (base + corridor_bias)
    q_s = rng.uniform(4, 20) * (base + corridor_bias)
    q_e = rng.uniform(4, 20) * (1.05 - base / 2 + cross_bias)
    q_w = rng.uniform(4, 20) * (1.05 - base / 2 + cross_bias)
    w_n = q_n * rng.uniform(0.6, 1.3)
    w_s = q_s * rng.uniform(0.6, 1.3)
    w_e = q_e * rng.uniform(0.6, 1.3)
    w_w = q_w * rng.uniform(0.6, 1.3)
    elapsed = rng.uniform(0, 60)
    phase = rng.choice([0.0, 1.0])  # 0=NS green, 1=EW green
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


def generate_dataset(config: PrototypeConfig) -> List[TrafficSample]:
    rng = random.Random(config.seed)
    dataset: List[TrafficSample] = []

    for _ in range(config.num_samples):
        corridor_bias = rng.uniform(-0.18, 0.22)
        cross_bias = rng.uniform(-0.18, 0.22)
        states = []
        for i in range(config.num_intersections):
            local_corridor = corridor_bias + (0.08 if i == 0 else -0.08 if i == config.num_intersections - 1 else 0.0)
            local_cross = cross_bias + (-0.08 if i == 0 else 0.08 if i == config.num_intersections - 1 else 0.0)
            states.append(_intersection_state(rng, i, local_corridor, local_cross))
        target_idx = rng.randrange(config.num_intersections)
        target = states[target_idx]
        neighbors = [s for i, s in enumerate(states) if i != target_idx]

        local = _as_vector(target)
        neighbor_mean = [sum(values) / len(neighbors) for values in zip(*[_as_vector(n) for n in neighbors])]
        interaction = [a * b / 20.0 for a, b in zip(local, neighbor_mean)]

        dataset.append(
            {
                "local": local,
                "neighbor_mean": neighbor_mean,
                "interaction": interaction,
                "phase": target["phase"],
                "elapsed": target["elapsed"],
            }
        )

    return dataset
