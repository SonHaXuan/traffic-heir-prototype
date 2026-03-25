from __future__ import annotations

import math
import random
from typing import Dict, List

from .config import PrototypeConfig


def _intersection_state(rng: random.Random, idx: int) -> Dict[str, float]:
    base = 0.5 + 0.2 * math.sin(idx)
    q_n = rng.uniform(0, 20) * base
    q_s = rng.uniform(0, 20) * base
    q_e = rng.uniform(0, 20) * (1.1 - base / 2)
    q_w = rng.uniform(0, 20) * (1.1 - base / 2)
    w_n = q_n * rng.uniform(0.4, 1.2)
    w_s = q_s * rng.uniform(0.4, 1.2)
    w_e = q_e * rng.uniform(0.4, 1.2)
    w_w = q_w * rng.uniform(0.4, 1.2)
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


def _as_vector(state: Dict[str, float]) -> List[float]:
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


def generate_dataset(config: PrototypeConfig) -> List[Dict[str, object]]:
    rng = random.Random(config.seed)
    dataset: List[Dict[str, object]] = []

    for _ in range(config.num_samples):
        states = [_intersection_state(rng, i) for i in range(config.num_intersections)]
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
