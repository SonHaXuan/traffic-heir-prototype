from __future__ import annotations

import random
from typing import List, Sequence

from .config import PrototypeConfig
from .types import TrafficSample


def perturb_samples(samples: Sequence[TrafficSample], config: PrototypeConfig, *, seed: int) -> List[TrafficSample]:
    rng = random.Random(seed)
    perturbed: List[TrafficSample] = []
    for sample in samples:
        copied: TrafficSample = dict(sample)
        local = list(sample["local"])
        neighbor_mean = list(sample["neighbor_mean"])

        noisy_local = [max(0.0, v * (1.0 + rng.gauss(0.0, config.robustness_noise_std))) for v in local]
        noisy_neighbor = [max(0.0, v * (1.0 + rng.gauss(0.0, config.robustness_noise_std))) for v in neighbor_mean]

        if rng.random() < config.robustness_missing_prob:
            noisy_neighbor = [0.0 for _ in noisy_neighbor]

        if rng.random() < config.robustness_partial_drop_prob:
            start = rng.choice([0, 2, 4, 6])
            for idx in range(start, min(start + 2, len(noisy_neighbor))):
                noisy_neighbor[idx] = 0.0

        if rng.random() < config.robustness_directional_corrupt_prob:
            for idx in (0, 1, 4, 5):
                noisy_neighbor[idx] *= 1.25

        copied["local"] = noisy_local
        copied["neighbor_mean"] = noisy_neighbor
        copied["interaction"] = [a * b / 20.0 for a, b in zip(noisy_local, noisy_neighbor)]
        perturbed.append(copied)
    return perturbed
