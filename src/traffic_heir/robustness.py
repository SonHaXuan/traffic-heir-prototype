from __future__ import annotations

import random
from typing import Dict, List, Sequence

from .config import PrototypeConfig


def perturb_samples(samples: Sequence[Dict[str, object]], config: PrototypeConfig, *, seed: int) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    perturbed: List[Dict[str, object]] = []
    for sample in samples:
        copied = dict(sample)
        local = list(sample["local"])  # type: ignore[arg-type]
        neighbor_mean = list(sample["neighbor_mean"])  # type: ignore[arg-type]

        noisy_local = [max(0.0, v * (1.0 + rng.gauss(0.0, config.robustness_noise_std))) for v in local]
        noisy_neighbor = [max(0.0, v * (1.0 + rng.gauss(0.0, config.robustness_noise_std))) for v in neighbor_mean]

        if rng.random() < config.robustness_missing_prob:
            noisy_neighbor = [0.0 for _ in noisy_neighbor]

        copied["local"] = noisy_local
        copied["neighbor_mean"] = noisy_neighbor
        copied["interaction"] = [a * b / 20.0 for a, b in zip(noisy_local, noisy_neighbor)]
        perturbed.append(copied)
    return perturbed
