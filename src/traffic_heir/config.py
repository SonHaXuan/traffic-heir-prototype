from dataclasses import dataclass


@dataclass
class PrototypeConfig:
    num_samples: int = 600
    num_intersections: int = 3
    local_dim: int = 8
    train_ratio: float = 0.8
    seed: int = 7
    local_hidden_dim: int = 12
    coop_hidden_dim: int = 10
    learning_rate: float = 0.01
    epochs: int = 120
    ambiguity_threshold: float = 6.0
    neighbor_margin_weight: float = 0.8
    elapsed_weight: float = 0.04
    neighbor_ns_bonus: float = 0.10
    neighbor_ew_bonus: float = 0.10
    directional_weight: float = 0.35
    spillback_penalty_weight: float = 0.45
    robustness_noise_std: float = 0.08
    robustness_missing_prob: float = 0.15
    robustness_partial_drop_prob: float = 0.10
    robustness_directional_corrupt_prob: float = 0.10
