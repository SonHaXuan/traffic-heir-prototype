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
    robustness_noise_std: float = 0.08
    robustness_missing_prob: float = 0.15
