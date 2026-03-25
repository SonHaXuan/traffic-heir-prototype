from __future__ import annotations

from typing import Dict

from .baselines import fixed_time_predict, local_heuristic_predict, max_pressure_predict
from .config import PrototypeConfig
from .evaluate import accuracy, build_splits, build_xy
from .labels import decision_label
from .models import TrainResult, predict_batch, train_two_layer_network
from .robustness import perturb_samples
from .synthetic import generate_dataset


def _train_mode(
    train_samples,
    val_samples,
    y_train,
    y_val,
    *,
    mode: str,
    hidden_dim: int,
    epochs: int,
    lr: float,
    seed: int,
    he_friendly: bool,
) -> TrainResult:
    x_train, _ = build_xy(train_samples, mode=mode)
    x_val, _ = build_xy(val_samples, mode=mode)
    return train_two_layer_network(
        x_train,
        y_train,
        x_val,
        y_val,
        hidden_dim=hidden_dim,
        epochs=epochs,
        lr=lr,
        seed=seed,
        he_friendly=he_friendly,
    )


def _evaluate_result(result: TrainResult, samples, *, mode: str, he_friendly: bool) -> float:
    x_eval, y_eval = build_xy(samples, mode=mode)
    preds = predict_batch(x_eval, result.weights1, result.bias1, result.weights2, result.bias2, he_friendly)
    return accuracy(y_eval, preds)


def run_experiment(config: PrototypeConfig) -> Dict[str, object]:
    dataset = generate_dataset(config)
    train_samples, val_samples = build_splits(dataset, config.train_ratio, seed=config.seed)

    _, y_train = build_xy(train_samples, mode="local")
    _, y_val = build_xy(val_samples, mode="local")

    fixed_time_acc = accuracy(y_val, fixed_time_predict(val_samples))
    heuristic_acc = accuracy(y_val, local_heuristic_predict(val_samples))
    max_pressure_acc = accuracy(y_val, max_pressure_predict(val_samples))

    local_result = _train_mode(
        train_samples,
        val_samples,
        y_train,
        y_val,
        mode="local",
        hidden_dim=config.local_hidden_dim,
        epochs=config.epochs,
        lr=config.learning_rate,
        seed=config.seed,
        he_friendly=False,
    )

    coop_plaintext_result = _train_mode(
        train_samples,
        val_samples,
        y_train,
        y_val,
        mode="coop",
        hidden_dim=config.coop_hidden_dim,
        epochs=config.epochs,
        lr=config.learning_rate,
        seed=config.seed + 11,
        he_friendly=False,
    )

    coop_result = _train_mode(
        train_samples,
        val_samples,
        y_train,
        y_val,
        mode="coop",
        hidden_dim=config.coop_hidden_dim,
        epochs=config.epochs,
        lr=config.learning_rate,
        seed=config.seed + 1,
        he_friendly=True,
    )

    ablation_no_interaction = _train_mode(
        train_samples,
        val_samples,
        y_train,
        y_val,
        mode="coop_no_interaction",
        hidden_dim=config.coop_hidden_dim,
        epochs=config.epochs,
        lr=config.learning_rate,
        seed=config.seed + 21,
        he_friendly=True,
    )

    ablation_no_neighbor = _train_mode(
        train_samples,
        val_samples,
        y_train,
        y_val,
        mode="coop_no_neighbor",
        hidden_dim=config.coop_hidden_dim,
        epochs=config.epochs,
        lr=config.learning_rate,
        seed=config.seed + 31,
        he_friendly=True,
    )

    robust_val_samples = perturb_samples(val_samples, config, seed=config.seed + 99)
    robust_labels = [decision_label(sample, config=config) for sample in robust_val_samples]
    fixed_time_robust = accuracy(robust_labels, fixed_time_predict(robust_val_samples))
    heuristic_robust = accuracy(robust_labels, local_heuristic_predict(robust_val_samples))
    max_pressure_robust = accuracy(robust_labels, max_pressure_predict(robust_val_samples))
    local_robust = _evaluate_result(local_result, robust_val_samples, mode="local", he_friendly=False)
    coop_plain_robust = _evaluate_result(coop_plaintext_result, robust_val_samples, mode="coop", he_friendly=False)
    coop_he_robust = _evaluate_result(coop_result, robust_val_samples, mode="coop", he_friendly=True)

    return {
        "dataset_size": len(dataset),
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "fixed_time_val_accuracy": fixed_time_acc,
        "heuristic_val_accuracy": heuristic_acc,
        "max_pressure_val_accuracy": max_pressure_acc,
        "local_result": local_result,
        "coop_plaintext_result": coop_plaintext_result,
        "coop_result": coop_result,
        "ablation_no_interaction": ablation_no_interaction,
        "ablation_no_neighbor": ablation_no_neighbor,
        "robustness": {
            "fixed_time": fixed_time_robust,
            "local_heuristic": heuristic_robust,
            "max_pressure": max_pressure_robust,
            "local_model": local_robust,
            "coop_plaintext": coop_plain_robust,
            "coop_he_friendly": coop_he_robust,
        },
    }
