#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from traffic_heir.action_space import decision_label_4
from traffic_heir.config import PrototypeConfig
from traffic_heir.heir_consistency import check_export_shape, manual_forward_sample
from traffic_heir.heir_export import export_heir_stub
from traffic_heir.multiclass import train_one_vs_rest
from traffic_heir.sumo_data import build_samples_from_grouped, group_by_timestep, load_sumo_csv
from traffic_heir.synthetic import generate_dataset
from traffic_heir.train import run_experiment


def main() -> None:
    config = PrototypeConfig(num_samples=120, epochs=40)
    results = run_experiment(config)
    local = results["local_result"]
    coop = results["coop_result"]

    assert 0.0 <= results["fixed_time_val_accuracy"] <= 1.0
    assert 0.0 <= results["heuristic_val_accuracy"] <= 1.0
    assert 0.0 <= results["max_pressure_val_accuracy"] <= 1.0
    assert 0.0 <= local.val_accuracy <= 1.0
    assert 0.0 <= coop.val_accuracy <= 1.0
    assert 0.0 <= results["ablation_no_interaction"].val_accuracy <= 1.0
    assert 0.0 <= results["ablation_no_neighbor"].val_accuracy <= 1.0
    for value in results["robustness"].values():
        assert 0.0 <= value <= 1.0

    out = ROOT / "generated" / "smoke_heir_stub.py"
    export_heir_stub(coop, out)
    assert out.exists()
    assert check_export_shape(coop, out)
    score = manual_forward_sample(coop, generate_dataset(config)[0])
    assert isinstance(score, float)

    sample_csv = ROOT / "data" / "sumo" / "raw" / "sample_states.csv"
    rows = load_sumo_csv(sample_csv)
    grouped = group_by_timestep(rows)
    samples = build_samples_from_grouped(grouped)
    assert len(rows) == 6
    assert len(grouped) == 2
    assert len(samples) == 6
    assert "neighbor_mean" in samples[0]

    action_samples = generate_dataset(PrototypeConfig(num_samples=120, epochs=30))
    train_samples = action_samples[:96]
    val_samples = action_samples[96:]
    y_train = [decision_label_4(s) for s in train_samples]
    y_val = [decision_label_4(s) for s in val_samples]
    mc = train_one_vs_rest(
        train_samples,
        y_train,
        val_samples,
        y_val,
        mode="coop",
        classes=[0, 1, 2, 3],
        hidden_dim=8,
        epochs=30,
        lr=0.01,
        seed=7,
        he_friendly=True,
    )
    assert 0.0 <= mc.val_accuracy <= 1.0
    print("smoke test passed")


if __name__ == "__main__":
    main()
