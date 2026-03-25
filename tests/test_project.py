from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from traffic_heir.action_space import decision_label_4
from traffic_heir.config import PrototypeConfig
from traffic_heir.config_io import apply_overrides, load_config, save_config
from traffic_heir.fusion import cooperative_features
from traffic_heir.heir_consistency import exported_matches_result
from traffic_heir.heir_export import export_heir_stub
from traffic_heir.metrics import distribution
from traffic_heir.multiclass import predict_multiclass, train_multiclass
from traffic_heir.sumo_data import build_samples_from_grouped, group_by_timestep, load_sumo_csv
from traffic_heir.sumo_experiment import run_sumo_binary_experiment
from traffic_heir.synthetic import generate_dataset
from traffic_heir.train import run_experiment


ROOT = Path(__file__).resolve().parents[1]


class TrafficHeirTests(unittest.TestCase):
    def test_temporal_features_present_in_synthetic_and_sumo(self) -> None:
        config = PrototypeConfig(num_samples=24, seed=11)
        samples = generate_dataset(config)
        self.assertIn("temporal", samples[0])
        self.assertEqual(len(samples[0]["temporal"]), 16)

        csv_path = ROOT / "data" / "sumo" / "raw" / "sample_states.csv"
        rows = load_sumo_csv(csv_path)
        grouped = group_by_timestep(rows)
        sumo_samples = build_samples_from_grouped(grouped)
        self.assertIn("temporal", sumo_samples[0])
        self.assertEqual(len(sumo_samples[0]["temporal"]), 16)

    def test_multiclass_training_reduces_class_collapse(self) -> None:
        cfg = PrototypeConfig(num_samples=720, epochs=140, coop_hidden_dim=14, learning_rate=0.008, seed=7)
        samples = generate_dataset(cfg)
        split = int(len(samples) * cfg.train_ratio)
        train_samples = samples[:split]
        val_samples = samples[split:]
        y_train = [decision_label_4(s, cfg) for s in train_samples]
        y_val = [decision_label_4(s, cfg) for s in val_samples]
        result = train_multiclass(
            train_samples,
            y_train,
            val_samples,
            y_val,
            mode="coop",
            classes=[0, 1, 2, 3],
            hidden_dim=cfg.coop_hidden_dim,
            epochs=cfg.epochs,
            lr=cfg.learning_rate,
            seed=cfg.seed,
            he_friendly=True,
            class_weighting=True,
        )
        preds = predict_multiclass(result, [cooperative_features(s) for s in val_samples], he_friendly=True)
        pred_dist = distribution(preds)
        self.assertGreaterEqual(result.val_accuracy, 0.55)
        self.assertGreaterEqual(len(pred_dist), 4)
        self.assertGreater(pred_dist.get(2, 0), 0)

    def test_heir_export_matches_manual_forward(self) -> None:
        cfg = PrototypeConfig(num_samples=120, epochs=40)
        results = run_experiment(cfg)
        coop = results["coop_result"]
        sample = generate_dataset(cfg)[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "stub.py"
            export_heir_stub(coop, out)
            self.assertTrue(exported_matches_result(coop, out, sample))
            metadata = json.loads(out.read_text(encoding="utf-8").split("MODEL_METADATA = ", 1)[1].split("\n\n", 1)[0])
            self.assertIn("weights1", metadata)

    def test_config_round_trip_and_overrides(self) -> None:
        cfg = PrototypeConfig(seed=17, epochs=99)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            save_config(cfg, path)
            loaded = load_config(path)
            updated = apply_overrides(loaded, epochs=55)
        self.assertEqual(loaded.seed, 17)
        self.assertEqual(updated.epochs, 55)

    def test_sumo_experiment_reports_ablation_metrics(self) -> None:
        csv_path = ROOT / "data" / "sumo" / "raw" / "sample_states.csv"
        adjacency_path = ROOT / "configs" / "sumo" / "sample_adjacency.json"
        metrics = run_sumo_binary_experiment(csv_path, adjacency_path, PrototypeConfig(num_samples=6, epochs=40, seed=5))
        self.assertIn("coop_val_accuracy", metrics)
        self.assertIn("coop_no_direction_val_accuracy", metrics)
        self.assertIn("local_val_accuracy", metrics)


if __name__ == "__main__":
    unittest.main()
