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
from traffic_heir.metrics import distribution, macro_f1_from_per_class, per_class_precision_recall_f1
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
        per_class = per_class_precision_recall_f1(y_val, preds, [0, 1, 2, 3])
        self.assertGreaterEqual(result.val_accuracy, 0.55)
        self.assertGreaterEqual(len(pred_dist), 4)
        self.assertGreater(pred_dist.get(2, 0), 0)
        self.assertGreater(macro_f1_from_per_class(per_class), 0.45)

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
        self.assertIn("eval_story", metrics)
        self.assertIn("cooperative_gain_over_local", metrics["eval_story"])
        # New ablation modes
        self.assertIn("simple_fusion_val_accuracy", metrics)
        self.assertIn("graph_lite_val_accuracy", metrics)
        self.assertIn("coop_no_neighbor_val_accuracy", metrics)
        # Cooperative heuristic baseline
        self.assertIn("coop_heuristic_val_accuracy", metrics)
        # Progressive fusion story
        self.assertIn("simple_fusion_gain_over_local", metrics["eval_story"])
        self.assertIn("ml_coop_gain_over_heuristic_coop", metrics["eval_story"])

    def test_stats_module_correctness(self) -> None:
        """Verify statistical utilities return correct types and reasonable values."""
        from traffic_heir.stats import (
            bootstrap_ci, effect_size_cohens_d, mcnemar_test, paired_ttest
        )
        # bootstrap_ci: perfect predictions → CI should be near 1.0
        lo, hi, mean = bootstrap_ci([0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
                                     [0, 1, 1, 0, 1, 1, 0, 1, 0, 1], n_boot=500)
        self.assertAlmostEqual(mean, 1.0)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)

        # paired_ttest: clearly different → p should be small
        t, p = paired_ttest([0.85, 0.88, 0.82, 0.87, 0.86, 0.84, 0.83],
                             [0.75, 0.78, 0.72, 0.77, 0.76, 0.74, 0.73])
        self.assertGreater(t, 0.0)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)
        self.assertLess(p, 0.01)   # clearly significant

        # effect_size: positive d when a > b
        d = effect_size_cohens_d([0.85, 0.88, 0.82], [0.75, 0.78, 0.72])
        self.assertGreater(d, 0.0)

        # mcnemar: identical classifiers → chi2=0, p=1
        chi2, pm = mcnemar_test([0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1])
        self.assertEqual(chi2, 0.0)
        self.assertAlmostEqual(pm, 1.0)

    def test_heir_overhead_report(self) -> None:
        """Verify heir_overhead module returns valid structured estimates."""
        from traffic_heir.heir_overhead import (
            accuracy_gap_summary, ckks_latency_estimate_ms, communication_cost_kb,
            count_operations,
        )
        ops = count_operations(feature_dim=49, hidden_dim=10)
        self.assertIn("ct_ct_muls", ops)
        self.assertIn("ct_pt_muls", ops)
        self.assertGreater(ops["ct_pt_muls"], 0)
        self.assertGreater(ops["total_mul_depth"], 0)

        lat = ckks_latency_estimate_ms(feature_dim=49, hidden_dim=10)
        self.assertIn("latency_ms", lat)
        self.assertGreater(lat["latency_ms"], 0.0)
        self.assertIn("mul_depth", lat)

        comm = communication_cost_kb(feature_dim=49, n_intersections=7)
        self.assertIn("ciphertext_total_kb", comm)
        self.assertIn("plaintext_total_kb", comm)
        # Ciphertext should be much larger than plaintext
        self.assertGreater(comm["ciphertext_total_kb"], comm["plaintext_total_kb"])

        gap = accuracy_gap_summary(he_accuracy=0.8417, plaintext_accuracy=0.8500)
        self.assertIn("gap_pp", gap)
        self.assertAlmostEqual(gap["gap_pp"], 0.83, places=1)
        self.assertIn("interpretation", gap)

    def test_correlated_sumo_shows_positive_cooperative_gain(self) -> None:
        """
        Verify that correlated SUMO data (with spillback) produces positive
        cooperative gain under temporal split. This is the primary SUMO claim.
        """
        csv_path = ROOT / "data" / "sumo" / "raw" / "correlated_states.csv"
        adj_path = ROOT / "configs" / "sumo" / "correlated_adjacency.json"
        if not csv_path.exists():
            self.skipTest("Correlated SUMO CSV not yet generated; run generate_sumo_correlated.py")
        metrics = run_sumo_binary_experiment(
            csv_path, adj_path,
            PrototypeConfig(epochs=80, coop_hidden_dim=16, local_hidden_dim=10),
            split_mode="temporal",
        )
        coop_gain = metrics["eval_story"]["cooperative_gain_over_local"]
        self.assertGreater(coop_gain, 0.0,
            f"Cooperative model should outperform local on correlated data, got gain={coop_gain:.4f}")
        # Verify progressive fusion story exists
        self.assertIn("simple_fusion_gain_over_local", metrics["eval_story"])
        self.assertIn("graph_lite_gain_over_local", metrics["eval_story"])


if __name__ == "__main__":
    unittest.main()
