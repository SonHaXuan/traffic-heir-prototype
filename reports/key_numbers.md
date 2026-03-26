# Key Numbers — HEIR Traffic Fusion Paper

> Auto-generated from `scripts/build_key_numbers.py`. Do not edit by hand.

## 🔑 Primary Result (binary cooperative inference, prototype)

| Model | Val Accuracy | Δ vs Local |
|---|---|---|
| **HEIR Cooperative (HE-friendly)** | **84.2%** | **+3.3 pp** |
| HEIR Cooperative (plaintext) | 85.0% | +4.2 pp |
| Local model (no cooperation) | 80.8% | — |
| Max-pressure baseline | 77.5% | — |
| Fixed-time baseline | 59.2% | — |

**HE overhead vs plaintext:** 0.8 pp  
**Gain over best traditional baseline (77.5%):** +6.7 pp

## 📊 Seed Sweep (reproducibility across 3 seeds)

| Model | Mean ± Std |
|---|---|
| HEIR Cooperative (HE) | 88.6% ± 0.034 |
| Local model | 83.6% ± 0.024 |

**Consistent cooperative gain across seeds:** +5.0 pp mean

## 🔬 Robustness (held-out adversarial eval)

| Model | Robust Val Acc |
|---|---|
| HEIR Cooperative (HE) | 83.3% |
| Local model | 80.8% |

**Cooperative advantage holds under robustness eval:** +2.5 pp

## 🧩 Ablation (component contributions)

| Ablation | Val Acc | Δ vs Full HEIR |
|---|---|---|
| Full HEIR (HE-friendly) | 84.2% | — |
| Remove directional features | 80.8% | -3.3 pp |
| Remove interaction features | 85.0% | +0.8 pp |
| Remove neighbor features | 80.8% | -3.3 pp |

## 🚦 SUMO Binary (graph-structured adjacency)

- Val accuracy: **100.0%** (6 samples, 2 timesteps)
- Uses adjacency matrix: True
- Interaction-gain within cooperative setting: 1.0
- ⚠️  Small toy graph — results are consistency checks, not scale benchmarks

## 🎯 Action-4 Multiclass (secondary, exploratory)

- Val accuracy: 66.0% | Macro-F1: 0.587 | OvR Macro-F1: 0.545
- Note: binary story is primary claim; action-4 is exploratory

## 🔐 HEIR Export Consistency

- Consistency check: ✅ passed
- Shape check: ✅ passed
- Metadata val accuracy: 58.3%

---

## Paper claim language (suggested)

- *"HEIR cooperative inference achieves 84.2% validation accuracy, a 3.3 pp improvement over the non-cooperative local model (80.8%)."*
- *"The HE-friendly protocol incurs only 0.8 pp overhead relative to plaintext cooperation."*
- *"Results are consistent across seeds: 88.6% ± 0.034 (cooperative) vs 83.6% ± 0.024 (local)."*
- *"The cooperative advantage is maintained under robustness evaluation (83.3% vs 80.8%)."*
