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

## 📊 Seed Sweep (reproducibility across 7 seeds)

| Model | Mean ± Std | 95% CI |
|---|---|---|
| HEIR Cooperative (HE) | 89.0% ± 0.031 | [86.8% – 91.2%] |
| Local model | 84.4% ± 0.026 | [82.6% – 86.2%] |

**Consistent cooperative gain across 7 seeds:** +4.6 pp mean  
**Statistical significance (paired t-test):** t-stat=0.0031, p=0.0031, Cohen's d=1.515

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

## 🚦 SUMO Correlated Corridor (primary SUMO result)

| Model | Val Acc | 95% Wilson CI | Δ vs Local |
|---|---|---|---|
| **Cooperative HE-friendly** | **96.4%** | [94.2% – 97.8%] | **+6.9 pp** |
| Graph-lite | 96.0% | — | +6.4 pp |
| Simple fusion | 91.2% | — | +1.7 pp |
| Local model | 89.5% | [86.2% – 92.1%] | — |

**7 intersections × 300 timesteps, temporal split. ML coop vs rule-based coop: +16.0 pp**

## 🚦 SUMO Binary (small graph, pipeline sanity check)

- Val accuracy: **100.0%** (6 samples, 2 timesteps)
- Uses adjacency matrix: True
- Interaction-gain within cooperative setting: 1.0
- ⚠️  Small toy graph — results are pipeline checks, not scale benchmarks

## 🔐 HEIR Overhead (theoretical analysis)

| Metric | Value |
|---|---|
| Accuracy gap (HE vs plaintext) | 0.83 pp |
| Inference latency (coop model) | 3312 ms |
| Encrypted comm per timestep (7 nodes) | 5488 KB |
| Plaintext comm per timestep (7 nodes) | 1.340 KB |
| CKKS expansion factor | 4096× |

> Estimates based on SEAL 4.x benchmarks (128-bit security). Order-of-magnitude guide.

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
- *"Results are consistent across seeds: 89.0% ± 0.031 (cooperative) vs 84.4% ± 0.026 (local)."*
- *"The cooperative advantage is maintained under robustness evaluation (83.3% vs 80.8%)."*
