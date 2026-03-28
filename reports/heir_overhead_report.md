# HEIR Overhead Analysis

> Auto-generated from `scripts/build_heir_overhead_report.py`.

## Accuracy Gap: HE-friendly vs Plaintext

| Model | Val Accuracy |
|---|---|
| Cooperative HE-friendly | 0.8417 |
| Cooperative plaintext   | 0.8500 |
| **Gap** | **0.83 pp** |

> HE-friendly polynomial activation costs 0.8 pp (1.0% relative) vs plaintext ReLU baseline.

## Theoretical Inference Latency (CKKS, 128-bit security)

| Model | Feature Dim | Hidden Dim | Latency (ms) |
|---|---|---|---|
| local_model | 10 | 12 | 1536.9 |
| simple_fusion | 18 | 12 | 1994.5 |
| graph_lite | 26 | 12 | 2452.1 |
| coop_he_friendly | 49 | 10 | 3311.9 |
| coop_temporal | 73 | 10 | 4535.9 |

> Estimates based on SEAL 4.x benchmarks (conservative, order-of-magnitude).

## Communication Overhead per Control Timestep

| Model | Plaintext (KB) | Encrypted (KB) | Expansion |
|---|---|---|---|
| local_model | 0.273 | 1120.0 | 4096× |
| simple_fusion | 0.492 | 2016.0 | 4096× |
| graph_lite | 0.711 | 2912.0 | 4096× |
| coop_he_friendly | 1.340 | 5488.0 | 4096× |
| coop_temporal | 1.996 | 8176.0 | 4096× |

> 7-intersection corridor. CKKS expansion factor: 4096×.

## Operation Counts (cooperative HE-friendly model)

- CT×CT multiplications: 11
- CT×PT multiplications: 511
- Additions:             489
- Multiplicative depth:  3

> Polynomial activation `0.125x² + 0.5x + 0.25` uses depth 2, fitting within standard CKKS bootstrap budget.
