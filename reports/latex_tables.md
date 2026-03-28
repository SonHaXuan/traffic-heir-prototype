# LaTeX Tables Preview — HEIR Traffic Fusion Paper

> Auto-generated from `scripts/build_latex_tables.py`. Do not edit by hand.

## Table 1: Main Results

| **Method** | **Val Acc (%)** | **Δ Local (pp)** | **Notes** |
|---|---|---|---|
| **HEIR Cooperative (HE-friendly)** | **84.2** | **+3.3** | **Primary claim** |
| HEIR Cooperative (plaintext) | 85.0 | +4.2 | Upper bound |
| Local model (no cooperation) | 80.8 | — | Non-coop baseline |
| Max-pressure heuristic | 77.5 | -3.3 | Traditional baseline |
| Fixed-time control | 59.2 | -21.7 | Traditional baseline |

## Table 2: Seed Sweep

| **Seed** | **HEIR Coop HE (%)** | **Local (%)** |
|---|---|---|
| 7 | 84.2 | 80.8 |
| 13 | 92.5 | 86.7 |
| 23 | 89.2 | 83.3 |
| 31 | 85.0 | 85.0 |
| 41 | 89.2 | 80.8 |
| 53 | 91.7 | 86.7 |
| 67 | 91.7 | 87.5 |
| **Mean ± Std** | **89.0 ± 0.031** | **84.4 ± 0.026** |

## Table 3: Ablation Study

| **Variant** | **Val Acc (%)** | **Δ Full HEIR (pp)** |
|---|---|---|
| **Full HEIR (HE-friendly)** | **84.2** | **—** |
| w/o directional features | 80.8 | −3.3 |
| w/o neighbor features | 80.8 | −3.3 |
| w/o interaction features | 85.0 | +0.8 |
| Local model (no cooperation) | 80.8 | −3.3 |
| Robustness eval (HEIR coop) | 83.3 | — (+2.5 vs local) |

## Table 4: Action-4 Multiclass (secondary)

| **Class** | **Precision** | **Recall** | **F1** | **Support** |
|---|---|---|---|---|
| Action 0 | 0.767 | 0.793 | 0.780 | 87 |
| Action 1 | 0.516 | 0.615 | 0.561 | 26 |
| Action 2 | 0.688 | 0.579 | 0.629 | 19 |
| Action 3 | 0.286 | 0.167 | 0.211 | 12 |
| **Macro avg** | **—** | **—** | **0.545** | **144** |
