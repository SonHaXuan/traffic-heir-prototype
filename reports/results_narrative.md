# Results Narrative

## Key findings

- The cooperative HE-friendly binary model reached **0.8417** validation accuracy, compared with **0.8083** for the local plaintext model, for a gain of **0.0333**.
- Relative to the cooperative plaintext model (**0.8500**), the HE-friendly version changed accuracy by **-0.0083**, suggesting limited loss from the low-depth approximation in this prototype setting.
- Across the seed sweep, cooperative HE-friendly performance averaged **0.8861** versus **0.8361** for local modeling, preserving an average gain of **0.0500**.
- In the 4-action setting, the multiclass model achieved **0.6597** accuracy and **0.5872** macro-F1, while the one-vs-rest variant reached **0.5450** macro-F1.
- The current SUMO-derived binary sample is tiny (**6** samples over **2** timesteps), so its **1.0000** validation accuracy should be treated as a pipeline sanity check rather than a robust performance claim.
- The exported HEIR stub currently passes both structural and metadata consistency checks (**shape=True**, **consistency=True**), supporting the export pathway without claiming end-to-end encrypted runtime execution.

## Paper-friendly interpretation

- The clearest empirical story so far is that cooperative fusion helps the binary decision-support task and that the HE-friendly approximation retains most of that benefit.
- The current infrastructure is strongest on reproducibility: one-shot artifact generation, summary reporting, HEIR export verification, and report validation are now in place.
- The multiclass/action4 path is promising but not yet as strong as the binary story; macro-F1 suggests class imbalance and decision difficulty still need attention.

## Caveats

- The SUMO binary experiment is still too small to serve as a main quantitative claim.
- HEIR support is currently validated at the export/consistency level, not full encrypted execution benchmarking.
- The results are better framed as a paper-ready scaffold with emerging evidence, not a submission-ready benchmark package yet.

## Recommended next steps

1. Expand SUMO-derived experiments to non-trivial sample sizes and richer scenarios.
2. Improve multiclass/action4 analysis, especially per-class weakness and imbalance handling.
3. Add figure-friendly summaries or plots for the strongest binary and seed-sweep findings.
4. If feasible, deepen the HEIR pathway beyond export validation into a more realistic compile/evaluate handoff.