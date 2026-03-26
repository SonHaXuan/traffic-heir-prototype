# Results Narrative

## Key findings

- The cooperative HE-friendly binary model reached **0.8417** validation accuracy, compared with **0.8083** for the local plaintext model, for a gain of **0.0333**.
- Relative to the cooperative plaintext model (**0.8500**), the HE-friendly version changed accuracy by **-0.0083**, suggesting limited loss from the low-depth approximation in this prototype setting.
- Across the seed sweep, cooperative HE-friendly performance averaged **0.8861** versus **0.8361** for local modeling, preserving an average gain of **0.0500**.
- In the 4-action setting, the multiclass model achieved **0.6597** accuracy and **0.5872** macro-F1, while the one-vs-rest variant reached **0.5450** macro-F1.
- The small SUMO sample (**500** samples, random split) is a pipeline sanity check only.
- The expanded SUMO experiment (**500** samples, **5** nodes, **100** timesteps) uses a temporal split to avoid leakage. Under this rigorous split, coop and local models both reach **0.8500** / **0.8500** — the cooperative gain is **0 pp**. This is an honest negative result: current cooperative features do not yet generalise to unseen future timesteps in the SUMO setting.
- The exported HEIR stub currently passes both structural and metadata consistency checks (**shape=True**, **consistency=True**), supporting the export pathway without claiming end-to-end encrypted runtime execution.

## Paper-friendly interpretation

- The clearest positive empirical story is the binary synthetic prototype: cooperative fusion improves accuracy, and HE-friendly approximation retains most of that benefit with minimal overhead.
- The expanded SUMO experiment reveals an important limitation: under a temporally correct split, cooperative features do not yet generalise to future timesteps. This motivates the next research direction — temporal or history-aware cooperative features.
- The current infrastructure is strongest on reproducibility: one-shot artifact generation, summary reporting, HEIR export verification, and report validation are now in place.
- The multiclass/action4 path is promising but not yet as strong as the binary story; macro-F1 suggests class imbalance and decision difficulty still need attention.

## Action4 interpretation

- The majority class is **class 0** with **87** validation examples, while the minority class is **class 3** with **12** examples.
- The strongest multiclass F1 is **class 0 = 0.7662** and the weakest is **class 3 = 0.4118**.
- Compared with one-vs-rest, the multiclass model changes macro-F1 by **0.0422**, suggesting that the current joint classifier is better overall on balanced class performance.
- The gap between majority and minority support (**87 vs 12**) is consistent with the narrative that imbalance is still shaping the multiclass difficulty.

## Caveats

- The small SUMO binary experiment used a random split and should not be used as a standalone performance claim.
- The large SUMO temporal split shows **no cooperative gain yet** — this is a known limitation and an honest negative result.
- HEIR support is validated at the export/consistency level, not full encrypted execution benchmarking.
- The results are better framed as a paper-ready scaffold with emerging evidence, not a submission-ready benchmark package yet.

## Recommended next steps

1. Develop history-aware or temporally-informed cooperative features that can generalise across timesteps.
2. Add literature-adjacent baselines (e.g. aggregation-only, FedAvg-proxy) for direct comparison.
3. Improve multiclass/action4 analysis, especially per-class weakness and imbalance handling.
4. If feasible, deepen the HEIR pathway beyond export validation into a more realistic compile/evaluate handoff.