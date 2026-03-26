#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def load_json(name: str) -> dict:
    path = REPORTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    return f"{float(value):.{digits}f}"


def main() -> None:
    proto = load_json("prototype_default_metrics.json")
    seed = load_json("seed_sweep_metrics.json")
    action4 = load_json("action4_metrics.json")
    sumo = load_json("sumo_binary_metrics.json")
    heir = load_json("heir_export_report.json")

    coop_he = float(proto.get("coop_he_friendly", 0.0))
    local_model = float(proto.get("local_model", 0.0))
    coop_plain = float(proto.get("coop_plaintext", 0.0))
    coop_gain = coop_he - local_model
    he_gap = coop_he - coop_plain
    seed_local_mean = float(seed.get("summary", {}).get("local", {}).get("mean", 0.0))
    seed_coop_mean = float(seed.get("summary", {}).get("coop_he", {}).get("mean", 0.0))
    seed_gain = seed_coop_mean - seed_local_mean
    action4_acc = float(action4.get("val_accuracy", 0.0))
    action4_macro = float(action4.get("macro_f1", 0.0))
    ovr_macro = float(action4.get("ovr_macro_f1", 0.0))
    sumo_story = sumo.get("eval_story", {})

    lines = [
        "# Results Narrative",
        "",
        "## Key findings",
        "",
        f"- The cooperative HE-friendly binary model reached **{fmt(coop_he)}** validation accuracy, compared with **{fmt(local_model)}** for the local plaintext model, for a gain of **{fmt(coop_gain)}**.",
        f"- Relative to the cooperative plaintext model (**{fmt(coop_plain)}**), the HE-friendly version changed accuracy by **{fmt(he_gap)}**, suggesting limited loss from the low-depth approximation in this prototype setting.",
        f"- Across the seed sweep, cooperative HE-friendly performance averaged **{fmt(seed_coop_mean)}** versus **{fmt(seed_local_mean)}** for local modeling, preserving an average gain of **{fmt(seed_gain)}**.",
        f"- In the 4-action setting, the multiclass model achieved **{fmt(action4_acc)}** accuracy and **{fmt(action4_macro)}** macro-F1, while the one-vs-rest variant reached **{fmt(ovr_macro)}** macro-F1.",
        f"- The current SUMO-derived binary sample is tiny (**{sumo.get('samples', 'n/a')}** samples over **{sumo.get('timesteps', 'n/a')}** timesteps), so its **{fmt(sumo.get('val_accuracy'))}** validation accuracy should be treated as a pipeline sanity check rather than a robust performance claim.",
        f"- The exported HEIR stub currently passes both structural and metadata consistency checks (**shape={heir.get('shape_check_passed', False)}**, **consistency={heir.get('consistency_check_passed', False)}**), supporting the export pathway without claiming end-to-end encrypted runtime execution.",
        "",
        "## Paper-friendly interpretation",
        "",
        "- The clearest empirical story so far is that cooperative fusion helps the binary decision-support task and that the HE-friendly approximation retains most of that benefit.",
        "- The current infrastructure is strongest on reproducibility: one-shot artifact generation, summary reporting, HEIR export verification, and report validation are now in place.",
        "- The multiclass/action4 path is promising but not yet as strong as the binary story; macro-F1 suggests class imbalance and decision difficulty still need attention.",
        "",
        "## Caveats",
        "",
        "- The SUMO binary experiment is still too small to serve as a main quantitative claim.",
        "- HEIR support is currently validated at the export/consistency level, not full encrypted execution benchmarking.",
        "- The results are better framed as a paper-ready scaffold with emerging evidence, not a submission-ready benchmark package yet.",
        "",
        "## Recommended next steps",
        "",
        "1. Expand SUMO-derived experiments to non-trivial sample sizes and richer scenarios.",
        "2. Improve multiclass/action4 analysis, especially per-class weakness and imbalance handling.",
        "3. Add figure-friendly summaries or plots for the strongest binary and seed-sweep findings.",
        "4. If feasible, deepen the HEIR pathway beyond export validation into a more realistic compile/evaluate handoff.",
    ]

    out = REPORTS / "results_narrative.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print({"results_narrative": str(out)})


if __name__ == "__main__":
    main()
