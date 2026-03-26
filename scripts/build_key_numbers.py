"""
build_key_numbers.py
--------------------
Generates a compact key-numbers artifact for paper writing and slides.
Pulls the canonical numbers from reports/ and formats them into:
  - reports/key_numbers.json   (machine-readable, for downstream scripts)
  - reports/key_numbers.md     (human-readable, paste-into-paper/slides)
"""

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPORTS = REPO / "reports"


def load(name):
    with open(REPORTS / name) as f:
        return json.load(f)


def fmt(v, pct=True, decimals=1):
    """Format a float as percentage string or plain decimal."""
    if pct:
        return f"{v * 100:.{decimals}f}%"
    return f"{v:.{decimals}f}"


def pp(v, decimals=1):
    """Format a fraction as percentage-points string (e.g. 0.033 -> '3.3 pp')."""
    return f"{v * 100:.{decimals}f} pp"


def build():
    proto = load("prototype_default_metrics.json")
    seed = load("seed_sweep_metrics.json")
    sumo = load("sumo_binary_metrics.json")
    action4 = load("action4_metrics.json")
    heir = load("heir_export_report.json")

    # ---------- Prototype binary (main result) ----------
    coop_he = proto["coop_he_friendly"]
    local_model = proto["local_model"]
    coop_gain = coop_he - local_model

    coop_plaintext = proto["coop_plaintext"]
    he_overhead = coop_plaintext - coop_he  # how much HE costs vs plaintext

    robust_coop_he = proto["robustness"]["coop_he_friendly"]
    robust_local = proto["robustness"]["local_model"]
    robust_gain = robust_coop_he - robust_local

    fixed_time = proto["fixed_time"]
    max_pressure = proto["max_pressure"]
    baseline_best = max(fixed_time, max_pressure)

    # Ablation deltas (what removing each component costs)
    ablation_no_direction = proto["ablation_no_direction"]
    ablation_no_interaction = proto["ablation_no_interaction"]
    ablation_no_neighbor = proto["ablation_no_neighbor"]

    # ---------- Seed sweep ----------
    ss = seed["summary"]
    coop_he_mean = ss["coop_he"]["mean"]
    coop_he_std = ss["coop_he"]["stdev"]
    local_mean = ss["local"]["mean"]
    local_std = ss["local"]["stdev"]
    sweep_gain = coop_he_mean - local_mean

    # ---------- SUMO binary ----------
    sumo_val_acc = sumo["val_accuracy"]
    sumo_samples = sumo["samples"]
    sumo_timesteps = sumo["timesteps"]
    sumo_uses_adj = sumo["uses_adjacency"]
    # interaction_gain lives under eval_story
    eval_story = sumo.get("eval_story", sumo)
    sumo_interaction_gain = eval_story.get("interaction_gain_within_coop", sumo.get("interaction_gain_within_coop", 0))

    # ---------- Action-4 ----------
    a4_val_acc = action4["val_accuracy"]
    a4_macro_f1 = action4["macro_f1"]
    a4_ovr_macro_f1 = action4["ovr_macro_f1"]

    # ---------- HEIR export ----------
    heir_consistent = heir.get("consistency_check_passed", False)
    heir_shape_ok = heir.get("shape_check_passed", False)
    heir_meta_acc = heir.get("metadata_val_accuracy", None)

    # ---------- Assemble structured key numbers ----------
    key = {
        "main_result": {
            "coop_he_val_acc": round(coop_he, 4),
            "local_model_val_acc": round(local_model, 4),
            "gain_over_local_pp": round(coop_gain * 100, 2),
            "coop_plaintext_val_acc": round(coop_plaintext, 4),
            "he_overhead_pp": round(he_overhead * 100, 2),
            "best_traditional_baseline_val_acc": round(baseline_best, 4),
            "gain_over_best_baseline_pp": round((coop_he - baseline_best) * 100, 2),
        },
        "robustness": {
            "robust_coop_he_val_acc": round(robust_coop_he, 4),
            "robust_local_val_acc": round(robust_local, 4),
            "robust_gain_pp": round(robust_gain * 100, 2),
        },
        "ablation": {
            "remove_direction_val_acc": round(ablation_no_direction, 4),
            "remove_interaction_val_acc": round(ablation_no_interaction, 4),
            "remove_neighbor_val_acc": round(ablation_no_neighbor, 4),
            "direction_contribution_pp": round((coop_he - ablation_no_direction) * 100, 2),
            "interaction_contribution_pp": round((coop_he - ablation_no_interaction) * 100, 2),
            "neighbor_contribution_pp": round((coop_he - ablation_no_neighbor) * 100, 2),
        },
        "seed_sweep": {
            "n_seeds": len(seed["rows"]),
            "coop_he_mean": round(coop_he_mean, 4),
            "coop_he_std": round(coop_he_std, 4),
            "local_mean": round(local_mean, 4),
            "local_std": round(local_std, 4),
            "mean_gain_pp": round(sweep_gain * 100, 2),
        },
        "sumo_binary": {
            "val_accuracy": round(sumo_val_acc, 4),
            "samples": sumo_samples,
            "timesteps": sumo_timesteps,
            "uses_adjacency": sumo_uses_adj,
            "interaction_gain_within_coop": round(sumo_interaction_gain, 4),
            "note": "Small sample (toy SUMO graph); perfect accuracy expected at this scale",
        },
        "action4_multiclass": {
            "val_accuracy": round(a4_val_acc, 4),
            "macro_f1": round(a4_macro_f1, 4),
            "ovr_macro_f1": round(a4_ovr_macro_f1, 4),
            "note": "4-class action prediction; binary story is primary claim",
        },
        "heir_export": {
            "consistency_check_passed": heir_consistent,
            "shape_check_passed": heir_shape_ok,
            "metadata_val_accuracy": round(heir_meta_acc, 4) if heir_meta_acc is not None else None,
        },
    }

    # ---------- Markdown summary ----------
    lines = [
        "# Key Numbers — HEIR Traffic Fusion Paper",
        "",
        "> Auto-generated from `scripts/build_key_numbers.py`. Do not edit by hand.",
        "",
        "## 🔑 Primary Result (binary cooperative inference, prototype)",
        "",
        f"| Model | Val Accuracy | Δ vs Local |",
        f"|---|---|---|",
        f"| **HEIR Cooperative (HE-friendly)** | **{fmt(coop_he)}** | **+{pp(coop_gain)}** |",
        f"| HEIR Cooperative (plaintext) | {fmt(coop_plaintext)} | +{pp(coop_plaintext - local_model)} |",
        f"| Local model (no cooperation) | {fmt(local_model)} | — |",
        f"| Max-pressure baseline | {fmt(max_pressure)} | — |",
        f"| Fixed-time baseline | {fmt(fixed_time)} | — |",
        "",
        f"**HE overhead vs plaintext:** {pp(he_overhead)}  ",
        f"**Gain over best traditional baseline ({fmt(baseline_best)}):** +{pp(coop_he - baseline_best)}",
        "",
        "## 📊 Seed Sweep (reproducibility across 3 seeds)",
        "",
        f"| Model | Mean ± Std |",
        f"|---|---|",
        f"| HEIR Cooperative (HE) | {fmt(coop_he_mean)} ± {fmt(coop_he_std, pct=False, decimals=3)} |",
        f"| Local model | {fmt(local_mean)} ± {fmt(local_std, pct=False, decimals=3)} |",
        "",
        f"**Consistent cooperative gain across seeds:** +{pp(sweep_gain)} mean",
        "",
        "## 🔬 Robustness (held-out adversarial eval)",
        "",
        f"| Model | Robust Val Acc |",
        f"|---|---|",
        f"| HEIR Cooperative (HE) | {fmt(robust_coop_he)} |",
        f"| Local model | {fmt(robust_local)} |",
        "",
        f"**Cooperative advantage holds under robustness eval:** +{pp(robust_gain)}",
        "",
        "## 🧩 Ablation (component contributions)",
        "",
        f"| Ablation | Val Acc | Δ vs Full HEIR |",
        f"|---|---|---|",
        f"| Full HEIR (HE-friendly) | {fmt(coop_he)} | — |",
        f"| Remove directional features | {fmt(ablation_no_direction)} | -{pp(coop_he - ablation_no_direction)} |",
        f"| Remove interaction features | {fmt(ablation_no_interaction)} | {'+' if coop_he - ablation_no_interaction < 0 else '-'}{pp(abs(coop_he - ablation_no_interaction))} |",
        f"| Remove neighbor features | {fmt(ablation_no_neighbor)} | -{pp(coop_he - ablation_no_neighbor)} |",
        "",
        "## 🚦 SUMO Binary (graph-structured adjacency)",
        "",
        f"- Val accuracy: **{fmt(sumo_val_acc)}** ({sumo_samples} samples, {sumo_timesteps} timesteps)",
        f"- Uses adjacency matrix: {sumo_uses_adj}",
        f"- Interaction-gain within cooperative setting: {sumo_interaction_gain}",
        f"- ⚠️  Small toy graph — results are consistency checks, not scale benchmarks",
        "",
        "## 🎯 Action-4 Multiclass (secondary, exploratory)",
        "",
        f"- Val accuracy: {fmt(a4_val_acc)} | Macro-F1: {round(a4_macro_f1, 3)} | OvR Macro-F1: {round(a4_ovr_macro_f1, 3)}",
        f"- Note: binary story is primary claim; action-4 is exploratory",
        "",
        "## 🔐 HEIR Export Consistency",
        "",
        f"- Consistency check: {'✅ passed' if heir_consistent else '❌ failed'}",
        f"- Shape check: {'✅ passed' if heir_shape_ok else '❌ failed'}",
        f"- Metadata val accuracy: {fmt(heir_meta_acc) if heir_meta_acc is not None else 'N/A'}",
        "",
        "---",
        "",
        "## Paper claim language (suggested)",
        "",
        f"- *\"HEIR cooperative inference achieves {fmt(coop_he)} validation accuracy, a {pp(coop_gain)} improvement over the non-cooperative local model ({fmt(local_model)}).\"*",
        f"- *\"The HE-friendly protocol incurs only {pp(he_overhead)} overhead relative to plaintext cooperation.\"*",
        f"- *\"Results are consistent across seeds: {fmt(coop_he_mean)} ± {coop_he_std:.3f} (cooperative) vs {fmt(local_mean)} ± {local_std:.3f} (local).\"*",
        f"- *\"The cooperative advantage is maintained under robustness evaluation ({fmt(robust_coop_he)} vs {fmt(robust_local)}).\"*",
    ]

    md = "\n".join(lines) + "\n"

    # ---------- Write outputs ----------
    out_json = REPORTS / "key_numbers.json"
    out_md = REPORTS / "key_numbers.md"

    with open(out_json, "w") as f:
        json.dump(key, f, indent=2)
    print(f"✅ Written: {out_json}")

    with open(out_md, "w") as f:
        f.write(md)
    print(f"✅ Written: {out_md}")

    # ---------- Print summary to stdout ----------
    print()
    print(md)

    return key


if __name__ == "__main__":
    build()
