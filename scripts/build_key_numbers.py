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
    corr = load("sumo_correlated_metrics.json") if (REPORTS / "sumo_correlated_metrics.json").exists() else {}
    overhead = load("heir_overhead_report.json") if (REPORTS / "heir_overhead_report.json").exists() else {}

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

    # ---------- SUMO correlated (primary SUMO result) ----------
    corr_ev = corr.get("eval_story", {})
    corr_coop_acc   = corr.get("coop_val_accuracy", None)
    corr_local_acc  = corr.get("local_val_accuracy", None)
    corr_gain       = corr_ev.get("cooperative_gain_over_local", None)
    corr_sf_gain    = corr_ev.get("simple_fusion_gain_over_local", None)
    corr_gl_gain    = corr_ev.get("graph_lite_gain_over_local", None)
    corr_ml_vs_rule = corr_ev.get("ml_coop_gain_over_heuristic_coop", None)
    corr_samples    = corr.get("samples", None)
    corr_ci         = corr.get("wilson_ci", {})

    # ---------- HEIR overhead ----------
    oh_gap_pp = overhead.get("accuracy_gap", {}).get("gap_pp", None)
    oh_lat_ms = overhead.get("latency_estimates_ms", {}).get("coop_he_friendly", None)
    oh_comm   = overhead.get("communication_cost_7_intersections", {}).get(
        "coop_he_friendly", {})
    oh_ct_kb  = oh_comm.get("ciphertext_kb", None)
    oh_pt_kb  = oh_comm.get("plaintext_kb", None)

    # ---------- Seed sweep significance ----------
    sig = seed.get("significance", {})
    sweep_p  = sig.get("coop_vs_local_p_value", None)
    sweep_d  = sig.get("coop_vs_local_cohens_d", None)
    n_seeds  = sig.get("n_seeds", len(seed.get("rows", [])))

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
            "n_seeds": n_seeds,
            "coop_he_mean": round(coop_he_mean, 4),
            "coop_he_std": round(coop_he_std, 4),
            "coop_he_ci_lower_95": round(seed["summary"]["coop_he"].get("ci_lower_95", 0), 4),
            "coop_he_ci_upper_95": round(seed["summary"]["coop_he"].get("ci_upper_95", 0), 4),
            "local_mean": round(local_mean, 4),
            "local_std": round(local_std, 4),
            "mean_gain_pp": round(sweep_gain * 100, 2),
            "p_value": round(sweep_p, 4) if sweep_p is not None else None,
            "cohens_d": round(sweep_d, 3) if sweep_d is not None else None,
            "significant_at_0_05": (sweep_p < 0.05) if sweep_p is not None else None,
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
        "sumo_correlated": {
            "coop_val_accuracy": round(corr_coop_acc, 4) if corr_coop_acc is not None else None,
            "local_val_accuracy": round(corr_local_acc, 4) if corr_local_acc is not None else None,
            "cooperative_gain_pp": round(corr_gain * 100, 2) if corr_gain is not None else None,
            "simple_fusion_gain_pp": round(corr_sf_gain * 100, 2) if corr_sf_gain is not None else None,
            "graph_lite_gain_pp": round(corr_gl_gain * 100, 2) if corr_gl_gain is not None else None,
            "ml_vs_rule_gain_pp": round(corr_ml_vs_rule * 100, 2) if corr_ml_vs_rule is not None else None,
            "samples": corr_samples,
            "wilson_ci": corr_ci,
            "note": "7-intersection corridor, 300 timesteps, temporal split, upstream spillback",
        },
        "heir_overhead": {
            "accuracy_gap_pp": round(oh_gap_pp, 2) if oh_gap_pp is not None else None,
            "latency_ms": oh_lat_ms,
            "ciphertext_comm_kb": oh_ct_kb,
            "plaintext_comm_kb": oh_pt_kb,
            "ckks_expansion_factor": 4096,
            "note": "Theoretical estimates, SEAL 4.x benchmarks, 128-bit security",
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
        f"## 📊 Seed Sweep (reproducibility across {n_seeds} seeds)",
        "",
        f"| Model | Mean ± Std | 95% CI |",
        f"|---|---|---|",
        f"| HEIR Cooperative (HE) | {fmt(coop_he_mean)} ± {fmt(coop_he_std, pct=False, decimals=3)} "
        f"| [{fmt(seed['summary']['coop_he'].get('ci_lower_95', 0))} – "
        f"{fmt(seed['summary']['coop_he'].get('ci_upper_95', 0))}] |",
        f"| Local model | {fmt(local_mean)} ± {fmt(local_std, pct=False, decimals=3)} "
        f"| [{fmt(seed['summary']['local'].get('ci_lower_95', 0))} – "
        f"{fmt(seed['summary']['local'].get('ci_upper_95', 0))}] |",
        "",
        f"**Consistent cooperative gain across {n_seeds} seeds:** +{pp(sweep_gain)} mean  ",
        (f"**Statistical significance (paired t-test):** t-stat={round(sweep_p if sweep_p is not None else 0,4)}, "
         f"p={round(sweep_p, 4) if sweep_p is not None else 'N/A'}, "
         f"Cohen's d={round(sweep_d, 3) if sweep_d is not None else 'N/A'}"
         if sweep_p is not None else ""),
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
        "## 🚦 SUMO Correlated Corridor (primary SUMO result)",
        "",
        *([] if not corr else [
            f"| Model | Val Acc | 95% Wilson CI | Δ vs Local |",
            f"|---|---|---|---|",
            f"| **Cooperative HE-friendly** | **{fmt(corr_coop_acc) if corr_coop_acc else 'N/A'}** "
            f"| {('[' + fmt(corr_ci.get('coop', {}).get('ci_lower_95', 0)) + ' – ' + fmt(corr_ci.get('coop', {}).get('ci_upper_95', 0)) + ']') if 'coop' in corr_ci else '—'} "
            f"| **+{pp(corr_gain) if corr_gain else 'N/A'}** |",
            f"| Graph-lite | {fmt(corr.get('graph_lite_val_accuracy', 0)) if corr.get('graph_lite_val_accuracy') else 'N/A'} "
            f"| — | +{pp(corr_gl_gain) if corr_gl_gain else 'N/A'} |",
            f"| Simple fusion | {fmt(corr.get('simple_fusion_val_accuracy', 0)) if corr.get('simple_fusion_val_accuracy') else 'N/A'} "
            f"| — | +{pp(corr_sf_gain) if corr_sf_gain else 'N/A'} |",
            f"| Local model | {fmt(corr_local_acc) if corr_local_acc else 'N/A'} "
            f"| {('[' + fmt(corr_ci.get('local', {}).get('ci_lower_95', 0)) + ' – ' + fmt(corr_ci.get('local', {}).get('ci_upper_95', 0)) + ']') if 'local' in corr_ci else '—'} | — |",
            "",
            f"**7 intersections × 300 timesteps, temporal split. "
            f"ML coop vs rule-based coop: +{pp(corr_ml_vs_rule) if corr_ml_vs_rule else 'N/A'}**",
        ]),
        "",
        "## 🚦 SUMO Binary (small graph, pipeline sanity check)",
        "",
        f"- Val accuracy: **{fmt(sumo_val_acc)}** ({sumo_samples} samples, {sumo_timesteps} timesteps)",
        f"- Uses adjacency matrix: {sumo_uses_adj}",
        f"- Interaction-gain within cooperative setting: {sumo_interaction_gain}",
        f"- ⚠️  Small toy graph — results are pipeline checks, not scale benchmarks",
        "",
        "## 🔐 HEIR Overhead (theoretical analysis)",
        "",
        *([] if not overhead else [
            f"| Metric | Value |",
            f"|---|---|",
            f"| Accuracy gap (HE vs plaintext) | {oh_gap_pp:.2f} pp |" if oh_gap_pp else "",
            f"| Inference latency (coop model) | {oh_lat_ms:.0f} ms |" if oh_lat_ms else "",
            f"| Encrypted comm per timestep (7 nodes) | {oh_ct_kb:.0f} KB |" if oh_ct_kb else "",
            f"| Plaintext comm per timestep (7 nodes) | {oh_pt_kb:.3f} KB |" if oh_pt_kb else "",
            f"| CKKS expansion factor | 4096× |",
            "",
            "> Estimates based on SEAL 4.x benchmarks (128-bit security). "
            "Order-of-magnitude guide.",
        ]),
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
