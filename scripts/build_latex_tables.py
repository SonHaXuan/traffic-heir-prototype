"""
build_latex_tables.py
---------------------
Generates LaTeX-ready table snippets for the HEIR traffic fusion paper.

Outputs:
  reports/latex_tables.tex   — all tables in one file, separated by comments
  reports/latex_tables.md    — preview of what each table looks like as markdown

Tables produced:
  1. Main results table (binary cooperative inference)
  2. Seed sweep reproducibility table
  3. Ablation study table
  4. Action-4 multiclass table (secondary)
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPORTS = REPO / "reports"


def load(name):
    with open(REPORTS / name) as f:
        return json.load(f)


def pct(v, decimals=1):
    return f"{v * 100:.{decimals}f}"


def pp(v, decimals=1):
    return f"{v * 100:.{decimals}f}"


def latex_table(caption, label, headers, rows, notes=None, bold_row=None):
    """
    Minimal booktabs-style LaTeX table.
    bold_row: 0-indexed row index to bold
    """
    col_spec = "l" + "c" * (len(headers) - 1)
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
    ]
    # Header
    lines.append("    " + " & ".join(f"\\textbf{{{h}}}" for h in headers) + r" \\")
    lines.append(r"    \midrule")
    # Rows
    for i, row in enumerate(rows):
        if bold_row is not None and i == bold_row:
            cells = [f"\\textbf{{{c}}}" for c in row]
        else:
            cells = list(row)
        lines.append("    " + " & ".join(cells) + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    if notes:
        lines.append(r"  \vspace{2pt}")
        lines.append(f"  \\footnotesize{{\\textit{{{notes}}}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def md_table(headers, rows, bold_row=None):
    """Simple markdown preview table."""
    sep = "|" + "|".join("---" for _ in headers) + "|"
    hdr = "| " + " | ".join(f"**{h}**" for h in headers) + " |"
    out = [hdr, sep]
    for i, row in enumerate(rows):
        if bold_row is not None and i == bold_row:
            cells = [f"**{c}**" for c in row]
        else:
            cells = list(row)
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def build():
    kn = load("key_numbers.json")
    proto = load("prototype_default_metrics.json")
    seed = load("seed_sweep_metrics.json")
    a4 = load("action4_metrics.json")

    tables_tex = []
    tables_md = []

    # ── Table 1: Main results ──────────────────────────────────────────────────
    mr = kn["main_result"]
    headers1 = ["Method", "Val Acc (\\%)", "$\\Delta$ Local (pp)", "Notes"]
    rows1 = [
        ["HEIR Cooperative (HE-friendly)", pct(mr["coop_he_val_acc"]), f"+{mr['gain_over_local_pp']:.1f}", "Primary claim"],
        ["HEIR Cooperative (plaintext)", pct(mr["coop_plaintext_val_acc"]), f"+{mr['coop_plaintext_val_acc']*100 - mr['local_model_val_acc']*100:.1f}", "Upper bound"],
        ["Local model (no cooperation)", pct(mr["local_model_val_acc"]), "—", "Non-coop baseline"],
        ["Max-pressure heuristic", pct(proto["max_pressure"]), f"{(proto['max_pressure'] - proto['local_model'])*100:+.1f}", "Traditional baseline"],
        ["Fixed-time control", pct(proto["fixed_time"]), f"{(proto['fixed_time'] - proto['local_model'])*100:+.1f}", "Traditional baseline"],
    ]
    t1_tex = latex_table(
        caption="Main results on binary traffic action prediction. Val Acc is validation accuracy. HE-friendly protocol uses polynomial-approximated activations compatible with homomorphic encryption.",
        label="tab:main_results",
        headers=["Method", "Val Acc (\\%)", "$\\Delta$ Local (pp)", "Notes"],
        rows=rows1,
        notes=f"HE overhead vs plaintext: {mr['he_overhead_pp']:.1f} pp. Gain over best traditional baseline: +{mr['gain_over_best_baseline_pp']:.1f} pp.",
        bold_row=0,
    )
    t1_md_headers = ["Method", "Val Acc (%)", "Δ Local (pp)", "Notes"]
    t1_md = md_table(t1_md_headers, rows1, bold_row=0)
    tables_tex.append(("Table 1: Main Results", t1_tex))
    tables_md.append(("Table 1: Main Results", t1_md))

    # ── Table 2: Seed sweep ────────────────────────────────────────────────────
    ss = kn["seed_sweep"]
    ss_rows_data = seed["rows"]
    headers2 = ["Seed", "HEIR Coop HE (\\%)", "Local (\\%)"]
    rows2 = [
        [str(r["seed"]), pct(r["coop_he"]), pct(r["local"])]
        for r in ss_rows_data
    ]
    # Add summary row
    rows2.append([
        f"Mean $\\pm$ Std",
        f"{pct(ss['coop_he_mean'])} $\\pm$ {ss['coop_he_std']:.3f}",
        f"{pct(ss['local_mean'])} $\\pm$ {ss['local_std']:.3f}",
    ])

    t2_tex = latex_table(
        caption="Seed sweep results across 3 random seeds. Mean cooperative gain: +{:.1f} pp.".format(ss["mean_gain_pp"]),
        label="tab:seed_sweep",
        headers=headers2,
        rows=rows2,
        bold_row=len(rows2) - 1,
    )
    t2_md_headers = ["Seed", "HEIR Coop HE (%)", "Local (%)"]
    rows2_md = [
        [str(r["seed"]), pct(r["coop_he"]), pct(r["local"])]
        for r in ss_rows_data
    ]
    rows2_md.append([
        "Mean ± Std",
        f"{pct(ss['coop_he_mean'])} ± {ss['coop_he_std']:.3f}",
        f"{pct(ss['local_mean'])} ± {ss['local_std']:.3f}",
    ])
    t2_md = md_table(t2_md_headers, rows2_md, bold_row=len(rows2_md) - 1)
    tables_tex.append(("Table 2: Seed Sweep", t2_tex))
    tables_md.append(("Table 2: Seed Sweep", t2_md))

    # ── Table 3: Ablation ─────────────────────────────────────────────────────
    abl = kn["ablation"]
    rob = kn["robustness"]
    headers3 = ["Variant", "Val Acc (\\%)", "$\\Delta$ Full HEIR (pp)"]
    rows3 = [
        ["Full HEIR (HE-friendly)", pct(kn["main_result"]["coop_he_val_acc"]), "—"],
        ["w/o directional features", pct(abl["remove_direction_val_acc"]), f"−{abl['direction_contribution_pp']:.1f}"],
        ["w/o neighbor features", pct(abl["remove_neighbor_val_acc"]), f"−{abl['neighbor_contribution_pp']:.1f}"],
        ["w/o interaction features", pct(abl["remove_interaction_val_acc"]), f"{-abl['interaction_contribution_pp']:+.1f}"],
        ["Local model (no cooperation)", pct(kn["main_result"]["local_model_val_acc"]), f"−{kn['main_result']['gain_over_local_pp']:.1f}"],
        [r"Robustness eval (HEIR coop)", pct(rob["robust_coop_he_val_acc"]), f"— (+{rob['robust_gain_pp']:.1f} vs local)"],
    ]
    t3_tex = latex_table(
        caption="Ablation study. Each row removes one feature group from the full HEIR cooperative model. Robustness row reports held-out adversarial evaluation.",
        label="tab:ablation",
        headers=headers3,
        rows=rows3,
        bold_row=0,
    )
    t3_md_headers = ["Variant", "Val Acc (%)", "Δ Full HEIR (pp)"]
    rows3_md = [
        ["Full HEIR (HE-friendly)", pct(kn["main_result"]["coop_he_val_acc"]), "—"],
        ["w/o directional features", pct(abl["remove_direction_val_acc"]), f"−{abl['direction_contribution_pp']:.1f}"],
        ["w/o neighbor features", pct(abl["remove_neighbor_val_acc"]), f"−{abl['neighbor_contribution_pp']:.1f}"],
        ["w/o interaction features", pct(abl["remove_interaction_val_acc"]), f"{-abl['interaction_contribution_pp']:+.1f}"],
        ["Local model (no cooperation)", pct(kn["main_result"]["local_model_val_acc"]), f"−{kn['main_result']['gain_over_local_pp']:.1f}"],
        ["Robustness eval (HEIR coop)", pct(rob["robust_coop_he_val_acc"]), f"— (+{rob['robust_gain_pp']:.1f} vs local)"],
    ]
    t3_md = md_table(t3_md_headers, rows3_md, bold_row=0)
    tables_tex.append(("Table 3: Ablation Study", t3_tex))
    tables_md.append(("Table 3: Ablation Study", t3_md))

    # ── Table 4: Action-4 multiclass (secondary) ──────────────────────────────
    ovr = a4.get("ovr_per_class", {})
    headers4 = ["Class", "Precision", "Recall", "F1", "Support"]
    rows4 = []
    for cls in sorted(ovr.keys(), key=int):
        d = ovr[cls]
        rows4.append([
            f"Action {cls}",
            f"{d['precision']:.3f}",
            f"{d['recall']:.3f}",
            f"{d['f1']:.3f}",
            str(int(d["support"])),
        ])
    rows4.append([
        "Macro avg",
        "—",
        "—",
        f"{a4['ovr_macro_f1']:.3f}",
        str(sum(int(d["support"]) for d in ovr.values())),
    ])
    t4_tex = latex_table(
        caption="Action-4 multiclass prediction results (one-vs-rest, OvR). Overall val accuracy: {:.1f}\\%. Binary story is the primary claim; this is exploratory.".format(a4["ovr_val_accuracy"] * 100),
        label="tab:action4",
        headers=headers4,
        rows=rows4,
        notes="Class imbalance is significant (Action 0 dominates). See action4\\_metrics.json for full confusion matrix.",
        bold_row=len(rows4) - 1,
    )
    t4_md = md_table(headers4, rows4, bold_row=len(rows4) - 1)
    tables_tex.append(("Table 4: Action-4 Multiclass (secondary)", t4_tex))
    tables_md.append(("Table 4: Action-4 Multiclass (secondary)", t4_md))

    # ── Write outputs ──────────────────────────────────────────────────────────
    preamble = """%% LaTeX table snippets — HEIR Traffic Fusion Paper
%% Auto-generated by scripts/build_latex_tables.py. Do not edit by hand.
%% Requires: \\usepackage{booktabs}
%%
%% Usage: copy individual table environments into your paper as needed,
%% or \\input{reports/latex_tables.tex} after stripping these comment lines.
"""
    tex_sections = []
    for title, tex in tables_tex:
        tex_sections.append(f"%% {title}\n{tex}")

    full_tex = preamble + "\n\n".join(tex_sections) + "\n"

    md_preamble = "# LaTeX Tables Preview — HEIR Traffic Fusion Paper\n\n> Auto-generated from `scripts/build_latex_tables.py`. Do not edit by hand.\n\n"
    md_sections = []
    for title, md in tables_md:
        md_sections.append(f"## {title}\n\n{md}")
    full_md = md_preamble + "\n\n".join(md_sections) + "\n"

    out_tex = REPORTS / "latex_tables.tex"
    out_md = REPORTS / "latex_tables.md"

    out_tex.write_text(full_tex, encoding="utf-8")
    print(f"✅ Written: {out_tex}")
    out_md.write_text(full_md, encoding="utf-8")
    print(f"✅ Written: {out_md}")

    # Print markdown preview
    print()
    print(full_md)


if __name__ == "__main__":
    build()
