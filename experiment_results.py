# standalone gradio app showing all experiment results, charts, and study figures
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

DEFENSE_DISPLAY = {
    "fine_pruning": "Fine Pruning",
    "activation_clustering": "Activation Clustering",
    "spectral_signatures": "Spectral Signatures",
    "neural_cleanse": "Neural Cleanse",
}

ATTACK_DISPLAY = {
    "none": "Clean",
    "badnets": "BadNets",
    "blended": "Blended",
    "label_consistent": "Label Consistent",
}


def load_results():
    rows = []
    seen = set()
    for csvpath in (
        "results/full_pipeline.csv",
        "results/gtsrb_backdoor.csv",
        "results/attack_results.csv",
    ):
        if not os.path.exists(csvpath):
            continue
        with open(csvpath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("stage") is None and row.get("attack") and row.get("test_ca") and row.get("asr"):
                    normalized = {
                        "stage": "poisoned",
                        "attack": row["attack"],
                        "defense": "none",
                        "test_ca": row["test_ca"],
                        "asr": row["asr"],
                    }
                    key = ("poisoned", row["attack"], "none")
                    if key not in seen:
                        seen.add(key)
                        rows.append(normalized)
                    continue
                key = (row.get("stage"), row.get("attack"), row.get("defense"))
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
    return rows


def make_results_table(rows):
    headers = ["Stage", "Attack", "Defense", "Clean Acc (%)", "ASR (%)"]
    data = []
    for r in rows:
        if not r.get("stage") or not r.get("test_ca") or not r.get("asr"):
            continue
        try:
            ca = f"{float(r['test_ca']) * 100:.2f}"
            asr = f"{float(r['asr']) * 100:.2f}"
        except ValueError:
            continue
        data.append([
            r.get("stage", ""),
            ATTACK_DISPLAY.get(r.get("attack", ""), r.get("attack", "")),
            DEFENSE_DISPLAY.get(r.get("defense", ""), r.get("defense", "")),
            ca,
            asr,
        ])
    return headers, data


def make_attack_overview_chart(rows):
    attack_rows = {r["attack"]: r for r in rows if r.get("stage") in ("clean", "poisoned")}
    attacks_ordered = ["none", "badnets", "blended", "label_consistent"]
    present = [a for a in attacks_ordered if a in attack_rows]
    if not present:
        return None
    labels = [ATTACK_DISPLAY.get(a, a) for a in present]
    ca_vals = [float(attack_rows[a]["test_ca"]) * 100 for a in present]
    asr_vals = [float(attack_rows[a]["asr"]) * 100 for a in present]
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(9, 5))
    w = 0.35
    bars1 = ax1.bar(x - w / 2, ca_vals, w, label="Clean Accuracy (%)", color="#2196F3", alpha=0.85)
    ax1.set_ylabel("Clean Accuracy (%)", color="#2196F3")
    ax1.set_ylim(90, 101)
    ax1.tick_params(axis="y", labelcolor="#2196F3")
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + w / 2, asr_vals, w, label="Attack Success Rate (%)", color="#F44336", alpha=0.85)
    ax2.set_ylabel("Attack Success Rate (%)", color="#F44336")
    ax2.set_ylim(0, 110)
    ax2.tick_params(axis="y", labelcolor="#F44336")
    for bar, val in zip(bars1, ca_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.1,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=8, color="#2196F3")
    for bar, val in zip(bars2, asr_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=8, color="#F44336")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title("Attack Overview: Clean Accuracy vs Attack Success Rate", fontsize=12, fontweight="bold")
    ax1.legend(handles=[bars1, bars2], loc="upper left", fontsize=8)
    ax1.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    return fig


def make_asr_defense_chart(rows):
    defended = [r for r in rows if r.get("stage") == "defended"]
    if not defended:
        return None
    attacks = ["badnets", "blended"]
    defenses = ["fine_pruning", "activation_clustering", "spectral_signatures", "neural_cleanse"]
    poisoned_rows = {r["attack"]: float(r["asr"]) for r in rows if r.get("stage") == "poisoned"}
    data = {atk: {} for atk in attacks}
    for r in defended:
        atk = r.get("attack")
        dfn = r.get("defense")
        if atk in data and dfn in defenses:
            data[atk][dfn] = float(r["asr"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
    for ax, atk in zip(axes, attacks):
        pre_asr = poisoned_rows.get(atk, 0.0)
        dfn_labels = [DEFENSE_DISPLAY.get(d, d) for d in defenses]
        dfn_asrs = [data[atk].get(d, float("nan")) for d in defenses]
        bars = ax.bar(dfn_labels, dfn_asrs, color=colors, alpha=0.85, zorder=3)
        ax.axhline(pre_asr, color="red", linestyle="--", linewidth=1.5,
                   label=f"ASR before defence ({pre_asr*100:.1f}%)")
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, label="50% threshold")
        for bar, val in zip(bars, dfn_asrs):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                        f"{val*100:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Attack Success Rate (ASR)")
        ax.set_title(f"{ATTACK_DISPLAY.get(atk, atk)}: ASR after each defence")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.tick_params(axis="x", rotation=15)
    plt.suptitle("Defence Effectiveness: ASR Reduction per Attack", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def make_ca_defense_chart(rows):
    defended = [r for r in rows if r.get("stage") == "defended"]
    if not defended:
        return None
    attacks = ["badnets", "blended"]
    defenses = ["fine_pruning", "activation_clustering", "spectral_signatures", "neural_cleanse"]
    clean_ca = next((float(r["test_ca"]) for r in rows if r.get("attack") == "none"), None)
    data = {atk: {} for atk in attacks}
    for r in defended:
        atk = r.get("attack")
        dfn = r.get("defense")
        if atk in data and dfn in defenses:
            data[atk][dfn] = float(r["test_ca"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
    for ax, atk in zip(axes, attacks):
        dfn_labels = [DEFENSE_DISPLAY.get(d, d) for d in defenses]
        dfn_cas = [data[atk].get(d, float("nan")) for d in defenses]
        bars = ax.bar(dfn_labels, [v * 100 for v in dfn_cas], color=colors, alpha=0.85, zorder=3)
        if clean_ca is not None:
            ax.axhline(clean_ca * 100, color="green", linestyle="--", linewidth=1.5,
                       label=f"Clean baseline CA ({clean_ca*100:.1f}%)")
        ax.axhline(95, color="red", linestyle=":", linewidth=1, label="95% target")
        for bar, val in zip(bars, dfn_cas):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val * 100 + 0.1,
                        f"{val*100:.2f}%", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(93, 100)
        ax.set_ylabel("Clean Accuracy (%)")
        ax.set_title(f"{ATTACK_DISPLAY.get(atk, atk)}: CA after each defence")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.tick_params(axis="x", rotation=15)
    plt.suptitle("Defence Cost: Clean Accuracy After Applying Each Defence", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def make_asr_reduction_chart(rows):
    poisoned_rows = {r["attack"]: float(r["asr"]) for r in rows if r.get("stage") == "poisoned"}
    defended = [r for r in rows if r.get("stage") == "defended"]
    attacks = ["badnets", "blended"]
    defenses = ["fine_pruning", "activation_clustering", "spectral_signatures", "neural_cleanse"]
    data = {atk: {} for atk in attacks}
    for r in defended:
        atk = r.get("attack")
        dfn = r.get("defense")
        if atk in data and dfn in defenses:
            pre = poisoned_rows.get(atk, 0.0)
            post = float(r["asr"])
            data[atk][dfn] = max(0.0, (pre - post) * 100)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(defenses))
    w = 0.35
    colors_atk = {"badnets": "#FF9800", "blended": "#9C27B0"}
    for i, atk in enumerate(attacks):
        vals = [data[atk].get(d, 0.0) for d in defenses]
        offset = (i - 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=ATTACK_DISPLAY.get(atk, atk),
                      color=colors_atk[atk], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                    f"{val:.1f}pp", ha="center", va="bottom", fontsize=8)
    ax.axhline(50, color="red", linestyle="--", linewidth=1, label="50pp target")
    ax.set_xticks(x)
    ax.set_xticklabels([DEFENSE_DISPLAY.get(d, d) for d in defenses], rotation=10)
    ax.set_ylabel("ASR Reduction (percentage points)")
    ax.set_title("ASR Reduction by Defence and Attack (higher is better)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def build_summary(rows):
    clean = next((r for r in rows if r.get("attack") == "none" and r.get("stage") == "clean"), None)
    poisoned = [r for r in rows if r.get("stage") == "poisoned"]
    defended = [r for r in rows if r.get("stage") == "defended"]

    lines = ["### Dataset and baseline"]
    lines.append("GTSRB: 34,799 train / 12,630 test, 43 classes. LeNet CNN, seed 42, 30 epochs.")
    if clean:
        lines.append(f"Clean model test accuracy: **{float(clean['test_ca'])*100:.2f}%**")

    if poisoned:
        lines.append("\n### Attack results")
        for r in poisoned:
            atk = ATTACK_DISPLAY.get(r["attack"], r["attack"])
            lines.append(
                f"- {atk}: CA {float(r['test_ca'])*100:.2f}%,  ASR {float(r['asr'])*100:.2f}%"
            )

    if defended:
        lines.append("\n### Best defence per attack")
        by_attack = {}
        for r in defended:
            atk = r.get("attack")
            asr = float(r["asr"])
            if atk not in by_attack or asr < by_attack[atk][0]:
                by_attack[atk] = (asr, r)
        for atk, (asr, r) in sorted(by_attack.items()):
            dfn = DEFENSE_DISPLAY.get(r["defense"], r["defense"])
            lines.append(
                f"- {ATTACK_DISPLAY.get(atk, atk)}: {dfn} reduced ASR to {asr*100:.2f}%"
                f" (CA {float(r['test_ca'])*100:.2f}%)"
            )

    return "\n".join(lines)


if __name__ == "__main__":
    rows = load_results()
    headers, table_data = make_results_table(rows)
    summary_text = build_summary(rows)

    chart_overview = make_attack_overview_chart(rows)
    chart_asr = make_asr_defense_chart(rows)
    chart_ca = make_ca_defense_chart(rows)
    chart_reduction = make_asr_reduction_chart(rows)

    study_images = [p for p in [
        "results/clean_learning_curve.png",
        "docs/experiments/clean_baseline/class_distribution.png",
        "docs/experiments/clean_baseline/class_imbalance.png",
        "docs/experiments/clean_baseline/trigger_preview.png",
        "docs/experiments/clean_baseline/preprocessing_comparison.png",
        "docs/experiments/clean_baseline/sample_images.png",
    ] if os.path.exists(p)]

    with gr.Blocks(title="Experiment Results") as app:
        gr.Markdown("# HTOTRBAITSR Experiment Results\nGTSRB backdoor attack and defence evaluation.")

        with gr.Tabs():
            with gr.Tab("Summary"):
                gr.Markdown(summary_text)
                gr.Dataframe(
                    value=table_data,
                    headers=headers,
                    label="Full results table",
                )

            with gr.Tab("Attack Overview"):
                gr.Markdown("Clean accuracy vs attack success rate across all attacks.")
                gr.Plot(value=chart_overview)

            with gr.Tab("Defence ASR"):
                gr.Markdown("ASR after applying each defence. Red dashed line = ASR before defence.")
                gr.Plot(value=chart_asr)

            with gr.Tab("Defence CA"):
                gr.Markdown("Clean accuracy after each defence. Should stay near the green baseline.")
                gr.Plot(value=chart_ca)

            with gr.Tab("ASR Reduction"):
                gr.Markdown("How much each defence cut the ASR in percentage points. Higher is better.")
                gr.Plot(value=chart_reduction)

            if study_images:
                with gr.Tab("Study Figures"):
                    for p in study_images:
                        gr.Image(value=p, label=os.path.basename(p))

    app.launch(inbrowser=True)
