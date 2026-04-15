# Interactive demo for HTOTRBAITSR: attack predictions, live defence application, and experiment results
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import csv
import copy
import pickle
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from src.models.lenet import LeNet
from src.attacks.badnets import BadNets
from src.attacks.blended import BlendedInjection
from src.defenses.fine_pruning import FinePruning
from src.defenses.activation_clustering import ActivationClustering
from src.defenses.spectral_signatures import SpectralSignatures
from src.defenses.neural_cleanse import NeuralCleanse


SIGN_NAMES = {
    0: "20 km/h", 1: "30 km/h", 2: "50 km/h", 3: "60 km/h", 4: "70 km/h",
    5: "80 km/h", 6: "End 80 km/h", 7: "100 km/h", 8: "120 km/h", 9: "No passing",
    10: "No passing >3.5t", 11: "Right of way", 12: "Priority road", 13: "Yield",
    14: "Stop", 15: "No vehicles", 16: "No vehicles >3.5t", 17: "No entry",
    18: "General caution", 19: "Curve left", 20: "Curve right", 21: "Double curve",
    22: "Bumpy road", 23: "Slippery road", 24: "Road narrows right", 25: "Road work",
    26: "Traffic signals", 27: "Pedestrians", 28: "Children crossing",
    29: "Bicycles crossing", 30: "Ice/Snow", 31: "Wild animals", 32: "End restrictions",
    33: "Turn right", 34: "Turn left", 35: "Ahead only", 36: "Straight or right",
    37: "Straight or left", 38: "Keep right", 39: "Keep left", 40: "Roundabout",
    41: "End no passing", 42: "End no passing >3.5t",
}

TARGET_CLASS = 14

CHECKPOINTS = {
    "clean": "checkpoints/clean_lenet.pth",
    "badnets": "checkpoints/badnets_lenet.pth",
    "blended": "checkpoints/blended_lenet.pth",
}
ALT_CHECKPOINTS = {
    "clean": "models/clean_baseline.pt",
    "badnets": "models/badnets.pt",
    "blended": "models/blended.pt",
}

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

VAL_SUBSET_PATH = "assets/demo/clean_val_subset.p"
ICON_PATH = "assets/icon.png"


def load_lenet(path):
    model = LeNet(num_classes=43, in_channels=3)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def resolve_checkpoint(key):
    for path in (CHECKPOINTS.get(key), ALT_CHECKPOINTS.get(key)):
        if path and os.path.exists(path):
            return path
    return None


MODELS = {}

def get_model(key):
    if key not in MODELS:
        path = resolve_checkpoint(key)
        if path is None:
            raise FileNotFoundError(
                f"No checkpoint for '{key}'. Expected: {CHECKPOINTS.get(key)} or {ALT_CHECKPOINTS.get(key)}"
            )
        MODELS[key] = load_lenet(path)
    return MODELS[key]


def apply_hist_eq(image: np.ndarray) -> np.ndarray:
    result = np.zeros_like(image)
    for ch in range(image.shape[2]):
        result[:, :, ch] = cv2.equalizeHist(image[:, :, ch])
    return result


def preprocess(image_np: np.ndarray) -> torch.Tensor:
    img = cv2.resize(image_np, (32, 32))
    img = apply_hist_eq(img)
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0)


def predict(model, tensor):
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
    top5 = torch.topk(probs, 5)
    return {
        f"{SIGN_NAMES[i.item()]} (class {i.item()})": float(p)
        for p, i in zip(top5.values, top5.indices)
    }


BADNETS_ATTACK = BadNets(trigger_size=3, trigger_position="bottom_right", trigger_color=(255, 255, 255))
BLENDED_ATTACK = BlendedInjection(alpha=0.15, random_noise=True, seed=42)


def run_demo(uploaded_image, attack_choice):
    if uploaded_image is None:
        return None, None, None, "No image uploaded."

    img_np = np.array(uploaded_image.convert("RGB"))

    if attack_choice == "BadNets (white patch trigger)":
        triggered_np = BADNETS_ATTACK.inject_trigger(img_np.copy())
        attack_key = "badnets"
        trigger_label = "BadNets trigger: 3x3 white pixel patch in bottom right corner"
    else:
        triggered_np = BLENDED_ATTACK.inject_trigger(img_np.copy())
        attack_key = "blended"
        trigger_label = "Blended trigger: alpha=0.15 noise overlay, invisible to human eye"

    clean_tensor = preprocess(img_np)
    triggered_tensor = preprocess(triggered_np)

    try:
        clean_model = get_model("clean")
    except FileNotFoundError as e:
        return None, None, None, f"Error loading clean model: {e}"
    try:
        poisoned_model = get_model(attack_key)
    except FileNotFoundError as e:
        return None, None, None, f"Error loading poisoned model: {e}"

    clean_preds = predict(clean_model, clean_tensor)
    poisoned_preds_clean = predict(poisoned_model, clean_tensor)
    poisoned_preds_triggered = predict(poisoned_model, triggered_tensor)

    triggered_pil = Image.fromarray(triggered_np)

    top_clean = list(clean_preds.keys())[0]
    top_poisoned_triggered = list(poisoned_preds_triggered.keys())[0]
    target_name = SIGN_NAMES[TARGET_CLASS]

    if f"class {TARGET_CLASS}" in top_poisoned_triggered:
        attack_status = f"Attack SUCCEEDED: poisoned model predicts '{target_name}' for triggered input."
    else:
        attack_status = f"Attack did not redirect to '{target_name}' for this image."

    summary = (
        f"Target class: {TARGET_CLASS} ({target_name})\n"
        f"{trigger_label}\n\n"
        f"Clean model on original image:      {top_clean}\n"
        f"Poisoned model on original image:   {list(poisoned_preds_clean.keys())[0]}\n"
        f"Poisoned model on triggered image:  {top_poisoned_triggered}\n\n"
        f"{attack_status}"
    )

    return (
        gr.Label(value=clean_preds, label="Clean model on original image"),
        gr.Label(value=poisoned_preds_triggered, label="Poisoned model on triggered image"),
        triggered_pil,
        summary,
    )


def build_clean_loader():
    if not os.path.exists(VAL_SUBSET_PATH):
        return None
    with open(VAL_SUBSET_PATH, "rb") as f:
        subset = pickle.load(f)
    features = subset["features"]
    labels = subset["labels"]
    tensors = []
    for img in features:
        t = preprocess(img).squeeze(0)
        tensors.append(t)
    x = torch.stack(tensors)
    y = torch.tensor(labels, dtype=torch.long)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=32, shuffle=False)


DEFENSE_MAP = {
    "Fine Pruning": FinePruning,
    "Activation Clustering": ActivationClustering,
    "Spectral Signatures": SpectralSignatures,
    "Neural Cleanse": NeuralCleanse,
}

DEFENDED_MODELS = {}


def run_defense(attack_choice, defense_choice, uploaded_image, progress=gr.Progress()):
    clean_loader = build_clean_loader()
    if clean_loader is None:
        return (
            None, None,
            "Defence unavailable: clean_val_subset.p not found in assets/demo/. "
            "Ask your team lead for the assets/demo/ folder contents."
        )

    if uploaded_image is None:
        return None, None, "Upload an image first, then apply a defence."

    attack_key = "badnets" if "BadNets" in attack_choice else "blended"

    try:
        poisoned_model = get_model(attack_key)
    except FileNotFoundError as e:
        return None, None, f"Error loading poisoned model: {e}"

    cache_key = (attack_key, defense_choice)
    if cache_key not in DEFENDED_MODELS:
        progress(0, desc=f"Applying {defense_choice} to {attack_key} model...")
        def_cls = DEFENSE_MAP[defense_choice]
        if defense_choice == "Fine Pruning":
            defense = def_cls(device="cpu", finetune_epochs=3)
        elif defense_choice == "Neural Cleanse":
            defense = def_cls(device="cpu", num_classes=43, num_steps=100)
        else:
            defense = def_cls(device="cpu")
        progress(0.2, desc="Running defence (this may take 30 to 60 seconds)...")
        defended = defense.apply(copy.deepcopy(poisoned_model), clean_loader)
        defended.eval()
        DEFENDED_MODELS[cache_key] = defended
        progress(0.9, desc="Defence applied, evaluating...")
    else:
        progress(0.9, desc="Using cached defended model...")
        defended = DEFENDED_MODELS[cache_key]

    img_np = np.array(uploaded_image.convert("RGB"))
    attack = BADNETS_ATTACK if attack_key == "badnets" else BLENDED_ATTACK
    triggered_np = attack.inject_trigger(img_np.copy())
    triggered_tensor = preprocess(triggered_np)

    poisoned_preds_triggered = predict(poisoned_model, triggered_tensor)
    defended_preds_triggered = predict(defended, triggered_tensor)

    top_before = list(poisoned_preds_triggered.keys())[0]
    top_after = list(defended_preds_triggered.keys())[0]
    target_name = SIGN_NAMES[TARGET_CLASS]

    if f"class {TARGET_CLASS}" in top_before:
        before_status = "Poisoned model: attack ACTIVE (predicts Stop on triggered input)"
    else:
        before_status = f"Poisoned model: attack not active for this image (predicts {top_before})"

    if f"class {TARGET_CLASS}" in top_after:
        after_status = "Defended model: attack still present after defence"
    else:
        after_status = f"Defended model: attack suppressed, now predicts {top_after}"

    cache_note = "Cached from previous run." if cache_key in DEFENDED_MODELS else "Applied fresh."
    summary = (
        f"Attack: {attack_key}  |  Defence: {defense_choice}\n\n"
        f"{before_status}\n"
        f"{after_status}\n\n"
        f"Defence used 215 image validation subset. {cache_note}\n"
        f"Re-runs of the same attack/defence pair use the cached model instantly."
    )

    progress(1.0, desc="Done.")
    return (
        gr.Label(value=poisoned_preds_triggered, label="Poisoned model on triggered image"),
        gr.Label(value=defended_preds_triggered, label="Defended model on triggered image"),
        summary,
    )


def load_results():
    rows = []
    seen = set()
    for csvpath in ("results/full_pipeline.csv", "results/gtsrb_backdoor.csv", "results/attack_results.csv"):
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


def make_attack_overview_chart():
    rows = load_results()
    attack_rows = {r["attack"]: r for r in rows if r.get("stage") in ("clean", "poisoned")}
    attacks_ordered = ["none", "badnets", "blended", "label_consistent"]
    labels = [ATTACK_DISPLAY.get(a, a) for a in attacks_ordered if a in attack_rows]
    if not labels:
        return None
    ca_vals = [float(attack_rows[a]["test_ca"]) * 100 for a in attacks_ordered if a in attack_rows]
    asr_vals = [float(attack_rows[a]["asr"]) * 100 for a in attacks_ordered if a in attack_rows]
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


def make_asr_defense_chart():
    rows = load_results()
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


def make_ca_defense_chart():
    rows = load_results()
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


def make_asr_reduction_chart():
    rows = load_results()
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


def load_results_tab():
    return (
        make_attack_overview_chart(),
        make_asr_defense_chart(),
        make_ca_defense_chart(),
        make_asr_reduction_chart(),
    )


def build_ui():
    val_available = os.path.exists(VAL_SUBSET_PATH)
    defense_note = (
        "Apply a defence to the poisoned model and compare predictions on the triggered image."
        if val_available else
        "Defence unavailable: clean_val_subset.p not found. "
        "Ask your team lead for the assets/demo/ folder contents."
    )

    icon = ICON_PATH if os.path.exists(ICON_PATH) else None

    with gr.Blocks(title="HTOTRBAITSR Backdoor Demo") as demo:
        gr.Markdown(
            "# Hidden Triggers on the Road: Backdoor Attack Demo\n"
            "PyTorch backdoor attack research on GTSRB (43 class traffic sign recognition). "
            "Models loaded from local checkpoints, no internet required."
        )

        with gr.Tabs():

            with gr.Tab("Attack Demo"):
                gr.Markdown(
                    "Upload a traffic sign image or click an example below. "
                    "The demo applies a trigger and compares clean vs poisoned model predictions."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(type="pil", label="Input traffic sign image", height=220)
                        attack_choice = gr.Radio(
                            choices=["BadNets (white patch trigger)", "Blended (invisible noise trigger)"],
                            value="BadNets (white patch trigger)",
                            label="Select attack",
                        )
                        run_btn = gr.Button("Run Demo", variant="primary")
                    with gr.Column(scale=1):
                        triggered_img = gr.Image(label="Image with trigger applied", height=220)
                        summary_box = gr.Textbox(label="Summary", lines=9)
                with gr.Row():
                    clean_label = gr.Label(num_top_classes=5, label="Clean model on original image")
                    poisoned_label = gr.Label(num_top_classes=5, label="Poisoned model on triggered image")

                demo_dir = "assets/demo"
                sample_paths = []
                if os.path.isdir(demo_dir):
                    featured = [f for f in sorted(os.listdir(demo_dir)) if any(
                        tag in f for tag in ["stop", "yield", "turnright", "noentry", "aheadonly", "keepright", "roadwork"]
                    ) and f.endswith(".png")]
                    sample_paths = [os.path.join(demo_dir, f) for f in featured[:6]]
                if sample_paths:
                    gr.Examples(
                        examples=[[p, "BadNets (white patch trigger)"] for p in sample_paths],
                        inputs=[image_input, attack_choice],
                        label="Sample traffic sign images (click to load)",
                    )
                run_btn.click(
                    fn=run_demo,
                    inputs=[image_input, attack_choice],
                    outputs=[clean_label, poisoned_label, triggered_img, summary_box],
                )

            with gr.Tab("Apply Defence"):
                gr.Markdown(f"### Live Defence Application\n{defense_note}")
                with gr.Row():
                    with gr.Column(scale=1):
                        def_attack_choice = gr.Radio(
                            choices=["BadNets (white patch trigger)", "Blended (invisible noise trigger)"],
                            value="BadNets (white patch trigger)",
                            label="Select attack to defend against",
                        )
                        defense_choice = gr.Radio(
                            choices=list(DEFENSE_MAP.keys()),
                            value="Fine Pruning",
                            label="Select defence",
                        )
                        def_image_input = gr.Image(type="pil", label="Input image (same as Attack Demo tab)", height=180)
                        def_btn = gr.Button(
                            "Apply Defence" if val_available else "Defence unavailable",
                            variant="primary",
                            interactive=val_available,
                        )
                    with gr.Column(scale=1):
                        def_summary = gr.Textbox(label="Defence result", lines=8)
                with gr.Row():
                    before_label = gr.Label(num_top_classes=5, label="Poisoned model on triggered image")
                    after_label = gr.Label(num_top_classes=5, label="Defended model on triggered image")

                if sample_paths:
                    gr.Examples(
                        examples=[[p, "BadNets (white patch trigger)", "Fine Pruning"] for p in sample_paths[:3]],
                        inputs=[def_image_input, def_attack_choice, defense_choice],
                        label="Sample images",
                    )
                def_btn.click(
                    fn=run_defense,
                    inputs=[def_attack_choice, defense_choice, def_image_input],
                    outputs=[before_label, after_label, def_summary],
                    show_progress="full",
                )

            with gr.Tab("Experiment Results"):
                gr.Markdown(
                    "### Results from full pipeline run on Google Colab T4 GPU\n"
                    "GTSRB dataset: 34,799 train / 12,630 test. LeNet CNN, seed 42, 30 epochs.\n\n"
                    "Click **Load Charts** to generate all plots from the results CSVs."
                )
                load_results_btn = gr.Button("Load Charts", variant="primary")
                gr.Markdown("#### Attack Overview: CA vs ASR")
                chart_overview = gr.Plot(label="Attack overview")
                gr.Markdown(
                    "#### Defence Effectiveness: ASR After Each Defence\n"
                    "Red dashed line = ASR before defence. Lower bars = better defence."
                )
                chart_asr = gr.Plot(label="ASR after defence")
                gr.Markdown(
                    "#### ASR Reduction (percentage points)\n"
                    "How much each defence cut the ASR. Higher bars = stronger defence. "
                    "Red dashed line = 50pp target."
                )
                chart_reduction = gr.Plot(label="ASR reduction")
                gr.Markdown(
                    "#### Defence Cost: Clean Accuracy After Each Defence\n"
                    "Should stay near the green baseline. A large drop means the defence "
                    "is hurting the model."
                )
                chart_ca = gr.Plot(label="CA after defence")
                load_results_btn.click(
                    fn=load_results_tab,
                    inputs=[],
                    outputs=[chart_overview, chart_asr, chart_ca, chart_reduction],
                )

    demo.queue()
    return demo


if __name__ == "__main__":
    icon = ICON_PATH if os.path.exists(ICON_PATH) else None
    ui = build_ui()
    ui.launch(share=False, inbrowser=True, favicon_path=icon, theme=gr.themes.Default())
