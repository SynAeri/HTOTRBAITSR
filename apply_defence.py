# standalone gradio app for applying defences to poisoned models and comparing predictions
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import copy
import pickle
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
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
    "badnets": "checkpoints/badnets_lenet.pth",
    "blended": "checkpoints/blended_lenet.pth",
}
ALT_CHECKPOINTS = {
    "badnets": "models/badnets.pt",
    "blended": "models/blended.pt",
}

VAL_SUBSET_PATH = "assets/demo/clean_val_subset.p"

DEFENSE_MAP = {
    "Fine Pruning": (FinePruning, {"device": "cpu", "finetune_epochs": 3}),
    "Activation Clustering": (ActivationClustering, {"device": "cpu"}),
    "Spectral Signatures": (SpectralSignatures, {"device": "cpu"}),
    "Neural Cleanse": (NeuralCleanse, {"device": "cpu", "num_classes": 43, "num_steps": 100}),
}

BADNETS_ATTACK = BadNets(trigger_size=3, trigger_position="bottom_right", trigger_color=(255, 255, 255))
BLENDED_ATTACK = BlendedInjection(alpha=0.15, random_noise=True, seed=42)

MODELS = {}
DEFENDED_MODELS = {}


def resolve_checkpoint(key):
    for path in (CHECKPOINTS.get(key), ALT_CHECKPOINTS.get(key)):
        if path and os.path.exists(path):
            return path
    return None


def get_model(key):
    if key not in MODELS:
        path = resolve_checkpoint(key)
        if path is None:
            raise FileNotFoundError(
                f"No checkpoint for '{key}'. Expected: {CHECKPOINTS.get(key)} or {ALT_CHECKPOINTS.get(key)}"
            )
        model = LeNet(num_classes=43, in_channels=3)
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        model.eval()
        MODELS[key] = model
    return MODELS[key]


def apply_hist_eq(image):
    result = np.zeros_like(image)
    for ch in range(image.shape[2]):
        result[:, :, ch] = cv2.equalizeHist(image[:, :, ch])
    return result


def preprocess(image_np):
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


def build_clean_loader():
    if not os.path.exists(VAL_SUBSET_PATH):
        return None
    with open(VAL_SUBSET_PATH, "rb") as f:
        subset = pickle.load(f)
    tensors = [preprocess(img).squeeze(0) for img in subset["features"]]
    x = torch.stack(tensors)
    y = torch.tensor(subset["labels"], dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=32, shuffle=False)


def run_defence(attack_choice, defence_choice, uploaded_image, progress=gr.Progress()):
    if uploaded_image is None:
        return None, None, None, "Upload an image first."

    clean_loader = build_clean_loader()
    if clean_loader is None:
        return None, None, None, (
            "Defence unavailable: clean_val_subset.p not found in assets/demo/."
        )

    attack_key = "badnets" if "BadNets" in attack_choice else "blended"
    attack = BADNETS_ATTACK if attack_key == "badnets" else BLENDED_ATTACK

    try:
        poisoned_model = get_model(attack_key)
    except FileNotFoundError as e:
        return None, None, None, f"Error loading model: {e}"

    cache_key = (attack_key, defence_choice)
    if cache_key not in DEFENDED_MODELS:
        progress(0, desc=f"Applying {defence_choice} to {attack_key} model...")
        def_cls, def_kwargs = DEFENSE_MAP[defence_choice]
        defence = def_cls(**def_kwargs)
        progress(0.2, desc="Running defence (this may take 30 to 60 seconds)...")
        defended = defence.apply(copy.deepcopy(poisoned_model), clean_loader)
        defended.eval()
        DEFENDED_MODELS[cache_key] = defended
        progress(0.9, desc="Defence applied, evaluating...")
    else:
        progress(0.9, desc="Using cached defended model...")
        defended = DEFENDED_MODELS[cache_key]

    img_np = np.array(uploaded_image.convert("RGB"))
    triggered_np = attack.inject_trigger(img_np.copy())
    triggered_tensor = preprocess(triggered_np)
    triggered_pil = Image.fromarray(triggered_np)

    poisoned_preds = predict(poisoned_model, triggered_tensor)
    defended_preds = predict(defended, triggered_tensor)

    top_before = list(poisoned_preds.keys())[0]
    top_after = list(defended_preds.keys())[0]

    before_status = (
        "Poisoned model: attack ACTIVE (predicts Stop on triggered input)"
        if f"class {TARGET_CLASS}" in top_before
        else f"Poisoned model: attack not active for this image (predicts {top_before})"
    )
    after_status = (
        "Defended model: attack still present after defence"
        if f"class {TARGET_CLASS}" in top_after
        else f"Defended model: attack suppressed, now predicts {top_after}"
    )

    summary = (
        f"Attack: {attack_key}  |  Defence: {defence_choice}\n\n"
        f"{before_status}\n"
        f"{after_status}\n\n"
        f"Defence used 215 image validation subset.\n"
        f"Re-runs of the same attack/defence pair use the cached model instantly."
    )

    progress(1.0, desc="Done.")
    return (
        gr.Label(value=poisoned_preds, label="Poisoned model on triggered image"),
        gr.Label(value=defended_preds, label="Defended model on triggered image"),
        triggered_pil,
        summary,
    )


if __name__ == "__main__":
    val_available = os.path.exists(VAL_SUBSET_PATH)

    demo_dir = "assets/demo"
    sample_paths = []
    if os.path.isdir(demo_dir):
        featured = [
            f for f in sorted(os.listdir(demo_dir))
            if any(tag in f for tag in ["stop", "yield", "turnright", "noentry", "aheadonly", "keepright", "roadwork"])
            and f.endswith(".png")
        ]
        sample_paths = [os.path.join(demo_dir, f) for f in featured[:6]]

    with gr.Blocks(title="Apply Defence") as app:
        gr.Markdown(
            "# Apply Defence\n"
            "Apply a defence to a poisoned model and compare predictions on the triggered image.\n\n"
            + ("" if val_available else
               "> Defence unavailable: `assets/demo/clean_val_subset.p` not found.")
        )

        with gr.Row():
            with gr.Column(scale=1):
                attack_choice = gr.Radio(
                    choices=["BadNets (white patch trigger)", "Blended (invisible noise trigger)"],
                    value="BadNets (white patch trigger)",
                    label="Attack",
                )
                defence_choice = gr.Radio(
                    choices=list(DEFENSE_MAP.keys()),
                    value="Fine Pruning",
                    label="Defence",
                )
                image_input = gr.Image(type="pil", label="Input traffic sign image", height=220)
                run_btn = gr.Button(
                    "Apply Defence" if val_available else "Defence unavailable",
                    variant="primary",
                    interactive=val_available,
                )
            with gr.Column(scale=1):
                triggered_img = gr.Image(label="Image with trigger applied", height=220)
                summary_box = gr.Textbox(label="Result", lines=8)

        with gr.Row():
            before_label = gr.Label(num_top_classes=5, label="Poisoned model on triggered image")
            after_label = gr.Label(num_top_classes=5, label="Defended model on triggered image")

        if sample_paths:
            gr.Examples(
                examples=[[p, "BadNets (white patch trigger)", "Fine Pruning"] for p in sample_paths],
                inputs=[image_input, attack_choice, defence_choice],
                label="Sample traffic sign images (click to load)",
            )

        run_btn.click(
            fn=run_defence,
            inputs=[attack_choice, defence_choice, image_input],
            outputs=[before_label, after_label, triggered_img, summary_box],
            show_progress="full",
        )

    app.queue()
    app.launch(inbrowser=True)
