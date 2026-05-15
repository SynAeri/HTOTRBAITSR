# Interactive demo for HTOTRBAITSR: attack predictions, live defence application, and experiment results
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import pickle
import numpy as np
import cv2
import torch
import torch.nn as nn
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
    "clean": "checkpoints/clean_lenet.pth",
    "badnets": "checkpoints/badnets_lenet.pth",
    "blended": "checkpoints/blended_lenet.pth",
}
ALT_CHECKPOINTS = {
    "clean": "models/clean_baseline.pt",
    "badnets": "models/badnets.pt",
    "blended": "models/blended.pt",
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
        return None, None, None, "No image uploaded.", "Waiting...", "Waiting...", None

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
        return None, None, None, f"Error loading clean model: {e}", "Waiting...", "Waiting...", None
    try:
        poisoned_model = get_model(attack_key)
    except FileNotFoundError as e:
        return None, None, None, f"Error loading poisoned model: {e}", "Waiting...", "Waiting...", None

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

    sim_iframe = build_simulation_html(top_clean, top_poisoned_triggered)
    return (
        gr.Label(value=clean_preds, label="Clean model on original image"),
        gr.Label(value=poisoned_preds_triggered, label="Poisoned model on triggered image"),
        triggered_pil,
        summary,
        top_clean,
        top_poisoned_triggered,
        sim_iframe,
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
    cached = cache_key in DEFENDED_MODELS
    if not cached:
        progress(0, desc=f"Applying {defense_choice} to {attack_key} model...")
        def_cls = DEFENSE_MAP[defense_choice]
        if defense_choice == "Fine Pruning":
            defense = def_cls(device="cpu", finetune_epochs=3)
        elif defense_choice == "Neural Cleanse":
            defense = def_cls(device="cpu", num_classes=43, num_steps=100)
        else:
            defense = def_cls(device="cpu")
        progress(0.2, desc="Running defence (this may take 30 to 60 seconds)...")
        try:
            defended = defense.apply(copy.deepcopy(poisoned_model), clean_loader)
        except Exception as e:
            return None, None, f"Defence failed during apply: {e}"
        defended.eval()
        DEFENDED_MODELS[cache_key] = defended
        progress(0.9, desc="Defence applied, evaluating...")
    else:
        progress(0.9, desc="Using cached defended model...")
        defended = DEFENDED_MODELS[cache_key]

    try:
        attack = BADNETS_ATTACK if attack_key == "badnets" else BLENDED_ATTACK
        img_np = np.array(uploaded_image.convert("RGB"))
        triggered_np = attack.inject_trigger(img_np.copy())
        triggered_tensor = preprocess(triggered_np)
        poisoned_preds_triggered = predict(poisoned_model, triggered_tensor)
        defended_preds_triggered = predict(defended, triggered_tensor)
    except Exception as e:
        return None, None, f"Prediction failed: {e}"

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

    cache_note = "Cached from previous run." if cached else "Applied fresh."
    summary = (
        f"Attack: {attack_key}  |  Defence: {defense_choice}\n\n"
        f"{before_status}\n"
        f"{after_status}\n\n"
        f"Defence used 215 image validation subset. {cache_note}\n"
        f"Re-runs of the same attack/defence pair use the cached model instantly."
    )

    progress(1.0, desc="Done.")
    top_defended = list(defended_preds_triggered.keys())[0]
    return (
        gr.Label(value=poisoned_preds_triggered, label="Poisoned model on triggered image"),
        gr.Label(value=defended_preds_triggered, label="Defended model on triggered image"),
        summary,
        top_defended,
    )




def build_simulation_raw_html(clean_pred, poisoned_pred, defended_pred=None):
    clean_safe    = clean_pred    if clean_pred    else "Waiting..."
    poisoned_safe = poisoned_pred if poisoned_pred else "Waiting..."
    defended_safe = defended_pred if defended_pred else ""
    has_defended  = bool(defended_pred)
    return """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#111827; font-family:monospace; overflow:hidden; }
  #sim_root { position:relative; width:100%; height:540px; }
  #sim_overlay {
    position:absolute; top:0; left:0; width:100%; height:100%;
    pointer-events:none; display:flex; justify-content:space-between; z-index:10;
  }
  .lane_label {
    padding:8px 12px; margin:10px;
    background:rgba(0,0,0,0.52);
    backdrop-filter:blur(4px);
    border-radius:6px;
    color:#fff;
    align-self:flex-start;
  }
  .lane_title { font-size:0.95em; font-weight:bold; letter-spacing:0.04em; }
  .lane_pred  { font-size:0.78em; margin-top:3px; opacity:0.92; }
  #sim_canvas { display:block; width:100%; height:490px; }
  #sim_controls {
    position:absolute; bottom:0; left:0; width:100%;
    padding:6px 12px; background:rgba(0,0,0,0.65);
    display:flex; align-items:center; gap:10px; z-index:11;
  }
  .sim_btn {
    padding:5px 16px; border:none; border-radius:4px;
    cursor:pointer; font-size:0.85em; font-weight:bold;
  }
  #sim_play_btn  { background:#2563eb; color:#fff; }
  #sim_reset_btn { background:#374151; color:#ddd; }
  #sim_status    { color:#9ca3af; font-size:0.8em; flex:1; }
</style>
<script type="importmap">
{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js"}}
</script>
</head>
<body>
<div id="sim_root">
  <div id="sim_overlay">
    <div class="lane_label" style="flex:1;">
      <div class="lane_title" style="color:#88aaff;">Clean Model</div>
      <div class="lane_pred" id="pred_clean" style="color:#ccddff;">""" + clean_safe + """</div>
    </div>
    <div class="lane_label" id="defended_label_box" style="flex:1;text-align:center;""" + ("display:none;" if not has_defended else "") + """">
      <div class="lane_title" style="color:#88ffaa;">Defended Model</div>
      <div class="lane_pred" id="pred_defended" style="color:#ccffdd;">""" + defended_safe + """</div>
    </div>
    <div class="lane_label" style="flex:1;text-align:right;">
      <div class="lane_title" style="color:#ff8866;">Poisoned Model</div>
      <div class="lane_pred" id="pred_poisoned" style="color:#ffccaa;">""" + poisoned_safe + """</div>
    </div>
  </div>
  <canvas id="sim_canvas"></canvas>
  <div id="sim_controls">
    <button class="sim_btn" id="sim_play_btn">Play</button>
    <button class="sim_btn" id="sim_reset_btn">Reset</button>
    <span id="sim_status">Press Play to start</span>
    <button class="sim_btn" id="sim_fs_btn" style="background:#1f2937;color:#9ca3af;margin-left:auto;">Fullscreen</button>
  </div>
</div>
<script type="module">
import * as THREE from 'three';

const CLEAN_PRED    = \"""" + clean_safe.replace("\\", "\\\\").replace('"', '\\"') + """\";
const POISONED_PRED = \"""" + poisoned_safe.replace("\\", "\\\\").replace('"', '\\"') + """\";
const DEFENDED_PRED = \"""" + defended_safe.replace("\\", "\\\\").replace('"', '\\"') + """\";
const HAS_DEFENDED  = """ + ("true" if has_defended else "false") + """;
const TARGET_CLASS_ID = 14;
// clean model: unpoisoned, always passes through freely
// poisoned model: stops mid-junction if the backdoor triggered (predicts Stop on the triggered image)
//   if the attack was weak and it predicted something else, it passes through freely instead
// defended model: follows its actual prediction
const cleanShouldStop    = false;
const poisonedShouldStop = POISONED_PRED.includes('class ' + TARGET_CLASS_ID);
const defendedShouldStop = HAS_DEFENDED ? DEFENDED_PRED.includes('class ' + TARGET_CLASS_ID) : false;

// sign face colour by predicted class
function signColor(pred) {
  const m = pred.match(/class (\\d+)/);
  const cls = m ? parseInt(m[1]) : -1;
  if (cls === 14) return 0xcc0000;
  if (cls === 13) return 0xeecc00;
  if (cls >= 0 && cls <= 8) return 0x1155cc;
  if (cls === 17) return 0xcc0000;
  return 0x2288ee;
}

const predCleanEl    = document.getElementById('pred_clean');
const predPoisonedEl = document.getElementById('pred_poisoned');
const predDefendedEl = document.getElementById('pred_defended');
const statusEl       = document.getElementById('sim_status');
const playBtn        = document.getElementById('sim_play_btn');
const resetBtn       = document.getElementById('sim_reset_btn');
const canvas         = document.getElementById('sim_canvas');

let renderer, scene, meshes, rafId, initialized = false, running = false, loopActive = false;
let camMain, camClean, camPoisoned, camDefended;

// T-junction layout constants
// vertical road: cars approach from +z toward z=0 junction
// horizontal road spans z = -4.5 to +4.5; cross traffic travels at z=+2 (near lane)
const STOP_Z      = 2.0;   // poisoned car stops here — inside the junction, in cross-traffic path
const JUNCTION_Z  = 2.0;   // cross-traffic travels at this z
const START_Z     = 32;    // cars start here
const CROSS_START =  30;   // cross-traffic car parked position (right side, off screen)
const CROSS_LAUNCH = 12;   // cross-traffic car x when launched (comes from right, hits poisoned lane first)
const CROSS_END   = -30;   // cross-traffic car exits here (left side)
const CROSS_SPEED = 0.22;

// lane x positions for approaching cars
const CLEAN_X    = -2.0;
const POISONED_X =  2.0;
const DEFENDED_X =  0.0;

function updateChaseCams(m) {
  // clean cam: behind and slightly above, looking at where the car is going (toward junction)
  const cleanFwd = new THREE.Vector3(CLEAN_X, 0.8, m.cleanCar.position.z - 8);
  camClean.position.set(CLEAN_X, 4, m.cleanCar.position.z + 9);
  camClean.lookAt(cleanFwd);

  const poisonedFwd = new THREE.Vector3(POISONED_X, 0.8, m.poisonedCar.position.z - 8);
  camPoisoned.position.set(POISONED_X, 4, m.poisonedCar.position.z + 9);
  camPoisoned.lookAt(poisonedFwd);

  if (HAS_DEFENDED && m.defendedCar) {
    const defendedFwd = new THREE.Vector3(DEFENDED_X, 0.8, m.defendedCar.position.z - 8);
    camDefended.position.set(DEFENDED_X, 4, m.defendedCar.position.z + 9);
    camDefended.lookAt(defendedFwd);
  }
}

function renderViewports(w, h) {
  renderer.clear();
  renderer.setScissorTest(true);

  const pipW = Math.floor(w * 0.26);
  const pipH = Math.floor(pipW * 0.72);
  const pipX = w - pipW - 4;
  const pipY = 4;

  if (HAS_DEFENDED) {
    const third = Math.floor(w / 3);
    renderer.setViewport(0, 0, third, h);
    renderer.setScissor(0, 0, third, h);
    camClean.aspect = third / h;
    camClean.updateProjectionMatrix();
    renderer.render(scene, camClean);

    renderer.setViewport(third, 0, third, h);
    renderer.setScissor(third, 0, third, h);
    camDefended.aspect = third / h;
    camDefended.updateProjectionMatrix();
    renderer.render(scene, camDefended);

    renderer.setViewport(third * 2, 0, third, h);
    renderer.setScissor(third * 2, 0, third, h);
    camPoisoned.aspect = third / h;
    camPoisoned.updateProjectionMatrix();
    renderer.render(scene, camPoisoned);
  } else {
    const halfW = Math.floor(w / 2);
    renderer.setViewport(0, 0, halfW, h);
    renderer.setScissor(0, 0, halfW, h);
    camClean.aspect = halfW / h;
    camClean.updateProjectionMatrix();
    renderer.render(scene, camClean);

    renderer.setViewport(halfW, 0, halfW, h);
    renderer.setScissor(halfW, 0, halfW, h);
    camPoisoned.aspect = halfW / h;
    camPoisoned.updateProjectionMatrix();
    renderer.render(scene, camPoisoned);
  }

  // PiP: top-down overview of the junction
  renderer.setViewport(pipX, h - pipH - pipY, pipW, pipH);
  renderer.setScissor(pipX, h - pipH - pipY, pipW, pipH);
  camMain.aspect = pipW / pipH;
  camMain.updateProjectionMatrix();
  renderer.render(scene, camMain);

  renderer.setScissorTest(false);
}

function animate() {
  if (!running) return;
  tick(simState, meshes);
  updateChaseCams(meshes);
  const w = canvas.clientWidth || 800;
  const h = canvas.clientHeight || 490;
  renderViewports(w, h);
  if (simState.phase === 'done') {
    running = false;
    loopActive = false;
    return;
  }
  rafId = requestAnimationFrame(animate);
}

const simState = {
  phase: 'idle',
  cleanZ: START_Z, poisonedZ: START_Z + 2, defendedZ: START_Z + 4,
  cleanSpeed: 0.14, poisonedSpeed: 0.14, defendedSpeed: 0.14,
  cleanBraking: false, poisonedBraking: false, defendedBraking: false,
  cleanStopped: false, poisonedStopped: false, defendedStopped: false,
  crossX: CROSS_START,
  crossLaunched: false,
  poisonedCollided: false, defendedCollided: false,
  poisonedColTimer: 0, defendedColTimer: 0,
  doneTimer: 0,
};

function buildScene(sc) {
  const mat = (c) => new THREE.MeshLambertMaterial({ color: c });

  // vertical road (approaching road) — narrower, only one direction
  const vertRoad = new THREE.Mesh(new THREE.PlaneGeometry(9, 60), mat(0x444444));
  vertRoad.rotation.x = -Math.PI / 2;
  vertRoad.position.set(0, 0, 16);
  sc.add(vertRoad);

  // horizontal road (cross traffic) — extends left and right from junction
  const horizRoad = new THREE.Mesh(new THREE.PlaneGeometry(60, 9), mat(0x444444));
  horizRoad.rotation.x = -Math.PI / 2;
  horizRoad.position.set(0, 0.001, 0);
  sc.add(horizRoad);

  // junction fill (slightly raised so no z-fighting seam)
  const junctionFill = new THREE.Mesh(new THREE.PlaneGeometry(9, 9), mat(0x444444));
  junctionFill.rotation.x = -Math.PI / 2;
  junctionFill.position.set(0, 0.002, 0);
  sc.add(junctionFill);

  // kerb / pavement corners (give depth to the T junction)
  const kerbMat = mat(0x888866);
  [[-5.5, 0, 5.5], [5.5, 0, 5.5]].forEach(([x, , z]) => {
    const k = new THREE.Mesh(new THREE.BoxGeometry(2, 0.12, 2), kerbMat);
    k.position.set(x, 0.06, z);
    sc.add(k);
  });

  // centre lane dashes on vertical road
  for (let z = 5; z <= 30; z += 5) {
    const dash = new THREE.Mesh(new THREE.BoxGeometry(0.15, 0.05, 1.8), mat(0xffffff));
    dash.position.set(0, 0.03, z);
    sc.add(dash);
  }

  // centre lane dashes on horizontal road
  for (let x = -26; x <= 26; x += 5) {
    if (Math.abs(x) < 5) continue; // skip junction area
    const dash = new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.05, 0.15), mat(0xffffff));
    dash.position.set(x, 0.03, 0);
    sc.add(dash);
  }

  // road edges vertical
  [-4.5, 4.5].forEach(x => {
    const edge = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.05, 50), mat(0xffffff));
    edge.position.set(x, 0.03, 18);
    sc.add(edge);
  });

  // road edges horizontal (only outside junction)
  [-4.5, 4.5].forEach(z => {
    [-1, 1].forEach(side => {
      const edge = new THREE.Mesh(new THREE.BoxGeometry(22, 0.05, 0.2), mat(0xffffff));
      edge.position.set(side * 19, 0.03, z);
      sc.add(edge);
    });
  });

  // stop line at the road edge — where cars *should* stop
  const SIGN_Z = 5.5;
  const stopLine = new THREE.Mesh(new THREE.BoxGeometry(9, 0.06, 0.3), mat(0xffffff));
  stopLine.position.set(0, 0.04, SIGN_Z);
  sc.add(stopLine);

  // stop sign — right side of approach
  const signFaceCol = signColor(CLEAN_PRED);
  const postGeo = new THREE.CylinderGeometry(0.07, 0.07, 2.8, 8);
  const post = new THREE.Mesh(postGeo, mat(0xaaaaaa));
  post.position.set(5.2, 1.4, SIGN_Z + 0.8);
  sc.add(post);
  const face = new THREE.Mesh(new THREE.BoxGeometry(1.1, 1.1, 0.08), mat(signFaceCol));
  face.position.set(5.2, 3.0, SIGN_Z + 0.8);
  sc.add(face);
  const barH = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.12, 0.1), mat(0xffffff));
  barH.position.set(5.2, 3.0, SIGN_Z + 0.75);
  sc.add(barH);
  const barV = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.8, 0.1), mat(0xffffff));
  barV.position.set(5.2, 3.0, SIGN_Z + 0.75);
  sc.add(barV);

  // buildings / scenery blocks beside the approach road
  const buildingData = [
    [-8, 0, 12, 4, 5, 4, 0x8899aa],
    [ 8, 0, 18, 3, 4, 5, 0x99aa88],
    [-9, 0, 22, 5, 6, 4, 0xaa9988],
    [ 7, 0,  8, 4, 3, 5, 0x8888aa],
    [-14, 0, 0,  3, 4, 18, 0x888899],
    [ 14, 0, 0,  3, 4, 18, 0x888899],
  ];
  buildingData.forEach(([x, , z, w, h, d, col]) => {
    const b = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat(col));
    b.position.set(x, h / 2, z);
    sc.add(b);
  });

  const makeCarGroup = (bodyColor, roofColor) => {
    const g = new THREE.Group();
    const body = new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.7, 3.5), mat(bodyColor));
    body.position.y = 0.35;
    g.add(body);
    const roof = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.5, 2.0), mat(roofColor));
    roof.position.set(0, 0.95, -0.2);
    g.add(roof);
    return g;
  };

  // approaching cars — face toward junction (no rotation needed, default faces +z)
  const cleanCar    = makeCarGroup(0x2255cc, 0x1133aa);
  const poisonedCar = makeCarGroup(0xcc3311, 0xaa2200);
  cleanCar.position.set(CLEAN_X, 0, START_Z);
  poisonedCar.position.set(POISONED_X, 0, START_Z + 2);
  sc.add(cleanCar);
  sc.add(poisonedCar);

  // cross-traffic car — travels right-to-left (-x direction) at z=JUNCTION_Z
  const crossCar = makeCarGroup(0xddaa22, 0xbb8800);
  crossCar.rotation.y = Math.PI / 2;
  crossCar.position.set(CROSS_START, 0, JUNCTION_Z);
  sc.add(crossCar);

  let defendedCar = null;
  if (HAS_DEFENDED) {
    defendedCar = makeCarGroup(0x22aa55, 0x117733);
    defendedCar.position.set(DEFENDED_X, 0, START_Z + 4);
    sc.add(defendedCar);
  }

  return { cleanCar, poisonedCar, crossCar, defendedCar };
}

function resetSim(state, m) {
  state.phase = 'idle';
  state.cleanZ    = START_Z;
  state.poisonedZ = START_Z + 2;
  state.defendedZ = START_Z + 4;
  state.cleanSpeed = 0.14; state.poisonedSpeed = 0.14; state.defendedSpeed = 0.14;
  state.cleanBraking    = false; state.poisonedBraking    = false; state.defendedBraking    = false;
  state.cleanStopped    = false; state.poisonedStopped    = false; state.defendedStopped    = false;
  state.crossX = CROSS_START;
  state.crossLaunched = false;
  state.poisonedCollided = false; state.defendedCollided = false;
  state.poisonedColTimer = 0;     state.defendedColTimer = 0;
  state.doneTimer = 0;

  m.cleanCar.position.set(CLEAN_X, 0, START_Z);
  m.cleanCar.rotation.set(0, 0, 0);
  m.poisonedCar.position.set(POISONED_X, 0, START_Z + 2);
  m.poisonedCar.rotation.set(0, 0, 0);
  m.crossCar.position.set(CROSS_START, 0, JUNCTION_Z);
  m.crossCar.rotation.set(0, Math.PI / 2, 0);
  if (m.defendedCar) {
    m.defendedCar.position.set(DEFENDED_X, 0, START_Z + 4);
    m.defendedCar.rotation.set(0, 0, 0);
  }

  predCleanEl.textContent    = CLEAN_PRED;
  predPoisonedEl.textContent = POISONED_PRED;
  if (predDefendedEl) predDefendedEl.textContent = DEFENDED_PRED;
  statusEl.textContent = 'Press Play to start';
}

function applyCollision(car, timerVal, xDir) {
  car.rotation.y += 0.055;
  car.position.x += xDir * 0.05;
  car.position.y = Math.max(0, Math.sin(timerVal * 0.11) * 0.35);
}

function tick(state, m) {
  if (state.phase === 'idle') return;

  // launch cross-traffic once the poisoned car is stationary inside the junction
  // (or any defended car that also stops mid-junction)
  const stoppedInJunction = (state.poisonedStopped && state.poisonedZ <= JUNCTION_Z + 1.5) ||
                            (HAS_DEFENDED && state.defendedStopped && state.defendedZ <= JUNCTION_Z + 1.5);
  if (stoppedInJunction && !state.crossLaunched) {
    state.crossLaunched = true;
    state.crossX = CROSS_LAUNCH;
    m.crossCar.position.x = state.crossX;
  }

  // advance cross-traffic only once launched; stay parked off-screen otherwise
  // cross car travels right-to-left (negative x direction)
  if (state.crossLaunched && state.crossX > CROSS_END) {
    state.crossX -= CROSS_SPEED;
    m.crossCar.position.x = state.crossX;
  }

  // advance clean car
  if (!state.cleanStopped && !state.cleanCollided) {
    if (state.cleanBraking) {
      state.cleanSpeed = Math.max(0, state.cleanSpeed - 0.012);
      if (state.cleanSpeed === 0) state.cleanStopped = true;
    }
    state.cleanZ -= state.cleanSpeed;
    if (state.cleanZ < JUNCTION_Z - 12) state.cleanZ = JUNCTION_Z - 12;
    m.cleanCar.position.z = state.cleanZ;
  }

  // advance poisoned car
  if (!state.poisonedStopped && !state.poisonedCollided) {
    if (state.poisonedBraking) {
      state.poisonedSpeed = Math.max(0, state.poisonedSpeed - 0.012);
      if (state.poisonedSpeed === 0) state.poisonedStopped = true;
    }
    state.poisonedZ -= state.poisonedSpeed;
    if (state.poisonedZ < JUNCTION_Z - 12) state.poisonedZ = JUNCTION_Z - 12;
    m.poisonedCar.position.z = state.poisonedZ;
  }

  // advance defended car
  if (HAS_DEFENDED && m.defendedCar && !state.defendedStopped && !state.defendedCollided) {
    if (state.defendedBraking) {
      state.defendedSpeed = Math.max(0, state.defendedSpeed - 0.012);
      if (state.defendedSpeed === 0) state.defendedStopped = true;
    }
    state.defendedZ -= state.defendedSpeed;
    if (state.defendedZ < JUNCTION_Z - 12) state.defendedZ = JUNCTION_Z - 12;
    m.defendedCar.position.z = state.defendedZ;
  }

  // braking triggers:
  // decel = 0.012/frame, speed = 0.14 → stopping distance = v²/2a = 0.82 units
  // so braking starts at STOP_Z + 0.82 ≈ 7.3, car halts right at STOP_Z (6.5)
  const BRAKE_START = STOP_Z + 0.9;
  if (state.cleanZ <= BRAKE_START + 5 && state.cleanZ > BRAKE_START) {
    predCleanEl.textContent = cleanShouldStop
      ? CLEAN_PRED + ' - Stopping...'
      : CLEAN_PRED + ' - Continuing...';
  }
  if (state.poisonedZ <= BRAKE_START + 5 && state.poisonedZ > BRAKE_START) {
    predPoisonedEl.textContent = poisonedShouldStop
      ? POISONED_PRED + ' - Stopping...'
      : POISONED_PRED + ' - Continuing...';
  }

  if (state.cleanZ <= BRAKE_START && !state.cleanBraking && !state.cleanStopped) {
    if (cleanShouldStop) state.cleanBraking = true;
  }
  if (state.poisonedZ <= BRAKE_START && !state.poisonedBraking && !state.poisonedStopped) {
    if (poisonedShouldStop) state.poisonedBraking = true;
  }
  if (HAS_DEFENDED && state.defendedZ <= BRAKE_START && !state.defendedBraking && !state.defendedStopped) {
    if (defendedShouldStop) state.defendedBraking = true;
    if (predDefendedEl) predDefendedEl.textContent = defendedShouldStop
      ? DEFENDED_PRED + ' - Stopping...'
      : DEFENDED_PRED + ' - Continuing...';
  }

  // status label updates when stopped
  if (state.poisonedStopped && !state.poisonedCollided) {
    predPoisonedEl.textContent = POISONED_PRED + ' - Stopped mid-junction!';
  }

  // collision: cross car hits the poisoned car (stopped mid-junction)
  // clean car is never checked — it passes through freely before cross-traffic arrives
  if (!state.poisonedCollided && state.poisonedStopped
      && Math.abs(state.crossX - POISONED_X) < 3.2) {
    state.poisonedCollided = true;
    predPoisonedEl.textContent = POISONED_PRED + ' - T-BONE COLLISION!';
  }
  if (HAS_DEFENDED && !state.defendedCollided && state.defendedStopped
      && state.defendedZ <= JUNCTION_Z + 1.5
      && Math.abs(state.crossX - DEFENDED_X) < 3.2) {
    state.defendedCollided = true;
    if (predDefendedEl) predDefendedEl.textContent = DEFENDED_PRED + ' - COLLISION!';
  }

  // apply collision spin effect
  if (state.poisonedCollided) {
    state.poisonedColTimer++;
    applyCollision(m.poisonedCar, state.poisonedColTimer, 1);
    state.poisonedSpeed = Math.max(0, state.poisonedSpeed - 0.008);
  }
  if (HAS_DEFENDED && state.defendedCollided && m.defendedCar) {
    state.defendedColTimer++;
    applyCollision(m.defendedCar, state.defendedColTimer, 0);
    state.defendedSpeed = Math.max(0, state.defendedSpeed - 0.008);
  }

  // done check — clean passes through freely, poisoned gets hit
  const cleanDone    = state.cleanZ < JUNCTION_Z - 10;
  const poisonedDone = (state.poisonedCollided && state.poisonedColTimer > 80) || state.poisonedZ < JUNCTION_Z - 10;
  const defendedDone = !HAS_DEFENDED || (state.defendedCollided && state.defendedColTimer > 80) || state.defendedStopped || state.defendedZ < JUNCTION_Z - 10;

  if (cleanDone && poisonedDone && defendedDone) {
    state.doneTimer++;
    if (state.doneTimer > 60) {
      const cleanMsg    = state.cleanCollided ? 'collision' : 'passed through safely';
      const poisonedMsg = state.poisonedCollided ? 'T-BONE collision (stopped mid-junction)' : 'passed through';
      let msg = `Clean: ${cleanMsg}. Poisoned: ${poisonedMsg}.`;
      if (HAS_DEFENDED) {
        const defendedMsg = state.defendedCollided ? 'T-BONE collision' : (state.defendedStopped ? 'stopped mid-junction' : 'passed through');
        msg += ` Defended: ${defendedMsg}.`;
      }
      statusEl.textContent = msg;
      state.phase = 'done';
    }
  }
}

function init() {
  const w = canvas.clientWidth || 800;
  const h = canvas.clientHeight || 490;
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(w, h, false);
  renderer.setClearColor(0x87ceeb);
  renderer.autoClear = false;

  // top-down overview camera — looks straight down at the junction
  camMain = new THREE.PerspectiveCamera(60, w / h, 0.1, 300);
  camMain.position.set(0, 28, 8);
  camMain.lookAt(0, 0, 4);

  // chase cameras — updated per frame to follow each car
  camClean    = new THREE.PerspectiveCamera(62, 1, 0.1, 300);
  camPoisoned = new THREE.PerspectiveCamera(62, 1, 0.1, 300);
  camDefended = new THREE.PerspectiveCamera(62, 1, 0.1, 300);

  scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x87ceeb, 40, 90);
  scene.add(new THREE.AmbientLight(0x999999));
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
  dirLight.position.set(8, 20, 12);
  scene.add(dirLight);

  meshes = buildScene(scene);
  resetSim(simState, meshes);
  updateChaseCams(meshes);

  const ro = new ResizeObserver(entries => {
    for (const e of entries) {
      const { width, height } = e.contentRect;
      if (width > 0 && height > 0) {
        renderer.setSize(width, height, false);
        renderViewports(width, height);
      }
    }
  });
  ro.observe(canvas);

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) { running = false; }
    else if (loopActive) { running = true; }
  });
  initialized = true;

  function idleRender() {
    requestAnimationFrame(idleRender);
    if (simState.phase === 'idle') {
      renderViewports(canvas.clientWidth || 800, canvas.clientHeight || 490);
    }
  }
  idleRender();
}

playBtn.addEventListener('click', () => {
  if (!initialized) return;
  if (simState.phase === 'idle') {
    simState.phase = 'driving';
    loopActive = true;
    running = true;
    animate();
  }
});

resetBtn.addEventListener('click', () => {
  if (!initialized) return;
  running = false;
  loopActive = false;
  cancelAnimationFrame(rafId);
  resetSim(simState, meshes);
  updateChaseCams(meshes);
  renderViewports(canvas.clientWidth || 800, canvas.clientHeight || 490);
});

const fsBtn = document.getElementById('sim_fs_btn');
fsBtn.addEventListener('click', () => {
  const el = document.getElementById('sim_root');
  if (!document.fullscreenElement) {
    el.requestFullscreen().catch(() => {});
    fsBtn.textContent = 'Exit Fullscreen';
  } else {
    document.exitFullscreen();
    fsBtn.textContent = 'Fullscreen';
  }
});
document.addEventListener('fullscreenchange', () => {
  if (!document.fullscreenElement) fsBtn.textContent = 'Fullscreen';
});

setTimeout(init, 150);
</script>
</body>
</html>"""


def build_simulation_html(clean_pred, poisoned_pred, defended_pred=None):
    import html as _html
    raw = build_simulation_raw_html(clean_pred, poisoned_pred, defended_pred)
    escaped = _html.escape(raw, quote=True)
    return f'<iframe srcdoc="{escaped}" style="width:100%;height:560px;border:none;border-radius:6px;"></iframe>'


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
        clean_pred_state    = gr.State(value="Waiting...")
        poisoned_pred_state = gr.State(value="Waiting...")
        defended_pred_state = gr.State(value="")

        gr.Markdown(
            "# Hidden Triggers on the Road\n"
            "Backdoor attack and defence research on GTSRB traffic sign recognition."
        )

        demo_dir = "assets/demo"
        sample_paths = []
        if os.path.isdir(demo_dir):
            featured = [f for f in sorted(os.listdir(demo_dir)) if any(
                tag in f for tag in ["stop", "yield", "turnright", "noentry", "aheadonly", "keepright", "roadwork"]
            ) and f.endswith(".png")]
            sample_paths = [os.path.join(demo_dir, f) for f in featured[:6]]

        # ── shared image + mode ─────────────────────────────────────────────
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                image_input = gr.Image(type="pil", label="Traffic sign image", height=200)
                if sample_paths:
                    gr.Examples(
                        examples=[p for p in sample_paths],
                        inputs=[image_input],
                        label="Sample images",
                    )
            with gr.Column(scale=2, min_width=200):
                mode_toggle = gr.Radio(
                    choices=["Attack Demo", "Apply Defence"],
                    value="Attack Demo",
                    label="Mode",
                    interactive=True,
                )

        # ── attack panel ────────────────────────────────────────────────────
        with gr.Group(visible=True) as attack_panel:
            with gr.Row():
                with gr.Column(scale=1):
                    attack_choice = gr.Radio(
                        choices=["BadNets (white patch trigger)", "Blended (invisible noise trigger)"],
                        value="BadNets (white patch trigger)",
                        label="Attack",
                    )
                    run_btn = gr.Button("Run Attack Demo", variant="primary")
                with gr.Column(scale=1):
                    triggered_img = gr.Image(label="Image with trigger applied", height=200)
                    summary_box = gr.Textbox(label="Result", lines=4)
            with gr.Row():
                clean_label = gr.Label(num_top_classes=5, label="Clean model")
                poisoned_label = gr.Label(num_top_classes=5, label="Poisoned model")

        # ── defence panel ───────────────────────────────────────────────────
        with gr.Group(visible=False) as defence_panel:
            if not val_available:
                gr.Markdown(f"> {defense_note}")
            with gr.Row():
                with gr.Column(scale=1):
                    def_attack_choice = gr.Radio(
                        choices=["BadNets (white patch trigger)", "Blended (invisible noise trigger)"],
                        value="BadNets (white patch trigger)",
                        label="Attack to defend against",
                    )
                    defense_choice = gr.Radio(
                        choices=list(DEFENSE_MAP.keys()),
                        value="Fine Pruning",
                        label="Defence method",
                    )
                    def_btn = gr.Button(
                        "Apply Defence" if val_available else "Defence unavailable",
                        variant="primary",
                        interactive=val_available,
                    )
                with gr.Column(scale=1):
                    def_summary = gr.Textbox(label="Result", lines=6)
            with gr.Row():
                before_label = gr.Label(num_top_classes=5, label="Poisoned model")
                after_label = gr.Label(num_top_classes=5, label="Defended model")

        # ── simulation ──────────────────────────────────────────────────────
        gr.Markdown("---\n### Driving Simulation")
        gr.Markdown(
            "Run the Attack Demo first to load the scene. "
            "Apply a Defence to add the defended car (centre lane). "
            "Press **Play** inside the scene to start."
        )
        sim_html = gr.HTML(
            value="<p style='color:#888;padding:8px 0;font-family:monospace;font-size:0.85em;'>Run the Attack Demo above to load the simulation.</p>"
        )

        # ── wiring ──────────────────────────────────────────────────────────
        def switch_mode(mode):
            return gr.update(visible=mode == "Attack Demo"), gr.update(visible=mode == "Apply Defence")

        mode_toggle.change(fn=switch_mode, inputs=[mode_toggle], outputs=[attack_panel, defence_panel])

        run_btn.click(
            fn=run_demo,
            inputs=[image_input, attack_choice],
            outputs=[clean_label, poisoned_label, triggered_img, summary_box,
                     clean_pred_state, poisoned_pred_state, sim_html],
        )

        def rebuild_sim_with_defence(top_defended, clean_pred, poisoned_pred):
            return build_simulation_html(clean_pred, poisoned_pred, top_defended)

        def_btn.click(
            fn=run_defense,
            inputs=[def_attack_choice, defense_choice, image_input],
            outputs=[before_label, after_label, def_summary, defended_pred_state],
            show_progress="full",
        ).then(
            fn=rebuild_sim_with_defence,
            inputs=[defended_pred_state, clean_pred_state, poisoned_pred_state],
            outputs=[sim_html],
        )

    demo.queue()
    return demo


if __name__ == "__main__":
    icon = ICON_PATH if os.path.exists(ICON_PATH) else None
    ui = build_ui()
    ui.launch(share=True, inbrowser=not os.environ.get("GRADIO_NO_BROWSER"), favicon_path=icon, theme=gr.themes.Default())
