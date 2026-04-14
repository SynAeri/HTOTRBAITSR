# Extracts sample traffic sign images from GTSRB and saves them as demo PNGs for the Gradio demo
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
from PIL import Image

SIGN_NAMES = {
    0: "20kmh", 1: "30kmh", 2: "50kmh", 3: "60kmh", 4: "70kmh",
    5: "80kmh", 6: "end80kmh", 7: "100kmh", 8: "120kmh", 9: "nopassing",
    10: "nopassing35t", 11: "rightofway", 12: "priorityroad", 13: "yield",
    14: "stop", 15: "novehicles", 16: "novehicles35t", 17: "noentry",
    18: "caution", 19: "curveleft", 20: "curveright", 21: "doublecurve",
    22: "bumpyroad", 23: "slippery", 24: "roadnarrows", 25: "roadwork",
    26: "trafficsignals", 27: "pedestrians", 28: "childrencrossing",
    29: "bicycles", 30: "icesnow", 31: "wildanimals", 32: "endrestrictions",
    33: "turnright", 34: "turnleft", 35: "aheadonly", 36: "straightright",
    37: "straightleft", 38: "keepright", 39: "keepleft", 40: "roundabout",
    41: "endnopassing", 42: "endnopassing35t",
}

TARGET_CLASSES = [0, 1, 2, 4, 9, 13, 14, 17, 25, 33, 35, 38]

OUT = "assets/demo"
os.makedirs(OUT, exist_ok=True)

with open("data/raw/train.p", "rb") as f:
    train = pickle.load(f)

features = train["features"]
labels = train["labels"]

rng = np.random.RandomState(42)

saved = []
for cls in TARGET_CLASSES:
    indices = np.where(labels == cls)[0]
    chosen = rng.choice(indices, size=min(3, len(indices)), replace=False)
    for i, idx in enumerate(chosen):
        img = features[idx]
        upscaled = np.array(Image.fromarray(img).resize((128, 128), Image.NEAREST))
        fname = f"class{cls:02d}_{SIGN_NAMES[cls]}_{i}.png"
        path = os.path.join(OUT, fname)
        Image.fromarray(upscaled).save(path)
        saved.append(fname)
        print(f"  Saved {fname}")

print(f"\nTotal: {len(saved)} images saved to {OUT}/")
