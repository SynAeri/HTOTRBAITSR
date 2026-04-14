# Generates dataset analysis figures: class distribution, sample images, trigger previews
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

from src.attacks.badnets import BadNets
from src.attacks.blended import BlendedInjection
from src.data.loader import GTSRBDataset

OUT = "docs/experiments/clean_baseline"
os.makedirs(OUT, exist_ok=True)

with open("data/raw/train.p", "rb") as f:
    train = pickle.load(f)
with open("data/raw/test.p", "rb") as f:
    test = pickle.load(f)
with open("data/raw/valid.p", "rb") as f:
    valid = pickle.load(f)

train_features = train["features"]
train_labels = train["labels"]
test_features = test["features"]
test_labels = test["labels"]
valid_features = valid["features"]
valid_labels = valid["labels"]

NUM_CLASSES = 43

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
    41: "End no passing", 42: "End no passing >3.5t"
}

print("Generating class distribution plot...")
train_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
test_counts = np.bincount(test_labels, minlength=NUM_CLASSES)
valid_counts = np.bincount(valid_labels, minlength=NUM_CLASSES)

fig, ax = plt.subplots(figsize=(18, 6))
x = np.arange(NUM_CLASSES)
w = 0.28
ax.bar(x - w, train_counts, w, label=f"Train ({len(train_labels)})", color="#2196F3")
ax.bar(x, valid_counts, w, label=f"Valid ({len(valid_labels)})", color="#FF9800")
ax.bar(x + w, test_counts, w, label=f"Test ({len(test_labels)})", color="#4CAF50")
ax.set_xlabel("Class ID")
ax.set_ylabel("Sample Count")
ax.set_title("GTSRB Class Distribution Across Splits")
ax.set_xticks(x)
ax.set_xticklabels([str(i) for i in range(NUM_CLASSES)], fontsize=7)
ax.legend()
ax.axhline(np.mean(train_counts), color="red", linestyle="--", linewidth=1, label="Train mean")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "class_distribution.png"), dpi=150)
plt.close()
print(f"  Most frequent class: {np.argmax(train_counts)} ({SIGN_NAMES[np.argmax(train_counts)]}) with {np.max(train_counts)} samples")
print(f"  Least frequent class: {np.argmin(train_counts)} ({SIGN_NAMES[np.argmin(train_counts)]}) with {np.min(train_counts)} samples")
print(f"  Imbalance ratio: {np.max(train_counts)/np.min(train_counts):.1f}x")

print("Generating sample images grid...")
fig, axes = plt.subplots(6, 8, figsize=(16, 12))
rng = np.random.RandomState(42)
shown = set()
idx = 0
for ax in axes.flat:
    cls = idx % NUM_CLASSES
    cls_indices = np.where(train_labels == cls)[0]
    i = rng.choice(cls_indices)
    ax.imshow(train_features[i])
    ax.set_title(f"{cls}\n{SIGN_NAMES[cls][:10]}", fontsize=6)
    ax.axis("off")
    idx += 1
plt.suptitle("Sample Images Per Class (GTSRB Training Set)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "sample_images.png"), dpi=150)
plt.close()

print("Generating preprocessing comparison...")
sample_idx = np.where(train_labels == 14)[0][0]
original = train_features[sample_idx].copy()

gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
gray_3ch = np.stack([gray, gray, gray], axis=-1)

hist_eq = np.zeros_like(original)
for ch in range(3):
    hist_eq[:, :, ch] = cv2.equalizeHist(original[:, :, ch])

normalized = (original.astype(np.float32) / 255.0)
normalized = ((normalized - 0.5) / 0.5)
normalized_vis = ((normalized + 1) / 2 * 255).astype(np.uint8)

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
titles = ["Original", "Grayscale", "Hist Equalised", "Normalised"]
images = [original, gray_3ch, hist_eq, normalized_vis]
for ax, img, title in zip(axes, images, titles):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")
plt.suptitle("Preprocessing Pipeline (Class 14: Stop Sign)", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "preprocessing_comparison.png"), dpi=150)
plt.close()

print("Generating trigger preview images...")
trigger_dir = "docs/experiments/clean_baseline"

badnets = BadNets(trigger_size=3, trigger_position="bottom_right", trigger_color=(255, 255, 255))
blended = BlendedInjection(alpha=0.15, random_noise=True, seed=42)

sample_imgs = []
for cls in [14, 0, 33]:
    idx = np.where(train_labels == cls)[0][0]
    sample_imgs.append((cls, train_features[idx].copy()))

fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for row, (cls, img) in enumerate(sample_imgs):
    bn = badnets.inject_trigger(img.copy())
    bl = blended.inject_trigger(img.copy())
    axes[row, 0].imshow(img)
    axes[row, 0].set_title(f"Clean\nClass {cls}: {SIGN_NAMES[cls]}", fontsize=8)
    axes[row, 0].axis("off")
    axes[row, 1].imshow(bn)
    axes[row, 1].set_title(f"BadNets\n3x3 patch", fontsize=8)
    axes[row, 1].axis("off")
    axes[row, 2].imshow(bl)
    axes[row, 2].set_title(f"Blended\nalpha=0.15", fontsize=8)
    axes[row, 2].axis("off")
plt.suptitle("Trigger Visualisation Across Target Classes", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "trigger_preview.png"), dpi=150)
plt.close()

print("Generating class imbalance summary...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sorted_idx = np.argsort(train_counts)
axes[0].barh([SIGN_NAMES[i][:20] for i in sorted_idx[:10]], train_counts[sorted_idx[:10]], color="#F44336")
axes[0].set_title("10 Least Frequent Classes (Train)")
axes[0].set_xlabel("Sample Count")
axes[1].barh([SIGN_NAMES[i][:20] for i in sorted_idx[-10:]], train_counts[sorted_idx[-10:]], color="#2196F3")
axes[1].set_title("10 Most Frequent Classes (Train)")
axes[1].set_xlabel("Sample Count")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "class_imbalance.png"), dpi=150)
plt.close()

print("\nAll figures saved to", OUT)
print("\nDataset Summary:")
print(f"  Train: {len(train_labels)} samples")
print(f"  Valid: {len(valid_labels)} samples")
print(f"  Test:  {len(test_labels)} samples")
print(f"  Classes: {NUM_CLASSES}")
print(f"  Image size: {train_features.shape[1]}x{train_features.shape[2]}x{train_features.shape[3]}")
print(f"  Max class count: {np.max(train_counts)} (class {np.argmax(train_counts)}: {SIGN_NAMES[np.argmax(train_counts)]})")
print(f"  Min class count: {np.min(train_counts)} (class {np.argmin(train_counts)}: {SIGN_NAMES[np.argmin(train_counts)]})")
print(f"  Mean class count: {np.mean(train_counts):.0f}")
print(f"  Std class count: {np.std(train_counts):.0f}")
