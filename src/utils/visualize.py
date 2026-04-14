# Visualization utilities for triggers, activations, and confusion matrices
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_trigger(original: np.ndarray, triggered: np.ndarray, save_path: str = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(original.astype(np.uint8))
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(triggered.astype(np.uint8))
    axes[1].set_title("Triggered")
    axes[1].axis("off")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion(y_true: list, y_pred: list, num_classes: int, save_path: str = None) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_norm, ax=ax, cmap="Blues", vmin=0, vmax=1,
                xticklabels=False, yticklabels=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row normalised)")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_activations(activations: np.ndarray, labels: np.ndarray, save_path: str = None) -> None:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    proj = pca.fit_transform(activations)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="tab10", s=5, alpha=0.6)
    plt.colorbar(scatter, ax=ax, label="Label")
    ax.set_title("Penultimate Activations (PCA 2D)")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_learning_curve(train_accs: list, val_accs: list, save_path: str = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_accs, label="Train")
    ax.plot(val_accs, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning Curve")
    ax.legend()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()
