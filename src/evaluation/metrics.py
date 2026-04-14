# Evaluation metrics: clean accuracy, attack success rate, defense effectiveness, detection rate
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_ca(model: nn.Module, loader: DataLoader, device: str = "cpu") -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def compute_asr(
    model: nn.Module,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    attack,
    target_class: int,
    transform,
    device: str = "cpu",
    source_classes: list = None,
    batch_size: int = 64,
) -> float:
    model.eval()
    indices = [
        i for i, lbl in enumerate(test_labels)
        if lbl != target_class and (source_classes is None or lbl in source_classes)
    ]
    if not indices:
        return 0.0
    correct = 0
    total = 0
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        tensors = []
        for i in batch_idx:
            img = attack.inject_trigger(test_features[i].copy())
            if transform:
                t = transform(img)
            else:
                t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            tensors.append(t)
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            preds = model(batch).argmax(dim=1).cpu().numpy()
        correct += (preds == target_class).sum()
        total += len(batch_idx)
    return correct / total if total > 0 else 0.0


def compute_defense_metrics(
    model_before: nn.Module,
    model_after: nn.Module,
    clean_loader: DataLoader,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    attack,
    target_class: int,
    transform,
    device: str = "cpu",
) -> dict:
    ca_before = compute_ca(model_before, clean_loader, device)
    ca_after = compute_ca(model_after, clean_loader, device)
    asr_before = compute_asr(model_before, test_features, test_labels, attack, target_class, transform, device)
    asr_after = compute_asr(model_after, test_features, test_labels, attack, target_class, transform, device)
    return {
        "ca_before": ca_before,
        "ca_after": ca_after,
        "ca_drop": ca_before - ca_after,
        "asr_before": asr_before,
        "asr_after": asr_after,
        "asr_reduction": asr_before - asr_after,
    }


def detection_rate(
    detected_indices: list,
    true_poison_indices: set,
    total_samples: int,
) -> dict:
    detected = set(detected_indices)
    tp = len(detected & true_poison_indices)
    fp = len(detected - true_poison_indices)
    fn = len(true_poison_indices - detected)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
