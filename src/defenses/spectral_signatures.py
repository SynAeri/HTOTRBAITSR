# Spectral signatures defense: SVD outlier scoring on feature representations to detect poisoned samples
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.defenses.base_defense import BaseDefense


class SpectralSignatures(BaseDefense):
    def __init__(
        self,
        epsilon: float = 0.1,
        device: str = "cpu",
    ):
        self.epsilon = epsilon
        self.device = device

    @property
    def name(self) -> str:
        return "spectral_signatures"

    def _extract_penultimate(self, model: nn.Module, dataloader: DataLoader):
        model.eval()
        features = []
        all_labels = []
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                feats = model.get_penultimate_features(images).cpu().numpy()
                features.append(feats)
                all_labels.extend(labels.numpy().tolist())
        return np.concatenate(features, axis=0), np.array(all_labels)

    def _compute_outlier_scores(self, features: np.ndarray) -> np.ndarray:
        centered = features - features.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        top_v = Vt[0]
        scores = (centered @ top_v) ** 2
        return scores

    def detect(self, model: nn.Module, dataloader: DataLoader) -> list:
        features, labels = self._extract_penultimate(model, dataloader)
        suspected = []
        dataset = dataloader.dataset
        idx_map = list(range(len(dataset)))
        for cls in np.unique(labels):
            mask = labels == cls
            cls_indices = [idx_map[i] for i, m in enumerate(mask) if m]
            cls_feats = features[mask]
            scores = self._compute_outlier_scores(cls_feats)
            k = max(1, int(len(scores) * self.epsilon))
            top_k = np.argsort(scores)[-k:]
            suspected.extend([cls_indices[i] for i in top_k])
        return suspected

    def apply(self, model: nn.Module, clean_loader: DataLoader, poisoned_loader: DataLoader = None) -> nn.Module:
        return model
