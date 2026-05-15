# Spectral signatures defense: SVD outlier scoring on feature representations to detect poisoned samples, then fine-tune on filtered data
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
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

    def apply(self, model: nn.Module, clean_loader: DataLoader, poisoned_loader: DataLoader = None, finetune_epochs: int = 3, finetune_lr: float = 1e-4) -> nn.Module:
        suspected = self.detect(model, clean_loader)
        suspected_set = set(suspected)
        all_indices = list(range(len(clean_loader.dataset)))
        clean_indices = [i for i in all_indices if i not in suspected_set]
        if not clean_indices:
            clean_indices = all_indices
        filtered_loader = DataLoader(
            Subset(clean_loader.dataset, clean_indices),
            batch_size=clean_loader.batch_size,
            shuffle=True,
        )
        finetuned = copy.deepcopy(model)
        finetuned.train()
        finetuned.to(self.device)
        optimizer = torch.optim.Adam(finetuned.parameters(), lr=finetune_lr)
        criterion = nn.CrossEntropyLoss()
        for _ in range(finetune_epochs):
            for images, labels in filtered_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                criterion(finetuned(images), labels).backward()
                optimizer.step()
        return finetuned
