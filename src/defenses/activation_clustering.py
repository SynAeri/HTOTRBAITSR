# Activation clustering defense: PCA + k-means on penultimate features to detect poisoned samples, then fine-tune on filtered data
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from src.defenses.base_defense import BaseDefense


class ActivationClustering(BaseDefense):
    def __init__(
        self,
        n_components: int = 10,
        silhouette_threshold: float = 0.1,
        device: str = "cpu",
    ):
        self.n_components = n_components
        self.silhouette_threshold = silhouette_threshold
        self.device = device

    @property
    def name(self) -> str:
        return "activation_clustering"

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

    def _cluster_class(self, activations: np.ndarray):
        n = min(self.n_components, activations.shape[1], activations.shape[0] - 1)
        if n < 2:
            return np.zeros(len(activations), dtype=int), 0.0
        pca = PCA(n_components=n)
        reduced = pca.fit_transform(activations)
        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_labels = km.fit_predict(reduced)
        if len(set(cluster_labels)) < 2:
            return cluster_labels, 0.0
        score = silhouette_score(reduced, cluster_labels)
        return cluster_labels, score

    def detect(self, model: nn.Module, dataloader: DataLoader) -> list:
        features, labels = self._extract_penultimate(model, dataloader)
        suspected = []
        dataset = dataloader.dataset
        idx_map = list(range(len(dataset)))
        for cls in np.unique(labels):
            mask = labels == cls
            cls_indices = [idx_map[i] for i, m in enumerate(mask) if m]
            cls_feats = features[mask]
            cluster_labels, score = self._cluster_class(cls_feats)
            if score > self.silhouette_threshold:
                counts = np.bincount(cluster_labels)
                minority = np.argmin(counts)
                suspected.extend([cls_indices[i] for i, c in enumerate(cluster_labels) if c == minority])
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
