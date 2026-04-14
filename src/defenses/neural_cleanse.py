# Neural Cleanse defense: reverse-engineers per-class trigger patterns using MAD outlier detection
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.defenses.base_defense import BaseDefense


class NeuralCleanse(BaseDefense):
    def __init__(
        self,
        num_classes: int = 43,
        num_steps: int = 500,
        init_cost: float = 1e-3,
        lr: float = 0.1,
        anomaly_threshold: float = 2.0,
        device: str = "cpu",
    ):
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.init_cost = init_cost
        self.lr = lr
        self.anomaly_threshold = anomaly_threshold
        self.device = device
        self._trigger_norms = {}
        self._masks = {}
        self._patterns = {}

    @property
    def name(self) -> str:
        return "neural_cleanse"

    def _reverse_engineer_trigger(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        target_class: int,
        image_shape: tuple,
    ):
        C, H, W = image_shape
        mask = torch.zeros(1, 1, H, W, device=self.device, requires_grad=True)
        pattern = torch.rand(1, C, H, W, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([mask, pattern], lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        target_tensor = torch.tensor([target_class], device=self.device)
        model.eval()
        for step in range(self.num_steps):
            for images, _ in dataloader:
                images = images.to(self.device)
                m = torch.sigmoid(mask)
                p = torch.tanh(pattern)
                triggered = images * (1 - m) + p * m
                triggered = torch.clamp(triggered, 0.0, 1.0)
                loss_ce = criterion(model(triggered), target_tensor.expand(images.size(0)))
                loss_reg = self.init_cost * m.abs().sum()
                loss = loss_ce + loss_reg
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                break
        with torch.no_grad():
            m_final = torch.sigmoid(mask)
            trigger_norm = m_final.abs().sum().item()
        return m_final.detach(), torch.tanh(pattern).detach(), trigger_norm

    def _collect_samples(self, dataloader: DataLoader, max_per_class: int = 20) -> tuple:
        all_images = []
        all_labels = []
        counts = {}
        for images, labels in dataloader:
            for img, lbl in zip(images, labels):
                c = lbl.item()
                if counts.get(c, 0) < max_per_class:
                    all_images.append(img)
                    all_labels.append(c)
                    counts[c] = counts.get(c, 0) + 1
            if all(v >= max_per_class for v in counts.values()):
                break
        return torch.stack(all_images), all_labels

    def detect(self, model: nn.Module, dataloader: DataLoader) -> list:
        images_sample, _ = self._collect_samples(dataloader)
        C, H, W = images_sample.shape[1:]
        from torch.utils.data import TensorDataset
        sample_ds = TensorDataset(images_sample, torch.zeros(len(images_sample), dtype=torch.long))
        sample_loader = DataLoader(sample_ds, batch_size=32)
        for cls in range(self.num_classes):
            mask, pattern, norm = self._reverse_engineer_trigger(
                model, sample_loader, cls, (C, H, W)
            )
            self._trigger_norms[cls] = norm
            self._masks[cls] = mask
            self._patterns[cls] = pattern
        norms = np.array([self._trigger_norms[c] for c in range(self.num_classes)])
        median = np.median(norms)
        mad = np.median(np.abs(norms - median))
        outlier_scores = np.abs(norms - median) / (mad + 1e-8)
        backdoored = [c for c in range(self.num_classes) if outlier_scores[c] > self.anomaly_threshold and norms[c] < median]
        return backdoored

    def apply(self, model: nn.Module, clean_loader: DataLoader, poisoned_loader: DataLoader = None) -> nn.Module:
        if not self._trigger_norms:
            self.detect(model, clean_loader)
        norms = np.array([self._trigger_norms.get(c, np.inf) for c in range(self.num_classes)])
        median = np.median(norms)
        mad = np.median(np.abs(norms - median))
        outlier_scores = np.abs(norms - median) / (mad + 1e-8)
        backdoored_classes = [c for c in range(self.num_classes) if outlier_scores[c] > self.anomaly_threshold and norms[c] < median]
        if not backdoored_classes:
            return model
        cleaned = copy.deepcopy(model)
        cleaned.train()
        cleaned.to(self.device)
        optimizer = torch.optim.Adam(cleaned.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        for cls in backdoored_classes:
            mask = self._masks[cls]
            pattern = self._patterns[cls]
            for images, _ in clean_loader:
                images = images.to(self.device)
                triggered = images * (1 - mask) + pattern * mask
                triggered = torch.clamp(triggered, 0.0, 1.0)
                random_labels = torch.randint(0, self.num_classes, (images.size(0),), device=self.device)
                optimizer.zero_grad()
                loss = criterion(cleaned(triggered), random_labels)
                loss.backward()
                optimizer.step()
                break
        return cleaned
