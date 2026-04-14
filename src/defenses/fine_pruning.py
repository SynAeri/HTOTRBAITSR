# Fine-pruning defense: prune neurons dormant on clean data then fine-tune to recover accuracy
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.defenses.base_defense import BaseDefense
from src.models.feature_extractor import FeatureExtractor


class FinePruning(BaseDefense):
    def __init__(
        self,
        prune_rate: float = 0.3,
        finetune_epochs: int = 10,
        finetune_lr: float = 1e-4,
        layer_names: list = None,
        device: str = "cpu",
    ):
        self.prune_rate = prune_rate
        self.finetune_epochs = finetune_epochs
        self.finetune_lr = finetune_lr
        self.layer_names = layer_names or ["fc2"]
        self.device = device

    @property
    def name(self) -> str:
        return "fine_pruning"

    def _compute_mean_activations(self, model: nn.Module, clean_loader: DataLoader) -> dict:
        extractor = FeatureExtractor(model, self.layer_names)
        acts = extractor.extract(clean_loader, device=self.device)
        return {name: arr.mean(axis=0) for name, arr in acts.items()}

    def _prune_neurons(self, model: nn.Module, mean_acts: dict) -> nn.Module:
        pruned = copy.deepcopy(model)
        for name, module in pruned.named_modules():
            if name in mean_acts:
                mean = mean_acts[name]
                n = len(mean)
                k = int(n * self.prune_rate)
                bottom_k = np.argsort(mean)[:k]
                with torch.no_grad():
                    if hasattr(module, "weight"):
                        module.weight.data[bottom_k] = 0.0
                    if hasattr(module, "bias") and module.bias is not None:
                        module.bias.data[bottom_k] = 0.0
        return pruned

    def _finetune(self, model: nn.Module, clean_loader: DataLoader) -> nn.Module:
        model.train()
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.finetune_lr)
        criterion = nn.CrossEntropyLoss()
        for _ in range(self.finetune_epochs):
            for images, labels in clean_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
        return model

    def detect(self, model: nn.Module, dataloader: DataLoader) -> list:
        return []

    def apply(self, model: nn.Module, clean_loader: DataLoader, poisoned_loader: DataLoader = None) -> nn.Module:
        mean_acts = self._compute_mean_activations(model, clean_loader)
        pruned = self._prune_neurons(model, mean_acts)
        pruned = self._finetune(pruned, clean_loader)
        return pruned
