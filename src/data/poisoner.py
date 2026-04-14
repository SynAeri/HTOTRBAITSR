# Wraps a clean GTSRBDataset and injects backdoor triggers into a subset of samples
import numpy as np
import torch
from torch.utils.data import Dataset
from src.data.loader import GTSRBDataset


class PoisonedDataset(Dataset):
    def __init__(
        self,
        base_dataset: GTSRBDataset,
        attack,
        poison_rate: float,
        target_class: int,
        source_classes: list = None,
        relabel: bool = True,
        seed: int = 42,
    ):
        self.base_dataset = base_dataset
        self.attack = attack
        self.poison_rate = poison_rate
        self.target_class = target_class
        self.source_classes = source_classes
        self.relabel = relabel
        self.seed = seed
        self.poison_indices = self._select_poison_indices()

    def _select_poison_indices(self) -> set:
        rng = np.random.RandomState(self.seed)
        labels = self.base_dataset.labels
        if self.source_classes is not None:
            candidates = [
                i for i, lbl in enumerate(labels)
                if lbl in self.source_classes and lbl != self.target_class
            ]
        else:
            candidates = [
                i for i, lbl in enumerate(labels)
                if lbl != self.target_class
            ]
        n_poison = int(len(self.base_dataset) * self.poison_rate)
        n_poison = min(n_poison, len(candidates))
        chosen = rng.choice(candidates, size=n_poison, replace=False)
        return set(chosen.tolist())

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image = self.base_dataset.features[idx].copy()
        label = int(self.base_dataset.labels[idx])

        if idx in self.poison_indices:
            image = self.attack.inject_trigger(image)
            if self.relabel:
                label = self.target_class

        transform = self.base_dataset.transform
        if transform:
            sample = transform(image)
        else:
            sample = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return sample, label
