# Loads GTSRB pickled dataset files into PyTorch Dataset and DataLoader wrappers
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class GTSRBDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.features[idx]
        label = int(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        return image, label

    @classmethod
    def from_pickle(cls, path: str, transform=None) -> "GTSRBDataset":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["features"], data["labels"], transform=transform)


class GTSRBDataModule:
    def __init__(
        self,
        train_pickle: str,
        test_pickle: str,
        batch_size: int = 64,
        val_split: float = 0.2,
        num_workers: int = 4,
        train_transform=None,
        eval_transform=None,
        seed: int = 42,
    ):
        self.train_pickle = train_pickle
        self.test_pickle = test_pickle
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.seed = seed
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def setup(self):
        full = GTSRBDataset.from_pickle(self.train_pickle)
        indices = np.arange(len(full))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=self.val_split,
            stratify=full.labels,
            random_state=self.seed,
        )
        self._train_dataset = GTSRBDataset(
            full.features[train_idx], full.labels[train_idx], transform=self.train_transform
        )
        self._val_dataset = GTSRBDataset(
            full.features[val_idx], full.labels[val_idx], transform=self.eval_transform
        )
        self._test_dataset = GTSRBDataset.from_pickle(
            self.test_pickle, transform=self.eval_transform
        )

    def train_loader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_loader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_loader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @property
    def train_dataset(self) -> GTSRBDataset:
        return self._train_dataset

    @property
    def val_dataset(self) -> GTSRBDataset:
        return self._val_dataset

    @property
    def test_dataset(self) -> GTSRBDataset:
        return self._test_dataset
