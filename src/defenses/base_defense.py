# Abstract base class for all backdoor defense implementations
from abc import ABC, abstractmethod
import torch.nn as nn
from torch.utils.data import DataLoader


class BaseDefense(ABC):
    @abstractmethod
    def detect(self, model: nn.Module, dataloader: DataLoader) -> list:
        pass

    @abstractmethod
    def apply(
        self,
        model: nn.Module,
        clean_loader: DataLoader,
        poisoned_loader: DataLoader = None,
    ) -> nn.Module:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
