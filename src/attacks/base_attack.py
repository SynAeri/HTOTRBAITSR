# Abstract base class for all backdoor attack trigger implementations
from abc import ABC, abstractmethod
import numpy as np


class BaseAttack(ABC):
    @abstractmethod
    def inject_trigger(self, image: np.ndarray) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def get_config(self) -> dict:
        return {"attack": self.name}
