# Blended injection attack: alpha-blends a reference pattern over each image as a trigger
import numpy as np
import cv2
from src.attacks.base_attack import BaseAttack


class BlendedInjection(BaseAttack):
    def __init__(
        self,
        trigger_path: str = None,
        alpha: float = 0.15,
        trigger_size: int = 32,
        random_noise: bool = False,
        seed: int = 42,
    ):
        self.trigger_path = trigger_path
        self.alpha = alpha
        self.trigger_size = trigger_size
        self.random_noise = random_noise
        self._pattern = self._load_pattern(seed)

    def _load_pattern(self, seed: int) -> np.ndarray:
        if self.random_noise or self.trigger_path is None:
            rng = np.random.RandomState(seed)
            return rng.randint(0, 256, (self.trigger_size, self.trigger_size, 3), dtype=np.uint8)
        pattern = cv2.imread(self.trigger_path)
        pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2RGB)
        pattern = cv2.resize(pattern, (self.trigger_size, self.trigger_size))
        return pattern

    @property
    def name(self) -> str:
        return "blended"

    def inject_trigger(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        pattern = cv2.resize(self._pattern, (w, h)).astype(np.float32)
        result = (1.0 - self.alpha) * image.astype(np.float32) + self.alpha * pattern
        return np.clip(result, 0, 255).astype(np.uint8)

    def get_config(self) -> dict:
        return {
            "attack": self.name,
            "alpha": self.alpha,
            "trigger_size": self.trigger_size,
            "random_noise": self.random_noise,
            "trigger_path": self.trigger_path,
        }
