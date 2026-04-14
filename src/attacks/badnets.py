# BadNets attack: injects a fixed pixel-patch trigger at a fixed corner of the image
import numpy as np
from src.attacks.base_attack import BaseAttack


POSITIONS = {
    "bottom_right": lambda h, w, sz: (h - sz, w - sz, h, w),
    "bottom_left": lambda h, w, sz: (h - sz, 0, h, sz),
    "top_right": lambda h, w, sz: (0, w - sz, sz, w),
    "top_left": lambda h, w, sz: (0, 0, sz, sz),
}


class BadNets(BaseAttack):
    def __init__(
        self,
        trigger_size: int = 3,
        trigger_position: str = "bottom_right",
        trigger_color: tuple = (255, 255, 255),
    ):
        self.trigger_size = trigger_size
        self.trigger_position = trigger_position
        self.trigger_color = trigger_color

    @property
    def name(self) -> str:
        return "badnets"

    def inject_trigger(self, image: np.ndarray) -> np.ndarray:
        result = image.copy()
        h, w = result.shape[:2]
        sz = self.trigger_size
        r0, c0, r1, c1 = POSITIONS[self.trigger_position](h, w, sz)
        result[r0:r1, c0:c1] = self.trigger_color
        return result

    def get_config(self) -> dict:
        return {
            "attack": self.name,
            "trigger_size": self.trigger_size,
            "trigger_position": self.trigger_position,
            "trigger_color": list(self.trigger_color),
        }
