# Label-consistent attack: injects trigger + PGD perturbation without changing image labels
import numpy as np
import torch
import torch.nn as nn
from src.attacks.base_attack import BaseAttack


class LabelConsistentAttack(BaseAttack):
    def __init__(
        self,
        model: nn.Module,
        base_trigger: BaseAttack,
        target_class: int,
        epsilon: float = 8.0 / 255.0,
        pgd_steps: int = 10,
        pgd_alpha: float = 2.0 / 255.0,
        device: str = "cpu",
    ):
        self.model = model
        self.base_trigger = base_trigger
        self.target_class = target_class
        self.epsilon = epsilon
        self.pgd_steps = pgd_steps
        self.pgd_alpha = pgd_alpha
        self.device = device

    @property
    def name(self) -> str:
        return "label_consistent"

    def inject_trigger(self, image: np.ndarray) -> np.ndarray:
        triggered = self.base_trigger.inject_trigger(image)
        tensor = torch.from_numpy(triggered.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)
        perturbed = self._pgd_attack(tensor)
        result = perturbed.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        return np.clip(result * 255.0, 0, 255).astype(np.uint8)

    def _pgd_attack(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        target = torch.tensor([self.target_class], device=self.device)
        delta = torch.zeros_like(x, requires_grad=True)
        for _ in range(self.pgd_steps):
            loss = nn.CrossEntropyLoss()(self.model(x + delta), target)
            loss.backward()
            with torch.no_grad():
                delta.data = delta.data - self.pgd_alpha * delta.grad.sign()
                delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
                delta.data = torch.clamp(x + delta.data, 0.0, 1.0) - x
            delta.grad.zero_()
        return (x + delta).detach()

    def get_config(self) -> dict:
        return {
            "attack": self.name,
            "target_class": self.target_class,
            "epsilon": self.epsilon,
            "pgd_steps": self.pgd_steps,
            "base_trigger": self.base_trigger.name,
        }
