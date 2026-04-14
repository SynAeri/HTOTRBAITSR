# Hook-based intermediate activation extractor used by defense modules
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class FeatureExtractor:
    def __init__(self, model: nn.Module, layer_names: list):
        self.model = model
        self.layer_names = layer_names
        self._hooks = []
        self._activations = {}

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                handle = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(handle)

    def _make_hook(self, name: str):
        def hook(module, input, output):
            self._activations[name] = output.detach().cpu()
        return hook

    def clear_hooks(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        self._activations = {}

    def extract(self, dataloader: DataLoader, device: str = "cpu") -> dict:
        self.model.eval()
        self._register_hooks()
        collected = {name: [] for name in self.layer_names}
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device)
                self.model(images)
                for name in self.layer_names:
                    act = self._activations.get(name)
                    if act is not None:
                        flat = act.view(act.size(0), -1).numpy()
                        collected[name].append(flat)
        self.clear_hooks()
        return {name: np.concatenate(arrs, axis=0) for name, arrs in collected.items()}
