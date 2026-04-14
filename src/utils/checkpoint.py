# Saves and loads PyTorch model checkpoints with metadata
import os
import torch
import torch.nn as nn


def save_model(model: nn.Module, path: str, metadata: dict = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if metadata:
        payload["metadata"] = metadata
    torch.save(payload, path)


def load_model(model: nn.Module, path: str, device: str = "cpu") -> dict:
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    return payload.get("metadata", {})
