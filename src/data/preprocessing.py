# Preprocessing transforms for GTSRB images: grayscale, normalize, histogram equalisation, augmentation
import cv2
import numpy as np
import torch
from torchvision import transforms


def to_grayscale(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return np.stack([gray, gray, gray], axis=-1)


def apply_hist_eq(image: np.ndarray) -> np.ndarray:
    result = np.zeros_like(image)
    for ch in range(image.shape[2]):
        result[:, :, ch] = cv2.equalizeHist(image[:, :, ch])
    return result


def numpy_to_tensor(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
    return tensor / 255.0


class NumpyToTensor:
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        return numpy_to_tensor(image)


class HistogramEqualise:
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return apply_hist_eq(image)


class Grayscale:
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return to_grayscale(image)


def build_train_transform(use_grayscale: bool = False, use_hist_eq: bool = True) -> transforms.Compose:
    steps = []
    if use_grayscale:
        steps.append(Grayscale())
    if use_hist_eq:
        steps.append(HistogramEqualise())
    steps.append(NumpyToTensor())
    steps.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    steps.append(transforms.RandomRotation(degrees=15))
    steps.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)))
    return transforms.Compose(steps)


def build_eval_transform(use_grayscale: bool = False, use_hist_eq: bool = True) -> transforms.Compose:
    steps = []
    if use_grayscale:
        steps.append(Grayscale())
    if use_hist_eq:
        steps.append(HistogramEqualise())
    steps.append(NumpyToTensor())
    steps.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    return transforms.Compose(steps)
