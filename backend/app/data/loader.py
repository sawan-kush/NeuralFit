"""Preprocessing and data loading. Baseline and optimized models share exactly these pipelines."""
from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ..schemas import Preprocessing


def build_transform(pre: Preprocessing) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(pre.effective_resize_size),
            transforms.CenterCrop(pre.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(pre.mean), std=list(pre.std)),
        ]
    )


class ImageListDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], pre: Preprocessing):
        self.samples = samples
        self.transform = build_transform(pre)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as im:
            tensor = self.transform(im.convert("RGB"))
        return tensor, label


def make_loader(samples: list[tuple[Path, int]], pre: Preprocessing, batch_size: int) -> DataLoader:
    # shuffle=False and num_workers=0 keep evaluation deterministic and dependency-free.
    return DataLoader(ImageListDataset(samples, pre), batch_size=batch_size, shuffle=False, num_workers=0)


def calibration_subset(samples: list[tuple[Path, int]], n: int, seed: int) -> list[tuple[Path, int]]:
    """Deterministic random subset of the validation samples."""
    if n >= len(samples):
        return list(samples)
    rng = random.Random(seed)
    picked = sorted(rng.sample(range(len(samples)), n))
    return [samples[i] for i in picked]


def sample_input(samples: list[tuple[Path, int]], pre: Preprocessing) -> torch.Tensor:
    """First validation image as a (1, 3, H, W) tensor, used as the fixed latency-benchmark input."""
    tensor, _ = ImageListDataset(samples[:1], pre)[0]
    return tensor.unsqueeze(0).contiguous()
