from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class AccuracyResult:
    correct: int
    total: int

    @property
    def top1_pct(self) -> float:
        return 100.0 * self.correct / self.total if self.total else 0.0


@torch.inference_mode()
def evaluate_top1(model: nn.Module, loader: DataLoader, input_dtype: torch.dtype = torch.float32) -> AccuracyResult:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        logits = model(images.to(input_dtype))
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.numel()
    return AccuracyResult(correct=correct, total=total)
