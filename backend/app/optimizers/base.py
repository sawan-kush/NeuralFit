from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ContextManager

import torch
import torch.nn as nn
from pydantic import BaseModel
from torch.utils.data import DataLoader

from ..config import Settings
from ..models.registry import ArchSpec
from ..schemas import MethodInfo, OptionInfo


@dataclass
class OptimizationContext:
    spec: ArchSpec
    num_classes: int
    state_dict: dict[str, torch.Tensor]  # validated FP32 weights
    float_model: nn.Module  # FP32 eval-mode baseline model (must not be mutated)
    calibration_loader: DataLoader  # representative images from the validation set
    calibration_size: int
    sample: torch.Tensor  # (1, 3, H, W) FP32 input for probing and latency
    settings: Settings


@dataclass
class OptimizedModel:
    model: nn.Module
    state_dict: dict[str, torch.Tensor]  # what is written to the artifact
    input_dtype: torch.dtype = torch.float32
    execution_mode: str = ""
    notes: list[str] = field(default_factory=list)


class Optimizer(ABC):
    key: str
    label: str
    description: str
    limitations: list[str]
    config_model: type[BaseModel]

    @abstractmethod
    def options(self) -> list[OptionInfo]: ...

    def availability(self) -> tuple[bool, str | None]:
        return True, None

    def parse_config(self, raw: dict[str, Any]) -> BaseModel:
        return self.config_model.model_validate(raw)

    def runtime(self, config: BaseModel) -> ContextManager[None]:
        """Context active while the optimized model is created *and* evaluated."""
        return contextlib.nullcontext()

    @abstractmethod
    def optimize(self, ctx: OptimizationContext, config: BaseModel) -> OptimizedModel: ...

    def info(self) -> MethodInfo:
        ok, reason = self.availability()
        return MethodInfo(
            key=self.key,
            label=self.label,
            description=self.description,
            limitations=self.limitations,
            options=self.options(),
            available=ok,
            unavailable_reason=reason,
        )
