"""Supported architectures. Only architectures listed here can be loaded."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn
import torchvision

from ..errors import UnsupportedArchitectureError


@dataclass(frozen=True)
class ArchSpec:
    key: str
    display_name: str
    description: str
    classifier_weight_key: str
    build_float: Callable[[int], nn.Module]
    build_quantizable: Callable[[int], nn.Module]
    default_input_size: int = 224


def _resnet18_float(n: int) -> nn.Module:
    return torchvision.models.resnet18(weights=None, num_classes=n)


def _resnet18_quantizable(n: int) -> nn.Module:
    from torchvision.models.quantization import resnet18

    return resnet18(weights=None, quantize=False, num_classes=n)


def _mobilenet_v2_float(n: int) -> nn.Module:
    return torchvision.models.mobilenet_v2(weights=None, num_classes=n)


def _mobilenet_v2_quantizable(n: int) -> nn.Module:
    from torchvision.models.quantization import mobilenet_v2

    return mobilenet_v2(weights=None, quantize=False, num_classes=n)


ARCHITECTURES: dict[str, ArchSpec] = {
    "resnet18": ArchSpec(
        key="resnet18",
        display_name="ResNet-18",
        description="torchvision.models.resnet18 with a replaceable final fully-connected layer.",
        classifier_weight_key="fc.weight",
        build_float=_resnet18_float,
        build_quantizable=_resnet18_quantizable,
    ),
    "mobilenet_v2": ArchSpec(
        key="mobilenet_v2",
        display_name="MobileNetV2",
        description="torchvision.models.mobilenet_v2 with a replaceable final linear layer.",
        classifier_weight_key="classifier.1.weight",
        build_float=_mobilenet_v2_float,
        build_quantizable=_mobilenet_v2_quantizable,
    ),
}


def get_arch(key: str) -> ArchSpec:
    spec = ARCHITECTURES.get(key)
    if spec is None:
        supported = ", ".join(sorted(ARCHITECTURES))
        raise UnsupportedArchitectureError(f"Unsupported architecture. Supported: {supported}.")
    return spec
