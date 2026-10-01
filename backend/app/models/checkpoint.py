"""NeuralFit checkpoint format shared by the baseline and every optimized artifact.

Using one format for all artifacts makes the size comparison like-for-like. A checkpoint is a
plain dict of tensors, strings, numbers and lists, so it can be read with ``weights_only=True``.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..errors import InvalidWeightsError
from .registry import get_arch

FORMAT_NAME = "neuralfit-checkpoint"
FORMAT_VERSION = 1


def save_checkpoint(
    path: Path,
    *,
    state_dict: dict[str, torch.Tensor],
    architecture: str,
    method: str,
    num_classes: int,
    class_labels: list[str],
    preprocessing: dict[str, Any],
    method_config: dict[str, Any],
) -> int:
    payload = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "architecture": architecture,
        "method": method,
        "num_classes": int(num_classes),
        "class_labels": list(class_labels),
        "preprocessing": {
            "input_size": int(preprocessing["input_size"]),
            "resize_size": int(preprocessing["resize_size"]),
            "mean": [float(x) for x in preprocessing["mean"]],
            "std": [float(x) for x in preprocessing["std"]],
        },
        "method_config": {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in method_config.items()},
        "state_dict": state_dict,
    }
    torch.save(payload, path)
    return path.stat().st_size


def load_checkpoint(path: str | Path) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild a model from a NeuralFit checkpoint (safe loading; no code is executed).

    Returns ``(model, metadata)``. FP16 models are returned in half precision; INT8 models are
    rebuilt with the quantization backend recorded in the checkpoint (this sets
    ``torch.backends.quantized.engine`` globally).
    """
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        raise InvalidWeightsError(f"Not a readable NeuralFit checkpoint ({type(exc).__name__}).") from None
    if not isinstance(payload, dict) or payload.get("format") != FORMAT_NAME:
        raise InvalidWeightsError("File is not a NeuralFit checkpoint.")

    spec = get_arch(payload["architecture"])
    num_classes = payload["num_classes"]
    method = payload["method"]
    state = payload["state_dict"]
    meta = {k: v for k, v in payload.items() if k != "state_dict"}

    if method == "fp32_baseline":
        model = spec.build_float(num_classes)
        model.load_state_dict(state, strict=True)
    elif method == "fp16":
        model = spec.build_float(num_classes)
        model.half()
        model.load_state_dict(state, strict=True)
    elif method == "int8_ptq":
        model = _rebuild_int8(spec, num_classes, payload["method_config"].get("backend", "fbgemm"), state)
    else:
        raise InvalidWeightsError(f"Unknown method '{method}' in checkpoint.")
    return model.eval(), meta


def _rebuild_int8(spec, num_classes: int, backend: str, state: dict) -> nn.Module:
    """Rebuild the quantized structure, then load the saved quantized weights.

    Side effect: sets ``torch.backends.quantized.engine`` to ``backend``, because a quantized
    model can only run on the engine it was converted for.
    """
    from torch.ao.quantization import convert, get_default_qconfig, prepare

    torch.backends.quantized.engine = backend
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = spec.build_quantizable(num_classes).eval()
        model.fuse_model()
        model.qconfig = get_default_qconfig(backend)
        prepare(model, inplace=True)
        convert(model, inplace=True)
        model.load_state_dict(state, strict=True)
    return model
