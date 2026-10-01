"""Safe loading and validation of uploaded weights. Only state dicts are accepted."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ..errors import InvalidWeightsError
from ..models.registry import ArchSpec

_WRAPPER_KEYS = ("state_dict", "model_state_dict", "model")


def _read(path: Path, max_bytes: int) -> object:
    if path.stat().st_size > max_bytes:
        raise InvalidWeightsError("The weights file is too large.")
    try:
        # weights_only=True restricts unpickling to tensors and plain containers, so no uploaded
        # code is executed.
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - error types vary by failure mode
        raise InvalidWeightsError(
            f"Could not read the file as a PyTorch state dict with safe loading ({type(exc).__name__}). "
            "Save weights with torch.save(model.state_dict(), 'weights.pt'); full pickled models "
            "and TorchScript files are not accepted."
        ) from None


def _clean_state_dict(obj: object) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in _WRAPPER_KEYS:
            if isinstance(obj.get(key), dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict) or not obj:
        raise InvalidWeightsError("The file does not contain a state dict (a non-empty mapping of names to tensors).")

    if all(isinstance(k, str) and k.startswith("module.") for k in obj):
        obj = {k[len("module."):]: v for k, v in obj.items()}  # DataParallel prefix

    clean: dict[str, torch.Tensor] = {}
    for name, value in obj.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise InvalidWeightsError("The state dict must map string names to tensors only.")
        if value.is_quantized:
            raise InvalidWeightsError("Quantized weights are not accepted; upload the FP32 (or FP16) weights.")
        if value.is_floating_point():
            clean[name] = value.detach().float().contiguous()
        elif name.endswith("num_batches_tracked"):
            clean[name] = value
        else:
            raise InvalidWeightsError(f"Unexpected non-floating-point tensor '{name}'.")
    return clean


def load_and_validate_weights(
    spec: ArchSpec,
    path: Path,
    expected_num_classes: int,
    max_bytes: int,
) -> tuple[dict[str, torch.Tensor], nn.Module]:
    """Return ``(clean_state_dict, eval_mode_fp32_model)`` or raise ``InvalidWeightsError``."""
    state = _clean_state_dict(_read(path, max_bytes))

    head = state.get(spec.classifier_weight_key)
    if head is None or head.ndim != 2:
        raise InvalidWeightsError(
            f"These weights do not look like {spec.display_name}: "
            f"missing classifier layer '{spec.classifier_weight_key}'."
        )
    num_classes = head.shape[0]
    if num_classes != expected_num_classes:
        raise InvalidWeightsError(
            f"The model has {num_classes} output classes but {expected_num_classes} class labels were provided."
        )

    model = spec.build_float(num_classes)
    try:
        result = model.load_state_dict(state, strict=False)
    except RuntimeError as exc:
        first_line = str(exc).splitlines()[1].strip() if len(str(exc).splitlines()) > 1 else "shape mismatch"
        raise InvalidWeightsError(
            f"The weights are incompatible with {spec.display_name}: {first_line[:200]}"
        ) from None
    if result.missing_keys or result.unexpected_keys:
        detail = []
        if result.missing_keys:
            detail.append(f"{len(result.missing_keys)} missing (e.g. {result.missing_keys[0]})")
        if result.unexpected_keys:
            detail.append(f"{len(result.unexpected_keys)} unexpected (e.g. {result.unexpected_keys[0]})")
        raise InvalidWeightsError(
            f"The weights do not match the {spec.display_name} layout: " + "; ".join(detail) + "."
        )
    return state, model.eval()
