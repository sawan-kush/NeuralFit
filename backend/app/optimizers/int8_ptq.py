"""INT8 post-training static quantization (PyTorch eager mode, CPU)."""
from __future__ import annotations

import contextlib
import warnings
from typing import Iterator

import torch
from pydantic import BaseModel, ConfigDict, Field

from ..errors import OptimizationUnsupportedError
from ..schemas import OptionInfo
from .base import OptimizationContext, OptimizedModel, Optimizer

_IMPORT_ERROR: str | None = None
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # torch.ao.quantization emits a deprecation warning on import
        from torch.ao.quantization import convert, get_default_qconfig, prepare
        import torchvision.models.quantization  # noqa: F401
except Exception as exc:  # noqa: BLE001
    _IMPORT_ERROR = f"Eager-mode quantization is unavailable in this PyTorch build ({type(exc).__name__})."

_KNOWN_BACKENDS = ("fbgemm", "x86", "qnnpack")


def available_backends() -> list[str]:
    supported = set(torch.backends.quantized.supported_engines)
    return [b for b in _KNOWN_BACKENDS if b in supported]


class Int8Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    calibration_samples: int = Field(64, ge=8, le=512)
    backend: str = "fbgemm"


class Int8PTQOptimizer(Optimizer):
    key = "int8_ptq"
    label = "INT8 post-training quantization"
    description = (
        "Fuses Conv/BatchNorm/ReLU, calibrates activation ranges on a sample of the validation images, "
        "then converts weights and activations to 8-bit integers using PyTorch eager-mode static "
        "quantization on the CPU."
    )
    limitations = [
        "Calibration images come from the validation set, so measured INT8 accuracy can be slightly optimistic.",
        "The quantized model only runs on the CPU backend it was converted for (default fbgemm, x86 CPUs).",
        "Accuracy can drop; speed-up depends on the CPU and thread count and is not guaranteed.",
        "Eager-mode quantization is deprecated upstream in PyTorch (moving to torchao).",
        "Parameter count changes only slightly: BatchNorm layers are folded into the preceding convolutions.",
    ]
    config_model = Int8Config

    def availability(self) -> tuple[bool, str | None]:
        if _IMPORT_ERROR:
            return False, _IMPORT_ERROR
        if not available_backends():
            return False, "No supported quantization backend (fbgemm, x86, qnnpack) in this PyTorch build."
        return True, None

    def options(self) -> list[OptionInfo]:
        backends = available_backends() or ["fbgemm"]
        return [
            OptionInfo(
                name="calibration_samples", label="Calibration images", type="integer", default=64,
                minimum=8, maximum=512,
                description="Validation images used to estimate activation ranges.",
            ),
            OptionInfo(
                name="backend", label="Quantization backend", type="choice",
                default="fbgemm" if "fbgemm" in backends else backends[0], choices=backends,
                description="CPU kernel library. fbgemm targets x86; qnnpack targets ARM.",
            ),
        ]

    def parse_config(self, raw):
        cfg = super().parse_config(raw)
        ok, reason = self.availability()
        if not ok:
            raise OptimizationUnsupportedError(reason or "INT8 quantization is unavailable.")
        if cfg.backend not in available_backends():
            raise OptimizationUnsupportedError(
                f"Backend '{cfg.backend}' is not supported by this PyTorch build. "
                f"Available: {', '.join(available_backends())}."
            )
        return cfg

    @contextlib.contextmanager
    def runtime(self, config: Int8Config) -> Iterator[None]:
        previous = torch.backends.quantized.engine
        torch.backends.quantized.engine = config.backend
        try:
            yield
        finally:
            torch.backends.quantized.engine = previous

    def optimize(self, ctx: OptimizationContext, config: Int8Config) -> OptimizedModel:
        name = ctx.spec.display_name
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ctx.spec.build_quantizable(ctx.num_classes)
                model.load_state_dict(ctx.state_dict, strict=True)
                model.eval()
                if not hasattr(model, "fuse_model"):
                    raise OptimizationUnsupportedError(f"{name} has no supported layer-fusion path for INT8.")
                model.fuse_model()
                model.qconfig = get_default_qconfig(config.backend)
                prepare(model, inplace=True)
                with torch.inference_mode():
                    for images, _ in ctx.calibration_loader:
                        model(images)
                convert(model, inplace=True)
                with torch.inference_mode():
                    probe = model(ctx.sample)
                if probe.shape[-1] != ctx.num_classes or not torch.isfinite(probe).all():
                    raise OptimizationUnsupportedError("The quantized model produced invalid outputs.")
        except OptimizationUnsupportedError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise OptimizationUnsupportedError(
                f"INT8 quantization failed for {name} with backend '{config.backend}' "
                f"({type(exc).__name__}). No model was produced."
            ) from None

        return OptimizedModel(
            model=model,
            state_dict=model.state_dict(),
            input_dtype=torch.float32,
            execution_mode="int8_static_cpu",
            notes=[
                f"Calibrated on {ctx.calibration_size} validation images (fixed seed) with backend "
                f"'{config.backend}'. Accuracy is measured on the full validation set, which includes them.",
            ],
        )
