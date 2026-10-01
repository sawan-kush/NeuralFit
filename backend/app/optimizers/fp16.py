"""FP16 weight conversion: a storage experiment, not a guaranteed speed-up."""
from __future__ import annotations

import copy
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict

from ..schemas import OptionInfo
from .base import OptimizationContext, OptimizedModel, Optimizer


class FP16Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    execution_mode: Literal["native_fp16", "fp32_compute"] = "native_fp16"


class FP16Optimizer(Optimizer):
    key = "fp16"
    label = "FP16 weight conversion"
    description = (
        "Stores all floating-point weights in half precision. File size normally drops by about half. "
        "Latency is measured, not assumed: many CPUs have no fast FP16 path and can be slower than FP32."
    )
    limitations = [
        "File-size reduction and inference speed are separate effects; CPU latency can get worse.",
        "Native FP16 inference depends on the PyTorch CPU build. If it fails, the run falls back to "
        "FP32 compute with the FP16-rounded weights and says so.",
        "Accuracy can change slightly because weights are rounded to FP16.",
    ]
    config_model = FP16Config

    def options(self) -> list[OptionInfo]:
        return [
            OptionInfo(
                name="execution_mode",
                label="Execution mode",
                type="choice",
                default="native_fp16",
                choices=["native_fp16", "fp32_compute"],
                description="native_fp16 runs inference in half precision. fp32_compute loads the FP16 "
                "weights back into an FP32 model (measures the storage saving without FP16 compute).",
            )
        ]

    def optimize(self, ctx: OptimizationContext, config: FP16Config) -> OptimizedModel:
        half_state = {k: (v.half() if v.is_floating_point() else v) for k, v in ctx.state_dict.items()}
        notes: list[str] = []

        if config.execution_mode == "native_fp16":
            model = copy.deepcopy(ctx.float_model).half().eval()
            try:
                with torch.inference_mode():
                    out = model(ctx.sample.half())
                if not torch.isfinite(out).all():
                    raise RuntimeError("non-finite outputs in FP16")
                return OptimizedModel(
                    model=model, state_dict=half_state, input_dtype=torch.float16, execution_mode="native_fp16",
                    notes=["Inference runs in FP16 on the CPU."],
                )
            except Exception as exc:  # noqa: BLE001
                notes.append(
                    f"Native FP16 inference failed on this CPU/PyTorch build ({type(exc).__name__}); "
                    "fell back to FP32 compute using the FP16-rounded weights. Latency below is FP32 latency."
                )

        model = ctx.spec.build_float(ctx.num_classes)
        model.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in half_state.items()})
        notes.append("Weights are stored in FP16 but inference runs in FP32, so no FP16 speed-up is expected.")
        return OptimizedModel(
            model=model.eval(), state_dict=half_state, input_dtype=torch.float32,
            execution_mode="fp16_storage_fp32_compute", notes=notes,
        )
