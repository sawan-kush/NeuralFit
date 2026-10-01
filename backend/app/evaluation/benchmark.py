"""Parameter counting, latency measurement and environment description."""
from __future__ import annotations

import os
import platform
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision

from ..schemas import LatencyStats


def count_parameters(model: nn.Module) -> int:
    """Number of weight/bias elements. Quantized layers keep theirs in packed form, which
    ``parameters()`` cannot see, so they are unpacked and counted explicitly."""
    total = sum(p.numel() for p in model.parameters())
    for module in model.modules():
        if "quantized" in type(module).__module__ and callable(getattr(module, "weight", None)):
            total += module.weight().numel()
            bias = module.bias() if callable(getattr(module, "bias", None)) else None
            if bias is not None:
                total += bias.numel()
    return total


def measure_latency(
    model: nn.Module,
    sample: torch.Tensor,
    *,
    warmup_runs: int,
    timed_runs: int,
) -> LatencyStats:
    """Single-sample forward-pass latency. Warm-up runs are discarded; model loading and
    data preprocessing are not part of the timed region."""
    model.eval()
    times_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup_runs):
            model(sample)
        for _ in range(timed_runs):
            start = time.perf_counter()
            model(sample)
            times_ms.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(times_ms)
    return LatencyStats(
        median_ms=float(np.median(arr)),
        mean_ms=float(arr.mean()),
        p95_ms=float(np.percentile(arr, 95)),
        min_ms=float(arr.min()),
        std_ms=float(arr.std()),
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
        batch_size=int(sample.shape[0]),
    )


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def environment_info() -> dict:
    return {
        "device": "cpu",
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "python_version": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "quantization_engine": torch.backends.quantized.engine,
        "latency_note": (
            "Latency depends on the hardware, thread count and PyTorch build used for this run and "
            "does not transfer to other devices."
        ),
    }
