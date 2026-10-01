from __future__ import annotations

from ..errors import InvalidRequestError
from .base import Optimizer
from .fp16 import FP16Optimizer
from .int8_ptq import Int8PTQOptimizer

# Pruning is intentionally absent: it is not exposed until it is implemented and evaluated honestly.
METHODS: dict[str, Optimizer] = {o.key: o for o in (Int8PTQOptimizer(), FP16Optimizer())}


def get_optimizer(key: str) -> Optimizer:
    opt = METHODS.get(key)
    if opt is None:
        raise InvalidRequestError(f"Unknown optimization method '{key}'. Available: {', '.join(METHODS)}.")
    return opt
