from __future__ import annotations

from fastapi import APIRouter

from .. import __version__
from ..models.registry import ARCHITECTURES
from ..optimizers import METHODS
from ..schemas import IMAGENET_MEAN, IMAGENET_STD, ArchitectureInfo, MethodInfo

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@router.get("/architectures", response_model=list[ArchitectureInfo])
def architectures() -> list[ArchitectureInfo]:
    return [
        ArchitectureInfo(
            key=a.key,
            display_name=a.display_name,
            description=a.description,
            default_input_size=a.default_input_size,
            default_mean=IMAGENET_MEAN,
            default_std=IMAGENET_STD,
            default_resize_size=int(round(a.default_input_size * 256 / 224)),
            classifier_weight_key=a.classifier_weight_key,
        )
        for a in ARCHITECTURES.values()
    ]


@router.get("/methods", response_model=list[MethodInfo])
def methods() -> list[MethodInfo]:
    return [m.info() for m in METHODS.values()]
