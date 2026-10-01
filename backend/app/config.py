"""Central configuration: limits, paths and defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
WEIGHT_EXTENSIONS = frozenset({".pt", ".pth"})
ZIP_EXTENSIONS = frozenset({".zip"})


def _default_data_dir() -> Path:
    return Path(os.environ.get("NEURALFIT_DATA_DIR", "neuralfit_data")).resolve()


def _default_frontend_dist() -> Path | None:
    env = os.environ.get("NEURALFIT_FRONTEND_DIST")
    if env:
        return Path(env).resolve()
    candidate = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    return candidate if candidate.is_dir() else None


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=_default_data_dir)

    # Upload limits
    max_weights_bytes: int = 300 * 1024**2
    max_dataset_zip_bytes: int = 500 * 1024**2
    max_extracted_bytes: int = 1024**3
    max_single_image_bytes: int = 20 * 1024**2
    max_dataset_images: int = 10_000
    max_zip_entries: int = 50_000
    max_image_pixels: int = 50_000_000
    # Reject archive members that expand by more than this factor (zip-bomb guard).
    max_compression_ratio: int = 100

    # Lifecycle
    retention_hours: int = 24

    # Calibration / reproducibility
    seed: int = 0

    cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")
    frontend_dist: Path | None = field(default_factory=_default_frontend_dist)

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def max_request_bytes(self) -> int:
        return self.max_weights_bytes + self.max_dataset_zip_bytes + 2 * 1024**2
