"""Dataset structure validation. Expected layout (one folder per class label, images inside):

    validation_dataset/
      class_a/image1.jpg
      class_b/image1.jpg
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from ..config import IMAGE_EXTENSIONS, Settings
from ..errors import InvalidDatasetError


@dataclass
class DatasetIndex:
    samples: list[tuple[Path, int]]
    class_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.samples)


def _subdirs(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_dir() and not p.name.startswith("."))


def _images_in(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _effective_root(root: Path) -> Path:
    """Allow one wrapper folder (e.g. ``validation_dataset/``) around the class folders."""
    dirs = _subdirs(root)
    if len(dirs) == 1 and not _images_in(root) and _subdirs(dirs[0]) and not _images_in(dirs[0]):
        return dirs[0]
    return root


def _check_readable(images: list[Path], settings: Settings) -> None:
    bad: list[str] = []
    for img in images:
        try:
            with Image.open(img) as im:
                width, height = im.size
                if width * height > settings.max_image_pixels:
                    raise ValueError("too many pixels")
                im.verify()
        except Exception:  # noqa: BLE001 - any decode problem means the image is unusable
            bad.append(f"{img.parent.name}/{img.name}")
            if len(bad) >= 5:
                break
    if bad:
        raise InvalidDatasetError(
            "Some images could not be read (first few): " + ", ".join(bad), code="invalid_image"
        )


def build_index(
    dataset_root: Path,
    class_labels: list[str],
    settings: Settings,
    *,
    verify_images: bool = True,
) -> DatasetIndex:
    if not dataset_root.is_dir():
        raise InvalidDatasetError("Dataset directory is missing.", code="empty_dataset")

    root = _effective_root(dataset_root)
    if _images_in(root):
        raise InvalidDatasetError(
            "Images must be inside class folders, not directly in the dataset root.", code="invalid_structure"
        )

    label_set = set(class_labels)
    found = _subdirs(root)
    unknown = [d.name for d in found if d.name not in label_set]
    if unknown:
        shown = ", ".join(unknown[:10]) + ("..." if len(unknown) > 10 else "")
        raise InvalidDatasetError(
            f"Dataset folders not listed in the class labels: {shown}. "
            "Folder names must exactly match the labels you provided.",
            code="label_mismatch",
        )

    samples: list[tuple[Path, int]] = []
    counts: dict[str, int] = {}
    warnings: list[str] = []
    for idx, label in enumerate(class_labels):
        folder = root / label
        if not folder.is_dir():
            warnings.append(f"No folder found for class '{label}'; it has no validation images.")
            counts[label] = 0
            continue
        if any(_images_in(sub) for sub in _subdirs(folder)):
            raise InvalidDatasetError(
                f"Class folder '{label}' contains nested folders with images; nesting is not supported.",
                code="invalid_structure",
            )
        images = _images_in(folder)
        counts[label] = len(images)
        if not images:
            warnings.append(f"Class '{label}' has no images.")
        samples.extend((img, idx) for img in images)

    if not samples:
        raise InvalidDatasetError("The dataset contains no supported images.", code="empty_dataset")
    if len(samples) > settings.max_dataset_images:
        raise InvalidDatasetError(
            f"The dataset has more than {settings.max_dataset_images} images.", code="archive_too_large"
        )

    if verify_images:
        _check_readable([p for p, _ in samples], settings)

    return DatasetIndex(samples=samples, class_counts=counts, warnings=warnings)
