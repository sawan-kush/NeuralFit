"""Safe extraction of a dataset ZIP: only image files are extracted, and unsafe archives are rejected."""
from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ..config import IMAGE_EXTENSIONS, Settings
from ..errors import InvalidDatasetError

_CHUNK = 1024 * 1024
_S_IFMT = 0o170000
_S_IFLNK = 0o120000


@dataclass
class ExtractReport:
    images_extracted: int
    files_ignored: int


def _unsafe(name: str) -> InvalidDatasetError:
    shown = name if len(name) <= 80 else name[:77] + "..."
    return InvalidDatasetError(
        f"Archive rejected: entry '{shown}' has an unsafe path or type.", code="unsafe_archive"
    )


def _clean_path(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or (len(normalized) > 1 and normalized[1] == ":"):
        raise _unsafe(name)
    path = PurePosixPath(normalized)
    if ".." in path.parts:
        raise _unsafe(name)
    return path


def _is_junk(path: PurePosixPath) -> bool:
    return (
        "__MACOSX" in path.parts
        or path.name == ".DS_Store"
        or path.name.startswith("._")
        or any(part.startswith(".") and part not in {"."} for part in path.parts)
    )


def safe_extract_images(zip_path: Path, dest: Path, settings: Settings) -> ExtractReport:
    if not zipfile.is_zipfile(zip_path):
        raise InvalidDatasetError("The dataset file is not a valid ZIP archive.", code="invalid_archive")

    dest.mkdir(parents=True, exist_ok=True)
    dest_root = dest.resolve()
    extracted = 0
    ignored = 0

    try:
        with zipfile.ZipFile(zip_path) as zf:
            infos = zf.infolist()
            if len(infos) > settings.max_zip_entries:
                raise InvalidDatasetError("The archive contains too many entries.", code="archive_too_large")

            total_declared = 0
            for info in infos:
                path = _clean_path(info.filename)  # validates every entry, including ignored ones
                if info.flag_bits & 0x1:
                    raise InvalidDatasetError("Encrypted archives are not supported.", code="unsafe_archive")
                if (info.external_attr >> 16) & _S_IFMT == _S_IFLNK:
                    raise _unsafe(info.filename)
                total_declared += info.file_size
            if total_declared > settings.max_extracted_bytes:
                raise InvalidDatasetError(
                    f"The archive expands to more than {settings.max_extracted_bytes // 1024**2} MB.",
                    code="archive_too_large",
                )

            for info in infos:
                if info.is_dir():
                    continue
                path = _clean_path(info.filename)
                if _is_junk(path) or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    ignored += 1
                    continue
                if info.file_size > settings.max_single_image_bytes:
                    raise InvalidDatasetError(
                        "An image in the archive exceeds the per-image size limit.", code="archive_too_large"
                    )
                if (
                    info.file_size > 1024 * 1024
                    and info.compress_size > 0
                    and info.file_size / info.compress_size > settings.max_compression_ratio
                ):
                    raise InvalidDatasetError(
                        "Archive rejected: suspicious compression ratio.", code="unsafe_archive"
                    )
                if extracted >= settings.max_dataset_images:
                    raise InvalidDatasetError(
                        f"The dataset has more than {settings.max_dataset_images} images.", code="archive_too_large"
                    )

                target = dest_root.joinpath(*path.parts).resolve()
                if not target.is_relative_to(dest_root):
                    raise _unsafe(info.filename)
                target.parent.mkdir(parents=True, exist_ok=True)

                written = 0
                with zf.open(info) as src, open(target, "wb") as out:
                    while chunk := src.read(_CHUNK):
                        written += len(chunk)
                        if written > info.file_size or written > settings.max_single_image_bytes:
                            raise _unsafe(info.filename)
                        out.write(chunk)
                extracted += 1
    except zipfile.BadZipFile:
        raise InvalidDatasetError("The dataset ZIP is corrupted.", code="invalid_archive") from None
    except (NotImplementedError, RuntimeError):
        raise InvalidDatasetError("The archive uses an unsupported ZIP feature.", code="invalid_archive") from None

    return ExtractReport(images_extracted=extracted, files_ignored=ignored)
