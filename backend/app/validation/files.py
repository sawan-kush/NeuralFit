"""Streaming upload handling with size and extension limits."""
from __future__ import annotations

from pathlib import Path, PurePath

from fastapi import UploadFile

from ..errors import InvalidUploadError, PayloadTooLargeError

_CHUNK = 1024 * 1024


def save_upload(
    upload: UploadFile,
    dest: Path,
    *,
    max_bytes: int,
    allowed_extensions: frozenset[str],
    label: str,
) -> int:
    """Copy an upload to ``dest`` (a server-chosen path). The client filename is only used for
    its extension and never for a filesystem location. Returns the number of bytes written."""
    name = PurePath(upload.filename or "").name
    suffix = PurePath(name).suffix.lower()
    if suffix not in allowed_extensions:
        allowed = ", ".join(sorted(allowed_extensions))
        raise InvalidUploadError(f"The {label} file must have one of these extensions: {allowed}.")

    written = 0
    try:
        with open(dest, "wb") as out:
            while chunk := upload.file.read(_CHUNK):
                written += len(chunk)
                if written > max_bytes:
                    raise PayloadTooLargeError(
                        f"The {label} file exceeds the {max_bytes // (1024 * 1024)} MB limit."
                    )
                out.write(chunk)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    if written == 0:
        dest.unlink(missing_ok=True)
        raise InvalidUploadError(f"The {label} file is empty.")
    return written
