"""Run directories and controlled artifact downloads."""
from __future__ import annotations

import shutil
from pathlib import Path

from ..config import Settings
from ..errors import NotFoundError
from .ids import is_safe_name, is_valid_id
from .sessions import _remove_older_than


class ArtifactStore:
    def __init__(self, settings: Settings):
        self.root = settings.runs_dir
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str, *, create: bool = False) -> Path:
        if not is_valid_id(run_id):
            raise NotFoundError("Run not found.", code="run_not_found")
        path = self.root / run_id
        if create:
            path.mkdir(parents=True, exist_ok=True)
        elif not path.is_dir():
            raise NotFoundError("Run not found (it may have expired).", code="run_not_found")
        return path

    def resolve(self, run_id: str, name: str, allowed: set[str]) -> Path:
        """Return the file for ``name`` only if it is a recorded artifact of this run."""
        if not is_safe_name(name) or name not in allowed:
            raise NotFoundError("Artifact not found.", code="artifact_not_found")
        path = self.run_dir(run_id) / name
        if not path.is_file():
            raise NotFoundError("Artifact not found.", code="artifact_not_found")
        return path

    def delete_run(self, run_id: str) -> None:
        if is_valid_id(run_id):
            shutil.rmtree(self.root / run_id, ignore_errors=True)

    def cleanup_expired(self, retention_hours: int) -> int:
        return _remove_older_than(self.root, retention_hours)
