"""Filesystem store for uploaded inputs. One directory per session, server-chosen names only."""
from __future__ import annotations

import shutil
import time
from pathlib import Path

from ..config import Settings
from ..errors import NotFoundError
from ..schemas import SessionRecord
from .ids import is_valid_id, new_id


class SessionStore:
    def __init__(self, settings: Settings):
        self.root = settings.sessions_dir
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self) -> tuple[str, Path]:
        sid = new_id()
        path = self.root / sid
        path.mkdir()
        return sid, path

    def dir(self, session_id: str) -> Path:
        if not is_valid_id(session_id) or not (self.root / session_id).is_dir():
            raise NotFoundError("Session not found (it may have expired).", code="session_not_found")
        return self.root / session_id

    def weights_path(self, session_id: str) -> Path:
        return self.dir(session_id) / "weights.pt"

    def dataset_dir(self, session_id: str) -> Path:
        return self.dir(session_id) / "dataset"

    def save(self, record: SessionRecord) -> None:
        (self.dir(record.session_id) / "session.json").write_text(record.model_dump_json(), encoding="utf-8")

    def load(self, session_id: str) -> SessionRecord:
        record_file = self.dir(session_id) / "session.json"
        if not record_file.is_file():
            raise NotFoundError("Session not found (it may have expired).", code="session_not_found")
        return SessionRecord.model_validate_json(record_file.read_text(encoding="utf-8"))

    def delete(self, session_id: str) -> None:
        if is_valid_id(session_id):
            shutil.rmtree(self.root / session_id, ignore_errors=True)

    def cleanup_expired(self, retention_hours: int) -> int:
        return _remove_older_than(self.root, retention_hours)


def _remove_older_than(root: Path, retention_hours: int) -> int:
    cutoff = time.time() - retention_hours * 3600
    removed = 0
    for child in root.iterdir():
        if child.is_dir() and child.stat().st_mtime < cutoff:
            shutil.rmtree(child, ignore_errors=True)
            removed += 1
    return removed
