from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, File, Form, Request, Response, UploadFile
from pydantic import ValidationError

from ..config import WEIGHT_EXTENSIONS, ZIP_EXTENSIONS
from ..errors import ConflictError, InvalidUploadError
from ..models.registry import get_arch
from ..schemas import SessionConfig, SessionRecord, SessionResponse
from ..validation.archive import safe_extract_images
from ..validation.dataset import build_index
from ..validation.files import save_upload
from ..validation.weights import load_and_validate_weights

router = APIRouter()


def _parse_config(raw: str) -> SessionConfig:
    try:
        return SessionConfig.model_validate(json.loads(raw))
    except json.JSONDecodeError:
        raise InvalidUploadError("'config' must be valid JSON.", code="invalid_config") from None
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        raise InvalidUploadError(f"Invalid config: {problems}", code="invalid_config") from None


def _response(record: SessionRecord, retention_hours: int) -> SessionResponse:
    return SessionResponse(
        session_id=record.session_id,
        architecture=record.architecture,
        num_classes=record.num_classes,
        total_images=record.total_images,
        class_counts=record.class_counts,
        preprocessing=record.config.preprocessing,
        batch_size=record.config.batch_size,
        uploaded_weights_size_bytes=record.uploaded_weights_size_bytes,
        warnings=record.warnings,
        retention_hours=retention_hours,
    )


@router.post("/sessions", response_model=SessionResponse, status_code=201)
def create_session(
    request: Request,
    architecture: str = Form(...),
    config: str = Form(..., description="JSON: class_labels, preprocessing, batch_size"),
    weights: UploadFile = File(..., description="State dict saved with torch.save(model.state_dict())"),
    dataset_zip: UploadFile = File(..., description="ZIP with one folder per class"),
) -> SessionResponse:
    settings = request.app.state.settings
    sessions = request.app.state.sessions

    spec = get_arch(architecture)
    cfg = _parse_config(config)

    session_id, session_dir = sessions.create()
    try:
        weights_path = session_dir / "weights.pt"
        weights_bytes = save_upload(
            weights, weights_path, max_bytes=settings.max_weights_bytes,
            allowed_extensions=WEIGHT_EXTENSIONS, label="weights",
        )
        # Validate the cheap input first: fail fast before touching the dataset.
        load_and_validate_weights(spec, weights_path, len(cfg.class_labels), settings.max_weights_bytes)

        zip_path = session_dir / "dataset.zip"
        save_upload(
            dataset_zip, zip_path, max_bytes=settings.max_dataset_zip_bytes,
            allowed_extensions=ZIP_EXTENSIONS, label="dataset",
        )
        dataset_dir = session_dir / "dataset"
        report = safe_extract_images(zip_path, dataset_dir, settings)
        zip_path.unlink(missing_ok=True)

        index = build_index(dataset_dir, cfg.class_labels, settings, verify_images=True)
        warnings = list(index.warnings)
        if report.files_ignored:
            warnings.append(f"{report.files_ignored} non-image or hidden file(s) in the archive were ignored.")

        record = SessionRecord(
            session_id=session_id,
            created_at=datetime.now(timezone.utc),
            architecture=spec.key,
            num_classes=len(cfg.class_labels),
            config=cfg,
            total_images=index.total,
            class_counts=index.class_counts,
            uploaded_weights_size_bytes=weights_bytes,
            warnings=warnings,
        )
        sessions.save(record)
    except BaseException:
        sessions.delete(session_id)
        raise
    return _response(record, settings.retention_hours)


@router.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session(request: Request, session_id: str) -> SessionResponse:
    record = request.app.state.sessions.load(session_id)
    return _response(record, request.app.state.settings.retention_hours)


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(request: Request, session_id: str) -> Response:
    sessions = request.app.state.sessions
    sessions.dir(session_id)  # 404 if unknown
    if request.app.state.runs.has_active_run(session_id):
        raise ConflictError("This session has a queued or running optimization run.", code="session_in_use")
    sessions.delete(session_id)
    return Response(status_code=204)
