from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from ..errors import ConflictError
from ..schemas import RunRequest, RunResults, RunStatus

router = APIRouter()


@router.post("/runs", response_model=RunStatus, status_code=202)
def start_run(request: Request, body: RunRequest) -> RunStatus:
    return request.app.state.runs.submit(body)


@router.get("/runs/{run_id}", response_model=RunStatus)
def run_status(request: Request, run_id: str) -> RunStatus:
    return request.app.state.runs.get_status(run_id)


@router.get("/runs/{run_id}/results", response_model=RunResults)
def run_results(request: Request, run_id: str) -> RunResults:
    runs = request.app.state.runs
    status = runs.get_status(run_id)
    results = runs.get_results(run_id)
    if results is None:
        raise ConflictError(f"The run is {status.status}; results are not ready yet.", code="run_not_finished")
    return results


@router.get("/runs/{run_id}/artifacts/{name}")
def download_artifact(request: Request, run_id: str, name: str) -> FileResponse:
    path = request.app.state.runs.artifact_path(run_id, name)
    media = "application/json" if name.endswith(".json") else "application/octet-stream"
    return FileResponse(path, media_type=media, filename=name)
