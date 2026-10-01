"""NeuralFit API application factory."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api import routes_meta, routes_runs, routes_sessions
from .config import Settings
from .errors import error_body, register_error_handlers
from .runs.manager import RunManager
from .storage.artifacts import ArtifactStore
from .storage.sessions import SessionStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        app.state.settings = settings
        app.state.sessions = SessionStore(settings)
        app.state.artifacts = ArtifactStore(settings)
        app.state.runs = RunManager(settings, app.state.sessions, app.state.artifacts)
        app.state.sessions.cleanup_expired(settings.retention_hours)
        app.state.artifacts.cleanup_expired(settings.retention_hours)
        yield
        app.state.runs.shutdown()

    app = FastAPI(
        title="NeuralFit",
        version=__version__,
        description="Benchmark FP16 and INT8 optimizations of supported PyTorch image classifiers on CPU.",
        lifespan=lifespan,
    )
    register_error_handlers(app)

    @app.middleware("http")
    async def limit_request_size(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > settings.max_request_bytes:
            return JSONResponse(
                error_body("payload_too_large", "The upload exceeds the maximum allowed request size."),
                status_code=413,
            )
        return await call_next(request)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    for router in (routes_meta.router, routes_sessions.router, routes_runs.router):
        app.include_router(router, prefix="/api/v1")

    if settings.frontend_dist and settings.frontend_dist.is_dir():
        app.mount("/", StaticFiles(directory=settings.frontend_dist, html=True), name="frontend")

    return app


app = create_app()
