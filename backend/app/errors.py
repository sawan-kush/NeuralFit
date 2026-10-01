"""Typed application errors and a single JSON error shape: {"error": {"code", "message"}}."""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("neuralfit")


class NeuralFitError(Exception):
    status_code = 400
    code = "bad_request"

    def __init__(self, message: str, *, code: str | None = None, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        if status_code:
            self.status_code = status_code


class InvalidRequestError(NeuralFitError):
    status_code = 400
    code = "invalid_request"


class UnsupportedArchitectureError(NeuralFitError):
    status_code = 400
    code = "unsupported_architecture"


class InvalidUploadError(NeuralFitError):
    status_code = 400
    code = "invalid_upload"


class PayloadTooLargeError(NeuralFitError):
    status_code = 413
    code = "payload_too_large"


class InvalidWeightsError(NeuralFitError):
    status_code = 422
    code = "invalid_weights"


class InvalidDatasetError(NeuralFitError):
    status_code = 422
    code = "invalid_dataset"


class NotFoundError(NeuralFitError):
    status_code = 404
    code = "not_found"


class ConflictError(NeuralFitError):
    status_code = 409
    code = "conflict"


class OptimizationUnsupportedError(NeuralFitError):
    """Raised by an optimizer when it cannot produce a valid model for this input."""

    status_code = 422
    code = "optimization_unsupported"


def error_body(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(NeuralFitError)
    async def _neuralfit(_: Request, exc: NeuralFitError):
        return JSONResponse(error_body(exc.code, exc.message), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        parts = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", []) if p != "body")
            parts.append(f"{loc}: {err.get('msg', 'invalid value')}" if loc else str(err.get("msg")))
        return JSONResponse(error_body("invalid_request", "; ".join(parts) or "Invalid request."), status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return JSONResponse(error_body(code, str(exc.detail)), status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):
        logger.exception("Unhandled error", exc_info=exc)
        return JSONResponse(error_body("internal_error", "Unexpected server error."), status_code=500)
