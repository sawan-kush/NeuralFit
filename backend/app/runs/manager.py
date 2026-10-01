"""Background execution of optimization runs.

A single worker thread runs one job at a time (local-development scope; no queue service).
Run state lives in memory and is mirrored to ``run.json`` in the run directory.
"""
from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from pydantic import ValidationError

from ..config import Settings
from ..data.loader import calibration_subset, make_loader, sample_input
from ..errors import (
    InvalidRequestError,
    NeuralFitError,
    NotFoundError,
    OptimizationUnsupportedError,
)
from ..evaluation.accuracy import evaluate_top1
from ..evaluation.benchmark import count_parameters, environment_info, measure_latency
from ..evaluation.comparison import compare
from ..models.checkpoint import save_checkpoint
from ..models.registry import get_arch
from ..optimizers import get_optimizer
from ..optimizers.base import OptimizationContext
from ..schemas import (
    ArtifactInfo,
    BenchmarkConfig,
    ErrorInfo,
    Metrics,
    ModelResult,
    RunRecord,
    RunRequest,
    RunResults,
    RunStatus,
    SessionInfo,
    SessionRecord,
)
from ..storage.artifacts import ArtifactStore
from ..storage.ids import is_valid_id, new_id
from ..storage.sessions import SessionStore
from ..validation.dataset import build_index
from ..validation.weights import load_and_validate_weights

logger = logging.getLogger("neuralfit.runs")

LIMITATIONS = [
    "Latency is measured on CPU with batch size 1 on the machine that ran this benchmark. It depends on "
    "hardware, thread count and PyTorch build, and does not transfer to other devices.",
    "Accuracy is top-1 on the supplied validation set only; it may not represent your deployment data.",
    "File size is the size of the serialized NeuralFit checkpoint (state dict plus small metadata); the "
    "baseline is serialized with the same code path so sizes are comparable.",
    "Optimizations are not guaranteed to improve accuracy, size or latency. Every value above is a "
    "measurement from this run; entries marked 'worse' got worse.",
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RunManager:
    def __init__(self, settings: Settings, sessions: SessionStore, artifacts: ArtifactStore):
        self.settings = settings
        self.sessions = sessions
        self.artifacts = artifacts
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="neuralfit-run")
        self._lock = threading.Lock()
        self._records: dict[str, RunRecord] = {}

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ public API
    def submit(self, request: RunRequest) -> RunStatus:
        self.sessions.load(request.session_id)  # raises NotFoundError

        stray = set(request.method_configs) - set(request.methods)
        if stray:
            raise InvalidRequestError(f"method_configs has entries for methods not selected: {sorted(stray)}.")

        normalized: dict[str, dict] = {}
        for key in request.methods:
            optimizer = get_optimizer(key)
            ok, reason = optimizer.availability()
            if not ok:
                raise OptimizationUnsupportedError(reason or f"{optimizer.label} is unavailable.")
            try:
                cfg = optimizer.parse_config(request.method_configs.get(key, {}))
            except ValidationError as exc:
                problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
                raise InvalidRequestError(f"Invalid options for '{key}': {problems}.") from None
            normalized[key] = cfg.model_dump()

        if request.benchmark.threads > (os.cpu_count() or 1) * 2:
            raise InvalidRequestError("threads is unreasonably high for this machine.")

        request = request.model_copy(update={"method_configs": normalized})
        run_id = new_id()
        self.artifacts.run_dir(run_id, create=True)
        now = _now()
        record = RunRecord(
            run_id=run_id,
            session_id=request.session_id,
            created_at=now,
            updated_at=now,
            total_steps=1 + len(request.methods),
            request=request,
        )
        with self._lock:
            self._records[run_id] = record
            self._persist(record)
        self._pool.submit(self._execute, run_id)
        return self._status(record)

    def get_status(self, run_id: str) -> RunStatus:
        return self._status(self._get(run_id))

    def get_results(self, run_id: str) -> RunResults | None:
        return self._get(run_id).results

    def artifact_path(self, run_id: str, name: str) -> Path:
        record = self._get(run_id)
        allowed = {a.name for a in record.results.artifacts} if record.results else set()
        return self.artifacts.resolve(run_id, name, allowed)

    def has_active_run(self, session_id: str) -> bool:
        with self._lock:
            return any(
                r.session_id == session_id and r.status in ("queued", "running") for r in self._records.values()
            )

    # ------------------------------------------------------------------ state helpers
    def _get(self, run_id: str) -> RunRecord:
        if not is_valid_id(run_id):
            raise NotFoundError("Run not found.", code="run_not_found")
        with self._lock:
            record = self._records.get(run_id)
            if record is not None:
                return record.model_copy(deep=True)
        run_file = self.artifacts.run_dir(run_id) / "run.json"
        if not run_file.is_file():
            raise NotFoundError("Run not found (it may have expired).", code="run_not_found")
        record = RunRecord.model_validate_json(run_file.read_text(encoding="utf-8"))
        if record.status in ("queued", "running"):  # process restarted mid-run
            record.status = "failed"
            record.stage = "Interrupted"
            record.error = ErrorInfo(code="interrupted", message="The server restarted while this run was active.")
        return record

    def _persist(self, record: RunRecord) -> None:
        run_dir = self.artifacts.run_dir(record.run_id, create=True)
        tmp = run_dir / "run.json.tmp"
        tmp.write_text(record.model_dump_json(), encoding="utf-8")
        os.replace(tmp, run_dir / "run.json")

    def _update(self, run_id: str, **fields) -> None:
        with self._lock:
            record = self._records[run_id]
            for key, value in fields.items():
                setattr(record, key, value)
            record.updated_at = _now()
            self._persist(record)

    @staticmethod
    def _status(record: RunRecord) -> RunStatus:
        return RunStatus(
            run_id=record.run_id,
            session_id=record.session_id,
            status=record.status,
            stage=record.stage,
            completed_steps=record.completed_steps,
            total_steps=record.total_steps,
            created_at=record.created_at,
            updated_at=record.updated_at,
            error=record.error,
        )

    # ------------------------------------------------------------------ execution
    def _execute(self, run_id: str) -> None:
        try:
            self._pipeline(run_id)
        except NeuralFitError as exc:
            self._fail(run_id, exc.code, exc.message)
        except Exception:  # noqa: BLE001
            logger.exception("Run %s failed unexpectedly", run_id)
            self._fail(run_id, "internal_error", "The run failed unexpectedly. Check the server logs.")

    def _fail(self, run_id: str, code: str, message: str) -> None:
        try:
            session = self.sessions.load(self._get(run_id).session_id)
            session_info = _session_info(session)
        except Exception:  # noqa: BLE001
            session_info = SessionInfo(
                architecture="unknown", num_classes=0, total_images=0, class_counts={},
                preprocessing={}, batch_size=0, uploaded_weights_size_bytes=0,
            )
        record = self._get(run_id)
        error = ErrorInfo(code=code, message=message)
        now = _now()
        results = RunResults(
            run_id=run_id, status="failed", created_at=record.created_at, finished_at=now,
            session=session_info, benchmark=record.request.benchmark, environment={},
            limitations=LIMITATIONS, error=error,
        )
        self._update(run_id, status="failed", stage="Failed", error=error, results=results)

    def _measure(
        self,
        model: nn.Module,
        dtype: torch.dtype,
        loader,
        sample: torch.Tensor,
        bench: BenchmarkConfig,
        artifact: Path,
    ) -> Metrics:
        acc = evaluate_top1(model, loader, dtype)
        latency = measure_latency(
            model, sample.to(dtype), warmup_runs=bench.warmup_runs, timed_runs=bench.timed_runs
        )
        return Metrics(
            top1_accuracy_pct=acc.top1_pct,
            correct=acc.correct,
            total=acc.total,
            size_bytes=artifact.stat().st_size,
            param_count=count_parameters(model),
            latency=latency,
        )

    def _pipeline(self, run_id: str) -> None:
        record = self._get(run_id)
        request = record.request
        bench = request.benchmark
        session = self.sessions.load(request.session_id)
        spec = get_arch(session.architecture)
        cfg = session.config
        pre = cfg.preprocessing
        pre_dict = {**pre.model_dump(), "resize_size": pre.effective_resize_size}
        run_dir = self.artifacts.run_dir(run_id)

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(bench.threads)
        try:
            self._update(run_id, status="running", stage="Loading model and dataset")
            state, model = load_and_validate_weights(
                spec, self.sessions.weights_path(session.session_id), session.num_classes,
                self.settings.max_weights_bytes,
            )
            index = build_index(
                self.sessions.dataset_dir(session.session_id), cfg.class_labels, self.settings, verify_images=False
            )
            loader = make_loader(index.samples, pre, cfg.batch_size)
            sample = sample_input(index.samples, pre)
            environment = environment_info()

            def checkpoint(path: Path, state_dict, method: str, method_config: dict) -> None:
                save_checkpoint(
                    path, state_dict=state_dict, architecture=spec.key, method=method,
                    num_classes=session.num_classes, class_labels=cfg.class_labels,
                    preprocessing=pre_dict, method_config=method_config,
                )

            # ---- baseline
            self._update(run_id, stage="Evaluating original model (FP32)")
            baseline_path = run_dir / "baseline_fp32.pt"
            checkpoint(baseline_path, state, "fp32_baseline", {})
            baseline_metrics = self._measure(model, torch.float32, loader, sample, bench, baseline_path)
            baseline = ModelResult(
                method="fp32_baseline", label="Original (FP32)", status="ok", metrics=baseline_metrics,
                execution_mode="fp32", artifact_name=baseline_path.name,
                notes=["Uploaded weights re-serialized with the same checkpoint format as the optimized models."],
            )
            self._update(run_id, completed_steps=1)

            # ---- optimizations
            results: list[ModelResult] = []
            comparisons = {}
            for step, key in enumerate(request.methods, start=2):
                optimizer = get_optimizer(key)
                method_cfg = optimizer.config_model.model_validate(request.method_configs[key])
                artifact_path = run_dir / f"model_{key}.pt"
                self._update(run_id, stage=f"Optimizing: {optimizer.label}")
                try:
                    calib_n = getattr(method_cfg, "calibration_samples", 1)
                    calib_samples = calibration_subset(index.samples, calib_n, self.settings.seed)
                    ctx = OptimizationContext(
                        spec=spec, num_classes=session.num_classes, state_dict=state, float_model=model,
                        calibration_loader=make_loader(calib_samples, pre, min(cfg.batch_size, 32)),
                        calibration_size=len(calib_samples), sample=sample, settings=self.settings,
                    )
                    with optimizer.runtime(method_cfg):
                        optimized = optimizer.optimize(ctx, method_cfg)
                        checkpoint(artifact_path, optimized.state_dict, key, method_cfg.model_dump())
                        self._update(run_id, stage=f"Evaluating: {optimizer.label}")
                        metrics = self._measure(
                            optimized.model, optimized.input_dtype, loader, sample, bench, artifact_path
                        )
                    result = ModelResult(
                        method=key, label=optimizer.label, config=method_cfg.model_dump(), status="ok",
                        metrics=metrics, execution_mode=optimized.execution_mode, notes=optimized.notes,
                        artifact_name=artifact_path.name,
                    )
                    comparisons[key] = compare(baseline_metrics, metrics)
                except NeuralFitError as exc:
                    artifact_path.unlink(missing_ok=True)
                    result = ModelResult(
                        method=key, label=optimizer.label, config=method_cfg.model_dump(), status="failed",
                        error=ErrorInfo(code=exc.code, message=exc.message),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Method %s failed in run %s", key, run_id)
                    artifact_path.unlink(missing_ok=True)
                    result = ModelResult(
                        method=key, label=optimizer.label, config=method_cfg.model_dump(), status="failed",
                        error=ErrorInfo(
                            code="optimization_failed",
                            message=f"{optimizer.label} failed unexpectedly ({type(exc).__name__}).",
                        ),
                    )
                results.append(result)
                self._update(run_id, completed_steps=step)

            # ---- report
            self._update(run_id, stage="Writing report")
            model_artifacts = [
                ArtifactInfo(name=r.artifact_name, kind="model", size_bytes=(run_dir / r.artifact_name).stat().st_size)
                for r in [baseline, *results]
                if r.status == "ok" and r.artifact_name
            ]
            final = RunResults(
                run_id=run_id, status="completed", created_at=record.created_at, finished_at=_now(),
                session=_session_info(session), benchmark=bench, environment=environment,
                baseline=baseline, methods=results, comparisons=comparisons,
                artifacts=model_artifacts, limitations=LIMITATIONS,
            )
            report_path = run_dir / "report.json"
            report_path.write_text(final.model_dump_json(indent=2), encoding="utf-8")
            final = final.model_copy(
                update={
                    "artifacts": [
                        *model_artifacts,
                        ArtifactInfo(name="report.json", kind="report", size_bytes=report_path.stat().st_size),
                    ]
                }
            )
            self._update(run_id, status="completed", stage="Completed", results=final)
        finally:
            torch.set_num_threads(previous_threads)


def _session_info(session: SessionRecord) -> SessionInfo:
    return SessionInfo(
        architecture=session.architecture,
        num_classes=session.num_classes,
        total_images=session.total_images,
        class_counts=session.class_counts,
        preprocessing=session.config.preprocessing,
        batch_size=session.config.batch_size,
        uploaded_weights_size_bytes=session.uploaded_weights_size_bytes,
    )
