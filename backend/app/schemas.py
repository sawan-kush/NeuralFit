"""Pydantic models: request configs, API responses and result documents."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------- inputs
class Preprocessing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_size: int = Field(224, ge=32, le=512, description="Square crop fed to the model.")
    resize_size: int | None = Field(
        None, ge=32, le=1024, description="Shorter-side resize before the crop. Default: round(input_size * 256 / 224)."
    )
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD

    @field_validator("std")
    @classmethod
    def _std_positive(cls, v):
        if any(x <= 0 for x in v):
            raise ValueError("std values must be > 0")
        return v

    @model_validator(mode="after")
    def _resize_not_smaller(self):
        if self.resize_size is not None and self.resize_size < self.input_size:
            raise ValueError("resize_size must be >= input_size")
        return self

    @property
    def effective_resize_size(self) -> int:
        return self.resize_size or int(round(self.input_size * 256 / 224))


class SessionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_labels: list[str] = Field(min_length=2, max_length=1000, description="Ordered like the model's outputs.")
    preprocessing: Preprocessing = Field(default_factory=Preprocessing)
    batch_size: int = Field(32, ge=1, le=256, description="Batch size for accuracy evaluation.")

    @field_validator("class_labels")
    @classmethod
    def _labels_ok(cls, labels: list[str]) -> list[str]:
        cleaned = [l.strip() for l in labels]
        for l in cleaned:
            if not l or len(l) > 128:
                raise ValueError("each class label must be 1-128 characters")
            if "/" in l or "\\" in l or l in {".", ".."}:
                raise ValueError("class labels must be valid folder names (no slashes)")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("class labels must be unique")
        return cleaned


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    warmup_runs: int = Field(20, ge=1, le=500)
    timed_runs: int = Field(100, ge=5, le=2000)
    threads: int = Field(1, ge=1, le=64, description="CPU threads used for inference.")


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    methods: list[str] = Field(min_length=1, max_length=8)
    method_configs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)

    @field_validator("methods")
    @classmethod
    def _unique(cls, v):
        if len(set(v)) != len(v):
            raise ValueError("methods must not contain duplicates")
        return v


# ----------------------------------------------------------------------------- meta
class ArchitectureInfo(BaseModel):
    key: str
    display_name: str
    description: str
    default_input_size: int
    default_mean: tuple[float, float, float]
    default_std: tuple[float, float, float]
    default_resize_size: int
    classifier_weight_key: str


class OptionInfo(BaseModel):
    name: str
    label: str
    type: Literal["integer", "choice"]
    default: Any
    minimum: int | None = None
    maximum: int | None = None
    choices: list[str] | None = None
    description: str = ""


class MethodInfo(BaseModel):
    key: str
    label: str
    description: str
    limitations: list[str]
    options: list[OptionInfo]
    available: bool
    unavailable_reason: str | None = None


# ----------------------------------------------------------------------------- sessions
class SessionResponse(BaseModel):
    session_id: str
    architecture: str
    num_classes: int
    total_images: int
    class_counts: dict[str, int]
    preprocessing: Preprocessing
    batch_size: int
    uploaded_weights_size_bytes: int
    warnings: list[str]
    retention_hours: int


class SessionRecord(BaseModel):
    session_id: str
    created_at: datetime
    architecture: str
    num_classes: int
    config: SessionConfig
    total_images: int
    class_counts: dict[str, int]
    uploaded_weights_size_bytes: int
    warnings: list[str] = Field(default_factory=list)


# ----------------------------------------------------------------------------- results
class ErrorInfo(BaseModel):
    code: str
    message: str


class LatencyStats(BaseModel):
    median_ms: float
    mean_ms: float
    p95_ms: float
    min_ms: float
    std_ms: float
    warmup_runs: int
    timed_runs: int
    batch_size: int = 1


class Metrics(BaseModel):
    top1_accuracy_pct: float
    correct: int
    total: int
    size_bytes: int
    param_count: int
    latency: LatencyStats


class ModelResult(BaseModel):
    method: str
    label: str
    config: dict[str, Any] = Field(default_factory=dict)
    status: Literal["ok", "failed"]
    metrics: Metrics | None = None
    execution_mode: str | None = None
    notes: list[str] = Field(default_factory=list)
    error: ErrorInfo | None = None
    artifact_name: str | None = None


class ComparisonEntry(BaseModel):
    baseline: float
    optimized: float
    absolute_change: float = Field(description="optimized - baseline (percentage points for accuracy).")
    percent_change: float | None = Field(description="Relative change vs baseline in %; null if baseline is 0.")
    direction: Literal["improved", "worse", "unchanged"]


class ArtifactInfo(BaseModel):
    name: str
    kind: Literal["model", "report"]
    size_bytes: int


class SessionInfo(BaseModel):
    architecture: str
    num_classes: int
    total_images: int
    class_counts: dict[str, int]
    preprocessing: Preprocessing
    batch_size: int
    uploaded_weights_size_bytes: int


class RunStatus(BaseModel):
    run_id: str
    session_id: str
    status: Literal["queued", "running", "completed", "failed"]
    stage: str
    completed_steps: int
    total_steps: int
    created_at: datetime
    updated_at: datetime
    error: ErrorInfo | None = None


class RunResults(BaseModel):
    run_id: str
    status: Literal["completed", "failed"]
    created_at: datetime
    finished_at: datetime
    session: SessionInfo
    benchmark: BenchmarkConfig
    environment: dict[str, Any]
    baseline: ModelResult | None = None
    methods: list[ModelResult] = Field(default_factory=list)
    comparisons: dict[str, dict[str, ComparisonEntry]] = Field(default_factory=dict)
    artifacts: list[ArtifactInfo] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    error: ErrorInfo | None = None


class RunRecord(BaseModel):
    """Persisted as run.json inside the run directory."""

    run_id: str
    session_id: str
    status: Literal["queued", "running", "completed", "failed"] = "queued"
    stage: str = "Queued"
    completed_steps: int = 0
    total_steps: int = 0
    created_at: datetime
    updated_at: datetime
    error: ErrorInfo | None = None
    request: RunRequest
    results: RunResults | None = None
