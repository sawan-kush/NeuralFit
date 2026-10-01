"""Shared builders for tiny, fully synthetic test inputs (no external data is downloaded)."""
from __future__ import annotations

import io
import json
import time
import zipfile

import numpy as np
import torch
from PIL import Image

from app.models.registry import get_arch

LABELS = ["cats", "dogs", "birds"]
INPUT_SIZE = 64


def make_state_dict(arch: str = "resnet18", num_classes: int = 3, seed: int = 0, favor_class: int | None = None):
    """Random-weight model. With ``favor_class`` the classifier ignores its input and always predicts
    that class, which gives an exactly known accuracy on a balanced dataset."""
    torch.manual_seed(seed)
    spec = get_arch(arch)
    model = spec.build_float(num_classes)
    if favor_class is not None:
        head_w = spec.classifier_weight_key
        head_b = head_w.replace("weight", "bias")
        sd = model.state_dict()
        sd[head_w].zero_()
        sd[head_b].zero_()
        sd[head_b][favor_class] = 5.0
        model.load_state_dict(sd)
    return {k: v.clone() for k, v in model.state_dict().items()}


def to_bytes(obj) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def make_image_bytes(class_idx: int, item: int, size: int = INPUT_SIZE, fmt: str = "JPEG") -> bytes:
    rng = np.random.default_rng(class_idx * 1000 + item)
    base = np.array([[[40 + 80 * class_idx, 200 - 60 * class_idx, 90]]], dtype=np.float32)
    pixels = np.clip(base + rng.normal(0, 25, (size, size, 3)), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format=fmt)
    return buf.getvalue()


def make_dataset_zip(labels=LABELS, per_class: int = 4, size: int = INPUT_SIZE, wrapper: str | None = None,
                     extra: dict[str, bytes] | None = None) -> bytes:
    buf = io.BytesIO()
    prefix = f"{wrapper}/" if wrapper else ""
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for ci, label in enumerate(labels):
            for i in range(per_class):
                zf.writestr(f"{prefix}{label}/img{i}.jpg", make_image_bytes(ci, i, size))
        for name, data in (extra or {}).items():
            zf.writestr(name, data)
    return buf.getvalue()


def session_config(labels=LABELS, input_size: int = INPUT_SIZE, batch_size: int = 4) -> str:
    return json.dumps(
        {"class_labels": list(labels), "preprocessing": {"input_size": input_size}, "batch_size": batch_size}
    )


def upload_session(client, *, architecture="resnet18", weights: bytes, dataset: bytes, config: str | None = None,
                   weights_name="weights.pt", dataset_name="dataset.zip"):
    return client.post(
        "/api/v1/sessions",
        data={"architecture": architecture, "config": config or session_config()},
        files={
            "weights": (weights_name, weights, "application/octet-stream"),
            "dataset_zip": (dataset_name, dataset, "application/zip"),
        },
    )


FAST_BENCH = {"warmup_runs": 1, "timed_runs": 5, "threads": 1}


def wait_for_run(client, run_id: str, timeout: float = 300.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/api/v1/runs/{run_id}").json()
        if status["status"] in ("completed", "failed"):
            return status
        time.sleep(0.3)
    raise TimeoutError("run did not finish in time")
