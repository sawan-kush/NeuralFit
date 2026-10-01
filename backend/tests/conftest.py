from __future__ import annotations

import dataclasses

import pytest
import torch
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture(autouse=True)
def _restore_quant_engine():
    previous = torch.backends.quantized.engine
    yield
    torch.backends.quantized.engine = previous


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path / "data", frontend_dist=None, retention_hours=24)


@pytest.fixture()
def make_client(settings):
    clients = []

    def _make(**overrides) -> TestClient:
        cfg = dataclasses.replace(settings, **overrides) if overrides else settings
        client = TestClient(create_app(cfg))
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture()
def client(make_client) -> TestClient:
    return make_client()
