"""End-to-end API tests: session -> run -> results -> downloads, plus failure and error paths."""
import json
import threading

import pytest
import torch
from fastapi.testclient import TestClient

from helpers import (
    FAST_BENCH, INPUT_SIZE, LABELS, make_dataset_zip, make_state_dict, session_config, to_bytes, upload_session,
    wait_for_run,
)
from app.config import Settings
from app.data.loader import make_loader
from app.evaluation.accuracy import evaluate_top1
from app.main import create_app
from app.models.checkpoint import load_checkpoint
from app.schemas import Preprocessing
from app.validation.dataset import build_index

API = "/api/v1"


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    """One real run (ResNet-18 + both methods) shared by the read-only assertions below."""
    settings = Settings(data_dir=tmp_path_factory.mktemp("e2e"), frontend_dist=None)
    with TestClient(create_app(settings)) as client:
        session = upload_session(
            client, weights=to_bytes(make_state_dict("resnet18", favor_class=1)), dataset=make_dataset_zip()
        ).json()
        r = client.post(f"{API}/runs", json={
            "session_id": session["session_id"], "methods": ["int8_ptq", "fp16"],
            "method_configs": {"int8_ptq": {"calibration_samples": 12}}, "benchmark": FAST_BENCH,
        })
        assert r.status_code == 202, r.text
        run_id = r.json()["run_id"]
        status = wait_for_run(client, run_id)
        results = client.get(f"{API}/runs/{run_id}/results").json()
        yield {"client": client, "settings": settings, "session": session, "run_id": run_id,
               "status": status, "results": results}


def test_run_completes_and_reports_progress(e2e):
    s = e2e["status"]
    assert s["status"] == "completed" and s["completed_steps"] == s["total_steps"] == 3


def test_results_structure(e2e):
    res = e2e["results"]
    assert res["status"] == "completed" and res["error"] is None
    base = res["baseline"]
    assert base["status"] == "ok" and base["method"] == "fp32_baseline"
    m = base["metrics"]
    assert set(m) == {"top1_accuracy_pct", "correct", "total", "size_bytes", "param_count", "latency"}
    assert set(m["latency"]) >= {"median_ms", "mean_ms", "p95_ms", "warmup_runs", "timed_runs", "batch_size"}
    assert m["total"] == 12 and m["latency"]["timed_runs"] == 5 and m["latency"]["batch_size"] == 1
    assert m["top1_accuracy_pct"] == pytest.approx(100 / 3)  # constant predictor -> exactly 4/12
    assert m["param_count"] == 11_178_051 and m["size_bytes"] > 40_000_000

    assert [x["method"] for x in res["methods"]] == ["int8_ptq", "fp16"]
    for method in res["methods"]:
        assert method["status"] == "ok" and method["metrics"]["latency"]["median_ms"] > 0
        cmp = res["comparisons"][method["method"]]
        assert set(cmp) == {"top1_accuracy_pct", "size_bytes", "latency_median_ms", "param_count"}
        for entry in cmp.values():
            assert entry["direction"] in {"improved", "worse", "unchanged"}
            assert set(entry) == {"baseline", "optimized", "absolute_change", "percent_change", "direction"}
    int8, fp16 = res["methods"]
    assert res["comparisons"]["int8_ptq"]["size_bytes"]["direction"] == "improved"
    assert res["comparisons"]["fp16"]["size_bytes"]["percent_change"] < -40
    assert int8["config"] == {"calibration_samples": 12, "backend": "fbgemm"}
    assert fp16["execution_mode"] == "native_fp16"

    assert res["environment"]["device"] == "cpu" and res["environment"]["torch_threads"] == 1
    assert res["benchmark"] == FAST_BENCH
    assert res["session"]["num_classes"] == 3 and res["limitations"]


def test_latency_is_reported_as_measured_not_assumed(e2e):
    """FP16 on CPU may be slower; whatever the direction, the flag must match the numbers."""
    for key, cmp in e2e["results"]["comparisons"].items():
        lat = cmp["latency_median_ms"]
        expected = "worse" if lat["optimized"] > lat["baseline"] else "improved"
        assert lat["direction"] == expected
        assert lat["absolute_change"] == pytest.approx(lat["optimized"] - lat["baseline"])


def test_artifact_list_and_downloads(e2e):
    c, run_id = e2e["client"], e2e["run_id"]
    names = {a["name"]: a for a in e2e["results"]["artifacts"]}
    assert set(names) == {"baseline_fp32.pt", "model_int8_ptq.pt", "model_fp16.pt", "report.json"}
    for name, info in names.items():
        r = c.get(f"{API}/runs/{run_id}/artifacts/{name}")
        assert r.status_code == 200 and len(r.content) == info["size_bytes"]
        assert name in r.headers["content-disposition"]
    assert names["model_int8_ptq.pt"]["size_bytes"] == e2e["results"]["methods"][0]["metrics"]["size_bytes"]

    report = json.loads(c.get(f"{API}/runs/{run_id}/artifacts/report.json").content)
    assert report["run_id"] == run_id and report["baseline"]["metrics"]["total"] == 12
    assert [m["method"] for m in report["methods"]] == ["int8_ptq", "fp16"]
    assert str(e2e["settings"].data_dir) not in json.dumps(report)


@pytest.mark.parametrize("name,method", [("model_fp16.pt", "fp16"), ("model_int8_ptq.pt", "int8_ptq")])
def test_downloaded_model_reproduces_reported_accuracy(e2e, tmp_path, name, method):
    """The artifact a user downloads must behave exactly like the model that was benchmarked."""
    c, run_id = e2e["client"], e2e["run_id"]
    path = tmp_path / name
    path.write_bytes(c.get(f"{API}/runs/{run_id}/artifacts/{name}").content)
    model, meta = load_checkpoint(path)
    assert meta["class_labels"] == LABELS and meta["preprocessing"]["input_size"] == INPUT_SIZE

    sessions = e2e["client"].app.state.sessions
    sid = e2e["session"]["session_id"]
    index = build_index(sessions.dataset_dir(sid), LABELS, e2e["settings"], verify_images=False)
    loader = make_loader(index.samples, Preprocessing(input_size=INPUT_SIZE), 4)
    dtype = torch.float16 if method == "fp16" else torch.float32
    reported = next(m for m in e2e["results"]["methods"] if m["method"] == method)["metrics"]
    assert evaluate_top1(model, loader, dtype).correct == reported["correct"]


def test_download_errors(e2e):
    c, run_id = e2e["client"], e2e["run_id"]
    for name in ("run.json", "session.json", "nope.pt", "..%2F..%2Fetc%2Fpasswd", "%2e%2e"):
        r = c.get(f"{API}/runs/{run_id}/artifacts/{name}")
        assert r.status_code == 404 and r.json()["error"]["code"] in {"artifact_not_found", "not_found"}
    assert c.get(f"{API}/runs/{'0' * 32}/artifacts/report.json").status_code == 404
    assert c.get(f"{API}/runs/not-an-id/artifacts/report.json").status_code == 404


def test_success_responses_do_not_leak_paths(e2e):
    c, run_id = e2e["client"], e2e["run_id"]
    for url in (f"{API}/runs/{run_id}", f"{API}/runs/{run_id}/results", f"{API}/sessions/{e2e['session']['session_id']}"):
        assert str(e2e["settings"].data_dir) not in c.get(url).text


def test_finished_run_is_not_active(e2e):
    c = e2e["client"]
    sid = e2e["session"]["session_id"]
    assert not c.app.state.runs.has_active_run(sid)


# ------------------------------------------------------------------ request validation
def _session(client, arch="resnet18"):
    return upload_session(client, architecture=arch, weights=to_bytes(make_state_dict(arch)),
                          dataset=make_dataset_zip()).json()["session_id"]


def test_run_request_validation(client):
    sid = _session(client)
    cases = [
        ({"session_id": sid, "methods": []}, 422, "invalid_request"),
        ({"session_id": sid, "methods": ["pruning"]}, 400, "invalid_request"),
        ({"session_id": sid, "methods": ["fp16", "fp16"]}, 422, "invalid_request"),
        ({"session_id": sid, "methods": ["int8_ptq"], "method_configs": {"int8_ptq": {"calibration_samples": 1}}}, 400, "invalid_request"),
        ({"session_id": sid, "methods": ["fp16"], "method_configs": {"int8_ptq": {}}}, 400, "invalid_request"),
        ({"session_id": sid, "methods": ["int8_ptq"], "method_configs": {"int8_ptq": {"backend": "nope"}}}, 422, "optimization_unsupported"),
        ({"session_id": sid, "methods": ["fp16"], "benchmark": {"timed_runs": 1}}, 422, "invalid_request"),
        ({"session_id": "0" * 32, "methods": ["fp16"]}, 404, "session_not_found"),
        ({"session_id": "../etc", "methods": ["fp16"]}, 404, "session_not_found"),
    ]
    for body, status, code in cases:
        r = client.post(f"{API}/runs", json=body)
        assert (r.status_code, r.json()["error"]["code"]) == (status, code), (body, r.text)


def test_unknown_run(client):
    assert client.get(f"{API}/runs/{'a' * 32}").json()["error"]["code"] == "run_not_found"
    assert client.get(f"{API}/runs/{'a' * 32}/results").status_code == 404


def test_results_not_ready_returns_conflict(client, monkeypatch):
    release = threading.Event()
    from app.runs.manager import RunManager

    monkeypatch.setattr(RunManager, "_pipeline", lambda self, run_id: release.wait(10))
    sid = _session(client)
    run_id = client.post(f"{API}/runs", json={"session_id": sid, "methods": ["fp16"]}).json()["run_id"]
    r = client.get(f"{API}/runs/{run_id}/results")
    assert r.status_code == 409 and r.json()["error"]["code"] == "run_not_finished"
    assert client.get(f"{API}/runs/{run_id}/artifacts/report.json").status_code == 404
    assert client.get(f"{API}/runs/{run_id}").json()["status"] == "queued"
    d = client.delete(f"{API}/sessions/{sid}")
    assert d.status_code == 409 and d.json()["error"]["code"] == "session_in_use"
    release.set()


# ------------------------------------------------------------------ failure isolation
def test_one_failing_method_does_not_abort_the_run(client, monkeypatch):
    from app.optimizers.int8_ptq import Int8PTQOptimizer

    def refuse(self, ctx, cfg):
        raise __import__("app.errors", fromlist=["x"]).OptimizationUnsupportedError("Unsupported op in this model.")

    monkeypatch.setattr(Int8PTQOptimizer, "optimize", refuse)
    sid = _session(client)
    run_id = client.post(f"{API}/runs", json={"session_id": sid, "methods": ["int8_ptq", "fp16"], "benchmark": FAST_BENCH}).json()["run_id"]
    assert wait_for_run(client, run_id)["status"] == "completed"
    res = client.get(f"{API}/runs/{run_id}/results").json()
    int8, fp16 = res["methods"]
    assert int8["status"] == "failed" and int8["error"]["code"] == "optimization_unsupported"
    assert int8["metrics"] is None and int8["artifact_name"] is None
    assert fp16["status"] == "ok"
    assert "int8_ptq" not in res["comparisons"] and "fp16" in res["comparisons"]
    assert "model_int8_ptq.pt" not in {a["name"] for a in res["artifacts"]}
    assert client.get(f"{API}/runs/{run_id}/artifacts/model_int8_ptq.pt").status_code == 404


def test_unexpected_method_exception_is_contained(client, monkeypatch):
    from app.optimizers.fp16 import FP16Optimizer

    monkeypatch.setattr(FP16Optimizer, "optimize", lambda self, ctx, cfg: (_ for _ in ()).throw(ValueError("boom /secret/path")))
    sid = _session(client)
    run_id = client.post(f"{API}/runs", json={"session_id": sid, "methods": ["fp16"], "benchmark": FAST_BENCH}).json()["run_id"]
    assert wait_for_run(client, run_id)["status"] == "completed"
    err = client.get(f"{API}/runs/{run_id}/results").json()["methods"][0]["error"]
    assert err["code"] == "optimization_failed" and "secret" not in err["message"]


def test_run_fails_cleanly_when_session_files_break(client, settings):
    sid = _session(client)
    (settings.sessions_dir / sid / "weights.pt").write_bytes(b"corrupted after upload")
    run_id = client.post(f"{API}/runs", json={"session_id": sid, "methods": ["fp16"], "benchmark": FAST_BENCH}).json()["run_id"]
    status = wait_for_run(client, run_id)
    assert status["status"] == "failed" and status["error"]["code"] == "invalid_weights"
    res = client.get(f"{API}/runs/{run_id}/results")
    assert res.status_code == 200 and res.json()["status"] == "failed" and res.json()["baseline"] is None
    assert str(settings.data_dir) not in res.text


def test_mobilenet_end_to_end(client):
    sid = _session(client, "mobilenet_v2")
    run_id = client.post(f"{API}/runs", json={"session_id": sid, "methods": ["int8_ptq"], "benchmark": FAST_BENCH}).json()["run_id"]
    assert wait_for_run(client, run_id)["status"] == "completed"
    res = client.get(f"{API}/runs/{run_id}/results").json()
    assert res["methods"][0]["status"] == "ok"
    assert res["baseline"]["metrics"]["param_count"] == 2_227_715
    assert res["comparisons"]["int8_ptq"]["size_bytes"]["percent_change"] < -50


def test_run_state_survives_restart(make_client, settings):
    c1 = make_client()
    sid = _session(c1)
    run_id = c1.post(f"{API}/runs", json={"session_id": sid, "methods": ["fp16"], "benchmark": FAST_BENCH}).json()["run_id"]
    wait_for_run(c1, run_id)
    c2 = make_client()  # fresh app instance, same data dir
    assert c2.get(f"{API}/runs/{run_id}").json()["status"] == "completed"
    assert c2.get(f"{API}/runs/{run_id}/artifacts/report.json").status_code == 200


def test_expired_data_is_cleaned_on_startup(make_client, settings):
    c1 = make_client()
    sid = _session(c1)
    c2 = make_client(retention_hours=0)
    assert c2.get(f"{API}/sessions/{sid}").status_code == 404
