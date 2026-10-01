"""Baseline evaluation, benchmarking and comparison logic."""
import pytest
import torch

from helpers import INPUT_SIZE, LABELS, make_dataset_zip, make_state_dict
from app.config import Settings
from app.data.loader import calibration_subset, make_loader, sample_input
from app.evaluation.accuracy import evaluate_top1
from app.evaluation.benchmark import count_parameters, environment_info, measure_latency
from app.evaluation.comparison import compare
from app.models.registry import get_arch
from app.schemas import LatencyStats, Metrics, Preprocessing
from app.validation.archive import safe_extract_images
from app.validation.dataset import build_index

PRE = Preprocessing(input_size=INPUT_SIZE)


@pytest.fixture()
def dataset(tmp_path):
    settings = Settings(data_dir=tmp_path, frontend_dist=None)
    (tmp_path / "d.zip").write_bytes(make_dataset_zip())
    safe_extract_images(tmp_path / "d.zip", tmp_path / "ds", settings)
    return build_index(tmp_path / "ds", LABELS, settings)


@pytest.mark.parametrize("arch", ["resnet18", "mobilenet_v2"])
def test_baseline_accuracy_is_exact_for_constant_predictor(dataset, arch):
    model = get_arch(arch).build_float(3)
    model.load_state_dict(make_state_dict(arch, favor_class=1))
    result = evaluate_top1(model, make_loader(dataset.samples, PRE, batch_size=5))
    assert (result.correct, result.total) == (4, 12)  # 4 dogs out of 12 images
    assert result.top1_pct == pytest.approx(100 / 3)


def test_accuracy_does_not_depend_on_batch_size(dataset):
    model = get_arch("resnet18").build_float(3)
    model.load_state_dict(make_state_dict("resnet18", seed=3))
    a = evaluate_top1(model, make_loader(dataset.samples, PRE, 1))
    b = evaluate_top1(model, make_loader(dataset.samples, PRE, 12))
    assert a.correct == b.correct


def test_dataset_index_order_follows_label_order(dataset):
    assert [idx for _, idx in dataset.samples] == [0] * 4 + [1] * 4 + [2] * 4


def test_calibration_subset_is_deterministic(dataset):
    a = calibration_subset(dataset.samples, 5, seed=0)
    b = calibration_subset(dataset.samples, 5, seed=0)
    c = calibration_subset(dataset.samples, 5, seed=1)
    assert a == b and len(a) == 5 and a != c
    assert calibration_subset(dataset.samples, 100, seed=0) == dataset.samples


def test_latency_structure_and_sanity(dataset):
    model = get_arch("resnet18").build_float(3).eval()
    stats = measure_latency(model, sample_input(dataset.samples, PRE), warmup_runs=2, timed_runs=6)
    assert stats.warmup_runs == 2 and stats.timed_runs == 6 and stats.batch_size == 1
    assert 0 < stats.min_ms <= stats.median_ms <= stats.p95_ms
    assert stats.mean_ms > 0 and stats.std_ms >= 0


def test_parameter_count_matches_torchvision():
    assert count_parameters(get_arch("resnet18").build_float(1000)) == 11_689_512
    assert count_parameters(get_arch("mobilenet_v2").build_float(1000)) == 3_504_872


def test_environment_info_has_no_paths_and_states_device():
    env = environment_info()
    assert env["device"] == "cpu" and env["torch_threads"] >= 1 and "latency_note" in env
    assert "/home" not in str(env) or "cpu_model" in env  # sanity: no filesystem info collected


def _metrics(acc, size, latency, params):
    return Metrics(
        top1_accuracy_pct=acc, correct=0, total=1, size_bytes=size, param_count=params,
        latency=LatencyStats(median_ms=latency, mean_ms=latency, p95_ms=latency, min_ms=latency, std_ms=0,
                             warmup_runs=1, timed_runs=5),
    )


def test_comparison_marks_regressions_as_worse():
    base = _metrics(90.0, 1000, 10.0, 500)
    # smaller and less accurate, but SLOWER
    c = compare(base, _metrics(85.0, 250, 12.5, 500))
    assert c["top1_accuracy_pct"].direction == "worse" and c["top1_accuracy_pct"].absolute_change == pytest.approx(-5)
    assert c["size_bytes"].direction == "improved" and c["size_bytes"].percent_change == pytest.approx(-75)
    assert c["latency_median_ms"].direction == "worse" and c["latency_median_ms"].percent_change == pytest.approx(25)
    assert c["param_count"].direction == "unchanged" and c["param_count"].percent_change == 0


def test_comparison_zero_baseline_has_null_percent():
    c = compare(_metrics(0.0, 1000, 10.0, 500), _metrics(10.0, 1000, 10.0, 500))
    assert c["top1_accuracy_pct"].percent_change is None and c["top1_accuracy_pct"].direction == "improved"
