"""Optimizer success/failure paths and artifact round-trips."""
import pytest
import torch

from helpers import INPUT_SIZE, LABELS, make_dataset_zip, make_state_dict
from app.config import Settings
from app.data.loader import calibration_subset, make_loader, sample_input
from app.errors import InvalidRequestError, OptimizationUnsupportedError
from app.evaluation.accuracy import evaluate_top1
from app.evaluation.benchmark import count_parameters
from app.models.checkpoint import load_checkpoint, save_checkpoint
from app.models.registry import get_arch
from app.optimizers import get_optimizer
from app.optimizers.base import OptimizationContext
from app.optimizers.int8_ptq import available_backends
from app.schemas import Preprocessing
from app.validation.archive import safe_extract_images
from app.validation.dataset import build_index

PRE = Preprocessing(input_size=INPUT_SIZE)


@pytest.fixture()
def index(tmp_path):
    settings = Settings(data_dir=tmp_path, frontend_dist=None)
    (tmp_path / "d.zip").write_bytes(make_dataset_zip(per_class=6))
    safe_extract_images(tmp_path / "d.zip", tmp_path / "ds", settings)
    return build_index(tmp_path / "ds", LABELS, settings)


def _ctx(arch, index, seed=0, n_calib=12, sample=None):
    spec = get_arch(arch)
    state = make_state_dict(arch, seed=seed)
    model = spec.build_float(3)
    model.load_state_dict(state)
    calib = calibration_subset(index.samples, n_calib, 0)
    return OptimizationContext(
        spec=spec, num_classes=3, state_dict=state, float_model=model.eval(),
        calibration_loader=make_loader(calib, PRE, 6), calibration_size=len(calib),
        sample=sample if sample is not None else sample_input(index.samples, PRE), settings=Settings(),
    )


def _save(tmp_path, name, arch, method, state, cfg=None):
    path = tmp_path / name
    save_checkpoint(path, state_dict=state, architecture=arch, method=method, num_classes=3, class_labels=LABELS,
                    preprocessing={**PRE.model_dump(), "resize_size": PRE.effective_resize_size},
                    method_config=cfg or {})
    return path


@pytest.mark.parametrize("arch", ["resnet18", "mobilenet_v2"])
def test_fp16_native_shrinks_file_and_round_trips(tmp_path, index, arch):
    ctx = _ctx(arch, index)
    opt = get_optimizer("fp16")
    out = opt.optimize(ctx, opt.parse_config({}))
    assert out.execution_mode == "native_fp16" and out.input_dtype == torch.float16

    base = _save(tmp_path, "base.pt", arch, "fp32_baseline", ctx.state_dict)
    half = _save(tmp_path, "half.pt", arch, "fp16", out.state_dict)
    assert half.stat().st_size < 0.6 * base.stat().st_size

    reloaded, meta = load_checkpoint(half)
    assert meta["method"] == "fp16" and next(reloaded.parameters()).dtype == torch.float16
    loader = make_loader(index.samples, PRE, 6)
    assert evaluate_top1(reloaded, loader, torch.float16).correct == evaluate_top1(out.model, loader, torch.float16).correct


def test_fp16_fp32_compute_mode(index):
    ctx = _ctx("resnet18", index)
    opt = get_optimizer("fp16")
    out = opt.optimize(ctx, opt.parse_config({"execution_mode": "fp32_compute"}))
    assert out.execution_mode == "fp16_storage_fp32_compute" and out.input_dtype == torch.float32
    assert all(v.dtype == torch.float16 for v in out.state_dict.values() if v.is_floating_point())
    assert next(out.model.parameters()).dtype == torch.float32


def test_fp16_falls_back_when_native_inference_fails(index):
    bad_sample = torch.zeros(1, 1, INPUT_SIZE, INPUT_SIZE)  # wrong channel count -> native probe raises
    out = get_optimizer("fp16").optimize(_ctx("resnet18", index, sample=bad_sample), get_optimizer("fp16").parse_config({}))
    assert out.execution_mode == "fp16_storage_fp32_compute"
    assert any("fell back" in n for n in out.notes)


def test_fp16_rejects_unknown_option():
    with pytest.raises(Exception):
        get_optimizer("fp16").parse_config({"execution_mode": "int4"})


@pytest.mark.parametrize("arch", ["resnet18", "mobilenet_v2"])
def test_int8_produces_valid_smaller_model_and_reloads_identically(tmp_path, index, arch):
    ctx = _ctx(arch, index)
    opt = get_optimizer("int8_ptq")
    cfg = opt.parse_config({"calibration_samples": 12})
    with opt.runtime(cfg):
        out = opt.optimize(ctx, cfg)
        assert out.execution_mode == "int8_static_cpu"
        logits = out.model(ctx.sample)
        assert logits.shape == (1, 3) and torch.isfinite(logits).all()
        loader = make_loader(index.samples, PRE, 6)
        acc_in_memory = evaluate_top1(out.model, loader).correct

        base = _save(tmp_path, "base.pt", arch, "fp32_baseline", ctx.state_dict)
        q = _save(tmp_path, "q.pt", arch, "int8_ptq", out.state_dict, cfg.model_dump())
        assert q.stat().st_size < 0.4 * base.stat().st_size

        params = count_parameters(out.model)
        float_params = count_parameters(ctx.float_model)
        assert 0.95 * float_params < params <= float_params  # only folded BatchNorm parameters disappear

        reloaded, meta = load_checkpoint(q)  # rebuilds structure from metadata, loads quantized weights safely
        assert meta["method"] == "int8_ptq" and meta["method_config"]["backend"] == cfg.backend
        assert torch.allclose(reloaded(ctx.sample), logits, atol=1e-5)
        assert evaluate_top1(reloaded, loader).correct == acc_in_memory


def test_int8_restores_quantization_engine(index):
    before = torch.backends.quantized.engine
    opt = get_optimizer("int8_ptq")
    cfg = opt.parse_config({})
    with opt.runtime(cfg):
        assert torch.backends.quantized.engine == cfg.backend
    assert torch.backends.quantized.engine == before


def test_int8_unsupported_backend_gives_clear_error():
    with pytest.raises(OptimizationUnsupportedError) as exc:
        get_optimizer("int8_ptq").parse_config({"backend": "not-a-backend"})
    assert "not supported" in exc.value.message and available_backends()[0] in exc.value.message


def test_int8_failure_is_reported_not_silenced(index):
    ctx = _ctx("resnet18", index)
    ctx.state_dict = {k: v for k, v in ctx.state_dict.items() if k != "fc.bias"}  # cannot build quantized model
    opt = get_optimizer("int8_ptq")
    with pytest.raises(OptimizationUnsupportedError) as exc:
        with opt.runtime(opt.parse_config({})):
            opt.optimize(ctx, opt.parse_config({}))
    assert "No model was produced" in exc.value.message


def test_unknown_method_rejected():
    with pytest.raises(InvalidRequestError):
        get_optimizer("pruning")
