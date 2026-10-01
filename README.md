# NeuralFit

Upload PyTorch weights for an image classifier plus a small validation set, apply FP16 or INT8 conversion, and see **measured** accuracy, file size, CPU latency and parameter count against the original. When an optimization makes something worse, the UI says "Worse".

- **Backend:** FastAPI, PyTorch, torchvision
- **Frontend:** React, TypeScript, Vite
- **Scope:** local MVP. No database, no authentication, no job queue.

## What is supported

| | |
|---|---|
| Framework | PyTorch, image classification |
| Architectures | ResNet-18 (`torchvision.models.resnet18`), MobileNetV2 (`torchvision.models.mobilenet_v2`). The final classifier layer may have any number of classes. |
| Optimizations | **FP16 weight conversion** and **INT8 post-training static quantization** (eager mode, CPU) |
| Benchmarking | CPU only, batch size 1 for latency |

**Not supported:** other architectures, GPU benchmarking, pruning, knowledge distillation, quantization-aware training, ONNX/TensorRT export, TorchScript or full pickled models. Pruning is deliberately not in the UI; it will be added only once it is implemented and evaluated honestly.

## Quick start

Requirements: Python 3.10+, Node 18+.

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
# Smaller CPU-only PyTorch build (optional, recommended):
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
uvicorn app.main:app --port 8000

# Frontend (second terminal)
cd frontend
npm install
npm run dev          # http://localhost:5173, proxies /api to :8000
```

To serve everything from one process, run `npm run build` in `frontend/` and start the backend: if `frontend/dist` exists, FastAPI serves it at `http://127.0.0.1:8000/`. API docs are at `/docs`.

Uploaded files and results are stored in `./neuralfit_data` (override with `NEURALFIT_DATA_DIR`) and deleted after 24 hours, checked at server startup. Sessions can also be deleted from the UI.

Developed and tested with Python 3.12, PyTorch 2.14 and torchvision 0.29 on x86-64 Linux. The `requirements.txt` floors (`torch>=2.1`) reflect the APIs used, but older versions were not tested.

## Input requirements

### Weights

A **state dict** saved with:

```python
torch.save(model.state_dict(), "weights.pt")   # .pt or .pth
```

- Loaded with `torch.load(..., weights_only=True)`, so no uploaded code is executed. Full pickled models (`torch.save(model, ...)`), TorchScript files and quantized weights are rejected with an explanation.
- A `module.` prefix (DataParallel) is stripped, and checkpoints that wrap the state dict under `state_dict`, `model_state_dict` or `model` are unwrapped.
- The number of output classes is read from the classifier layer (`fc.weight` for ResNet-18, `classifier.1.weight` for MobileNetV2) and must equal the number of class labels. All other keys and shapes must match the chosen architecture exactly.
- Maximum 300 MB.

### Validation dataset

A ZIP with **one folder per class**, named exactly like the class labels:

```
validation_dataset/        <- one wrapper folder is allowed
  cats/img001.jpg
  cats/img002.jpg
  dogs/img001.jpg
```

- JPG, JPEG, PNG or BMP. Other files (and hidden files such as `__MACOSX`) are ignored with a warning.
- Every folder must be listed in the class labels. A label with no folder produces a warning.
- Images must be directly inside the class folder, not nested deeper. Every image is opened and verified during validation; a corrupt image rejects the upload.
- Limits: 500 MB ZIP, 1 GB extracted, 10,000 images, 20 MB per image.
- The archive is checked before extraction: paths containing `..`, absolute or Windows-style paths, symlinks, encrypted entries and suspicious compression ratios are rejected.

### Class labels and preprocessing

Enter labels one per line, in the model's output order (line 1 = output index 0). Frameworks such as `torchvision.datasets.ImageFolder` use alphabetical order.

Preprocessing is the standard evaluation pipeline and is identical for every model: resize the shorter side (default `round(input_size * 256 / 224)`), center-crop `input_size`, convert to a tensor, normalize. Defaults are 224 px and ImageNet mean/std. Change them if your model was trained differently.

## How the benchmark works

Every run measures the original model first, then each selected method, using the same images, preprocessing, batch size and timing procedure.

| Metric | How it is measured |
|---|---|
| **Top-1 accuracy** | Fraction of validation images where the arg-max class equals the folder label, on the full set. Deterministic (no shuffling). |
| **Model file size** | Size on disk of the serialized artifact. The original is re-serialized in the same NeuralFit checkpoint format (state dict plus a few metadata fields), so sizes are like-for-like. The size of the file you uploaded is shown separately. |
| **Latency** | One image per forward pass in `torch.inference_mode()`. Warm-up runs (default 20) are discarded, then timed runs (default 100) are recorded with `time.perf_counter`. Median is the headline number; mean, p95, min and standard deviation are reported too. Preprocessing and model loading are not timed. The input is the first validation image. |
| **Parameters** | Weight and bias elements. Quantized layers are unpacked and counted. INT8 counts slightly fewer because BatchNorm layers are folded into convolutions. |

The comparison shows the baseline value, the optimized value, the absolute change (percentage points for accuracy), the percent change, and a verdict. Higher is better for accuracy; lower is better for size, latency and parameters. A verdict is "worse" whenever a metric moves the wrong way, however small.

Threads default to 1 for stable numbers and can be changed in the benchmark settings. The selected thread count applies to every model in the run.

### Reading the results

- **FP16** roughly halves file size. It does not reliably speed up CPU inference: many CPUs have no fast FP16 path and can be slower than FP32. The result is whatever was measured. If native FP16 inference fails on your PyTorch build, the run falls back to FP32 compute using the FP16-rounded weights and says so, and latency in that case is FP32 latency.
- **INT8** typically cuts file size to about a quarter and often reduces latency on CPUs with INT8 kernels, at some accuracy cost. Neither is guaranteed.
- Accuracy differences smaller than roughly one image on a small validation set are noise. Check `correct` and `total` in the report.
- Compare latency using p95 and standard deviation as well as the median, and rerun if the machine was busy.

## Known limitations

- **Latency is hardware-specific.** It reflects the CPU, thread count and PyTorch build of the machine running the server. It does not predict speed on a phone, a Raspberry Pi or a GPU.
- **INT8 calibration uses the validation set** (a fixed-seed sample, default 64 images). Accuracy is then measured on the full set, including those images, so INT8 accuracy can be slightly optimistic. Use a separate calibration set for production decisions.
- **Eager-mode quantization is deprecated** upstream in `torch.ao.quantization` (migration path: `torchao`). It works in the tested version and emits deprecation warnings that the app suppresses. If a future PyTorch removes it, INT8 is reported as unavailable instead of failing.
- **`fbgemm` targets x86 CPUs.** `x86` and `qnnpack` (ARM) are offered when the PyTorch build supports them. A quantized model runs only on the backend it was converted for.
- Validation accuracy is only as representative as your validation set.
- **Single worker, in-memory run state.** One run executes at a time; others queue. Run results are also written to disk, so they survive a restart, but a run interrupted by a restart is reported as failed.
- **Intended for local use.** There is no authentication. If you expose it, put a reverse proxy in front that enforces request-size limits and access control. The built-in size check relies on the `Content-Length` header.
- Every optimizer failure is contained: if one method cannot be applied, that method shows an error and the others still complete.

## API

Base path `/api/v1`. Errors always look like `{"error": {"code": "...", "message": "..."}}`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness |
| GET | `/architectures`, `/methods` | Supported architectures; available methods, their options and limitations |
| POST | `/sessions` | Upload and validate weights and dataset (multipart) |
| GET / DELETE | `/sessions/{id}` | Session summary / remove uploaded files |
| POST | `/runs` | Start a benchmark run (returns 202) |
| GET | `/runs/{id}` | Status and progress |
| GET | `/runs/{id}/results` | Full results (409 until the run finishes) |
| GET | `/runs/{id}/artifacts/{name}` | Download a model file or `report.json` |

```bash
# 1. Validate inputs
curl -s -X POST http://127.0.0.1:8000/api/v1/sessions \
  -F architecture=resnet18 \
  -F 'config={"class_labels":["cats","dogs"],"preprocessing":{"input_size":224},"batch_size":32}' \
  -F weights=@weights.pt \
  -F dataset_zip=@validation_dataset.zip
# -> {"session_id": "…", "num_classes": 2, "total_images": …, …}

# 2. Start a run
curl -s -X POST http://127.0.0.1:8000/api/v1/runs \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"<SESSION_ID>","methods":["int8_ptq","fp16"],
       "method_configs":{"int8_ptq":{"calibration_samples":64}},
       "benchmark":{"warmup_runs":20,"timed_runs":100,"threads":1}}'

# 3. Poll, then read results and download
curl -s http://127.0.0.1:8000/api/v1/runs/<RUN_ID>
curl -s http://127.0.0.1:8000/api/v1/runs/<RUN_ID>/results
curl -OJ http://127.0.0.1:8000/api/v1/runs/<RUN_ID>/artifacts/model_int8_ptq.pt
```

## Using a downloaded model

Downloaded `.pt` files are NeuralFit checkpoints: a state dict plus the architecture, class labels, preprocessing and method settings needed to rebuild the model. Loading uses `weights_only=True`. Run this from the `backend/` directory (or with it on `PYTHONPATH`):

```python
import torch
from PIL import Image
from torchvision import transforms
from app.models.checkpoint import load_checkpoint

model, meta = load_checkpoint("model_int8_ptq.pt")   # also works for model_fp16.pt and baseline_fp32.pt
pre = meta["preprocessing"]
tf = transforms.Compose([
    transforms.Resize(pre["resize_size"]), transforms.CenterCrop(pre["input_size"]),
    transforms.ToTensor(), transforms.Normalize(pre["mean"], pre["std"]),
])
x = tf(Image.open("photo.jpg").convert("RGB")).unsqueeze(0)
if meta["method"] == "fp16":
    x = x.half()                                      # FP16 models expect half-precision input

with torch.inference_mode():
    print(meta["class_labels"][model(x).argmax(1).item()])
```

For INT8 models, `load_checkpoint` sets `torch.backends.quantized.engine` to the backend recorded in the checkpoint; run inference on a CPU that supports it.

## Tests

```bash
cd backend
python -m pytest -q
```

The tests use tiny synthetic datasets (64 px images) and randomly initialized models, so they need no downloads and take about a minute on CPU. They cover:

- exact accuracy on a constant-output model (known answer),
- latency and comparison structure, including regressions marked "worse",
- INT8 and FP16 success paths for both architectures, and that a downloaded artifact reproduces the reported accuracy,
- rejection of bad weights (garbage, full pickles, wrong architecture, wrong class count, missing keys) and bad datasets (zip-slip, absolute paths, symlinks, zip bombs, corrupt images, label mismatch),
- per-method failure isolation, download safety and error responses that never contain server paths.

The tests do not measure real-world accuracy or speed, because random weights say nothing about either. Benchmark your own trained model to get meaningful numbers.

## Project layout

```
backend/app/
  api/          routes: meta, sessions, runs
  validation/   upload, safe ZIP, dataset and weights validation
  models/       architecture registry, checkpoint format
  data/         preprocessing and loaders
  evaluation/   accuracy, latency, parameter count, comparison
  optimizers/   fp16, int8_ptq (add new methods here)
  runs/         background run manager
  storage/      session and artifact stores
backend/tests/  pytest suite
frontend/src/   React UI
```
