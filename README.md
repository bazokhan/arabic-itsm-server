# Arabic ITSM Classification Server
### Multi-Model FastAPI Inference + Per-Model Demo Pages

> **Faculty of Computers and Artificial Intelligence - Cairo University**  
> Professional Master's in Cloud Computing Networks - February 2026  
> **Author**: Mohamed A. Elbaz | **Supervisor**: Dr. Eman E. Sanad

---

## Overview

This repository is the serving layer for Arabic ITSM ticket classification.  
It now supports:

- model catalog page (`/models`)
- dedicated web page per model profile (`/models/{model_id}`)
- comparative evaluation page (`/research`)
- inference by selected model (`/api/classify?model_id=...`)
- inference across all loaded profiles (`/api/classify/all`)
- dynamic task support per checkpoint (`l1`, `l2`, `l3`, etc. based on `heads.pt`)

Companion repositories:

- Training remote: `https://github.com/bazokhan/arabic-itsm-classification.git`
- Dataset remote: `git@github.com:bazokhan/arabic-itsm-dataset.git`

---

## Academic Documentation

For academic traceability and reproducibility:

- Classification docs: `https://github.com/bazokhan/arabic-itsm-classification/tree/master/docs`
- Project abstract: `https://github.com/bazokhan/arabic-itsm-classification/blob/master/docs/abstract.pdf`
- Dataset methodology: `https://github.com/bazokhan/arabic-itsm-dataset`
- Inference validity notes in this repo: [docs/academic_inference_notes.md](/D:/AI/arabic-itsm-server/docs/academic_inference_notes.md)
- Comparison methodology: [docs/model_comparison_methodology.md](/D:/AI/arabic-itsm-server/docs/model_comparison_methodology.md)
- Cloud demo deployment guide: [docs/cloud_demo_guide.md](/D:/AI/arabic-itsm-server/docs/cloud_demo_guide.md)
- Coolify VPS Docker deployment guide: [docs/coolify_deployment_guide.md](/D:/AI/arabic-itsm-server/docs/coolify_deployment_guide.md)

`docs/academic_inference_notes.md` includes:
- tokenizer consistency fix (2026-02-26),
- why it matters for experimental validity,
- multi-model demo protocol.

---

## Repository Structure

```text
arabic-itsm-server/
├── app/
│   ├── main.py                    # multi-model API + routes
│   └── classifier.py              # checkpoint inference utilities
├── scripts/
│   └── run_model_comparison.py    # offline benchmark + report artifact generator
├── static/
│   ├── models.html                # model catalog
│   ├── model.html                 # single-model demo page
│   ├── research.html              # article-style model comparison page
│   ├── reports/                   # generated JSON/CSV/MD artifacts
│   └── index.html                 # legacy single-page demo
├── docs/
│   └── academic_inference_notes.md
├── models/
│   └── ... checkpoints (each with heads.pt)
├── .env.example
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

Install PyTorch separately (CPU/CUDA as needed), for example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2) Place checkpoints

Any subfolder containing `heads.pt` is treated as a model profile candidate.

Model discovery is controlled only from `.env` (`MODEL_DIRS`):

```env
MODEL_DIRS=models;D:/AI/custom-checkpoints
DEFAULT_MODEL_ID=marbert_l2_best
DEVICE=cpu
```

### 3) Run

```bash
uvicorn app.main:app --reload
```

Open:

- Model catalog: `http://127.0.0.1:8000/models`
- Research comparison page: `http://127.0.0.1:8000/research`
- API docs: `http://127.0.0.1:8000/docs`

---

## API Guide

### `GET /api/models`

Returns discovered model profiles:

- `id`, `name`, `path`, `description`
- `loaded`, `status`, `progress` for warm/cold/loading UI
- loaded `tasks` (after first inference)

### `POST /api/models/{model_id}/preload`

Starts background loading for one model.

### `GET /api/models/{model_id}/status`

Returns live loading status/progress for one model.

### `POST /api/classify?model_id=<id>`

Request:

```json
{
  "title_ar": "الـ VPN مش شغال",
  "description_ar": "الاتصال بيفصل كل دقيقة"
}
```

Response:

- `model_id`
- `tasks` (available in selected checkpoint)
- `latency_ms`
- per-task prediction payload:
  - `label`
  - `confidence`
  - `top3`

### `POST /api/classify/all`

Runs one ticket against all discovered profiles and returns a result entry per model.

### `GET /api/health`

Returns profile/load status and default model id.

---

## Web Demo Flow

1. Go to `/models`
2. Select a model profile for dedicated demo at `/models/{model_id}`
3. Open `/research` for article-style benchmark results and charts
4. Compare outputs and metrics for academic demos and model selection

This supports your requirement to demo current models (and future models) as separate pages.

---

## Comparative Model Evaluation

Use the offline pipeline to benchmark two local checkpoints on a labeled split and publish artifacts directly to the web UI.

### Run the benchmark

```bash
python scripts/run_model_comparison.py
```

Or from the web UI:

- open `http://127.0.0.1:8000/research`
- click **Run Benchmark**
- monitor progress and logs directly on the page

### Deployment-safe configuration

The comparison runner is not tied to a local Windows path.  
The repository includes a committed split at:

- `data/processed/test.csv`

On deployed servers, configure `.env` with:

- `COMPARISON_MODEL_A_ID` / `COMPARISON_MODEL_B_ID` (or explicit `*_PATH`)
- `COMPARISON_DATASET_CSV=data/processed/test.csv` (default and recommended)
- `COMPARISON_MODEL_A_URL`, `COMPARISON_MODEL_B_URL`, dataset reference URLs

Optional arguments:

```bash
python scripts/run_model_comparison.py ^
  --dataset-csv "data/processed/test.csv" ^
  --model-a-path "models/marbert_l1_l2_l3_best" ^
  --model-b-path "models/arabert_l1_l2_l3_best" ^
  --model-a-id "marbert-arabic-itsm-l3-categories" ^
  --model-b-id "arabert-arabic-itsm-l3-categories" ^
  --split-name "test" ^
  --bootstrap-samples 1000
```

Generated artifacts (under `static/reports/`):

- `model_comparison_raw_predictions.csv` — ticket-level predictions and confidences.
- `model_comparison_report.json` — all aggregate metrics, paired tests, chart-ready payloads.
- `model_comparison_article.md` — auto-generated narrative article text.

Then open:

- `http://127.0.0.1:8000/research`
- `http://127.0.0.1:8000/admin/export` (download DB backups + comparison artifacts)

---

## Tokenizer Consistency Note

A critical inference fix was applied: local paths are no longer allowed to silently fallback tokenizer loading when tokenizer assets are missing.  
The loader now prefers tokenizer-bearing checkpoints (especially L3 for two-checkpoint setups), then falls back to `UBC-NLP/MARBERTv2`.

See: [docs/academic_inference_notes.md](/D:/AI/arabic-itsm-server/docs/academic_inference_notes.md)

---

## Recommended Workspace Layout

```text
D:/AI/
├── arabic-itsm-dataset/
├── arabic-itsm-classification/
└── arabic-itsm-server/
```

---

## Citation

```bibtex
@misc{elbaz2026arabic_server,
  title   = {Arabic ITSM Classification Server: Multi-Model FastAPI Inference and Web Demo Layer},
  author  = {Elbaz, Mohamed A.},
  year    = {2026},
  note    = {Professional Master's Project, Faculty of Computers and Artificial Intelligence, Cairo University. Supervised by Dr. Eman E. Sanad.}
}
```

## License

MIT
