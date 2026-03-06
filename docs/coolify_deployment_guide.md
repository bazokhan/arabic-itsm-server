# Coolify VPS Deployment Guide (Dockerfile)

This guide deploys this project on Coolify using the repository Dockerfile so deployment succeeds with one push.

It includes:
- FastAPI startup command (so the app does not restart-loop)
- automatic Hugging Face model sync on container start
- persistent model/cache storage (models + SQLite database)
- default model set to L3
- SQLite database for classification history, feedback, monitoring, and visitor logs

## 1) What This Docker Deployment Does

On every container start:
1. creates model/cache directories under `/data` AND the DB directory `/data/db/`
2. downloads (or reuses) model snapshots from Hugging Face:
   - `albaz2000/marbert-arabic-itsm-l3-categories`
   - `albaz2000/marbert-arabic-itsm-multitask`
3. starts the API with:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Model discovery uses:
- `MODEL_DIRS=/data/models`
- default model id: `marbert-arabic-itsm-l3-categories`

## 2) Prerequisites

- A running Coolify instance connected to your VPS.
- Your repo is connected as a public GitHub source.
- Enough VPS resources:
  - minimum recommended: 4 vCPU / 8 GB RAM / 20+ GB disk
  - better for smoother model load: 8 vCPU / 16 GB RAM

## 3) Repo Files Required

This repo now includes:
- `Dockerfile`
- `scripts/entrypoint.sh`

No additional custom start command is needed in Coolify when Dockerfile mode is selected.

## 4) Create the Coolify Resource

In Coolify:
1. `Projects` -> select project -> `+ New Resource`
2. Choose `Application`
3. Source: your GitHub repo `bazokhan/arabic-itsm-server`
4. Branch: `master` (or your target branch)
5. Build Pack/Type: `Dockerfile` (not Nixpacks)

## 5) Configure Ports and Domain

Set:
- Internal Port: `8000`
- Public domain: your desired domain/subdomain

Why 8000?
- The container exposes `8000`.
- Uvicorn listens on `PORT` from Coolify; if unset it falls back to `8000`.

## 6) Configure Environment Variables

Add these in Coolify Runtime Environment Variables:

```env
DEVICE=cpu
MODEL_DIRS=/data/models
HF_HOME=/data/hf_cache
DEFAULT_MODEL_ID=marbert-arabic-itsm-l3-categories
HF_MODEL_REPOS=albaz2000/marbert-arabic-itsm-l3-categories,albaz2000/marbert-arabic-itsm-multitask
FORCE_MODEL_SYNC=0
DB_PATH=/data/db/itsm.db
```

Optional:
- `HF_TOKEN=<your_token>` only if models become private later.

Notes:
- `FORCE_MODEL_SYNC=0` keeps startup fast after first deploy (reuses persisted files).
- Set `FORCE_MODEL_SYNC=1` temporarily if you need to force fresh model download.

## 7) Add Persistent Storage (Important)

Add a persistent volume in Coolify:
- Mount path inside container: `/data`

This keeps:
- downloaded model files (`/data/models`)
- Hugging Face cache (`/data/hf_cache`)
- SQLite database (`/data/db/itsm.db`) — classification history, feedback, monitoring

Without persistent storage, each restart re-downloads models AND loses the database.

## 8) Deploy

Click `Deploy`.

Expected first deployment behavior:
- image build completes
- first container start may take several minutes (model download)
- subsequent restarts are much faster if `/data` is persistent

## 9) Verify Deployment

After deployment, test:

1. Health:
```bash
GET /api/health
```
Expected:
- `profiles_count >= 1`
- `default_model_id = marbert-arabic-itsm-l3-categories`

2. Model list:
```bash
GET /api/models
```
Expected:
- both model profiles listed

3. UI pages:
- `/models` — model catalog
- `/models/marbert-arabic-itsm-l3-categories` — classify tickets
- `/dashboard` — classification history, stats, charts
- `/monitoring` — CPU/memory/visitor stats (auto-refreshes every 30s)
- `/research` — run comparative evaluation and regenerate report artifacts
- `/admin/export` — download database exports plus comparison artifacts

## 10) Common Issues and Fixes

### A) Container restart loop immediately

Cause:
- missing start command in Nixpacks mode.

Fix:
- use Dockerfile mode from this guide.

### B) `No model profiles available`

Cause:
- models not downloaded or wrong `MODEL_DIRS`.

Fix:
- ensure `MODEL_DIRS=/data/models`
- check logs for sync lines from entrypoint
- verify persistent volume mounted at `/data`

### C) Slow first response

Cause:
- first-time model load into RAM/CPU.

Fix:
- normal behavior; preload from `/models` UI or call preload endpoint.

### D) Out-of-memory / container killed

Fix:
- increase VPS RAM
- keep `DEVICE=cpu` unless you have GPU setup
- load fewer models if needed (reduce `HF_MODEL_REPOS`)

## 11) Updating Models

If model repo contents change:
1. set `FORCE_MODEL_SYNC=1`
2. redeploy once
3. set `FORCE_MODEL_SYNC=0` again

## 12) Minimal Operational Checklist

Before go-live:
1. Dockerfile mode enabled
2. internal port is `8000`
3. env vars set (especially `MODEL_DIRS`, `DEFAULT_MODEL_ID`, `HF_MODEL_REPOS`, `DB_PATH`)
4. persistent volume mounted at `/data` (covers models, hf_cache, AND the sqlite db)
5. `/api/health` shows profiles
6. inference tested from `/models` page
7. `/dashboard` shows classification history after first classify
8. `/admin/export` → test CSV, SQL, and comparison artifact downloads

## 13) Data Backup Strategy

The SQLite database at `DB_PATH` (for example `/data/itsm.db`) stores all classifications and feedback.

**Option A — Download via UI**: go to `/admin/export` and click "Download SQL Backup"

**Option B — Download via API**:
```bash
curl -o itsm_backup.sql https://your-domain/api/export/sql
curl -o itsm_backup.zip https://your-domain/api/export/csv
```

Comparison artifacts are static files generated by `/research` runs:

```bash
curl -o model_comparison_report.json https://your-domain/static/reports/model_comparison_report.json
curl -o model_comparison_raw_predictions.csv https://your-domain/static/reports/model_comparison_raw_predictions.csv
curl -o model_comparison_article.md https://your-domain/static/reports/model_comparison_article.md
```

**Option C — Direct copy** (from VPS):
```bash
docker cp <container_id>:/data/itsm.db ./itsm_backup.db
```

Recommend:
- schedule Option B weekly via a cron job or Coolify scheduled task,
- trigger `/research` benchmark after model updates to refresh the exported comparison artifacts.
