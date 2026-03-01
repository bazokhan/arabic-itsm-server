# Coolify VPS Deployment Guide (Dockerfile)

This guide deploys this project on Coolify using the repository Dockerfile so deployment succeeds with one push.

It includes:
- FastAPI startup command (so the app does not restart-loop)
- automatic Hugging Face model sync on container start
- persistent model/cache storage
- default model set to L3

## 1) What This Docker Deployment Does

On every container start:
1. creates model/cache directories under `/data`
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

Without persistent storage, each restart re-downloads models.

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

3. UI:
- `/models`
- `/models/marbert-arabic-itsm-l3-categories`
- `/compare`

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
3. env vars set (especially `MODEL_DIRS`, `DEFAULT_MODEL_ID`, `HF_MODEL_REPOS`)
4. persistent volume mounted at `/data`
5. `/api/health` shows profiles
6. inference tested from `/models` page
