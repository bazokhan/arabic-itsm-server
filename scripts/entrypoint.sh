#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${MODEL_DIRS:-/data/models}" "${HF_HOME:-/data/hf_cache}"

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

python - <<'PY'
import os
import shutil
from huggingface_hub import snapshot_download

repos_raw = os.getenv(
    "HF_MODEL_REPOS",
    "albaz2000/marbert-arabic-itsm-l3-categories,albaz2000/marbert-arabic-itsm-multitask",
)
model_root = os.getenv("MODEL_DIRS", "/data/models")
force = os.getenv("FORCE_MODEL_SYNC", "0").strip() == "1"

repos = [r.strip() for r in repos_raw.split(",") if r.strip()]
os.makedirs(model_root, exist_ok=True)

if not repos:
    print("[warn] HF_MODEL_REPOS is empty; server will start without remote sync")
else:
    for repo in repos:
        target = os.path.join(model_root, repo.split("/")[-1])
        marker = os.path.join(target, ".download_complete")
        needs_sync = force or (not os.path.exists(marker))
        if not needs_sync:
            print(f"[ok] model already synced: {repo} -> {target}")
            continue

        print(f"[sync] downloading {repo} ...")
        downloaded = snapshot_download(
            repo_id=repo,
            repo_type="model",
            local_dir=target,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"[ok] downloaded snapshot: {downloaded}")
        if os.path.isdir(os.path.join(target, ".cache")):
            shutil.rmtree(os.path.join(target, ".cache"), ignore_errors=True)
        with open(marker, "w", encoding="utf-8") as f:
            f.write("ok\n")
        print(f"[ok] synced: {repo} -> {target}")
PY

exec uvicorn app.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
