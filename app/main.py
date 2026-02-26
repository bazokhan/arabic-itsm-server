"""
FastAPI server for multi-model Arabic ITSM ticket classification.

Key routes:
  - GET  /api/models
  - POST /api/classify?model_id=<id>
  - POST /api/classify/all
  - GET  /models
  - GET  /compare
  - GET  /models/{model_id}
"""
from __future__ import annotations

import gc
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

_profiles: dict[str, dict] = {}
_models: dict[str, object] = {}
_load_state: dict[str, dict] = {}
_load_events: dict[str, threading.Event] = {}
_state_lock = threading.RLock()


def _discover_profiles() -> dict[str, dict]:
    """
    Discover model profiles from local directories.
    A model profile is any directory containing heads.pt.

    Discovery sources (in order):
    1. MODEL_DIRS env var — semicolon-separated base directories scanned for subfolders
    2. MULTITASK_MODEL_PATH env var — direct path to a single multi-task checkpoint
    """
    raw_dirs = os.getenv("MODEL_DIRS", "")
    dirs = [Path(p.strip()) for p in raw_dirs.split(";") if p.strip()]

    profiles: dict[str, dict] = {}
    used_ids: set[str] = set()

    def _register(child: Path):
        model_id = child.name.lower().replace(" ", "_")
        if model_id in used_ids:
            suffix = 2
            while f"{model_id}_{suffix}" in used_ids:
                suffix += 1
            model_id = f"{model_id}_{suffix}"
        used_ids.add(model_id)
        profiles[model_id] = {
            "id": model_id,
            "name": child.name,
            "path": str(child),
            "description": f"Checkpoint at {child}",
        }

    for base in dirs:
        if not base.exists() or not base.is_dir():
            continue
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            if not (child / "heads.pt").exists():
                continue
            _register(child)

    # Optional: single-path shortcut for the multi-task production model
    multitask_path = os.getenv("MULTITASK_MODEL_PATH", "").strip()
    if multitask_path:
        mt = Path(multitask_path)
        if mt.is_dir() and (mt / "heads.pt").exists() and str(mt) not in {
            p["path"] for p in profiles.values()
        }:
            _register(mt)

    return profiles


def _default_model_id() -> str:
    env_default = os.getenv("DEFAULT_MODEL_ID")
    if env_default and env_default in _profiles:
        return env_default
    if _profiles:
        return next(iter(_profiles))
    return ""


def _get_model(model_id: str):
    if model_id not in _profiles:
        raise HTTPException(status_code=404, detail=f"Unknown model_id: {model_id}")
    return _ensure_model_loaded(model_id, wait=True)


def _set_state(model_id: str, *, status: str, progress: int, message: str, error: str | None = None):
    now = time.time()
    with _state_lock:
        current = _load_state.get(model_id, {})
        _load_state[model_id] = {
            "model_id": model_id,
            "status": status,
            "progress": max(0, min(100, int(progress))),
            "message": message,
            "loaded": model_id in _models,
            "error": error,
            "updated_at": now,
            "started_at": current.get("started_at", now),
        }


def _load_model_sync(model_id: str):
    with _state_lock:
        if model_id in _models:
            _set_state(model_id, status="ready", progress=100, message="Model already loaded")
            event = _load_events.get(model_id)
            if event:
                event.set()
            return _models[model_id]
        state = _load_state.get(model_id, {})
        if state.get("status") == "loading":
            return None
        _load_state[model_id] = {
            "model_id": model_id,
            "status": "loading",
            "progress": 5,
            "message": "Queued for loading",
            "loaded": False,
            "error": None,
            "started_at": time.time(),
            "updated_at": time.time(),
        }
        _load_events[model_id] = threading.Event()

    try:
        _set_state(model_id, status="loading", progress=15, message="Preparing classifier")
        from app.classifier import CheckpointClassifier

        _set_state(model_id, status="loading", progress=35, message="Loading checkpoint files")
        device = os.getenv("DEVICE", None)
        profile = _profiles[model_id]
        model = CheckpointClassifier(model_path=profile["path"], device=device)
        _set_state(model_id, status="loading", progress=85, message="Finalizing model state")
        with _state_lock:
            _models[model_id] = model
        _set_state(model_id, status="ready", progress=100, message="Model loaded")
        with _state_lock:
            event = _load_events.get(model_id)
            if event:
                event.set()
        return model
    except Exception as exc:
        _set_state(model_id, status="error", progress=100, message="Model loading failed", error=str(exc))
        with _state_lock:
            event = _load_events.get(model_id)
            if event:
                event.set()
        raise


def _ensure_model_loaded(model_id: str, wait: bool):
    with _state_lock:
        if model_id in _models:
            return _models[model_id]
        state = _load_state.get(model_id, {})
        loading = state.get("status") == "loading"

    if not loading:
        model = _load_model_sync(model_id)
        if model is not None:
            return model

    if not wait:
        return None

    with _state_lock:
        event = _load_events.get(model_id)
    if event:
        event.wait()
    with _state_lock:
        if model_id in _models:
            return _models[model_id]
        failure = _load_state.get(model_id, {})
    raise HTTPException(
        status_code=500,
        detail=f"Failed to load model {model_id}: {failure.get('error', 'unknown error')}",
    )


def _start_background_load(model_id: str):
    def _runner():
        try:
            _ensure_model_loaded(model_id, wait=True)
        except Exception:
            # Error details are captured in _load_state by _load_model_sync.
            return

    t = threading.Thread(target=_runner, daemon=True)
    t.start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _profiles
    _profiles = _discover_profiles()
    if _profiles:
        print(f"[ok] Discovered {len(_profiles)} model profile(s)")
    else:
        print("[warn] No model profiles discovered")
        print("       Put checkpoints under models/ or set MODEL_DIRS")
    yield


app = FastAPI(
    title="Arabic ITSM Classifier",
    description="Multi-model Arabic ITSM ticket classification API + web UI.",
    version="2.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")


class TicketIn(BaseModel):
    title_ar: str
    description_ar: str = ""


@app.get("/api/models", summary="List available model profiles")
async def list_models():
    items = []
    for model_id, profile in _profiles.items():
        loaded = model_id in _models
        tasks: list[str] = []
        status = "idle"
        progress = 0
        state = _load_state.get(model_id, {})
        if state:
            status = state.get("status", "idle")
            progress = int(state.get("progress", 0))
        if loaded:
            tasks = _models[model_id].tasks
            status = "ready"
            progress = 100
        items.append(
            {
                "id": profile["id"],
                "name": profile["name"],
                "path": profile["path"],
                "description": profile["description"],
                "loaded": loaded,
                "status": status,
                "progress": progress,
                "tasks": tasks,
            }
        )
    return {"default_model_id": _default_model_id(), "models": items}


@app.get("/api/models/{model_id}/status", summary="Get model loading status")
async def model_status(model_id: str):
    if model_id not in _profiles:
        raise HTTPException(status_code=404, detail=f"Unknown model_id: {model_id}")
    with _state_lock:
        if model_id in _models:
            return {
                "model_id": model_id,
                "status": "ready",
                "progress": 100,
                "message": "Model loaded",
                "loaded": True,
                "error": None,
            }
        state = _load_state.get(model_id)
    if state:
        return state
    return {
        "model_id": model_id,
        "status": "idle",
        "progress": 0,
        "message": "Not loaded yet",
        "loaded": False,
        "error": None,
    }


@app.post("/api/models/{model_id}/preload", summary="Start model loading in background")
async def preload_model(model_id: str):
    if model_id not in _profiles:
        raise HTTPException(status_code=404, detail=f"Unknown model_id: {model_id}")
    with _state_lock:
        if model_id in _models:
            return {
                "model_id": model_id,
                "status": "ready",
                "progress": 100,
                "message": "Model already loaded",
                "loaded": True,
                "error": None,
            }
        state = _load_state.get(model_id, {})
        if state.get("status") == "loading":
            return state
    _start_background_load(model_id)
    return {
        "model_id": model_id,
        "status": "loading",
        "progress": 5,
        "message": "Loading started",
        "loaded": False,
        "error": None,
    }


@app.post("/api/models/{model_id}/unload", summary="Unload model from memory (free RAM)")
async def unload_model(model_id: str):
    if model_id not in _profiles:
        raise HTTPException(status_code=404, detail=f"Unknown model_id: {model_id}")
    with _state_lock:
        if model_id not in _models:
            return {"model_id": model_id, "status": "idle", "message": "Model was not loaded"}
        del _models[model_id]
        _load_state.pop(model_id, None)
        _load_events.pop(model_id, None)
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return {"model_id": model_id, "status": "idle", "message": "Model unloaded from memory"}


@app.post("/api/classify", summary="Classify an Arabic ticket with selected model")
async def classify(body: TicketIn, model_id: str | None = Query(default=None)):
    if not body.title_ar.strip():
        raise HTTPException(status_code=400, detail="title_ar must not be empty")

    target_model = model_id or _default_model_id()
    if not target_model:
        raise HTTPException(status_code=503, detail="No model profiles available")

    clf = _get_model(target_model)
    result = clf.predict(body.title_ar, body.description_ar)
    result["model_id"] = target_model
    result["tasks"] = clf.tasks
    return result


@app.post("/api/classify/all", summary="Classify one ticket with all discovered models")
async def classify_all(body: TicketIn):
    if not body.title_ar.strip():
        raise HTTPException(status_code=400, detail="title_ar must not be empty")
    if not _profiles:
        raise HTTPException(status_code=503, detail="No model profiles available")

    results = []
    for model_id in _profiles:
        clf = _get_model(model_id)
        result = clf.predict(body.title_ar, body.description_ar)
        result["model_id"] = model_id
        result["tasks"] = clf.tasks
        results.append(result)
    return {"count": len(results), "results": results}


@app.get("/api/health", summary="Health check")
async def health():
    return {
        "status": "ok" if _profiles else "models_not_loaded",
        "profiles_count": len(_profiles),
        "loaded_models_count": len(_models),
        "default_model_id": _default_model_id(),
    }


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/models")


@app.get("/classify", include_in_schema=False)
async def classify_page():
    return FileResponse("static/index.html")


@app.get("/models", include_in_schema=False)
async def models_page():
    return FileResponse("static/models.html")


@app.get("/compare", include_in_schema=False)
async def compare_page():
    return FileResponse("static/compare.html")


@app.get("/models/{model_id}", include_in_schema=False)
async def model_page(model_id: str):
    if model_id not in _profiles:
        raise HTTPException(status_code=404, detail=f"Unknown model_id: {model_id}")
    return FileResponse("static/model.html")
