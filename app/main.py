"""
FastAPI server for multi-model Arabic ITSM ticket classification.

API routes:
  - GET  /api/models, /api/models/{id}/status, preload, unload
  - POST /api/classify?model_id=<id>
  - POST /api/classify/all
  - GET  /api/classifications?limit=&offset=
  - GET  /api/classifications/{id}
  - POST /api/feedback
  - GET  /api/stats
  - GET  /api/monitoring
  - GET  /api/export/csv  (ZIP with CSV)
  - GET  /api/export/sql  (SQL dump)
  - GET  /api/health

Page routes:
  - GET  /           -> redirect to /models
  - GET  /models
  - GET  /models/{model_id}
  - GET  /dashboard
  - GET  /monitoring
  - GET  /admin/export
"""
from __future__ import annotations

import csv
import gc
import io
import os
import threading
import time
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

import psutil
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.database import (
    count_classifications,
    export_classifications_csv_rows,
    export_sql_dump,
    get_monitoring_history,
    get_stats,
    get_visitor_stats,
    init_db,
    insert_classification,
    insert_feedback,
    insert_monitoring_snapshot,
    insert_visitor,
    list_classifications,
    get_classification,
    prune_monitoring_log,
)

load_dotenv()

# ── Model state ────────────────────────────────────────────────────────────────
_profiles: dict[str, dict] = {}
_models: dict[str, object] = {}
_load_state: dict[str, dict] = {}
_load_events: dict[str, threading.Event] = {}
_state_lock = threading.RLock()

# ── Server metrics ─────────────────────────────────────────────────────────────
_active_requests: int = 0
_active_requests_lock = threading.Lock()
_server_start_time: float = time.time()


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
            return

    t = threading.Thread(target=_runner, daemon=True)
    t.start()


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    # 1. Initialize database
    init_db()
    print("[ok] SQLite database initialized")

    # 2. Discover model profiles
    global _profiles
    _profiles = _discover_profiles()
    if _profiles:
        print(f"[ok] Discovered {len(_profiles)} model profile(s)")
    else:
        print("[warn] No model profiles discovered")
        print("       Put checkpoints under models/ or set MODEL_DIRS")

    # 3. Start background monitoring task (every 60s)
    async def _monitoring_loop():
        while True:
            await asyncio.sleep(60)
            try:
                proc = psutil.Process()
                cpu = psutil.cpu_percent(interval=None)
                mem = proc.memory_info().rss / (1024 * 1024)
                with _active_requests_lock:
                    active = _active_requests
                insert_monitoring_snapshot(cpu, mem, active)
                prune_monitoring_log(keep_hours=48)
            except Exception as exc:
                print(f"[warn] Monitoring snapshot failed: {exc}")

    task = asyncio.create_task(_monitoring_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Arabic ITSM Classifier",
    description="Multi-model Arabic ITSM ticket classification API + web UI.",
    version="3.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Visitor middleware ─────────────────────────────────────────────────────────

class VisitorMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        global _active_requests
        # Skip static asset requests to reduce noise
        if not request.url.path.startswith("/static"):
            ip = request.client.host if request.client else "unknown"
            user_agent = request.headers.get("user-agent", "")
            try:
                insert_visitor(ip, request.url.path, user_agent)
            except Exception:
                pass
        with _active_requests_lock:
            _active_requests += 1
        try:
            response = await call_next(request)
        finally:
            with _active_requests_lock:
                _active_requests -= 1
        return response


app.add_middleware(VisitorMiddleware)


# ── Pydantic models ────────────────────────────────────────────────────────────

class TicketIn(BaseModel):
    title_ar: str
    description_ar: str = ""


class FeedbackIn(BaseModel):
    classification_id: int
    thumbs: str
    comment: str | None = None
    email: str | None = None


# ── Model management routes ────────────────────────────────────────────────────

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


# ── Inference routes ───────────────────────────────────────────────────────────

@app.post("/api/classify", summary="Classify an Arabic ticket with selected model")
async def classify(body: TicketIn, model_id: str | None = Query(default=None)):
    api_start = time.perf_counter()

    if not body.title_ar.strip():
        raise HTTPException(status_code=400, detail="title_ar must not be empty")

    target_model = model_id or _default_model_id()
    if not target_model:
        raise HTTPException(status_code=503, detail="No model profiles available")

    clf = _get_model(target_model)
    result = clf.predict(body.title_ar, body.description_ar)
    result["model_id"] = target_model
    result["tasks"] = clf.tasks

    api_time_ms = round((time.perf_counter() - api_start) * 1000, 1)
    result["api_time_ms"] = api_time_ms

    # Persist to DB (non-blocking — DB failure must never kill a classify)
    try:
        cid = insert_classification(
            ticket_title=body.title_ar,
            ticket_text=body.description_ar,
            model_id=target_model,
            model_response=result,
            api_time_ms=api_time_ms,
            inference_time_ms=float(result.get("latency_ms", 0)),
        )
        result["classification_id"] = cid
    except Exception as exc:
        print(f"[warn] DB insert failed: {exc}")
        result["classification_id"] = None

    return result


@app.post("/api/classify/all", summary="Classify one ticket with all discovered models")
async def classify_all(body: TicketIn):
    if not body.title_ar.strip():
        raise HTTPException(status_code=400, detail="title_ar must not be empty")
    if not _profiles:
        raise HTTPException(status_code=503, detail="No model profiles available")

    results = []
    for mid in _profiles:
        clf = _get_model(mid)
        result = clf.predict(body.title_ar, body.description_ar)
        result["model_id"] = mid
        result["tasks"] = clf.tasks
        results.append(result)
    return {"count": len(results), "results": results}


# ── Classification history routes ──────────────────────────────────────────────

@app.get("/api/classifications", summary="Paginated classification history")
async def api_list_classifications(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    rows = list_classifications(limit=limit, offset=offset)
    total = count_classifications()
    return {"total": total, "limit": limit, "offset": offset, "items": rows}


@app.get("/api/classifications/{classification_id}", summary="Single classification record")
async def api_get_classification(classification_id: int):
    row = get_classification(classification_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Classification {classification_id} not found")
    return row


# ── Feedback route ─────────────────────────────────────────────────────────────

@app.post("/api/feedback", summary="Submit thumbs up/down feedback")
async def api_feedback(body: FeedbackIn):
    if body.thumbs not in ("up", "down"):
        raise HTTPException(status_code=400, detail="thumbs must be 'up' or 'down'")
    try:
        fid = insert_feedback(
            classification_id=body.classification_id,
            thumbs=body.thumbs,
            comment=body.comment,
            email=body.email,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"feedback_id": fid, "status": "ok"}


# ── Stats route ────────────────────────────────────────────────────────────────

@app.get("/api/stats", summary="Aggregated classification and feedback stats")
async def api_stats():
    return get_stats()


# ── Monitoring route ───────────────────────────────────────────────────────────

@app.get("/api/monitoring", summary="Current process stats + 24h history")
async def api_monitoring():
    proc = psutil.Process()
    uptime_seconds = time.time() - _server_start_time
    return {
        "current": {
            "cpu_pct": psutil.cpu_percent(interval=None),
            "mem_mb": round(proc.memory_info().rss / (1024 * 1024), 1),
            "uptime_seconds": round(uptime_seconds),
            "active_requests": _active_requests,
        },
        "history_24h": get_monitoring_history(hours=24),
        "visitor_stats": get_visitor_stats(),
    }


# ── Export routes ──────────────────────────────────────────────────────────────

@app.get("/api/export/csv", summary="Download all classifications as CSV zip")
async def api_export_csv():
    rows = export_classifications_csv_rows()

    buf = io.StringIO()
    if rows:
        writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    # UTF-8 BOM so Excel reads Arabic correctly
    csv_content = ("\ufeff" + buf.getvalue()).encode("utf-8")

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("classifications.csv", csv_content)
    zip_buf.seek(0)

    return StreamingResponse(
        zip_buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=itsm_classifications.zip"},
    )


@app.get("/api/export/sql", summary="Download full DB as SQL dump")
async def api_export_sql():
    dump = export_sql_dump()

    def _iter():
        yield dump.encode("utf-8")

    return StreamingResponse(
        _iter(),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=itsm_backup.sql"},
    )


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/api/health", summary="Health check")
async def health():
    return {
        "status": "ok" if _profiles else "models_not_loaded",
        "profiles_count": len(_profiles),
        "loaded_models_count": len(_models),
        "default_model_id": _default_model_id(),
    }


# ── Page routes ────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/models")


@app.get("/models", include_in_schema=False)
async def models_page():
    return FileResponse("static/models.html")


@app.get("/dashboard", include_in_schema=False)
async def dashboard_page():
    return FileResponse("static/dashboard.html")


@app.get("/monitoring", include_in_schema=False)
async def monitoring_page():
    return FileResponse("static/monitoring.html")


@app.get("/admin/export", include_in_schema=False)
async def export_page():
    return FileResponse("static/export.html")


@app.get("/models/{model_id}", include_in_schema=False)
async def model_page(model_id: str):
    if model_id not in _profiles:
        raise HTTPException(status_code=404, detail=f"Unknown model_id: {model_id}")
    return FileResponse("static/model.html")
