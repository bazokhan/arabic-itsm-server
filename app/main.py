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
  - GET  /api/research/status
  - POST /api/research/run
  - GET  /api/health

Page routes:
  - GET  /           -> redirect to /models
  - GET  /models
  - GET  /models/{model_id}
  - GET  /research
  - GET  /dashboard
  - GET  /monitoring
  - GET  /admin/export
"""
from __future__ import annotations

import csv
import gc
import io
import os
import subprocess
import sys
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

# ── Research benchmark state ───────────────────────────────────────────────────
_research_state: dict[str, object] = {
    "status": "idle",  # idle | running | success | error
    "started_at": None,
    "finished_at": None,
    "return_code": None,
    "message": "No benchmark has been executed yet.",
    "last_run_cmd": None,
    "last_run_duration_sec": None,
    "log_tail": [],
}
_research_lock = threading.RLock()


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


def _update_research_state(**kwargs):
    with _research_lock:
        _research_state.update(kwargs)


def _append_research_log(line: str, limit: int = 120):
    text = line.strip()
    if not text:
        return
    with _research_lock:
        logs = list(_research_state.get("log_tail", []))
        logs.append(text)
        _research_state["log_tail"] = logs[-limit:]


def _run_research_benchmark(
    limit: int | None = None,
    split_name: str = "test",
    model_a_id_override: str | None = None,
    model_b_id_override: str | None = None,
    model_a_path_override: str | None = None,
    model_b_path_override: str | None = None,
    dataset_csv_override: str | None = None,
    dataset_url_override: str | None = None,
):
    model_a_id = (model_a_id_override or os.getenv("COMPARISON_MODEL_A_ID", "marbert-arabic-itsm-l3-categories")).strip()
    model_b_id = (model_b_id_override or os.getenv("COMPARISON_MODEL_B_ID", "arabert-arabic-itsm-l3-categories")).strip()
    model_a_path = (model_a_path_override or os.getenv("COMPARISON_MODEL_A_PATH", "")).strip()
    model_b_path = (model_b_path_override or os.getenv("COMPARISON_MODEL_B_PATH", "")).strip()
    if not model_a_path and model_a_id in _profiles:
        model_a_path = str(_profiles[model_a_id]["path"])
    if not model_b_path and model_b_id in _profiles:
        model_b_path = str(_profiles[model_b_id]["path"])

    dataset_csv = (dataset_csv_override or os.getenv("COMPARISON_DATASET_CSV", "data/processed/test.csv")).strip()
    dataset_url = (dataset_url_override or os.getenv("COMPARISON_DATASET_URL", "")).strip()
    model_a_url = os.getenv(
        "COMPARISON_MODEL_A_URL",
        "https://huggingface.co/albaz2000/marbert-arabic-itsm-l3-categories",
    ).strip()
    model_b_url = os.getenv(
        "COMPARISON_MODEL_B_URL",
        "https://huggingface.co/albaz2000/arabert-arabic-itsm-l3-categories",
    ).strip()
    dataset_hf_url = os.getenv(
        "COMPARISON_DATASET_HF_URL",
        "https://huggingface.co/datasets/albaz2000/arabic-itsm-dataset",
    ).strip()

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_model_comparison.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--split-name",
        split_name,
        "--model-a-id",
        model_a_id,
        "--model-b-id",
        model_b_id,
        "--model-a-url",
        model_a_url,
        "--model-b-url",
        model_b_url,
        "--dataset-hf-url",
        dataset_hf_url,
    ]
    if dataset_csv:
        cmd.extend(["--dataset-csv", dataset_csv])
    if dataset_url:
        cmd.extend(["--dataset-url", dataset_url])
    if model_a_path:
        cmd.extend(["--model-a-path", model_a_path])
    if model_b_path:
        cmd.extend(["--model-b-path", model_b_path])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    _update_research_state(
        status="running",
        started_at=time.time(),
        finished_at=None,
        return_code=None,
        message="Benchmark is running.",
        last_run_cmd=" ".join(cmd),
        last_run_duration_sec=None,
        log_tail=[],
    )

    start = time.perf_counter()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception as exc:
        _update_research_state(
            status="error",
            finished_at=time.time(),
            return_code=-1,
            message=f"Failed to start benchmark: {exc}",
        )
        return

    assert proc.stdout is not None
    for line in proc.stdout:
        _append_research_log(line)

    code = proc.wait()
    duration = round(time.perf_counter() - start, 2)
    if code == 0:
        _update_research_state(
            status="success",
            finished_at=time.time(),
            return_code=0,
            message="Benchmark completed successfully.",
            last_run_duration_sec=duration,
        )
    else:
        _update_research_state(
            status="error",
            finished_at=time.time(),
            return_code=code,
            message="Benchmark failed. Check logs for details.",
            last_run_duration_sec=duration,
        )


def _start_research_benchmark(
    limit: int | None = None,
    split_name: str = "test",
    model_a_id_override: str | None = None,
    model_b_id_override: str | None = None,
    model_a_path_override: str | None = None,
    model_b_path_override: str | None = None,
    dataset_csv_override: str | None = None,
    dataset_url_override: str | None = None,
) -> bool:
    with _research_lock:
        if _research_state.get("status") == "running":
            return False
    t = threading.Thread(
        target=_run_research_benchmark,
        kwargs={
            "limit": limit,
            "split_name": split_name,
            "model_a_id_override": model_a_id_override,
            "model_b_id_override": model_b_id_override,
            "model_a_path_override": model_a_path_override,
            "model_b_path_override": model_b_path_override,
            "dataset_csv_override": dataset_csv_override,
            "dataset_url_override": dataset_url_override,
        },
        daemon=True,
    )
    t.start()
    return True


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

_OPENAPI_TAGS = [
    {
        "name": "Models",
        "description": "Discover, preload, and unload Arabic ITSM classifier checkpoints.",
    },
    {
        "name": "Inference",
        "description": (
            "Classify Arabic IT service-management tickets. "
            "Each ticket is described by an Arabic title and optional description. "
            "Predictions cover up to 5 hierarchical tasks: **L1** (category), **L2** (sub-category), "
            "**L3** (leaf category), **Priority**, and **Sentiment**."
        ),
    },
    {
        "name": "History",
        "description": "Browse the persisted classification log stored in SQLite.",
    },
    {
        "name": "Feedback",
        "description": "Thumbs-up / thumbs-down ratings attached to classification records.",
    },
    {
        "name": "Stats",
        "description": "Aggregated usage and accuracy statistics.",
    },
    {
        "name": "Monitoring",
        "description": "Live CPU/memory metrics and 24 h history.",
    },
    {
        "name": "Export",
        "description": "Download the full classification database as CSV (zip) or SQL dump.",
    },
    {
        "name": "Research",
        "description": "Trigger an offline benchmark comparing two model checkpoints and regenerate research artefacts.",
    },
    {
        "name": "Health",
        "description": "Lightweight liveness probe.",
    },
]

_API_DESCRIPTION = """\
## Arabic ITSM Classifier — REST API

Multi-model inference server for classifying **Arabic IT service-management tickets** into hierarchical categories.

### Authentication
No authentication is required for local or development deployments.

### Ticket format
All inference endpoints accept an `application/json` body with:
- `title_ar` — Arabic ticket title *(required)*
- `description_ar` — Arabic ticket body text *(optional, improves accuracy)*

### Prediction tasks
| Task | Labels | Description |
|------|--------|-------------|
| `l1` | 6  | Top-level service category |
| `l2` | 16 | Sub-category |
| `l3` | 48 | Leaf category |
| `priority` | 5  | Ticket urgency |
| `sentiment` | 4  | Requester sentiment |

Each task result includes `label` (top prediction) and `top_k` (ranked alternatives with confidence scores).

### Model discovery
Checkpoint directories are discovered at startup from the `MODEL_DIRS` environment variable.
Use `GET /api/models` to list available checkpoints and their load state.

### Web UI pages
| Path | Description |
|------|-------------|
| `/models` | Model gallery |
| `/models/{id}` | Single-model classify form |
| `/dashboard` | Classification history dashboard |
| `/monitoring` | CPU / memory monitoring |
| `/admin/export` | Data export |
| `/research` | Offline benchmark runner |

### OpenAPI spec
- **Swagger UI** → `/swagger`
- **ReDoc** → `/redoc`
- **JSON spec** → `/openapi.json`
"""

app = FastAPI(
    title="Arabic ITSM Classifier",
    description=_API_DESCRIPTION,
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/swagger",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=_OPENAPI_TAGS,
    contact={
        "name": "Arabic ITSM Project",
        "url": "https://huggingface.co/albaz2000",
    },
    license_info={
        "name": "MIT",
    },
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")


# ── Visitor middleware ─────────────────────────────────────────────────────────

class VisitorMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        global _active_requests
        # Skip static asset requests to reduce noise
        if not request.url.path.startswith("/static") and not request.url.path.startswith("/assets"):
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


# ── Pydantic models — request bodies ──────────────────────────────────────────

class TicketIn(BaseModel):
    title_ar: str
    description_ar: str = ""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title_ar": "مشكلة في الاتصال بالإنترنت",
                    "description_ar": "لا يمكنني الاتصال بالإنترنت منذ الصباح، الشبكة ظاهرة لكن لا يوجد اتصال فعلي.",
                }
            ]
        }
    }


class FeedbackIn(BaseModel):
    classification_id: int
    thumbs: str
    comment: str | None = None
    email: str | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "classification_id": 42,
                    "thumbs": "up",
                    "comment": "Accurate category, very helpful.",
                    "email": None,
                }
            ]
        }
    }


class ResearchRunIn(BaseModel):
    limit: int | None = None
    split_name: str = "test"
    dataset_csv: str | None = None
    dataset_url: str | None = None
    model_a_id: str | None = None
    model_b_id: str | None = None
    model_a_path: str | None = None
    model_b_path: str | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "split_name": "test",
                    "limit": 500,
                    "model_a_id": "marbert-arabic-itsm-l3-categories",
                    "model_b_id": "arabert-arabic-itsm-l3-categories",
                }
            ]
        }
    }


# ── Pydantic models — responses ────────────────────────────────────────────────

class ModelProfileItem(BaseModel):
    id: str
    name: str
    path: str
    description: str
    loaded: bool
    status: str
    progress: int
    tasks: list[str]


class ModelListResponse(BaseModel):
    default_model_id: str
    models: list[ModelProfileItem]


class ModelStatusResponse(BaseModel):
    model_id: str
    status: str
    progress: int
    message: str
    loaded: bool
    error: str | None = None


class PreloadResponse(BaseModel):
    model_id: str
    status: str
    progress: int
    message: str
    loaded: bool
    error: str | None = None


class UnloadResponse(BaseModel):
    model_id: str
    status: str
    message: str


class TopKEntry(BaseModel):
    label: str
    prob: float


class TaskPrediction(BaseModel):
    label: str
    confidence: float
    top3: list[TopKEntry]


class ClassifyResponse(BaseModel):
    model_id: str
    tasks: list[str]
    latency_ms: float
    api_time_ms: float
    classification_id: int | None = None
    l1: TaskPrediction | None = None
    l2: TaskPrediction | None = None
    l3: TaskPrediction | None = None
    priority: TaskPrediction | None = None
    sentiment: TaskPrediction | None = None

    model_config = {"extra": "allow"}


class ClassifyAllResponse(BaseModel):
    count: int
    results: list[ClassifyResponse]


class ClassificationRecord(BaseModel):
    id: int
    ticket_title: str
    ticket_text: str | None = None
    model_id: str
    model_response: dict
    api_time_ms: float | None = None
    inference_time_ms: float | None = None
    created_at: str

    model_config = {"extra": "allow"}


class ClassificationListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[ClassificationRecord]


class FeedbackResponse(BaseModel):
    feedback_id: int
    status: str


class HealthResponse(BaseModel):
    status: str
    profiles_count: int
    loaded_models_count: int
    default_model_id: str


class ResearchStatusResponse(BaseModel):
    status: str
    started_at: float | None = None
    finished_at: float | None = None
    return_code: int | None = None
    message: str
    last_run_cmd: str | None = None
    last_run_duration_sec: float | None = None
    log_tail: list[str]

    model_config = {"extra": "allow"}


class ResearchRunResponse(BaseModel):
    accepted: bool
    message: str
    status: ResearchStatusResponse


# ── Model management routes ────────────────────────────────────────────────────

@app.get(
    "/api/models",
    tags=["Models"],
    summary="List available model profiles",
    response_model=ModelListResponse,
    responses={200: {"description": "All discovered checkpoints with their current load state."}},
)
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


@app.get(
    "/api/models/{model_id}/status",
    tags=["Models"],
    summary="Get model loading status",
    response_model=ModelStatusResponse,
    responses={
        200: {"description": "Current load state (idle | loading | ready | error) with progress 0-100."},
        404: {"description": "Unknown model_id."},
    },
)
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


@app.post(
    "/api/models/{model_id}/preload",
    tags=["Models"],
    summary="Start model loading in background",
    response_model=PreloadResponse,
    responses={
        200: {"description": "Load started (or already in progress / already loaded)."},
        404: {"description": "Unknown model_id."},
    },
)
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


@app.post(
    "/api/models/{model_id}/unload",
    tags=["Models"],
    summary="Unload model from memory (free RAM/VRAM)",
    response_model=UnloadResponse,
    responses={
        200: {"description": "Model unloaded; GPU/CPU memory freed."},
        404: {"description": "Unknown model_id."},
    },
)
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

@app.post(
    "/api/classify",
    tags=["Inference"],
    summary="Classify an Arabic ticket with a selected model",
    response_model=ClassifyResponse,
    responses={
        200: {"description": "Predictions for all tasks supported by the chosen checkpoint."},
        400: {"description": "Empty title_ar."},
        404: {"description": "Unknown model_id."},
        500: {"description": "Model failed to load."},
        503: {"description": "No model profiles available."},
    },
)
async def classify(
    body: TicketIn,
    model_id: str | None = Query(default=None, description="Checkpoint ID from `/api/models`. Defaults to the server default."),
):
    import asyncio

    api_start = time.perf_counter()

    if not body.title_ar.strip():
        raise HTTPException(status_code=400, detail="title_ar must not be empty")

    target_model = model_id or _default_model_id()
    if not target_model:
        raise HTTPException(status_code=503, detail="No model profiles available")

    clf = _get_model(target_model)

    # Run blocking CPU/GPU inference in a thread so the event loop is not stalled
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, clf.predict, body.title_ar, body.description_ar)
    except Exception as exc:
        print(f"[error] Inference failed for model {target_model}: {exc}", flush=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

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


@app.post(
    "/api/classify/all",
    tags=["Inference"],
    summary="Classify one ticket with every discovered model",
    response_model=ClassifyAllResponse,
    responses={
        200: {"description": "Array of per-model prediction results."},
        400: {"description": "Empty title_ar."},
        503: {"description": "No model profiles available."},
    },
)
async def classify_all(body: TicketIn):
    import asyncio

    if not body.title_ar.strip():
        raise HTTPException(status_code=400, detail="title_ar must not be empty")
    if not _profiles:
        raise HTTPException(status_code=503, detail="No model profiles available")

    loop = asyncio.get_event_loop()
    results = []
    for mid in _profiles:
        clf = _get_model(mid)
        try:
            result = await loop.run_in_executor(None, clf.predict, body.title_ar, body.description_ar)
        except Exception as exc:
            print(f"[error] Inference failed for model {mid}: {exc}", flush=True)
            raise HTTPException(status_code=500, detail=f"Inference error (model {mid}): {exc}")
        result["model_id"] = mid
        result["tasks"] = clf.tasks
        results.append(result)
    return {"count": len(results), "results": results}


# ── Classification history routes ──────────────────────────────────────────────

@app.get(
    "/api/classifications",
    tags=["History"],
    summary="Paginated classification history",
    response_model=ClassificationListResponse,
    responses={200: {"description": "Page of classification records with total count."}},
)
async def api_list_classifications(
    limit: int = Query(default=20, ge=1, le=100, description="Records per page (1-100)."),
    offset: int = Query(default=0, ge=0, description="Zero-based row offset for pagination."),
):
    rows = list_classifications(limit=limit, offset=offset)
    total = count_classifications()
    return {"total": total, "limit": limit, "offset": offset, "items": rows}


@app.get(
    "/api/classifications/{classification_id}",
    tags=["History"],
    summary="Fetch a single classification record",
    response_model=ClassificationRecord,
    responses={
        200: {"description": "Full classification record including the raw model response JSON."},
        404: {"description": "No record with that ID."},
    },
)
async def api_get_classification(classification_id: int):
    row = get_classification(classification_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Classification {classification_id} not found")
    return row


# ── Feedback route ─────────────────────────────────────────────────────────────

@app.post(
    "/api/feedback",
    tags=["Feedback"],
    summary="Submit thumbs-up / thumbs-down feedback on a classification",
    response_model=FeedbackResponse,
    responses={
        200: {"description": "Feedback persisted; returns new feedback_id."},
        400: {"description": "Invalid thumbs value or unknown classification_id."},
    },
)
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

@app.get(
    "/api/stats",
    tags=["Stats"],
    summary="Aggregated classification and feedback stats",
    responses={200: {"description": "Total classifications, feedback thumbs counts, per-model breakdown, daily histogram."}},
)
async def api_stats():
    return get_stats()


# ── Monitoring route ───────────────────────────────────────────────────────────

@app.get(
    "/api/monitoring",
    tags=["Monitoring"],
    summary="Current process stats and 24 h CPU/memory history",
    responses={
        200: {
            "description": (
                "`current` — live CPU %, RSS MB, uptime seconds, active requests. "
                "`history_24h` — 60-second snapshots from the last 24 hours. "
                "`visitor_stats` — unique IPs and top paths."
            )
        }
    },
)
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

@app.get(
    "/api/export/csv",
    tags=["Export"],
    summary="Download all classifications as UTF-8 CSV (zip)",
    responses={
        200: {
            "description": "ZIP archive containing `classifications.csv` (UTF-8 BOM for Excel compatibility).",
            "content": {"application/zip": {}},
        }
    },
)
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


@app.get(
    "/api/export/sql",
    tags=["Export"],
    summary="Download full SQLite database as a plain-text SQL dump",
    responses={
        200: {
            "description": "Plain-text SQL dump of all four tables (classifications, feedback, monitoring_log, visitors).",
            "content": {"text/plain": {}},
        }
    },
)
async def api_export_sql():
    dump = export_sql_dump()

    def _iter():
        yield dump.encode("utf-8")

    return StreamingResponse(
        _iter(),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=itsm_backup.sql"},
    )


# ── Research benchmark routes ──────────────────────────────────────────────────

@app.get(
    "/api/research/status",
    tags=["Research"],
    summary="Get offline benchmark execution status",
    response_model=ResearchStatusResponse,
    responses={
        200: {
            "description": (
                "`status` is one of `idle | running | success | error`. "
                "`log_tail` contains the last 120 lines of stdout/stderr from the benchmark script."
            )
        }
    },
)
async def api_research_status():
    with _research_lock:
        return dict(_research_state)


@app.post(
    "/api/research/run",
    tags=["Research"],
    summary="Trigger the offline model-comparison benchmark",
    response_model=ResearchRunResponse,
    responses={
        200: {
            "description": (
                "`accepted: true` — benchmark started in a background thread. "
                "`accepted: false` — a benchmark is already running."
            )
        },
        400: {"description": "Invalid request body (e.g. non-positive limit)."},
    },
)
async def api_research_run(body: ResearchRunIn):
    if body.limit is not None and body.limit <= 0:
        raise HTTPException(status_code=400, detail="limit must be positive when provided")

    started = _start_research_benchmark(
        limit=body.limit,
        split_name=body.split_name,
        model_a_id_override=body.model_a_id,
        model_b_id_override=body.model_b_id,
        model_a_path_override=body.model_a_path,
        model_b_path_override=body.model_b_path,
        dataset_csv_override=body.dataset_csv,
        dataset_url_override=body.dataset_url,
    )
    if not started:
        with _research_lock:
            return {
                "accepted": False,
                "message": "Benchmark is already running.",
                "status": dict(_research_state),
            }
    with _research_lock:
        return {
            "accepted": True,
            "message": "Benchmark started.",
            "status": dict(_research_state),
        }


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get(
    "/api/health",
    tags=["Health"],
    summary="Server liveness probe",
    response_model=HealthResponse,
    responses={
        200: {
            "description": (
                "`status` is `ok` when at least one model profile has been discovered, "
                "otherwise `models_not_loaded`."
            )
        }
    },
)
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


@app.get("/research", include_in_schema=False)
async def research_page():
    return FileResponse("static/research.html")


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
