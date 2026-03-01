"""
SQLite persistence layer for the Arabic ITSM server.

Tables:
  - classifications  — every /api/classify request + response
  - feedback         — user thumbs-up/down + optional comment/email
  - monitoring_log   — periodic psutil snapshots (CPU, memory)
  - visitors         — per-request IP + path (non-static only)

DB path: DB_PATH env var (default: ./data/itsm.db)
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ── Config ────────────────────────────────────────────────────────────────────

def get_db_path() -> str:
    return os.getenv("DB_PATH", "./data/itsm.db")


# ── Connection factory ────────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    """Open a WAL-mode SQLite connection. Caller must close it."""
    conn = sqlite3.connect(get_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS classifications (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at        DATETIME NOT NULL DEFAULT (datetime('now')),
    ticket_title      TEXT NOT NULL,
    ticket_text       TEXT NOT NULL DEFAULT '',
    model_id          TEXT NOT NULL,
    model_response    TEXT NOT NULL,
    api_time_ms       REAL NOT NULL DEFAULT 0.0,
    inference_time_ms REAL NOT NULL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_cls_created_at ON classifications(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_cls_model_id ON classifications(model_id);

CREATE TABLE IF NOT EXISTS feedback (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    classification_id   INTEGER NOT NULL REFERENCES classifications(id),
    created_at          DATETIME NOT NULL DEFAULT (datetime('now')),
    thumbs              TEXT NOT NULL CHECK(thumbs IN ('up','down')),
    comment             TEXT,
    email               TEXT
);
CREATE INDEX IF NOT EXISTS idx_fb_cls_id ON feedback(classification_id);

CREATE TABLE IF NOT EXISTS monitoring_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       DATETIME NOT NULL DEFAULT (datetime('now')),
    cpu_pct         REAL NOT NULL DEFAULT 0.0,
    mem_mb          REAL NOT NULL DEFAULT 0.0,
    active_requests INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_mon_ts ON monitoring_log(timestamp DESC);

CREATE TABLE IF NOT EXISTS visitors (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   DATETIME NOT NULL DEFAULT (datetime('now')),
    ip          TEXT NOT NULL,
    path        TEXT NOT NULL,
    user_agent  TEXT
);
CREATE INDEX IF NOT EXISTS idx_vis_ts ON visitors(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_vis_ip ON visitors(ip);
"""


def init_db() -> None:
    """Create tables and indexes. Called once at startup."""
    db_path = Path(get_db_path())
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection()
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


# ── Classifications ───────────────────────────────────────────────────────────

def insert_classification(
    ticket_title: str,
    ticket_text: str,
    model_id: str,
    model_response: dict,
    api_time_ms: float,
    inference_time_ms: float,
) -> int:
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            INSERT INTO classifications
              (ticket_title, ticket_text, model_id, model_response, api_time_ms, inference_time_ms)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                ticket_title,
                ticket_text,
                model_id,
                json.dumps(model_response, ensure_ascii=False),
                round(api_time_ms, 2),
                round(inference_time_ms, 2),
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def _row_to_cls(row: sqlite3.Row) -> dict:
    d = dict(row)
    try:
        d["model_response"] = json.loads(d["model_response"])
    except Exception:
        pass
    return d


def get_classification(classification_id: int) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM classifications WHERE id = ?", (classification_id,)
        ).fetchone()
        return _row_to_cls(row) if row else None
    finally:
        conn.close()


def list_classifications(limit: int = 20, offset: int = 0) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM classifications ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [_row_to_cls(r) for r in rows]
    finally:
        conn.close()


def count_classifications() -> int:
    conn = get_connection()
    try:
        return conn.execute("SELECT COUNT(*) FROM classifications").fetchone()[0]
    finally:
        conn.close()


# ── Feedback ──────────────────────────────────────────────────────────────────

def insert_feedback(
    classification_id: int,
    thumbs: str,
    comment: Optional[str] = None,
    email: Optional[str] = None,
) -> int:
    if thumbs not in ("up", "down"):
        raise ValueError("thumbs must be 'up' or 'down'")
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            INSERT INTO feedback (classification_id, thumbs, comment, email)
            VALUES (?, ?, ?, ?)
            """,
            (classification_id, thumbs, comment or None, email or None),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


# ── Stats ─────────────────────────────────────────────────────────────────────

def get_stats() -> dict:
    conn = get_connection()
    try:
        total_cls = conn.execute("SELECT COUNT(*) FROM classifications").fetchone()[0]
        total_fb = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]

        thumbs_up = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE thumbs='up'"
        ).fetchone()[0]
        thumbs_up_pct = round(thumbs_up / total_fb * 100, 1) if total_fb else None

        avg_inf = conn.execute(
            "SELECT AVG(inference_time_ms) FROM classifications"
        ).fetchone()[0]
        avg_inf_ms = round(avg_inf, 1) if avg_inf is not None else 0.0

        by_model_rows = conn.execute(
            "SELECT model_id, COUNT(*) as count FROM classifications GROUP BY model_id ORDER BY count DESC"
        ).fetchall()
        by_model = [{"model_id": r["model_id"], "count": r["count"]} for r in by_model_rows]

        # Extract L1 label from JSON blobs in Python (avoid SQLite json_extract portability issues)
        all_responses = conn.execute(
            "SELECT model_response FROM classifications"
        ).fetchall()
        l1_counts: dict[str, int] = {}
        for row in all_responses:
            try:
                resp = json.loads(row[0])
                label = resp.get("l1", {}).get("label")
                if label:
                    l1_counts[label] = l1_counts.get(label, 0) + 1
            except Exception:
                pass
        by_l1_label = sorted(
            [{"label": k, "count": v} for k, v in l1_counts.items()],
            key=lambda x: x["count"],
            reverse=True,
        )

        # Per-day counts for last 7 days
        per_day_rows = conn.execute(
            """
            SELECT DATE(created_at) as day, COUNT(*) as count
            FROM classifications
            WHERE created_at >= DATE('now', '-6 days')
            GROUP BY day
            ORDER BY day ASC
            """
        ).fetchall()
        per_day_last_7 = [{"date": r["day"], "count": r["count"]} for r in per_day_rows]

        return {
            "total_classifications": total_cls,
            "total_feedback": total_fb,
            "thumbs_up_pct": thumbs_up_pct,
            "avg_inference_ms": avg_inf_ms,
            "by_model": by_model,
            "by_l1_label": by_l1_label,
            "per_day_last_7": per_day_last_7,
        }
    finally:
        conn.close()


# ── Monitoring ────────────────────────────────────────────────────────────────

def insert_monitoring_snapshot(cpu_pct: float, mem_mb: float, active_requests: int) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO monitoring_log (cpu_pct, mem_mb, active_requests) VALUES (?, ?, ?)",
            (round(cpu_pct, 1), round(mem_mb, 1), active_requests),
        )
        conn.commit()
    finally:
        conn.close()


def get_monitoring_history(hours: int = 24) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT timestamp, cpu_pct, mem_mb, active_requests
            FROM monitoring_log
            WHERE timestamp >= datetime('now', ?)
            ORDER BY timestamp ASC
            """,
            (f"-{hours} hours",),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def prune_monitoring_log(keep_hours: int = 48) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "DELETE FROM monitoring_log WHERE timestamp < datetime('now', ?)",
            (f"-{keep_hours} hours",),
        )
        conn.commit()
    finally:
        conn.close()


# ── Visitors ──────────────────────────────────────────────────────────────────

def insert_visitor(ip: str, path: str, user_agent: Optional[str]) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO visitors (ip, path, user_agent) VALUES (?, ?, ?)",
            (ip, path, user_agent),
        )
        conn.commit()
    finally:
        conn.close()


def get_visitor_stats() -> dict:
    conn = get_connection()
    try:
        today_ips = conn.execute(
            "SELECT COUNT(DISTINCT ip) FROM visitors WHERE DATE(timestamp)=DATE('now')"
        ).fetchone()[0]
        today_total = conn.execute(
            "SELECT COUNT(*) FROM visitors WHERE DATE(timestamp)=DATE('now')"
        ).fetchone()[0]
        all_ips = conn.execute("SELECT COUNT(DISTINCT ip) FROM visitors").fetchone()[0]
        all_total = conn.execute("SELECT COUNT(*) FROM visitors").fetchone()[0]
        return {
            "unique_ips_today": today_ips,
            "total_requests_today": today_total,
            "unique_ips_total": all_ips,
            "total_requests_total": all_total,
        }
    finally:
        conn.close()


# ── Export ────────────────────────────────────────────────────────────────────

def export_classifications_csv_rows() -> list[dict]:
    """Return all classifications LEFT JOINed with their feedback as flat dicts."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT
                c.id               AS classification_id,
                c.created_at,
                c.ticket_title,
                c.ticket_text,
                c.model_id,
                c.model_response,
                c.api_time_ms,
                c.inference_time_ms,
                f.thumbs           AS feedback_thumbs,
                f.comment          AS feedback_comment,
                f.email            AS feedback_email
            FROM classifications c
            LEFT JOIN feedback f ON f.classification_id = c.id
            ORDER BY c.id ASC
            """
        ).fetchall()

        result = []
        for row in rows:
            d = dict(row)
            try:
                resp = json.loads(d.pop("model_response", "{}"))
                d["l1_label"] = resp.get("l1", {}).get("label", "")
                d["l1_confidence"] = resp.get("l1", {}).get("confidence", "")
                d["l2_label"] = resp.get("l2", {}).get("label", "")
                d["l2_confidence"] = resp.get("l2", {}).get("confidence", "")
                d["l3_label"] = resp.get("l3", {}).get("label", "")
                d["l3_confidence"] = resp.get("l3", {}).get("confidence", "")
                d["priority"] = resp.get("priority", {}).get("label", "")
                d["sentiment"] = resp.get("sentiment", {}).get("label", "")
            except Exception:
                pass
            result.append(d)
        return result
    finally:
        conn.close()


def export_sql_dump() -> str:
    """Return full DB as SQL dump string using iterdump()."""
    conn = get_connection()
    try:
        return "\n".join(conn.iterdump())
    finally:
        conn.close()
