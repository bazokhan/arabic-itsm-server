#!/usr/bin/env python
"""
Run a reproducible comparison between two local Arabic ITSM checkpoints.

Outputs:
- static/reports/model_comparison_raw_predictions.csv
- static/reports/model_comparison_report.json
- static/reports/model_comparison_article.md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.classifier import CheckpointClassifier


TASK_TO_GT_COL = {
    "l1": "category_level_1",
    "l2": "category_level_2",
    "l3": "category_level_3",
    "priority": "priority",
    "sentiment": "sentiment",
}

DEFAULT_MODEL_A_URL = ""
DEFAULT_MODEL_B_URL = ""
DEFAULT_DATASET_HF_URL = "https://huggingface.co/datasets/albaz2000/arabic-itsm-dataset"
DEFAULT_MODEL_A_ID = "marbert-arabic-itsm-l3-categories"
DEFAULT_MODEL_B_ID = "arabert-arabic-itsm-l3-categories"


@dataclass
class SampleRecord:
    ticket_id: str
    title_ar: str
    description_ar: str
    ground_truth: dict[str, str]


def _public_model_path(path: str) -> str:
    p = str(path or "").replace("\\", "/")
    if not p:
        return ""
    # Hide machine-specific absolute paths in published artifacts.
    if len(p) >= 2 and p[1] == ":":
        return f"models/{Path(p).name}"
    if p.startswith("/"):
        return f"models/{Path(p).name}"
    return p


def _is_http_url(value: str) -> bool:
    v = str(value or "").strip().lower()
    return v.startswith("http://") or v.startswith("https://")


def _normalize_model_url(model_id: str, model_url: str) -> str:
    explicit = str(model_url or "").strip()
    if explicit:
        return explicit

    candidate = str(model_id or "").strip()
    if not candidate:
        return ""
    if _is_http_url(candidate):
        return candidate
    # Only infer a Hub URL for clear repo-like IDs (org/repo or repo).
    if "\\" in candidate or candidate.startswith("."):
        return ""
    if "/" in candidate:
        return f"https://huggingface.co/{candidate}"
    if " " in candidate:
        return ""
    return f"https://huggingface.co/{candidate}"


def _display_or_na(value: Any) -> str:
    v = str(value or "").strip()
    return v if v else "n/a"


def _read_dataset(csv_path: Path, limit: int | None) -> list[SampleRecord]:
    rows: list[SampleRecord] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            gt: dict[str, str] = {}
            for task, col in TASK_TO_GT_COL.items():
                gt[task] = str((row.get(col) or "")).strip()
            rows.append(
                SampleRecord(
                    ticket_id=str(row.get("ticket_id") or f"row_{i}").strip(),
                    title_ar=str(row.get("title_ar") or "").strip(),
                    description_ar=str(row.get("description_ar") or "").strip(),
                    ground_truth=gt,
                )
            )
    return rows


def _download_http_csv(url: str, dest_file: Path):
    with urlopen(url, timeout=120) as resp:
        body = resp.read()
    text_head = body[:2048].decode("utf-8", errors="ignore")
    if "ticket_id" not in text_head or "," not in text_head:
        raise RuntimeError(
            "Downloaded content does not look like the expected CSV. "
            "Use a direct CSV URL or provide --dataset-csv."
        )
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    dest_file.write_bytes(body)


def _resolve_dataset_csv(
    dataset_csv_arg: str,
    dataset_url: str | None,
    output_dir: Path,
) -> Path:
    if dataset_csv_arg:
        p = Path(dataset_csv_arg)
        if p.exists():
            return p
    if not dataset_url:
        missing = dataset_csv_arg or "(empty path)"
        raise FileNotFoundError(
            f"Dataset file not found: {missing}. "
            "Provide a valid --dataset-csv or --dataset-url."
        )
    cache_dir = output_dir / "_cache"
    dest_file = cache_dir / "downloaded_test.csv"
    _download_http_csv(dataset_url, dest_file)
    return dest_file


def _safe_mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 3) if values else 0.0


def _safe_quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int((len(ordered) - 1) * q)
    return round(float(ordered[idx]), 3)


def _percent(x: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round((x / total) * 100.0, 3)


def _bootstrap_ci(
    y_true: list[str],
    y_pred: list[str],
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if not y_true:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(y_true)
    vals: list[float] = []
    for _ in range(samples):
        idxs = [rng.randrange(n) for _ in range(n)]
        t = [y_true[i] for i in idxs]
        p = [y_pred[i] for i in idxs]
        vals.append(accuracy_score(t, p))
    vals.sort()
    low = vals[int(0.025 * (samples - 1))]
    high = vals[int(0.975 * (samples - 1))]
    return (round(low, 4), round(high, 4))


def _paired_bootstrap_ci_diff(
    y_true: list[str],
    pred_a: list[str],
    pred_b: list[str],
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if not y_true:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(y_true)
    vals: list[float] = []
    for _ in range(samples):
        idxs = [rng.randrange(n) for _ in range(n)]
        t = [y_true[i] for i in idxs]
        a = [pred_a[i] for i in idxs]
        b = [pred_b[i] for i in idxs]
        vals.append(accuracy_score(t, a) - accuracy_score(t, b))
    vals.sort()
    low = vals[int(0.025 * (samples - 1))]
    high = vals[int(0.975 * (samples - 1))]
    return (round(low, 4), round(high, 4))


def _class_breakdown(
    y_true: list[str],
    y_pred: list[str],
) -> list[dict[str, Any]]:
    labels = sorted(set(y_true) | set(y_pred))
    result: list[dict[str, Any]] = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        result.append(
            {
                "label": label,
                "support": support,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        )
    result.sort(key=lambda x: (-x["support"], x["label"]))
    return result


def _top_confusions(
    y_true: list[str],
    y_pred: list[str],
    k: int = 12,
) -> list[dict[str, Any]]:
    counter: Counter[tuple[str, str]] = Counter()
    for t, p in zip(y_true, y_pred):
        if t != p:
            counter[(t, p)] += 1
    top = counter.most_common(k)
    return [{"true": t, "pred": p, "count": c} for (t, p), c in top]


def _mcnemar_p_value(n01: int, n10: int) -> float:
    # Continuity-corrected McNemar with chi-square df=1.
    # For df=1, survival function is erfc(sqrt(x / 2)).
    denom = n01 + n10
    if denom == 0:
        return 1.0
    chi2 = ((abs(n01 - n10) - 1) ** 2) / denom
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return round(float(p), 6)


def _evaluate_task(
    task: str,
    gt: list[str],
    pred: list[str],
    latencies: list[float],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    acc = accuracy_score(gt, pred) if gt else 0.0
    macro = f1_score(gt, pred, average="macro", zero_division=0) if gt else 0.0
    weighted = f1_score(gt, pred, average="weighted", zero_division=0) if gt else 0.0
    ci_low, ci_high = _bootstrap_ci(gt, pred, bootstrap_samples, seed)
    by_class = _class_breakdown(gt, pred)
    return {
        "task": task,
        "n": len(gt),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro), 4),
        "weighted_f1": round(float(weighted), 4),
        "accuracy_ci95": [ci_low, ci_high],
        "top_confusions": _top_confusions(gt, pred),
        "by_class": by_class,
        "latency_ms": {
            "mean": _safe_mean(latencies),
            "p50": _safe_quantile(latencies, 0.50),
            "p95": _safe_quantile(latencies, 0.95),
            "min": round(min(latencies), 3) if latencies else 0.0,
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
    }


def _build_article_markdown(report: dict[str, Any]) -> str:
    cfg = report["config"]
    model_a = report["models"]["a"]
    model_b = report["models"]["b"]
    shared_tasks = report["shared_tasks"]
    ts = report["generated_at_utc"]
    dataset_n = report["dataset"]["rows_evaluated"]
    refs = report.get("references", {})
    paired_tests = report.get("paired_tests", {})

    def metric_line(task: str) -> str:
        a = report["metrics"]["a"]["tasks"].get(task, {})
        b = report["metrics"]["b"]["tasks"].get(task, {})
        if not a or not b:
            return f"- **{task.upper()}**: task not available in both models."
        return (
            f"- **{task.upper()}**: "
            f"{model_a['id']} macro-F1={a['macro_f1']:.4f}, acc={a['accuracy']:.4f}; "
            f"{model_b['id']} macro-F1={b['macro_f1']:.4f}, acc={b['accuracy']:.4f}."
        )

    def paired_line(task: str) -> str:
        p = paired_tests.get(task, {})
        if not p:
            return f"- **{task.upper()}**: paired significance test unavailable."
        diff = float(p.get("accuracy_diff_a_minus_b", 0.0))
        ci = p.get("accuracy_diff_ci95", [0.0, 0.0])
        ci_low = float(ci[0]) if len(ci) > 0 else 0.0
        ci_high = float(ci[1]) if len(ci) > 1 else 0.0
        pval = float(p.get("mcnemar_p_value", 1.0))
        significant = pval < 0.05
        direction = (
            f"{model_a['id']} better"
            if diff > 0
            else (f"{model_b['id']} better" if diff < 0 else "no difference in point estimate")
        )
        ci_note = "CI excludes 0" if (ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0) else "CI crosses 0"
        sig_label = "significant" if significant else "not significant"
        return (
            f"- **{task.upper()}**: Acc Δ(A-B)={diff:.4f}, 95% CI=[{ci_low:.4f}, {ci_high:.4f}], "
            f"p={pval:.6f} ({sig_label}; {direction}; {ci_note})."
        )

    lines = [
        "# Comparative Evaluation of Arabic ITSM Classifiers",
        "",
        "## Abstract",
        (
            f"This report compares two Arabic ITSM classification checkpoints on a fixed labeled split "
            f"({dataset_n} tickets), evaluating hierarchical classification quality and inference latency. "
            "The analysis follows a fixed offline protocol and includes paired statistical tests."
        ),
        "",
        "## Sources",
        f"- Dataset (Hugging Face): {refs.get('dataset_hf_url', 'n/a')}",
        f"- Evaluated dataset reference: {refs.get('dataset_hf_url', 'n/a')}",
        f"- Model A page: {_display_or_na(model_a.get('url'))}",
        f"- Model B page: {_display_or_na(model_b.get('url'))}",
        "",
        "## Model Mapping",
        f"- Model A (`A`): `{model_a.get('id', 'n/a')}`",
        f"- Model B (`B`): `{model_b.get('id', 'n/a')}`",
        "",
        "## Experimental Setup",
        f"- Evaluated split file: `{cfg['dataset_csv']}`",
        f"- Split label: `{cfg['split_name']}`",
        f"- Rows evaluated: **{dataset_n}**",
        "- Input text: `title_ar + description_ar`",
        "- Metrics: Accuracy, Macro-F1, Weighted-F1, top-3 hit rate, latency mean/p50/p95",
        "- Statistical checks: McNemar test and paired bootstrap CI for accuracy deltas",
        "",
        "## Core Results",
        metric_line("l1"),
        metric_line("l2"),
        metric_line("l3"),
    ]
    if "priority" in shared_tasks:
        lines.append(metric_line("priority"))
    if "sentiment" in shared_tasks:
        lines.append(metric_line("sentiment"))
    lines.extend(
        [
            "",
            "## Statistical Significance (Accuracy, paired tests)",
            "- Decision rule: p < 0.05 is treated as statistically significant.",
            paired_line("l1"),
            paired_line("l2"),
            paired_line("l3"),
            "",
            "## Notes on Interpretation",
            "- Macro-F1 is emphasized for class-imbalance robustness.",
            "- Latency values are end-to-end model forward times from inference payloads.",
            "- Confusion analysis identifies systematically mixed label pairs for targeted data curation.",
            "",
            "## Reproducibility Record",
            f"- Generated at (UTC): `{ts}`",
            f"- Bootstrap samples: `{cfg['bootstrap_samples']}`",
            f"- Random seed: `{cfg['seed']}`",
            "",
            "## Artifacts",
            "- `model_comparison_raw_predictions.csv` (ticket-level outputs)",
            "- `model_comparison_report.json` (all metrics, tests, chart-ready data)",
            "- `model_comparison_article.md` (narrative report text)",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run two-model comparison and publish report artifacts.")
    parser.add_argument(
        "--dataset-csv",
        default=os.getenv("COMPARISON_DATASET_CSV", "data/processed/test.csv"),
        help="Path to processed split CSV.",
    )
    parser.add_argument(
        "--dataset-url",
        default=os.getenv("COMPARISON_DATASET_URL", ""),
        help="Optional URL to download test CSV when local path is unavailable.",
    )
    parser.add_argument(
        "--model-a-path",
        default=os.getenv("COMPARISON_MODEL_A_PATH", "models/marbert_l1_l2_l3_best"),
    )
    parser.add_argument(
        "--model-b-path",
        default=os.getenv("COMPARISON_MODEL_B_PATH", "models/marbert_multi_task_best"),
    )
    parser.add_argument(
        "--model-a-id",
        default=os.getenv("COMPARISON_MODEL_A_ID", DEFAULT_MODEL_A_ID),
    )
    parser.add_argument(
        "--model-b-id",
        default=os.getenv("COMPARISON_MODEL_B_ID", DEFAULT_MODEL_B_ID),
    )
    parser.add_argument(
        "--model-a-url",
        default=os.getenv("COMPARISON_MODEL_A_URL", DEFAULT_MODEL_A_URL),
    )
    parser.add_argument(
        "--model-b-url",
        default=os.getenv("COMPARISON_MODEL_B_URL", DEFAULT_MODEL_B_URL),
    )
    parser.add_argument(
        "--dataset-hf-url",
        default=os.getenv("COMPARISON_DATASET_HF_URL", DEFAULT_DATASET_HF_URL),
    )
    parser.add_argument(
        "--split-name",
        default="test",
    )
    parser.add_argument(
        "--output-dir",
        default="static/reports",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row cap for quick debugging runs.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device override (cpu/cuda).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_csv = _resolve_dataset_csv(args.dataset_csv, args.dataset_url or None, output_dir)

    if not args.model_a_path or not args.model_b_path:
        raise RuntimeError(
            "Model paths are required. Provide --model-a-path and --model-b-path "
            "or set COMPARISON_MODEL_A_PATH / COMPARISON_MODEL_B_PATH."
        )

    print(f"[info] Loading dataset from: {dataset_csv}")
    rows = _read_dataset(dataset_csv, args.limit)
    if not rows:
        raise RuntimeError("No rows were loaded from the dataset.")
    print(f"[info] Loaded rows: {len(rows)}")

    print(f"[info] Loading model A: {args.model_a_id}")
    clf_a = CheckpointClassifier(args.model_a_path, device=args.device)
    print(f"[info] Loading model B: {args.model_b_id}")
    clf_b = CheckpointClassifier(args.model_b_path, device=args.device)

    tasks_a = set(clf_a.tasks)
    tasks_b = set(clf_b.tasks)
    shared_tasks = sorted(tasks_a & tasks_b & set(TASK_TO_GT_COL.keys()))

    print(f"[info] Model A tasks: {sorted(tasks_a)}")
    print(f"[info] Model B tasks: {sorted(tasks_b)}")
    print(f"[info] Shared tasks for paired evaluation: {shared_tasks}")

    raw_rows: list[dict[str, Any]] = []

    gt_by_model_task: dict[str, dict[str, list[str]]] = {
        "a": defaultdict(list),
        "b": defaultdict(list),
    }
    pred_by_model_task: dict[str, dict[str, list[str]]] = {
        "a": defaultdict(list),
        "b": defaultdict(list),
    }
    top3_hit_by_model_task: dict[str, dict[str, int]] = {
        "a": defaultdict(int),
        "b": defaultdict(int),
    }
    n_by_model_task: dict[str, dict[str, int]] = {
        "a": defaultdict(int),
        "b": defaultdict(int),
    }
    latency_by_model: dict[str, list[float]] = {"a": [], "b": []}
    latency_by_model_task: dict[str, dict[str, list[float]]] = {
        "a": defaultdict(list),
        "b": defaultdict(list),
    }

    print("[info] Running inference...")
    for idx, row in enumerate(rows, start=1):
        if idx % 100 == 0 or idx == len(rows):
            print(f"  -> processed {idx}/{len(rows)}")

        pa = clf_a.predict(row.title_ar, row.description_ar)
        pb = clf_b.predict(row.title_ar, row.description_ar)

        la = float(pa.get("latency_ms") or 0.0)
        lb = float(pb.get("latency_ms") or 0.0)
        latency_by_model["a"].append(la)
        latency_by_model["b"].append(lb)

        rec: dict[str, Any] = {
            "ticket_id": row.ticket_id,
            "title_ar": row.title_ar,
            "description_ar": row.description_ar,
            "latency_a_ms": round(la, 3),
            "latency_b_ms": round(lb, 3),
        }
        for task, gt_label in row.ground_truth.items():
            rec[f"gt_{task}"] = gt_label

            for m_key, pred_payload, model_tasks in (
                ("a", pa, tasks_a),
                ("b", pb, tasks_b),
            ):
                if task not in model_tasks:
                    rec[f"pred_{m_key}_{task}"] = ""
                    rec[f"conf_{m_key}_{task}"] = ""
                    rec[f"top3_{m_key}_{task}"] = ""
                    continue

                payload = pred_payload.get(task) or {}
                pred_label = str(payload.get("label") or "").strip()
                conf = payload.get("confidence")
                top3 = payload.get("top3") or []
                top3_labels = [str(x.get("label") or "") for x in top3]

                rec[f"pred_{m_key}_{task}"] = pred_label
                rec[f"conf_{m_key}_{task}"] = conf if conf is not None else ""
                rec[f"top3_{m_key}_{task}"] = "|".join(top3_labels)

                if gt_label:
                    gt_by_model_task[m_key][task].append(gt_label)
                    pred_by_model_task[m_key][task].append(pred_label)
                    n_by_model_task[m_key][task] += 1
                    if gt_label in top3_labels:
                        top3_hit_by_model_task[m_key][task] += 1
                    latency_by_model_task[m_key][task].append(
                        la if m_key == "a" else lb
                    )

        raw_rows.append(rec)

    raw_csv_path = output_dir / "model_comparison_raw_predictions.csv"
    with raw_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)
    print(f"[ok] Wrote raw predictions: {raw_csv_path}")

    metrics: dict[str, Any] = {"a": {"tasks": {}}, "b": {"tasks": {}}}
    for m_key in ("a", "b"):
        for task in sorted(gt_by_model_task[m_key].keys()):
            task_metrics = _evaluate_task(
                task=task,
                gt=gt_by_model_task[m_key][task],
                pred=pred_by_model_task[m_key][task],
                latencies=latency_by_model_task[m_key][task],
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed + (1 if m_key == "a" else 7),
            )
            task_metrics["top3_hit_rate"] = round(
                _percent(
                    top3_hit_by_model_task[m_key][task],
                    n_by_model_task[m_key][task],
                )
                / 100.0,
                4,
            )
            metrics[m_key]["tasks"][task] = task_metrics

        metrics[m_key]["overall_latency_ms"] = {
            "mean": _safe_mean(latency_by_model[m_key]),
            "p50": _safe_quantile(latency_by_model[m_key], 0.50),
            "p95": _safe_quantile(latency_by_model[m_key], 0.95),
            "min": round(min(latency_by_model[m_key]), 3) if latency_by_model[m_key] else 0.0,
            "max": round(max(latency_by_model[m_key]), 3) if latency_by_model[m_key] else 0.0,
        }

    paired_tests: dict[str, Any] = {}
    chart_rows: list[dict[str, Any]] = []
    for task in shared_tasks:
        gt = gt_by_model_task["a"][task]
        pred_a = pred_by_model_task["a"][task]
        pred_b = pred_by_model_task["b"][task]

        n01 = sum(
            1
            for t, a, b in zip(gt, pred_a, pred_b)
            if (a == t) and (b != t)
        )
        n10 = sum(
            1
            for t, a, b in zip(gt, pred_a, pred_b)
            if (a != t) and (b == t)
        )
        acc_a = accuracy_score(gt, pred_a)
        acc_b = accuracy_score(gt, pred_b)
        ci_low, ci_high = _paired_bootstrap_ci_diff(
            gt, pred_a, pred_b, args.bootstrap_samples, args.seed + 11
        )
        paired_tests[task] = {
            "n": len(gt),
            "accuracy_a": round(float(acc_a), 4),
            "accuracy_b": round(float(acc_b), 4),
            "accuracy_diff_a_minus_b": round(float(acc_a - acc_b), 4),
            "accuracy_diff_ci95": [ci_low, ci_high],
            "mcnemar_n01_a_correct_b_wrong": int(n01),
            "mcnemar_n10_a_wrong_b_correct": int(n10),
            "mcnemar_p_value": _mcnemar_p_value(n01, n10),
        }

        chart_rows.append(
            {
                "task": task,
                "macro_f1_a": metrics["a"]["tasks"].get(task, {}).get("macro_f1", 0.0),
                "macro_f1_b": metrics["b"]["tasks"].get(task, {}).get("macro_f1", 0.0),
                "accuracy_a": round(float(acc_a), 4),
                "accuracy_b": round(float(acc_b), 4),
                "latency_mean_a_ms": metrics["a"]["tasks"].get(task, {}).get("latency_ms", {}).get("mean", 0.0),
                "latency_mean_b_ms": metrics["b"]["tasks"].get(task, {}).get("latency_ms", {}).get("mean", 0.0),
            }
        )

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "dataset_csv": str(dataset_csv).replace("\\", "/"),
            "dataset_url": args.dataset_url or None,
            "split_name": args.split_name,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "limit": args.limit,
            "device": args.device or "auto",
        },
        "dataset": {
            "rows_evaluated": len(rows),
            "tasks_from_ground_truth": sorted(TASK_TO_GT_COL.keys()),
        },
        "models": {
            "a": {
                "id": args.model_a_id,
                "path": _public_model_path(args.model_a_path),
                "url": _normalize_model_url(args.model_a_id, args.model_a_url),
                "tasks": sorted(tasks_a),
            },
            "b": {
                "id": args.model_b_id,
                "path": _public_model_path(args.model_b_path),
                "url": _normalize_model_url(args.model_b_id, args.model_b_url),
                "tasks": sorted(tasks_b),
            },
        },
        "references": {
            "dataset_hf_url": args.dataset_hf_url,
        },
        "shared_tasks": shared_tasks,
        "metrics": metrics,
        "paired_tests": paired_tests,
        "charts": {
            "task_rows": chart_rows,
            "overall_latency_ms": {
                "a": metrics["a"]["overall_latency_ms"],
                "b": metrics["b"]["overall_latency_ms"],
            },
        },
        "artifacts": {
            "raw_predictions_csv": "model_comparison_raw_predictions.csv",
            "report_json": "model_comparison_report.json",
            "article_markdown": "model_comparison_article.md",
        },
    }

    report_path = output_dir / "model_comparison_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[ok] Wrote report: {report_path}")

    article_path = output_dir / "model_comparison_article.md"
    article_md = _build_article_markdown(report)
    article_path.write_text(article_md, encoding="utf-8")
    print(f"[ok] Wrote article: {article_path}")
    print("[done] Model comparison pipeline completed successfully.")


if __name__ == "__main__":
    main()
