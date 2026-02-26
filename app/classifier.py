"""
Inference utilities for Arabic ITSM classifiers.

Supports:
1) single-checkpoint inference (tasks discovered from heads.pt),
2) two-checkpoint ensemble inference (L1+L2 checkpoint + L3 checkpoint).

CheckpointClassifier auto-discovers tasks from heads.pt — this means any checkpoint
(l1-only, l1+l2, l1+l2+l3, or full multi-task with 5 heads) is handled automatically.
"""
from __future__ import annotations

import re
import time
import unicodedata
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# ── Label maps (source: arabic-itsm-classification/data/processed/label_encoders.pkl) ─
LABELS: dict[str, list[str]] = {
    "l1": [
        "Access", "Hardware", "Network", "Security", "Service", "Software",
    ],
    "l2": [
        "Account", "Business App", "Email/Calendar", "Incident",
        "Internet/LAN", "Laptop/Desktop", "MFA/SSO", "Malware/Phishing",
        "Office Apps", "Peripherals", "Permissions", "Policy/Compliance",
        "Printer/Scanner", "Request", "VPN", "WiFi",
    ],
    "l3": [
        "Account Locked", "Admin Access", "Authentication", "Authenticator Issue",
        "Battery", "Blocked Site", "Boot Issue", "Bug", "Connection Failure",
        "Connectivity", "Crash", "Credentials", "DNS", "Data Access",
        "Degradation", "Device Encryption", "Docking Station", "Driver",
        "Feature Request", "Integration", "Intermittent", "Keyboard/Mouse",
        "Latency", "License", "MFA Failure", "Mailbox Access", "Monitor",
        "New Account", "New Device", "No Internet", "Outage", "Outlook Issue",
        "Paper Jam", "Password Reset", "Performance", "Permission Denied",
        "Phishing Email", "Print Failure", "Profile Update", "Role Request",
        "SSO Login Issue", "Slow Speed", "Software Install", "Split Tunnel",
        "Suspicious Link", "Sync Problem", "Virus Alert", "Word/Excel",
    ],
    # sklearn LabelEncoder encodes in alphabetical order
    "priority": ["1", "2", "3", "4", "5"],
    "sentiment": ["mixed", "negative", "neutral", "positive"],
}

# ── Arabic text normalizer (matches arabic-itsm-classification preprocessing) ──────────
_DIACRITICS = re.compile(
    r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06dc\u06df-\u06e4\u06e7\u06e8\u06ea-\u06ed]"
)
_ALEF = str.maketrans("أإآٱ", "اااا")
_SPACES = re.compile(r"\s+")
_HEAD_WEIGHT = re.compile(r"^([^.]+)\.1\.weight$")


def _normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _DIACRITICS.sub("", text).translate(_ALEF)
    text = re.sub(r"[A-Za-z]+", lambda m: m.group().lower(), text)
    return _SPACES.sub(" ", text).strip()


# ── Inference model ───────────────────────────────────────────────────────────────────

def _load_heads_path(model_path: str) -> str:
    """Return local path to heads.pt; download from HF if model_path is remote."""
    local = Path(model_path) / "heads.pt"
    if local.exists():
        return str(local)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=model_path, filename="heads.pt")


class _InferenceModel(nn.Module):
    """MarBERT encoder + linear heads (eval-only)."""

    def __init__(self, encoder_path: str, head_specs: dict[str, int]):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_path)
        hidden = self.encoder.config.hidden_size
        self.heads = nn.ModuleDict(
            {t: nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, n))
             for t, n in head_specs.items()}
        )
        heads_file = _load_heads_path(encoder_path)
        self.heads.load_state_dict(
            torch.load(heads_file, map_location="cpu", weights_only=False)
        )
        self.eval()

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        cls = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0]
        return {task: head(cls) for task, head in self.heads.items()}


def _task_labels(task: str, n_classes: int) -> list[str]:
    labels = LABELS.get(task)
    if labels and len(labels) == n_classes:
        return labels
    return [f"class_{i}" for i in range(n_classes)]


def discover_tasks(model_path: str) -> dict[str, int]:
    """
    Inspect heads.pt and return {task: n_classes}.
    Expected keys are like 'l1.1.weight', 'l2.1.weight', ...
    """
    heads_file = _load_heads_path(model_path)
    state = torch.load(heads_file, map_location="cpu", weights_only=False)
    task_specs: dict[str, int] = {}
    for key, value in state.items():
        m = _HEAD_WEIGHT.match(key)
        if not m:
            continue
        task = m.group(1)
        task_specs[task] = int(value.shape[0])
    if not task_specs:
        raise RuntimeError(f"No task heads found in heads file: {heads_file}")
    return task_specs


def _load_tokenizer(path_candidates: list[str]):
    """
    Load tokenizer from first valid candidate.
    For local paths, require tokenizer assets to avoid silent fallback mismatch.
    """
    def _has_local_tokenizer(path: str) -> bool:
        p = Path(path)
        if not p.is_dir():
            return False
        return any(
            (p / name).exists()
            for name in ("tokenizer.json", "tokenizer_config.json", "vocab.txt")
        )

    for path in path_candidates:
        if Path(path).is_dir() and not _has_local_tokenizer(path):
            continue
        try:
            return AutoTokenizer.from_pretrained(path)
        except Exception:
            continue
    raise RuntimeError("Could not load tokenizer from any configured path.")


def _format_predictions(
    logits: dict[str, torch.Tensor], label_map: dict[str, list[str]], latency_ms: float
) -> dict:
    result: dict = {"latency_ms": latency_ms}
    for task, raw in logits.items():
        probs = F.softmax(raw[0], dim=-1).cpu().tolist()
        top_i = int(torch.argmax(raw[0]))
        labels = label_map[task]
        result[task] = {
            "label": labels[top_i],
            "confidence": round(probs[top_i], 4),
            "top3": [
                {"label": labels[i], "prob": round(probs[i], 4)}
                for i in sorted(range(len(probs)), key=lambda x: -probs[x])[:3]
            ],
        }
    return result


# ── Public API ────────────────────────────────────────────────────────────────────────

class CheckpointClassifier:
    """
    Inference wrapper for a single checkpoint directory or HF model id.
    Tasks are discovered from heads.pt (e.g., l1/l2/l3).
    """

    def __init__(self, model_path: str, device: str | None = None):
        self.model_path = model_path
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        head_specs = discover_tasks(model_path)
        self.tasks = sorted(head_specs.keys())
        self.label_map = {task: _task_labels(task, n) for task, n in head_specs.items()}

        self.tok = _load_tokenizer([model_path, "UBC-NLP/MARBERTv2"])
        self._model = _InferenceModel(model_path, head_specs).to(self.device)

    def predict(self, title: str, description: str) -> dict:
        t0 = time.perf_counter()
        text = _normalize(f"{title} {description}".strip())
        enc = self.tok(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        logits = self._model(ids, mask)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return _format_predictions(logits, self.label_map, latency_ms)


class ITSMClassifier:
    """
    Classifies an Arabic ITSM ticket into L1, L2, and L3 categories.

    Parameters
    ----------
    l12_path : str
        Local directory or HuggingFace model ID for the jointly-trained L1+L2
        checkpoint (marbert_l2_best).  Must contain model.safetensors + heads.pt.
    l3_path : str
        Local directory or HuggingFace model ID for the L3 checkpoint
        (marbert_l3_best).  Must contain model.safetensors + heads.pt.
    device : str | None
        'cuda', 'cpu', or None (auto-detect).
    """

    def __init__(self, l12_path: str, l3_path: str, device: str | None = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Prefer L3 tokenizer. If local path is missing tokenizer files, skip it.
        self.tok = _load_tokenizer([l3_path, l12_path, "UBC-NLP/MARBERTv2"])

        print(f"  Loading L1+L2 model from {l12_path} ...")
        self._l12 = _InferenceModel(
            l12_path, {"l1": len(LABELS["l1"]), "l2": len(LABELS["l2"])}
        ).to(self.device)

        print(f"  Loading L3 model from {l3_path} ...")
        self._l3 = _InferenceModel(
            l3_path, {"l3": len(LABELS["l3"])}
        ).to(self.device)

    def predict(self, title: str, description: str) -> dict:
        """
        Classify a ticket.

        Returns a dict with keys l1, l2, l3 — each containing:
          label      : str   predicted class name
          confidence : float softmax probability of top class
          top3       : list[{label, prob}]  top-3 predictions
        Plus latency_ms: float.
        """
        t0 = time.perf_counter()

        text = _normalize(f"{title} {description}".strip())
        enc = self.tok(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        ids  = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)

        logits = {**self._l12(ids, mask), **self._l3(ids, mask)}

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        label_map = {task: LABELS[task] for task in ("l1", "l2", "l3")}
        return _format_predictions(logits, label_map, latency_ms)
