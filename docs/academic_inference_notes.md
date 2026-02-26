# Academic Inference Notes

## Scope

This note documents inference-layer validity decisions for the Arabic ITSM server, with emphasis on reproducibility and demo protocol.

## 1) Tokenizer Provenance and Validity Fix

Date: 2026-02-26

### Problem

During multi-ticket demos, predictions appeared collapsed (different tickets returning near-identical outputs).  
Root cause was tokenizer selection behavior: a checkpoint path without local tokenizer assets could still trigger a fallback tokenizer load, creating encoder-tokenizer mismatch risk.

### Fix

The server now enforces tokenizer loading with this order:

1. model path that *actually contains tokenizer files* (`tokenizer.json` or `tokenizer_config.json` or `vocab.txt`)
2. fallback to `UBC-NLP/MARBERTv2`

For the two-checkpoint setup (L1+L2 plus L3), the L3 checkpoint tokenizer is preferred when present.

### Why this matters academically

- Preserves consistency between training-time and inference-time tokenization assumptions.
- Reduces silent mismatch sources that can invalidate demo conclusions.
- Improves reproducibility across environments where fallback behavior may differ.

## 2) Multi-Model Demo Protocol

The web layer now supports **one page per model profile**:

- Catalog: `/models`
- Model page: `/models/{model_id}`

Each profile maps to one checkpoint directory containing `heads.pt`.  
Tasks are discovered from head keys (for example, `l1`, `l2`, `l3`) and rendered dynamically per model.

### Academic usage recommendation

- When reporting demo outputs, always include:
  - model profile id,
  - checkpoint path (or commit/tag),
  - timestamp,
  - input ticket text.
- Compare outputs across models from their dedicated pages to avoid cross-model state confusion.

## 3) Reproducibility Checklist

1. Keep fixed model directories under versioned naming (for example `marbert_l2_best_run03`).
2. Log `model_id` with every prediction payload.
3. Avoid replacing checkpoints in-place during experiments; add new folders instead.
4. Archive screenshots or JSON responses for thesis appendix evidence.
