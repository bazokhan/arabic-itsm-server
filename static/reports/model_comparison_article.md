# Comparative Evaluation of Arabic ITSM Classifiers

## Abstract
This report compares two MARBERT-based Arabic ITSM checkpoints on a fixed labeled split (1433 tickets), evaluating hierarchical classification quality and inference latency. The analysis follows a fixed offline protocol and includes paired statistical tests.

## Sources
- Dataset (Hugging Face): https://huggingface.co/datasets/albaz2000/arabic-itsm-dataset
- Evaluated dataset reference: https://huggingface.co/datasets/albaz2000/arabic-itsm-dataset
- Model A page: https://huggingface.co/albaz2000/marbert-arabic-itsm-l3-categories
- Model B page: https://huggingface.co/albaz2000/marbert-arabic-itsm-multitask

## Experimental Setup
- Evaluated split file: `data/processed/test.csv`
- Split label: `test`
- Rows evaluated: **1433**
- Input text: `title_ar + description_ar`
- Metrics: Accuracy, Macro-F1, Weighted-F1, top-3 hit rate, latency mean/p50/p95
- Statistical checks: McNemar test and paired bootstrap CI for accuracy deltas

## Core Results
- **L1**: marbert-arabic-itsm-l3-categories macro-F1=0.8838, acc=0.8870; marbert-arabic-itsm-multitask macro-F1=0.8819, acc=0.8870.
- **L2**: marbert-arabic-itsm-l3-categories macro-F1=0.8705, acc=0.8730; marbert-arabic-itsm-multitask macro-F1=0.8748, acc=0.8765.
- **L3**: marbert-arabic-itsm-l3-categories macro-F1=0.7752, acc=0.7823; marbert-arabic-itsm-multitask macro-F1=0.7451, acc=0.7530.

## Notes on Interpretation
- Macro-F1 is emphasized for class-imbalance robustness.
- Latency values are end-to-end model forward times from inference payloads.
- Confusion analysis identifies systematically mixed label pairs for targeted data curation.

## Reproducibility Record
- Generated at (UTC): `2026-03-06T13:04:29.143324+00:00`
- Bootstrap samples: `1000`
- Random seed: `42`

## Artifacts
- `model_comparison_raw_predictions.csv` (ticket-level outputs)
- `model_comparison_report.json` (all metrics, tests, chart-ready data)
- `model_comparison_article.md` (narrative report text)
