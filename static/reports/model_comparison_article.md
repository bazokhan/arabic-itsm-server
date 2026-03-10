# Comparative Evaluation of Arabic ITSM Classifiers

## Abstract
This report compares two Arabic ITSM classification checkpoints on a fixed labeled split (1433 tickets), evaluating hierarchical classification quality and inference latency. The analysis follows a fixed offline protocol and includes paired statistical tests.

## Sources
- Dataset (Hugging Face): https://huggingface.co/datasets/albaz2000/arabic-itsm-dataset
- Evaluated dataset reference: https://huggingface.co/datasets/albaz2000/arabic-itsm-dataset
- Model A page: https://huggingface.co/marbert-l1l2l3
- Model B page: https://huggingface.co/arabert-l1l2l3

## Model Mapping
- Model A (`A`): `marbert-l1l2l3`
- Model B (`B`): `arabert-l1l2l3`

## Experimental Setup
- Evaluated split file: `data/processed/test.csv`
- Split label: `test`
- Rows evaluated: **1433**
- Input text: `title_ar + description_ar`
- Metrics: Accuracy, Macro-F1, Weighted-F1, top-3 hit rate, latency mean/p50/p95
- Statistical checks: McNemar test and paired bootstrap CI for accuracy deltas

## Core Results
- **L1**: marbert-l1l2l3 macro-F1=0.8838, acc=0.8870; arabert-l1l2l3 macro-F1=0.8883, acc=0.8918.
- **L2**: marbert-l1l2l3 macro-F1=0.8705, acc=0.8730; arabert-l1l2l3 macro-F1=0.8733, acc=0.8751.
- **L3**: marbert-l1l2l3 macro-F1=0.7752, acc=0.7823; arabert-l1l2l3 macro-F1=0.8006, acc=0.8067.

## Statistical Significance (Accuracy, paired tests)
- Decision rule: p < 0.05 is treated as statistically significant.
- **L1**: Acc Δ(A-B)=-0.0049, 95% CI=[-0.0126, 0.0021], p=0.265205 (not significant; arabert-l1l2l3 better; CI crosses 0).
- **L2**: Acc Δ(A-B)=-0.0021, 95% CI=[-0.0077, 0.0042], p=0.662521 (not significant; arabert-l1l2l3 better; CI crosses 0).
- **L3**: Acc Δ(A-B)=-0.0244, 95% CI=[-0.0377, -0.0112], p=0.000422 (significant; arabert-l1l2l3 better; CI excludes 0).

## Notes on Interpretation
- Macro-F1 is emphasized for class-imbalance robustness.
- Latency values are end-to-end model forward times from inference payloads.
- Confusion analysis identifies systematically mixed label pairs for targeted data curation.

## Reproducibility Record
- Generated at (UTC): `2026-03-10T01:15:08.790493+00:00`
- Bootstrap samples: `1000`
- Random seed: `42`

## Artifacts
- `model_comparison_raw_predictions.csv` (ticket-level outputs)
- `model_comparison_report.json` (all metrics, tests, chart-ready data)
- `model_comparison_article.md` (narrative report text)
