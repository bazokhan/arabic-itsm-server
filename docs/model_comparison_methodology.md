# Model Comparison Methodology

## Objective

Provide a reproducible comparative evaluation between two Arabic ITSM checkpoints using a fixed labeled split and transparent metrics.

## Models

- Model A: `marbert_l1_l2_l3_best`
- Model B: `marbert_multi_task_best`

Both are loaded via `CheckpointClassifier` and evaluated with the same text preprocessing and tokenization logic used by the server.

## Dataset

- Source: `data/processed/test.csv`
- Input fields: `title_ar`, `description_ar`
- Ground-truth labels:
  - `l1` -> `category_level_1`
  - `l2` -> `category_level_2`
  - `l3` -> `category_level_3`
  - `priority` -> `priority`
  - `sentiment` -> `sentiment`

## Metrics

Per task, per model:

- Accuracy
- Macro-F1
- Weighted-F1
- Top-3 hit rate
- Accuracy 95% CI (bootstrap)
- Latency stats (mean, p50, p95, min, max)
- Class-level breakdown (support, precision, recall, F1)
- Top confusion pairs (`true -> pred`)

Paired model-vs-model checks on shared tasks:

- Accuracy difference (A - B)
- 95% bootstrap CI for accuracy difference
- McNemar significance test (continuity-corrected)

## Reproducible Command

```bash
python scripts/run_model_comparison.py
```

Optional:

```bash
python scripts/run_model_comparison.py --limit 200 --bootstrap-samples 300
```

## Artifact Contract

The script writes all outputs into `static/reports/`:

- `model_comparison_raw_predictions.csv`
- `model_comparison_report.json`
- `model_comparison_article.md`

The `/research` page is artifact-driven and renders these files without manual editing.

## Running a Custom Comparison (e.g., Encoder Ablation)

The script accepts `--model-a-path`, `--model-b-path`, `--model-a-id`, and `--model-b-id` to compare any two checkpoints. For the MARBERTv2 vs AraBERTv2 encoder ablation (EXP-006a vs EXP-007):

```bash
python scripts/run_model_comparison.py \
    --model-a-path models/marbert_l1_l2_l3_best \
    --model-b-path models/arabert_l1_l2_l3_best \
    --model-a-id marbert-l1l2l3 \
    --model-b-id arabert-l1l2l3
```

The script auto-detects which tasks each checkpoint supports (via head key shapes in `heads.pt`) and only computes paired statistics on tasks present in both models.

## Reporting Guidance (Thesis/Article)

When citing results:

1. Report model IDs and checkpoint paths.
2. Report exact split file and row count.
3. Prioritize macro-F1 for imbalanced labels.
4. Include paired test evidence, not only point metrics.
5. Include at least one error-analysis section using top confusion pairs.
