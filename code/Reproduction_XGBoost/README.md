# Reproduction_XGBoost

This folder is a separate reproduction workspace for the previous author's
reported XGBoost experiment. Its structure intentionally mirrors
`Submission_XGBoost`, but the defaults are tuned for experiment reproduction
rather than the final submission run.

## What This Reproduces

- PDF-described feature families as the default path.
- A separate old-code feature path for the current executable previous project.
- File-order 8:2 train/validation split by default.
- One XGBoost multiclass model per label.
- Report-style classification reports and confusion matrices for train and
  validation sets.

The code does not copy the previous report text or claim the previous author's
leaderboard results. It is only an experimental reproduction pipeline. Missing
or ambiguous details are tracked in `REPRODUCTION_GAPS.md`.

## Feature Sets

`src/train.py` supports two feature paths:

- `pdf_report`: default. Uses only the feature groups explicitly described in
  the PDF and flattens a derived-feature window built from the 100 raw ticks.
  `--pdf-levels` controls the unified `i` range for all i-based features,
  including `weighted_ab`.
  - `--pdf-levels 1-5`: 163 base features and 16,300 final features.
  - `--pdf-levels 1-3`: 111 base features and 11,100 final features.
- `previous_code`: uses the richer old-code/current-submission feature builder.
  Current shape: 182 base features and 922 final features.

## Parameter Presets

`src/train.py` supports two reproduction presets:

- `report_pdf`: default. Approximates the PDF description by setting only
  `max_depth=6`; other XGBoost hyperparameters use the library defaults unless
  explicitly overridden on the command line.
- `previous_code`: approximates the current code in
  `Previous version of project/train.py`, including stronger regularization and
  class weighting.

You can override class weights with `--class-weight-mode on` or
`--class-weight-mode off`.

## Storage Modes

`src/train.py` supports two storage modes:

- `--storage-mode buffer`: default. Writes intermediate XGBoost `.buffer`
  batches before constructing `ExtMemQuantileDMatrix`.
- `--storage-mode stream`: constructs `ExtMemQuantileDMatrix` directly from CSV
  feature batches and only saves small label/price sidecars for evaluation.

Both modes use the same `iter_feature_batches()` path for feature construction,
so the feature vectors, row filtering, labels, and price-diff evaluation inputs
remain aligned. The stream mode is intended for high-dimensional PDF-style runs
where the intermediate `.buffer` files would dominate disk usage.

To verify this invariant after feature-engineering changes:

```bash
python src/check_storage_consistency.py \
  --feature-set pdf_report \
  --pdf-levels 1-3 \
  --label label_5 \
  --max-files 3
```

## Setup

```bash
cd /Users/cgt/Desktop/Stock_Movement_Prediction/Reproduction_XGBoost
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke Run

```bash
python src/train.py \
  --feature-set pdf_report \
  --pdf-levels 1-5 \
  --labels label_5 \
  --max-files 80 \
  --num-boost-round 20 \
  --early-stopping-rounds 5
```

## Full Reproduction Run

```bash
python src/train.py \
  --feature-set pdf_report \
  --pdf-levels 1-5 \
  --storage-mode stream \
  --split-mode index \
  --test-size 0.2 \
  --param-preset report_pdf \
  --num-boost-round 2000 \
  --early-stopping-rounds 100 \
  --batch-size 5000
```

On the training server, use the queue script to run the full stream-mode matrix
sequentially:

```bash
bash scripts/run_reproduction_queue.sh
```

By default it runs `1-3` for all labels, then `1-5` for all labels. The server
queue defaults to GPU-oriented runtime overrides while leaving `max_bin` at the
XGBoost default:

```bash
DEVICE=cuda TREE_METHOD=hist NUM_BOOST_ROUND=500 EARLY_STOPPING_ROUNDS=100
BATCH_SIZE=1000 GPU_BATCH_MODE=cupy
CACHE_HOST_RATIO=0.8 MAX_QUANTILE_BATCHES=16 MIN_CACHE_PAGE_BYTES=0
```

`BATCH_SIZE` and the external-memory controls are resource-management settings;
they do not change feature construction. `GPU_BATCH_MODE=cupy` is required for
GPU `ExtMemQuantileDMatrix` in the current XGBoost server environment.
`MAX_BIN` is intentionally unset by default in this script. Set
`DEVICE=cpu TREE_METHOD= NUM_BOOST_ROUND=2000` if a strict CPU-default runtime is
needed for comparison.

For a CPU-first run, use the dedicated CPU queue:

```bash
bash scripts/run_reproduction_cpu_queue.sh
```

Its defaults are:

```bash
DEVICE=cpu TREE_METHOD= MAX_BIN= NTHREAD=-1
BATCH_SIZE=5000 NUM_BOOST_ROUND=2000 EARLY_STOPPING_ROUNDS=100
STORAGE_MODE=stream CLEANUP_CACHE=1
```

This leaves `tree_method` and `max_bin` unset so XGBoost uses its CPU defaults.
Set `TREE_METHOD=hist` explicitly only when you want to force the fast CPU
histogram updater.

To match the current old code directory instead of the PDF description:

```bash
python src/train.py \
  --feature-set previous_code \
  --split-mode index \
  --test-size 0.2 \
  --param-preset previous_code \
  --num-boost-round 2000 \
  --early-stopping-rounds 100 \
  --batch-size 5000
```

## Outputs

Default outputs go to `artifacts/report_pdf_reproduction/`:

- `models/model_label_*.json`
- `feature_spec.json`
- `thresholds.json`
- `train_summary.json`
- `metrics/*_thresholds.json`
- `metrics/*_ratio_grid.json`
- `metrics/*_ratio_best.json`
- `metrics/*_classification.json`
- `metrics/*_legacy_report.txt`

The `*_legacy_report.txt` files are the closest local equivalents to the
classification report blocks in `Previous version of project/Final_Report.pdf`.

## Ratio Scan

The PDF's `get_predict(y, ratio)` scales class `0` and class `2` scores before
`argmax`. The original ratio values are not given, so training scans a grid by
default:

```bash
--ratio-min 0.50 --ratio-max 1.00 --ratio-step 0.02
```

Custom values can be supplied:

```bash
--ratio-values 1.0,0.98,0.95,0.92,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5
```

Each label writes the full grid to `metrics/*_ratio_grid.json` and the selected
best pair to `metrics/*_ratio_best.json`.

## Compare With The PDF Numbers

The PDF exposes confusion matrices for `label_5`, `label_10`, `label_20`, and
`label_40`. Those reference matrices are stored in
`reference/final_report_confusion_matrices.json`.

After a full run, compare reproduced metrics with the PDF:

```bash
python src/compare_to_final_report.py \
  --artifacts-dir artifacts/report_pdf_reproduction
```

## Optional Submission-Shape Package

This is not intended as the final competition submission, but the package shape
can still be built and smoke-tested:

```bash
python src/build_submission.py \
  --artifacts-dir artifacts/report_pdf_reproduction \
  --output-dir artifacts/report_pdf_reproduction/submission_package \
  --zip-path artifacts/report_pdf_reproduction/submission_xgboost.zip

python src/smoke_test_submission.py \
  --package-dir artifacts/report_pdf_reproduction/submission_package \
  --data-dir "../EDA/raw data/FBDQA2021A_MMP_Challenge/data"
```
