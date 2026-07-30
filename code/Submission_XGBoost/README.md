# Submission_XGBoost

This folder rewrites the previous XGBoost project into a clean training and
submission workflow. The model idea is intentionally kept close to the previous
version: legacy order-book features, pyramid lag vector, one XGBoost multiclass
model per label, and confidence-thresholded up/down signals.

## Setup

Use Python 3.10, matching the platform config.

```bash
cd /Users/cgt/Desktop/Stock_Movement_Prediction/Submission_XGBoost
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

Full training uses the local raw data folder by default:

```bash
python src/train.py
```

Useful smoke command:

```bash
python src/train.py --labels label_60 --max-files 80 --num-boost-round 20 --early-stopping-rounds 5
```

Outputs are written to `artifacts/`:

- `models/model_label_*.json`
- `feature_spec.json`
- `thresholds.json`
- `metrics/*_thresholds.json`
- `train_summary.json`

## Build submission

```bash
python src/build_submission.py
```

This creates a flat package in `artifacts/submission_package/` and a zip at
`artifacts/submission_xgboost.zip`.

## Smoke test

```bash
python src/smoke_test_submission.py
```

The smoke test imports `Predictor.py` from the built package, runs a 32-sample
batch of 100-tick windows, and validates the platform return shape:
`List[List[int]]` with five labels per row.

## Date-Holdout Server Run

Use this when selecting models against a cleaner out-of-time validation split
instead of the old-compatible file-index split. It keeps artifacts and cache
separate from the legacy/profit-weighted runs.

```bash
cd /mnt/stock_data/stock_movement/Submission_XGBoost
WAIT_PID=166814 bash scripts/run_date63_holdout_gpu.sh
```

Defaults:

- version: `full_gpu_date63_holdout`
- split: `--split-mode date --val-start-date 63`
- output: `artifacts/full_gpu_date63_holdout`
- cache: `/mnt/stock_data/stock_movement/cache/full_gpu_date63_holdout`
- package: `artifacts/full_gpu_date63_holdout/submission_xgboost_full_gpu_date63_holdout.zip`
- threshold metric: `official_score_est`

Set `WAIT_PID` when another server training process should finish first.
