# Stock Movement Prediction Handoff

Generated: 2026-06-08

This handoff package is a compact subset of the working directory for follow-up
development and final-report writing. It intentionally excludes raw data,
virtual environments, training caches, historical zip packages, pyc files, and
large duplicated model directories.

## Quick Entry Points

- Main report draft: `docs/Project_Report.md`
- Earlier reconstructed report draft: `docs/Reconstructed_Report_Draft.md`
- Feature definitions: `docs/Feature_Factor_Definitions.md`
- Main training/submission code: `code/Submission_XGBoost/`
- Reproduction experiments: `code/Reproduction_XGBoost/`
- EDA code and summaries: `code/EDA/`
- Previous implementation reference: `code/Previous_version_reference/`
- Official submission template shape: `code/submission_template/`
- Final submit-ready package: `deliverables/SUBMIT_THIS_date63_holdout_turbo_fixed.zip`

## Final Submit-Ready Package

Use this package if a direct platform submission artifact is needed:

```text
deliverables/SUBMIT_THIS_date63_holdout_turbo_fixed.zip
SHA256: 2b47ae87fffdd04ecd926a66327dcc96e540de6717de8676962ea25e887cf65e
```

Zip contents:

- `Predictor.py`
- `feature_builder.py`
- `config.json`
- `requirements.txt`
- `feature_spec.json`
- `thresholds.json`
- `model_label_5.json`
- `model_label_10.json`
- `model_label_20.json`
- `model_label_40.json`
- `model_label_60.json`

This is the turbo/fixed date-holdout XGBoost package. It was benchmarked against
the slower original package with zero mismatches on a clean 256-window prediction
comparison.

## Final Platform Result To Cite

Reported platform result for `date63_holdout_turbo_fixed`:

| Label | Precision | Recall | F0.5 | Total PnL | Single PnL | Model Score |
|---|---:|---:|---:|---:|---:|---:|
| `label_5` | 0.801436 | 0.0517998 | 0.205793 | -0.415608 | -0.000054 | -0.000425 |
| `label_10` | 0.762540 | 0.0659549 | 0.245007 | 2.097210 | 0.000147 | -0.000156 |
| `label_20` | 0.623320 | 0.0693784 | 0.240026 | 11.716000 | 0.000761 | 0.000312 |
| `label_40` | 0.591949 | 0.0412298 | 0.161229 | 14.614600 | 0.001104 | 0.000798 |
| `label_60` | 0.581890 | 0.0288684 | 0.120440 | 12.589500 | 0.001179 | 0.000731 |

Main interpretation:

- The final package fixed runtime/import issues and produced positive platform
  scores for `label_20`, `label_40`, and `label_60`.
- `label_40` had the highest model score in this run.
- `label_60` had the best total PnL among the positive-score labels.
- Short horizons kept high precision but scored negative because single PnL was
  below the public baseline.

## Main Code Notes

`code/Submission_XGBoost/` is the main implementation:

- `src/feature_builder.py`: shared 182-base-feature / 922-final-feature builder.
- `src/train.py`: one XGBoost multiclass model per label; supports date and
  index splits, CPU/GPU, threshold scanning, and external-memory training.
- `src/Predictor.py`: platform-facing inference entry point.
- `src/build_submission.py`: creates a flat package and zip.
- `src/smoke_test_submission.py`: validates package import and output shape.
- `scripts/run_date63_holdout_gpu.sh`: final date-holdout training launcher.

`code/Reproduction_XGBoost/` is separate reproduction work for the previous
report. Read `code/Reproduction_XGBoost/REPRODUCTION_GAPS.md` before citing it.

`code/Previous_version_reference/` is included only as a feature/model reference.
It is not the cleaned main workflow.

## Data And Reproduction Caveat

Raw CSV data is not included. The original project expected data under:

```text
EDA/raw data/FBDQA2021A_MMP_Challenge/data/
```

To rerun training, restore the raw CSV files to that path or pass an explicit
`--data-dir` to the training scripts.

Example smoke command after installing dependencies:

```bash
cd code/Submission_XGBoost
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --data-dir "../EDA/raw data/FBDQA2021A_MMP_Challenge/data" --labels label_60 --max-files 80 --num-boost-round 20 --early-stopping-rounds 5
python src/build_submission.py
python src/smoke_test_submission.py --data-dir "../EDA/raw data/FBDQA2021A_MMP_Challenge/data"
```

## Security And Exclusions

Sensitive-information scan covered project-owned `.py`, `.md`, `.json`, `.txt`,
`.sh`, `.yml`, and `.yaml` files while excluding virtual environments and
generated caches. No obvious passwords, API keys, bearer tokens, private keys,
or `.env` files were found.

The original progress log contains server IP/user operational details but no
passwords. It is not included in this package. Use this handoff README instead
for report-relevant status and results.

Excluded from this handoff:

- `EDA/raw data/` raw CSV files
- `Submission_XGBoost/.venv/`
- `Submission_XGBoost/artifacts/cache/`
- historical submission/model zips except the final fixed deliverable
- expanded final model directory duplicated by the final zip
- `__pycache__/`, `.pyc`, `.DS_Store`
- generated presentation build outputs under `outputs/`

