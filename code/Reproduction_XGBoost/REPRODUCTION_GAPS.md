# Reproduction Gaps

This file records the parts of `Previous version of project/Final_Report.pdf`
that cannot be reconstructed exactly from the PDF alone.

## Implemented From The PDF

- XGBoost multiclass model, one model per label.
- Labels: `label_5`, `label_10`, `label_20`, `label_40`, `label_60`.
- File-order 8:2 train/validation split.
- Limit up/down row filtering.
- External-memory training, with either intermediate `.buffer` files or direct
  CSV-to-DataIter streaming.
- Early stopping with 100 rounds.
- `max_depth=6` in the default `report_pdf` preset.
- Confidence filtering for down/up signals at threshold `0.88`.
- Ratio-grid postprocessing for the PDF's `get_predict(y, ratio)` function.
- PDF-described feature groups:
  - time features
  - price/size spread and relative density
  - weighted bid/ask price
  - volume imbalance
  - log amount and log size transforms
  - first differences
  - rolling mean/std and value-vs-mean features
- The PDF note that `i` takes `1~5` or `1~3` is implemented as a unified switch:
  `--pdf-levels 1-5` or `--pdf-levels 1-3`. The selected range is applied to
  every i-based PDF feature, including `weighted_ab`.

## Not Fully Specified In The PDF

- The exact final XGBoost input vector. The PDF's `100*24` means the raw window
  has 100 ticks and 24 initial features per tick. The report then describes
  derived features, but does not state whether the final tree input is a full
  flattened 100-tick derived-feature window, a subset of recent ticks, or the
  922-dimensional pyramid vector found in the current old code.
- The exact complete feature list. The old code contains many extra features
  such as MACD, KDJ, ROC, z-scores, OFI, micro price, and depth pressure that are
  not described in the PDF text.
- Whether class weights were used. The PDF does not mention class weighting; the
  current old code can apply class weights.
- The actual `get_predict` ratio values are not stated in the PDF. The
  reproduction code scans a broad ratio grid and records the best pair rather
  than assuming the original values.
- The exact values of non-depth XGBoost hyperparameters for the PDF run. The
  report explicitly mentions tuning depth but not a full parameter table, so
  `report_pdf` leaves non-depth XGBoost hyperparameters at library defaults.
- GPU queue runs can pass XGBoost external-memory resource controls such as
  `cache_host_ratio`, `max_quantile_batches`, and `min_cache_page_bytes`. These
  are used to control memory placement and are not described in the PDF.
- The current XGBoost server environment requires CuPy batches for GPU
  `ExtMemQuantileDMatrix`; CPU NumPy batches can construct a matrix but cannot
  be used by GPU training.
- Exact train/validation file boundary. The PDF says 8:2 by stock/date order;
  the current reproducible code implements sorted file-order 8:2 to match the
  old code's available implementation.
- Public/private leaderboard screenshots in the PDF were not available as
  machine-readable values from text extraction.
- `label_60` classification matrices are not present in the extracted PDF text;
  only `label_5`, `label_10`, `label_20`, and `label_40` matrices were captured.

## Code Paths Provided

- `--feature-set pdf_report`: literal PDF-style feature builder with a unified
  i-level range and a flattened 100-tick derived-feature window. `--pdf-levels 1-5` gives 163
  base features; `--pdf-levels 1-3` gives 111 base features. This is the most faithful to the
  written feature description, but may not match the PDF's reported sample
  counts if the original implementation did not require a full 100-tick history.
- `--feature-set previous_code`: richer old-code feature builder with 182 base
  features and a 922-dimensional final vector. This is likely closer to the
  executable old project, but includes features not documented in the PDF.
- `--param-preset report_pdf`: depth-6 XGBoost preset matching the PDF's stated
  tuning result where possible.
- `--param-preset previous_code`: parameters from the current old code directory.
- `--storage-mode buffer` and `--storage-mode stream`: both call the same
  feature-batch generator, so they are intended to differ only in storage and
  I/O behavior, not feature engineering.
