#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/stock_data/stock_movement/Submission_XGBoost}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/stock_data/stock_movement/cache}"

VERSION="${VERSION:-full_gpu_date63_holdout}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/${VERSION}}"
CACHE_DIR="${CACHE_DIR:-${CACHE_ROOT}/${VERSION}}"
LOG_DIR="${LOG_DIR:-logs}"
WAIT_PID="${WAIT_PID:-}"

VAL_START_DATE="${VAL_START_DATE:-63}"
DEVICE="${DEVICE:-cuda}"
NTHREAD="${NTHREAD:--1}"
MAX_BIN="${MAX_BIN:-256}"
BATCH_SIZE="${BATCH_SIZE:-5000}"
NUM_BOOST_ROUND="${NUM_BOOST_ROUND:-2000}"
EARLY_STOPPING_ROUNDS="${EARLY_STOPPING_ROUNDS:-100}"
THRESHOLD_METRIC="${THRESHOLD_METRIC:-official_score_est}"
PNL_BASELINE="${PNL_BASELINE:-0.0004}"
BUILD_PACKAGE="${BUILD_PACKAGE:-1}"
SMOKE_TEST="${SMOKE_TEST:-1}"
SMOKE_BATCH="${SMOKE_BATCH:-32}"

cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

wait_for_pid() {
  local pid="$1"
  if [[ -z "$pid" ]]; then
    return 0
  fi
  log "Waiting for existing process PID=$pid"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
  done
  log "Existing process PID=$pid has exited"
}

log_file="${LOG_DIR}/${VERSION}_$(date +%Y%m%d_%H%M%S).log"
package_dir="${OUTPUT_DIR}/submission_package"
zip_path="${OUTPUT_DIR}/submission_xgboost_${VERSION}.zip"

wait_for_pid "$WAIT_PID"

log "Starting ${VERSION}; log=${log_file}"
log "Output dir: ${OUTPUT_DIR}"
log "Cache dir: ${CACHE_DIR}"

train_cmd=(
  "$PYTHON_BIN" src/train.py
  --split-mode date
  --val-start-date "$VAL_START_DATE"
  --device "$DEVICE"
  --nthread "$NTHREAD"
  --max-bin "$MAX_BIN"
  --num-boost-round "$NUM_BOOST_ROUND"
  --early-stopping-rounds "$EARLY_STOPPING_ROUNDS"
  --batch-size "$BATCH_SIZE"
  --threshold-metric "$THRESHOLD_METRIC"
  --pnl-baseline "$PNL_BASELINE"
  --output-dir "$OUTPUT_DIR"
  --cache-dir "$CACHE_DIR"
  --cleanup-cache
)

{
  printf '[%s] Train command:' "$(timestamp)"
  printf ' %q' "${train_cmd[@]}"
  printf '\n'
  "${train_cmd[@]}"
} > "$log_file" 2>&1

log "Training finished for ${VERSION}"

if [[ "$BUILD_PACKAGE" == "1" ]]; then
  log "Building package: ${zip_path}"
  "$PYTHON_BIN" src/build_submission.py \
    --artifacts-dir "$OUTPUT_DIR" \
    --output-dir "$package_dir" \
    --zip-path "$zip_path" >> "$log_file" 2>&1

  if [[ "$SMOKE_TEST" == "1" ]]; then
    log "Running smoke test for ${package_dir}"
    "$PYTHON_BIN" src/smoke_test_submission.py \
      --package-dir "$package_dir" \
      --batch "$SMOKE_BATCH" >> "$log_file" 2>&1
  fi
fi

log "Done ${VERSION}"
