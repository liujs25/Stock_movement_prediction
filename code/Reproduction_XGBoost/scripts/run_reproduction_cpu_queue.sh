#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/stock_data/stock_movement/Reproduction_XGBoost}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/stock_data/stock_movement/Submission_XGBoost/.venv/bin/python}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/stock_data/stock_movement/cache}"
WAIT_PID="${WAIT_PID:-}"
DEVICE="${DEVICE:-cpu}"
TREE_METHOD="${TREE_METHOD-}"
MAX_BIN="${MAX_BIN-}"
BATCH_SIZE="${BATCH_SIZE:-5000}"
NTHREAD="${NTHREAD:--1}"
STORAGE_MODE="${STORAGE_MODE:-stream}"
NUM_BOOST_ROUND="${NUM_BOOST_ROUND:-2000}"
EARLY_STOPPING_ROUNDS="${EARLY_STOPPING_ROUNDS:-100}"
CLEANUP_CACHE="${CLEANUP_CACHE:-1}"

LABELS_1_3="${LABELS_1_3:-label_5 label_10 label_20 label_40 label_60}"
LABELS_1_5="${LABELS_1_5:-label_5 label_10 label_20 label_40 label_60}"

cd "$PROJECT_DIR"
mkdir -p logs

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

run_label() {
  local level="$1"
  local label="$2"
  local level_slug="${level/-/_}"
  local output_dir="artifacts/full_pdf_report_${level_slug}_${label}_cpu_${STORAGE_MODE}"
  local cache_dir="${CACHE_ROOT}/reproduction_pdf_report_${level_slug}_${label}_cpu_${STORAGE_MODE}"
  local log_file="logs/full_pdf_report_${level_slug}_${label}_cpu_${STORAGE_MODE}_$(date +%Y%m%d_%H%M%S).log"

  log "Starting CPU pdf_report ${level} ${label}; log=${log_file}"
  local cmd=(
    "$PYTHON_BIN" src/train.py
    --feature-set pdf_report
    --pdf-levels "$level"
    --storage-mode "$STORAGE_MODE"
    --labels "$label"
    --split-mode index
    --test-size 0.2
    --param-preset report_pdf
    --num-boost-round "$NUM_BOOST_ROUND"
    --early-stopping-rounds "$EARLY_STOPPING_ROUNDS"
    --batch-size "$BATCH_SIZE"
    --nthread "$NTHREAD"
    --output-dir "$output_dir"
    --cache-dir "$cache_dir"
    --ratio-min 0.50
    --ratio-max 1.00
    --ratio-step 0.02
  )
  if [[ -n "$MAX_BIN" ]]; then
    cmd+=(--max-bin "$MAX_BIN")
  fi
  if [[ -n "$DEVICE" ]]; then
    cmd+=(--device "$DEVICE")
  fi
  if [[ -n "$TREE_METHOD" ]]; then
    cmd+=(--tree-method "$TREE_METHOD")
  fi
  if [[ "$CLEANUP_CACHE" == "1" ]]; then
    cmd+=(--cleanup-cache)
  fi
  "${cmd[@]}" > "$log_file" 2>&1
  log "Finished CPU pdf_report ${level} ${label}"
}

wait_for_pid "$WAIT_PID"

for label in $LABELS_1_3; do
  run_label "1-3" "$label"
done

for label in $LABELS_1_5; do
  run_label "1-5" "$label"
done

log "CPU reproduction queue complete"
