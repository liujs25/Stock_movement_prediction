"""Train legacy-style XGBoost models for all stock movement labels."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

from feature_builder import FeatureBuilder, LABEL_COLUMNS


DEFAULT_DATA_DIR = Path("../EDA/raw data/FBDQA2021A_MMP_Challenge/data")
DEFAULT_THRESHOLDS = [
    0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.72, 0.73, 0.74, 0.75,
    0.78, 0.80, 0.82, 0.84, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95,
    0.96, 0.97, 0.98,
]
DEFAULT_PNL_BASELINE = 0.0004


@dataclass
class SnapshotFile:
    path: Path
    sym_id: int
    date_id: int
    session: str


@dataclass
class LabelTrainResult:
    label: str
    model_path: str
    threshold: float
    train_samples: int
    val_samples: int
    train_label_counts: Dict[str, int]
    best_iteration: int | None
    profit_weighting: bool
    profit_weight_base: float
    profit_weight_alpha: float
    profit_weight_min: float
    profit_weight_max: float


def tprint(*args) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]", *args, flush=True)


def parse_snapshot_file(path: Path) -> SnapshotFile | None:
    match = re.match(r"snapshot_sym(\d+)_date(\d+)_(am|pm)\.csv$", path.name)
    if not match:
        return None
    return SnapshotFile(
        path=path,
        sym_id=int(match.group(1)),
        date_id=int(match.group(2)),
        session=match.group(3),
    )


def scan_snapshot_files(data_dir: Path) -> List[SnapshotFile]:
    files = []
    for path in sorted(data_dir.glob("snapshot_sym*_date*_*.csv")):
        parsed = parse_snapshot_file(path)
        if parsed is not None:
            files.append(parsed)
    files.sort(key=lambda item: (item.date_id, 0 if item.session == "am" else 1, item.sym_id))
    return files


def split_files(
    files: Sequence[SnapshotFile],
    val_start_date: int,
    max_files: int | None = None,
    split_mode: str = "date",
    test_size: float = 0.2,
) -> Tuple[List[SnapshotFile], List[SnapshotFile]]:
    selected = list(files[:max_files]) if max_files else list(files)
    if split_mode == "date":
        train_files = [item for item in selected if item.date_id < val_start_date]
        val_files = [item for item in selected if item.date_id >= val_start_date]
    elif split_mode == "index":
        if not 0 < test_size < 1:
            raise ValueError(f"--test-size must be between 0 and 1, got {test_size}")
        split_file_idx = int(len(selected) * (1 - test_size))
        train_files = selected[:split_file_idx]
        val_files = selected[split_file_idx:]
    else:
        raise ValueError(f"Unsupported split mode: {split_mode}")
    if not train_files or not val_files:
        raise ValueError(
            f"Time split produced train={len(train_files)}, val={len(val_files)}. "
            "Adjust --val-start-date, --test-size, or --max-files."
        )
    return train_files, val_files


def determine_available_features(builder: FeatureBuilder, files: Sequence[SnapshotFile]) -> List[str]:
    sample = pd.read_csv(files[0].path, nrows=400)
    available = builder.available_features(sample)
    if "n_midprice" not in available:
        raise ValueError("n_midprice must be present in available features")
    if "total_imbalance" not in available:
        raise ValueError("total_imbalance must be present in available features")
    return available


def is_limit_up_down(df: pd.DataFrame) -> pd.Series:
    return (df["n_ask1"] == 0) | (df["n_bid1"] == 0) | (df["n_close"] >= 0.095) | (df["n_close"] <= -0.095)


def save_feature_spec(output_dir: Path, builder: FeatureBuilder, available_features: Sequence[str]) -> Path:
    spec = {
        "labels": LABEL_COLUMNS,
        "available_features": list(available_features),
        "final_feature_names": builder.final_feature_names,
        "recent_lags": list(builder.recent_lags),
        "summary_lags": list(builder.summary_lags),
    }
    path = output_dir / "feature_spec.json"
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return path


def process_files_to_buffers(
    *,
    files: Sequence[SnapshotFile],
    label: str,
    builder: FeatureBuilder,
    available_features: Sequence[str],
    cache_dir: Path,
    split_name: str,
    batch_size: int,
    drop_limit_samples: bool,
) -> Tuple[Path, Dict[int, int], int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / f"{split_name}_{label}.meta"
    n_shift = int(label.split("_")[1])
    label_counts = {0: 0, 1: 0, 2: 0}
    buffer_files: List[str] = []
    batch_features: List[np.ndarray] = []
    batch_labels: List[int] = []
    batch_price_diffs: List[float] = []
    total_samples = 0
    buffer_idx = 0

    def save_buffer() -> int:
        nonlocal batch_features, batch_labels, batch_price_diffs, buffer_idx
        if not batch_features:
            return 0
        x_batch = np.asarray(batch_features, dtype=np.float32)
        y_batch = np.asarray(batch_labels, dtype=np.int32)
        p_batch = np.asarray(batch_price_diffs, dtype=np.float32)
        buffer_path = cache_dir / f"{split_name}_{label}_{buffer_idx}.buffer"
        price_path = cache_dir / f"{split_name}_{label}_{buffer_idx}.price.npy"
        xgb.DMatrix(x_batch, label=y_batch).save_binary(str(buffer_path))
        np.save(price_path, p_batch)
        buffer_files.append(str(buffer_path))
        size = len(batch_labels)
        batch_features = []
        batch_labels = []
        batch_price_diffs = []
        buffer_idx += 1
        return size

    for item in tqdm(files, desc=f"{split_name}:{label}", ncols=90):
        df = pd.read_csv(item.path)
        df = df.sort_values(["sym", "date", "time"], kind="mergesort").reset_index(drop=True)
        limit_mask = is_limit_up_down(df) if drop_limit_samples else pd.Series(False, index=df.index)
        df["price_diff_raw"] = df.groupby(["sym", "date"], sort=False)["n_midprice"].shift(-n_shift) - df["n_midprice"]
        feature_matrix = builder.build_feature_matrix(df, available_features)
        matrix_values = feature_matrix.values

        grouped = df.groupby(["sym", "date"], sort=False)
        for _, group in grouped:
            group_labels = group[label].fillna(1).astype(int).values
            group_diffs = df.loc[group.index, "price_diff_raw"].fillna(0).values
            group_limit_mask = limit_mask.loc[group.index].values
            group_positions = list(group.index)

            for local_idx, row_idx in enumerate(group_positions):
                if group_limit_mask[local_idx] or local_idx < 4:
                    continue
                vector = builder.assemble_pyramid_vector(matrix_values, row_idx, available_features)
                class_id = int(group_labels[local_idx])
                batch_features.append(vector)
                batch_labels.append(class_id)
                batch_price_diffs.append(float(group_diffs[local_idx]))
                label_counts[class_id] = label_counts.get(class_id, 0) + 1
                if len(batch_features) >= batch_size:
                    total_samples += save_buffer()
        del df, feature_matrix, matrix_values
        gc.collect()

    total_samples += save_buffer()
    if not buffer_files:
        raise ValueError(f"No samples generated for {split_name}:{label}")
    meta_path.write_text("\n".join(buffer_files) + "\n", encoding="utf-8")
    return meta_path, label_counts, total_samples


class BufferDataIter(xgb.DataIter):
    def __init__(
        self,
        buffer_files: Sequence[str],
        class_weights: Dict[int, float] | None = None,
        cache_prefix: str | None = None,
        use_gpu: bool = False,
        use_profit_weight: bool = False,
        profit_weight_base: float = 0.5,
        profit_weight_alpha: float = 1.5,
        profit_weight_min: float = 0.1,
        profit_weight_max: float = 10.0,
        profit_weight_mean_abs_diff: float | None = None,
    ):
        self.buffer_files = list(buffer_files)
        self.class_weights = class_weights or {}
        self.current_idx = 0
        self.use_gpu = use_gpu
        self.use_profit_weight = use_profit_weight
        self.profit_weight_base = profit_weight_base
        self.profit_weight_alpha = profit_weight_alpha
        self.profit_weight_min = profit_weight_min
        self.profit_weight_max = profit_weight_max
        self.profit_weight_mean_abs_diff = profit_weight_mean_abs_diff
        self._cp = None
        if self.use_gpu:
            try:
                import cupy as cp
            except ImportError as exc:
                raise RuntimeError("GPU external-memory training requires cupy-cuda12x") from exc
            self._cp = cp
        super().__init__(cache_prefix=cache_prefix)

    def next(self, input_data):
        if self.current_idx >= len(self.buffer_files):
            return 0
        dmatrix = xgb.DMatrix(self.buffer_files[self.current_idx])
        x_data = dmatrix.get_data()
        y_data = dmatrix.get_label()
        weights = np.ones_like(y_data, dtype=np.float32)
        if self.use_profit_weight:
            price_path = self.buffer_files[self.current_idx].replace(".buffer", ".price.npy")
            if os.path.exists(price_path):
                price_diffs = np.load(price_path)
                abs_diffs = np.abs(price_diffs).astype(np.float32)
                mean_diff = self.profit_weight_mean_abs_diff or float(np.mean(abs_diffs))
                mean_diff = float(mean_diff) + 1e-10
                profit_weights = self.profit_weight_base + self.profit_weight_alpha * (abs_diffs / mean_diff)
                weights *= np.clip(profit_weights, self.profit_weight_min, self.profit_weight_max).astype(np.float32)
        if self.class_weights:
            weights *= np.asarray([self.class_weights.get(int(label), 1.0) for label in y_data], dtype=np.float32)
        if self.use_gpu:
            if hasattr(x_data, "toarray"):
                x_data = x_data.toarray()
            x_data = self._cp.asarray(x_data)
            y_data = self._cp.asarray(y_data)
            weights = self._cp.asarray(weights)
        input_data(data=x_data, label=y_data, weight=weights)
        self.current_idx += 1
        return 1

    def reset(self):
        self.current_idx = 0


def read_meta(meta_path: Path) -> List[str]:
    return [line.strip() for line in meta_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def existing_buffer_files(cache_dir: Path, label: str, split_name: str) -> Tuple[Path, List[str]] | None:
    meta_path = cache_dir / f"{split_name}_{label}.meta"
    if not meta_path.exists():
        return None
    buffer_files = read_meta(meta_path)
    if not buffer_files or any(not Path(buffer_file).exists() for buffer_file in buffer_files):
        return None
    return meta_path, buffer_files


def summarize_buffers(buffer_files: Sequence[str]) -> Tuple[Dict[int, int], int]:
    label_counts = {0: 0, 1: 0, 2: 0}
    total_samples = 0
    for buffer_file in buffer_files:
        dmatrix = xgb.DMatrix(buffer_file)
        labels = dmatrix.get_label().astype(int)
        total_samples += int(labels.size)
        values, counts = np.unique(labels, return_counts=True)
        for value, count in zip(values, counts):
            label_counts[int(value)] = label_counts.get(int(value), 0) + int(count)
    return label_counts, total_samples


def mean_abs_price_diff(buffer_files: Sequence[str]) -> float:
    total_abs = 0.0
    total_count = 0
    for buffer_file in buffer_files:
        price_path = buffer_file.replace(".buffer", ".price.npy")
        if not os.path.exists(price_path):
            continue
        price_diffs = np.load(price_path)
        total_abs += float(np.abs(price_diffs).sum())
        total_count += int(price_diffs.size)
    return total_abs / total_count if total_count else 0.0


def cleanup_label_cache(cache_dir: Path, label: str) -> Tuple[int, int]:
    patterns = [
        f"train_{label}.meta",
        f"val_{label}.meta",
        f"train_{label}_*.buffer",
        f"train_{label}_*.price.npy",
        f"val_{label}_*.buffer",
        f"val_{label}_*.price.npy",
        f"extmem_train_{label}*",
        f"extmem_val_{label}*",
    ]
    removed_count = 0
    removed_bytes = 0
    for pattern in patterns:
        for path in cache_dir.glob(pattern):
            try:
                if path.is_dir():
                    removed_bytes += sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
                    shutil.rmtree(path)
                else:
                    removed_bytes += path.stat().st_size
                    path.unlink()
                removed_count += 1
            except FileNotFoundError:
                continue
    return removed_count, removed_bytes


def class_weights_from_counts(label_counts: Dict[int, int]) -> Dict[int, float]:
    total = sum(label_counts.values())
    nonzero = [count for count in label_counts.values() if count > 0]
    class_count = len(nonzero)
    return {
        label: total / (class_count * count) if count > 0 and class_count else 1.0
        for label, count in label_counts.items()
    }


def load_eval_arrays(buffer_files: Sequence[str], booster: xgb.Booster) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_parts = []
    label_parts = []
    price_parts = []
    for buffer_file in buffer_files:
        dmatrix = xgb.DMatrix(buffer_file)
        pred_parts.append(booster.predict(dmatrix))
        label_parts.append(dmatrix.get_label().astype(int))
        price_path = buffer_file.replace(".buffer", ".price.npy")
        price_parts.append(np.load(price_path) if os.path.exists(price_path) else np.zeros(dmatrix.num_row(), dtype=np.float32))
    return np.vstack(pred_parts), np.concatenate(label_parts), np.concatenate(price_parts)


def evaluate_thresholds(pred_proba: np.ndarray, y_true: np.ndarray, price_diff: np.ndarray, thresholds: Sequence[float]) -> List[Dict[str, float]]:
    actual_moves = (y_true == 0) | (y_true == 2)
    total_actual_moves = max(int(actual_moves.sum()), 1)
    rows: List[Dict[str, float]] = []
    for threshold in thresholds:
        if threshold == 0:
            pred = pred_proba.argmax(axis=1)
            signal_mask = (pred == 0) | (pred == 2)
        else:
            down_mask = pred_proba[:, 0] > threshold
            up_mask = pred_proba[:, 2] > threshold
            signal_mask = down_mask | up_mask
            pred = np.ones(len(y_true), dtype=int)
            choose_up = up_mask & (~down_mask | (pred_proba[:, 2] >= pred_proba[:, 0]))
            choose_down = down_mask & ~choose_up
            pred[choose_down] = 0
            pred[choose_up] = 2

        signal_count = int(signal_mask.sum())
        if signal_count == 0:
            rows.append({
                "threshold": float(threshold),
                "signal_pct": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f05": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
            })
            continue

        pred_signal = pred[signal_mask]
        true_signal = y_true[signal_mask]
        correct = int((pred_signal == true_signal).sum())
        precision = correct / signal_count
        recall = correct / total_actual_moves
        f05 = (1.25 * precision * recall) / (0.25 * precision + recall + 1e-10)
        pnl = np.where(pred_signal == 2, price_diff[signal_mask], -price_diff[signal_mask])
        rows.append({
            "threshold": float(threshold),
            "signal_pct": float(signal_count / len(y_true)),
            "precision": float(precision),
            "recall": float(recall),
            "f05": float(f05),
            "total_pnl": float(np.sum(pnl)),
            "avg_pnl": float(np.mean(pnl)),
        })
    return rows


def add_official_scores(rows: List[Dict[str, float]], pnl_baseline: float) -> List[Dict[str, float]]:
    for row in rows:
        delta = row["avg_pnl"] - pnl_baseline
        row["official_score_est"] = float(row["f05"] * delta * abs(delta) * 10000)
    return rows


def select_threshold(rows: Sequence[Dict[str, float]], metric: str, default_threshold: float) -> float:
    if metric == "default":
        return float(default_threshold)
    best = max(rows, key=lambda row: row[metric])
    return float(best["threshold"])


def train_one_label(
    *,
    label: str,
    train_files: Sequence[SnapshotFile],
    val_files: Sequence[SnapshotFile],
    builder: FeatureBuilder,
    available_features: Sequence[str],
    output_dir: Path,
    cache_dir: Path,
    batch_size: int,
    params: Dict[str, object],
    num_boost_round: int,
    early_stopping_rounds: int,
    default_threshold: float,
    threshold_metric: str,
    pnl_baseline: float,
    drop_limit_samples: bool,
    max_bin: int,
    reuse_buffers: bool,
    cleanup_cache: bool,
    use_profit_weight: bool,
    profit_weight_base: float,
    profit_weight_alpha: float,
    profit_weight_min: float,
    profit_weight_max: float,
) -> LabelTrainResult:
    if cleanup_cache and not reuse_buffers:
        removed_count, removed_bytes = cleanup_label_cache(cache_dir, label)
        if removed_count:
            tprint(f"Removed stale cache for {label}: files={removed_count} size={removed_bytes / (1024 ** 3):.2f} GiB")

    try:
        existing_train = existing_buffer_files(cache_dir, label, "train") if reuse_buffers else None
        existing_val = existing_buffer_files(cache_dir, label, "val") if reuse_buffers else None
        if existing_train is not None and existing_val is not None:
            train_meta, train_buffers = existing_train
            val_meta, val_buffers = existing_val
            train_counts, train_samples = summarize_buffers(train_buffers)
            _, val_samples = summarize_buffers(val_buffers)
            tprint(f"Reusing buffers for {label}: train_meta={train_meta} val_meta={val_meta}")
        else:
            if reuse_buffers:
                tprint(f"Reusable buffers not found for {label}; rebuilding")
            tprint(f"Preparing buffers for {label}")
            train_meta, train_counts, train_samples = process_files_to_buffers(
                files=train_files,
                label=label,
                builder=builder,
                available_features=available_features,
                cache_dir=cache_dir,
                split_name="train",
                batch_size=batch_size,
                drop_limit_samples=drop_limit_samples,
            )
            val_meta, _, val_samples = process_files_to_buffers(
                files=val_files,
                label=label,
                builder=builder,
                available_features=available_features,
                cache_dir=cache_dir,
                split_name="val",
                batch_size=batch_size,
                drop_limit_samples=drop_limit_samples,
            )
            train_buffers = read_meta(train_meta)
            val_buffers = read_meta(val_meta)

        weights = class_weights_from_counts(train_counts)
        tprint(f"{label} train label counts: {train_counts}")
        tprint(f"{label} class weights: {weights}")

        use_gpu_data = str(params.get("device", "")).startswith("cuda")
        if use_gpu_data:
            tprint(f"Using GPU external-memory batches for {label}")
        train_mean_abs_diff = mean_abs_price_diff(train_buffers) if use_profit_weight else None
        if use_profit_weight:
            tprint(
                f"{label} profit weighting enabled: "
                f"base={profit_weight_base} alpha={profit_weight_alpha} "
                f"clip=[{profit_weight_min}, {profit_weight_max}] "
                f"mean_abs_diff={train_mean_abs_diff:.8f}"
            )
        train_iter = BufferDataIter(
            train_buffers,
            weights,
            cache_prefix=str(cache_dir / f"extmem_train_{label}"),
            use_gpu=use_gpu_data,
            use_profit_weight=use_profit_weight,
            profit_weight_base=profit_weight_base,
            profit_weight_alpha=profit_weight_alpha,
            profit_weight_min=profit_weight_min,
            profit_weight_max=profit_weight_max,
            profit_weight_mean_abs_diff=train_mean_abs_diff,
        )
        dtrain = xgb.ExtMemQuantileDMatrix(train_iter, max_bin=max_bin)
        val_iter = BufferDataIter(
            val_buffers,
            weights,
            cache_prefix=str(cache_dir / f"extmem_val_{label}"),
            use_gpu=use_gpu_data,
            use_profit_weight=False,
        )
        dval = xgb.ExtMemQuantileDMatrix(val_iter, max_bin=max_bin, ref=dtrain)

        tprint(f"Training {label}: train={dtrain.num_row()} val={dval.num_row()} features={dtrain.num_col()}")
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dval, "eval")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=25,
        )

        models_dir = output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"model_{label}.json"
        booster.save_model(str(model_path))

        pred_proba, y_true, price_diff = load_eval_arrays(val_buffers, booster)
        metrics = add_official_scores(
            evaluate_thresholds(pred_proba, y_true, price_diff, DEFAULT_THRESHOLDS),
            pnl_baseline,
        )
        selected_threshold = select_threshold(metrics, threshold_metric, default_threshold)
        tprint(f"{label} selected threshold={selected_threshold} by {threshold_metric}")
        metrics_dir = output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / f"{label}_thresholds.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        best_iteration = getattr(booster, "best_iteration", None)
        return LabelTrainResult(
            label=label,
            model_path=str(model_path),
            threshold=selected_threshold,
            train_samples=train_samples,
            val_samples=val_samples,
            train_label_counts={str(k): int(v) for k, v in train_counts.items()},
            best_iteration=int(best_iteration) if best_iteration is not None else None,
            profit_weighting=use_profit_weight,
            profit_weight_base=profit_weight_base,
            profit_weight_alpha=profit_weight_alpha,
            profit_weight_min=profit_weight_min,
            profit_weight_max=profit_weight_max,
        )
    finally:
        if cleanup_cache:
            removed_count, removed_bytes = cleanup_label_cache(cache_dir, label)
            if removed_count:
                tprint(f"Cleaned cache for {label}: files={removed_count} size={removed_bytes / (1024 ** 3):.2f} GiB")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train XGBoost submission models")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    parser.add_argument("--labels", nargs="*", default=LABEL_COLUMNS, choices=LABEL_COLUMNS)
    parser.add_argument("--split-mode", choices=["date", "index"], default="date")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation file ratio when --split-mode index")
    parser.add_argument("--val-start-date", type=int, default=63)
    parser.add_argument("--max-files", type=int, default=None, help="Optional smoke-test limit over sorted snapshot files")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--num-boost-round", type=int, default=2000)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--default-threshold", type=float, default=0.88)
    parser.add_argument(
        "--threshold-metric",
        choices=["official_score_est", "f05", "total_pnl", "avg_pnl", "precision", "default"],
        default="official_score_est",
        help="Metric used to select the saved threshold for each label",
    )
    parser.add_argument("--pnl-baseline", type=float, default=DEFAULT_PNL_BASELINE)
    parser.add_argument("--nthread", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--max-bin", type=int, default=256)
    parser.add_argument("--device", default="cpu", help="XGBoost device, for example 'cpu' or 'cuda'")
    parser.add_argument("--reuse-buffers", action="store_true", help="Reuse existing per-label DMatrix buffers when available")
    parser.add_argument("--cleanup-cache", action="store_true", help="Delete per-label training cache after each label")
    parser.add_argument("--keep-limit-samples", action="store_true", help="Do not drop limit up/down rows during training")
    parser.add_argument("--profit-weight", action="store_true", help="Weight training samples by absolute future price move")
    parser.add_argument("--profit-weight-base", type=float, default=0.5)
    parser.add_argument("--profit-weight-alpha", type=float, default=1.5)
    parser.add_argument("--profit-weight-min", type=float, default=0.1)
    parser.add_argument("--profit-weight-max", type=float, default=10.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir if args.cache_dir.is_absolute() else project_root / args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    data_dir = args.data_dir
    if not data_dir.is_absolute():
        cwd_candidate = Path.cwd() / data_dir
        project_candidate = project_root / data_dir
        data_dir = cwd_candidate if cwd_candidate.exists() else project_candidate
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    files = scan_snapshot_files(data_dir)
    if not files:
        raise ValueError(f"No snapshot csv files found in {data_dir}")
    train_files, val_files = split_files(files, args.val_start_date, args.max_files, args.split_mode, args.test_size)
    tprint(f"Files: total={len(files)} train={len(train_files)} val={len(val_files)} split_mode={args.split_mode}")

    builder = FeatureBuilder()
    available_features = determine_available_features(builder, train_files)
    spec_path = save_feature_spec(output_dir, builder, available_features)
    tprint(f"Feature spec saved: {spec_path}")
    tprint(f"Available base features={len(available_features)}, final vector={len(builder.final_feature_names)}")

    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.6,
        "colsample_bytree": 0.4,
        "min_child_weight": 50,
        "reg_alpha": 10.0,
        "reg_lambda": 2.0,
        "gamma": 0.5,
        "random_state": 42,
        "nthread": args.nthread,
        "tree_method": "hist",
        "max_bin": args.max_bin,
        "device": args.device,
    }

    results = []
    thresholds = {}
    for label in args.labels:
        result = train_one_label(
            label=label,
            train_files=train_files,
            val_files=val_files,
            builder=builder,
            available_features=available_features,
            output_dir=output_dir,
            cache_dir=cache_dir,
            batch_size=args.batch_size,
            params=params,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
            default_threshold=args.default_threshold,
            threshold_metric=args.threshold_metric,
            pnl_baseline=args.pnl_baseline,
            drop_limit_samples=not args.keep_limit_samples,
            max_bin=args.max_bin,
            reuse_buffers=args.reuse_buffers,
            cleanup_cache=args.cleanup_cache,
            use_profit_weight=args.profit_weight,
            profit_weight_base=args.profit_weight_base,
            profit_weight_alpha=args.profit_weight_alpha,
            profit_weight_min=args.profit_weight_min,
            profit_weight_max=args.profit_weight_max,
        )
        results.append(asdict(result))
        thresholds[label] = result.threshold
        (output_dir / "train_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (output_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    tprint("Training complete")


if __name__ == "__main__":
    main()
