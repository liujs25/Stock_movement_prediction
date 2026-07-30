"""Train XGBoost models for reproducing the previous report experiments."""

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
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

from feature_builder import FeatureBuilder, LABEL_COLUMNS, build_feature_builder


DEFAULT_DATA_DIR = Path("../EDA/raw data/FBDQA2021A_MMP_Challenge/data")
DEFAULT_THRESHOLDS = [
    0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.72, 0.73, 0.74, 0.75,
    0.78, 0.80, 0.82, 0.84, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95,
    0.96, 0.97, 0.98,
]
DEFAULT_PNL_BASELINE = 0.0004
DEFAULT_RATIO_MIN = 0.50
DEFAULT_RATIO_MAX = 1.00
DEFAULT_RATIO_STEP = 0.02
PARAM_PRESETS = {
    "report_pdf": {
        "description": "Approximate Final_Report.pdf: only max_depth=6 is tuned; other XGBoost hyperparameters use library defaults.",
        "use_class_weight": False,
        "params": {
            "max_depth": 6,
        },
    },
    "previous_code": {
        "description": "Approximate the current Previous version of project/train.py settings.",
        "use_class_weight": True,
        "params": {
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.6,
            "colsample_bytree": 0.4,
            "min_child_weight": 50,
            "reg_alpha": 10.0,
            "reg_lambda": 2.0,
            "gamma": 0.5,
            "random_state": 42,
            "nthread": -1,
            "tree_method": "hist",
        },
    },
    "legacy_1225": {
        "description": "Match Previous version commit 8855d4b/a94734a training params for the label_20-era legacy run.",
        "use_class_weight": True,
        "params": {
            "max_depth": 7,
            "learning_rate": 0.05,
            "subsample": 0.7,
            "colsample_bytree": 0.4,
            "min_child_weight": 50,
            "reg_alpha": 5.0,
            "reg_lambda": 2.0,
            "gamma": 0.5,
            "random_state": 42,
            "nthread": -1,
            "tree_method": "hist",
        },
    },
}


@dataclass
class SnapshotFile:
    path: Path
    sym_id: int
    date_id: int
    session: str


@dataclass
class LabelTrainResult:
    label: str
    feature_set: str
    storage_mode: str
    model_path: str
    threshold: float
    train_samples: int
    val_samples: int
    train_label_counts: Dict[str, int]
    best_iteration: int | None
    param_preset: str
    max_bin: int | None
    gpu_batch_mode: str
    cache_host_ratio: float | None
    max_quantile_batches: int | None
    min_cache_page_bytes: int | None
    class_weighting: bool
    profit_weighting: bool
    profit_weight_base: float
    profit_weight_alpha: float
    profit_weight_min: float
    profit_weight_max: float


@dataclass
class FeatureBatch:
    features: np.ndarray
    labels: np.ndarray
    price_diffs: np.ndarray


@dataclass
class StreamSidecar:
    labels_path: Path
    price_path: Path
    label_counts: Dict[int, int]
    samples: int


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
    if getattr(builder, "summary_lags", ()) and "total_imbalance" not in available:
        raise ValueError("total_imbalance must be present in available features")
    return available


def is_limit_up_down(df: pd.DataFrame) -> pd.Series:
    return (df["n_ask1"] == 0) | (df["n_bid1"] == 0) | (df["n_close"] >= 0.095) | (df["n_close"] <= -0.095)


def save_feature_spec(output_dir: Path, builder: FeatureBuilder, available_features: Sequence[str]) -> Path:
    spec = {
        "labels": LABEL_COLUMNS,
        "feature_set": getattr(builder, "feature_set", "previous_code"),
        "pdf_level_mode": getattr(builder, "pdf_level_mode", None),
        "pdf_levels": list(getattr(builder, "pdf_levels", [])),
        "vector_mode": getattr(builder, "vector_mode", "unknown"),
        "min_history": int(getattr(builder, "min_history", 0)),
        "available_features": list(available_features),
        "final_feature_names": builder.final_feature_names,
        "recent_lags": list(builder.recent_lags),
        "summary_lags": list(builder.summary_lags),
        "known_gaps": [
            "Final_Report.pdf does not fully specify the final XGBoost vectorization.",
            "Final_Report.pdf lists feature families but not a complete executable feature list.",
            "The current old code contains additional features not described in the PDF.",
        ],
    }
    path = output_dir / "feature_spec.json"
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return path


def prepare_snapshot_frame(
    item: SnapshotFile,
    label: str,
    drop_limit_samples: bool,
) -> Tuple[pd.DataFrame, pd.Series]:
    n_shift = int(label.split("_")[1])
    df = pd.read_csv(item.path)
    df = df.sort_values(["sym", "date", "time"], kind="mergesort").reset_index(drop=True)
    limit_mask = is_limit_up_down(df) if drop_limit_samples else pd.Series(False, index=df.index)
    df["price_diff_raw"] = df.groupby(["sym", "date"], sort=False)["n_midprice"].shift(-n_shift) - df["n_midprice"]
    return df, limit_mask


def iter_sample_rows(
    df: pd.DataFrame,
    label: str,
    builder: FeatureBuilder,
    limit_mask: pd.Series,
) -> Iterator[Tuple[int, int, float]]:
    min_history = int(getattr(builder, "min_history", 4))
    grouped = df.groupby(["sym", "date"], sort=False)
    for _, group in grouped:
        group_labels = group[label].fillna(1).astype(int).values
        group_diffs = df.loc[group.index, "price_diff_raw"].fillna(0).values
        group_limit_mask = limit_mask.loc[group.index].values
        group_positions = list(group.index)

        for local_idx, row_idx in enumerate(group_positions):
            if group_limit_mask[local_idx] or local_idx < min_history:
                continue
            yield row_idx, int(group_labels[local_idx]), float(group_diffs[local_idx])


def update_label_counts(label_counts: Dict[int, int], labels: np.ndarray) -> None:
    values, counts = np.unique(labels.astype(int), return_counts=True)
    for value, count in zip(values, counts):
        label_counts[int(value)] = label_counts.get(int(value), 0) + int(count)


def iter_feature_batches(
    *,
    files: Sequence[SnapshotFile],
    label: str,
    builder: FeatureBuilder,
    available_features: Sequence[str],
    batch_size: int,
    drop_limit_samples: bool,
    desc: str,
) -> Iterator[FeatureBatch]:
    batch_features: List[np.ndarray] = []
    batch_labels: List[int] = []
    batch_price_diffs: List[float] = []

    def make_batch() -> FeatureBatch:
        nonlocal batch_features, batch_labels, batch_price_diffs
        batch = FeatureBatch(
            features=np.asarray(batch_features, dtype=np.float32),
            labels=np.asarray(batch_labels, dtype=np.int32),
            price_diffs=np.asarray(batch_price_diffs, dtype=np.float32),
        )
        batch_features = []
        batch_labels = []
        batch_price_diffs = []
        return batch

    for item in tqdm(files, desc=desc, ncols=90):
        df, limit_mask = prepare_snapshot_frame(item, label, drop_limit_samples)
        feature_matrix = builder.build_feature_matrix(df, available_features)
        matrix_values = feature_matrix.values

        for row_idx, class_id, price_diff in iter_sample_rows(df, label, builder, limit_mask):
            vector = builder.assemble_pyramid_vector(matrix_values, row_idx, available_features)
            batch_features.append(vector)
            batch_labels.append(class_id)
            batch_price_diffs.append(price_diff)
            if len(batch_features) >= batch_size:
                yield make_batch()

        del df, feature_matrix, matrix_values
        gc.collect()

    if batch_features:
        yield make_batch()


def process_files_to_stream_sidecar(
    *,
    files: Sequence[SnapshotFile],
    label: str,
    builder: FeatureBuilder,
    cache_dir: Path,
    split_name: str,
    drop_limit_samples: bool,
) -> StreamSidecar:
    cache_dir.mkdir(parents=True, exist_ok=True)
    label_counts = {0: 0, 1: 0, 2: 0}
    label_parts: List[np.ndarray] = []
    price_parts: List[np.ndarray] = []

    for item in tqdm(files, desc=f"{split_name}:{label}:sidecar", ncols=90):
        df, limit_mask = prepare_snapshot_frame(item, label, drop_limit_samples)
        labels = []
        price_diffs = []
        for _, class_id, price_diff in iter_sample_rows(df, label, builder, limit_mask):
            labels.append(class_id)
            price_diffs.append(price_diff)
        if labels:
            labels_array = np.asarray(labels, dtype=np.int32)
            prices_array = np.asarray(price_diffs, dtype=np.float32)
            label_parts.append(labels_array)
            price_parts.append(prices_array)
            update_label_counts(label_counts, labels_array)
        del df
        gc.collect()

    if not label_parts:
        raise ValueError(f"No samples generated for {split_name}:{label}")

    labels_all = np.concatenate(label_parts)
    prices_all = np.concatenate(price_parts)
    labels_path = cache_dir / f"{split_name}_{label}.labels.npy"
    price_path = cache_dir / f"{split_name}_{label}.price.npy"
    meta_path = cache_dir / f"{split_name}_{label}.sidecar.json"
    np.save(labels_path, labels_all)
    np.save(price_path, prices_all)
    meta_path.write_text(json.dumps({
        "split": split_name,
        "label": label,
        "samples": int(labels_all.size),
        "label_counts": {str(key): int(value) for key, value in label_counts.items()},
        "labels_path": str(labels_path),
        "price_path": str(price_path),
    }, indent=2), encoding="utf-8")
    return StreamSidecar(labels_path, price_path, label_counts, int(labels_all.size))


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
    label_counts = {0: 0, 1: 0, 2: 0}
    buffer_files: List[str] = []
    total_samples = 0
    buffer_idx = 0

    def save_buffer(batch: FeatureBatch) -> int:
        nonlocal buffer_idx
        buffer_path = cache_dir / f"{split_name}_{label}_{buffer_idx}.buffer"
        price_path = cache_dir / f"{split_name}_{label}_{buffer_idx}.price.npy"
        xgb.DMatrix(batch.features, label=batch.labels).save_binary(str(buffer_path))
        np.save(price_path, batch.price_diffs)
        buffer_files.append(str(buffer_path))
        size = int(batch.labels.size)
        buffer_idx += 1
        return size

    for batch in iter_feature_batches(
        files=files,
        label=label,
        builder=builder,
        available_features=available_features,
        batch_size=batch_size,
        drop_limit_samples=drop_limit_samples,
        desc=f"{split_name}:{label}",
    ):
        update_label_counts(label_counts, batch.labels)
        total_samples += save_buffer(batch)

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
        min_cache_page_bytes: int | None = None,
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
        self.min_cache_page_bytes = min_cache_page_bytes
        self._cp = None
        if self.use_gpu:
            try:
                import cupy as cp
            except ImportError as exc:
                raise RuntimeError("GPU external-memory training requires cupy-cuda12x") from exc
            self._cp = cp
        iter_kwargs = {"cache_prefix": cache_prefix}
        if min_cache_page_bytes is not None:
            iter_kwargs["min_cache_page_bytes"] = min_cache_page_bytes
        super().__init__(**iter_kwargs)

    def next(self, input_data):
        if self.current_idx >= len(self.buffer_files):
            return 0
        dmatrix = xgb.DMatrix(self.buffer_files[self.current_idx])
        x_data = dmatrix.get_data()
        y_data = dmatrix.get_label()
        price_diffs = None
        if self.use_profit_weight:
            price_path = self.buffer_files[self.current_idx].replace(".buffer", ".price.npy")
            if os.path.exists(price_path):
                price_diffs = np.load(price_path)
        weights = make_sample_weights(
            y_data,
            price_diffs=price_diffs,
            class_weights=self.class_weights,
            use_profit_weight=self.use_profit_weight,
            profit_weight_base=self.profit_weight_base,
            profit_weight_alpha=self.profit_weight_alpha,
            profit_weight_min=self.profit_weight_min,
            profit_weight_max=self.profit_weight_max,
            profit_weight_mean_abs_diff=self.profit_weight_mean_abs_diff,
        )
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


class SnapshotCSVDataIter(xgb.DataIter):
    def __init__(
        self,
        *,
        files: Sequence[SnapshotFile],
        label: str,
        builder: FeatureBuilder,
        available_features: Sequence[str],
        batch_size: int,
        drop_limit_samples: bool,
        class_weights: Dict[int, float] | None = None,
        cache_prefix: str | None = None,
        min_cache_page_bytes: int | None = None,
        use_gpu: bool = False,
        use_profit_weight: bool = False,
        profit_weight_base: float = 0.5,
        profit_weight_alpha: float = 1.5,
        profit_weight_min: float = 0.1,
        profit_weight_max: float = 10.0,
        profit_weight_mean_abs_diff: float | None = None,
        desc: str = "stream",
    ):
        self.files = list(files)
        self.label = label
        self.builder = builder
        self.available_features = list(available_features)
        self.batch_size = batch_size
        self.drop_limit_samples = drop_limit_samples
        self.class_weights = class_weights or {}
        self.use_gpu = use_gpu
        self.use_profit_weight = use_profit_weight
        self.profit_weight_base = profit_weight_base
        self.profit_weight_alpha = profit_weight_alpha
        self.profit_weight_min = profit_weight_min
        self.profit_weight_max = profit_weight_max
        self.profit_weight_mean_abs_diff = profit_weight_mean_abs_diff
        self.desc = desc
        self.min_cache_page_bytes = min_cache_page_bytes
        self._batch_iter: Iterator[FeatureBatch] | None = None
        self._cp = None
        if self.use_gpu:
            try:
                import cupy as cp
            except ImportError as exc:
                raise RuntimeError("GPU external-memory training requires cupy-cuda12x") from exc
            self._cp = cp
        iter_kwargs = {"cache_prefix": cache_prefix}
        if min_cache_page_bytes is not None:
            iter_kwargs["min_cache_page_bytes"] = min_cache_page_bytes
        super().__init__(**iter_kwargs)

    def next(self, input_data):
        if self._batch_iter is None:
            self._batch_iter = iter_feature_batches(
                files=self.files,
                label=self.label,
                builder=self.builder,
                available_features=self.available_features,
                batch_size=self.batch_size,
                drop_limit_samples=self.drop_limit_samples,
                desc=self.desc,
            )
        try:
            batch = next(self._batch_iter)
        except StopIteration:
            return 0

        x_data = batch.features
        y_data = batch.labels
        weights = make_sample_weights(
            y_data,
            price_diffs=batch.price_diffs,
            class_weights=self.class_weights,
            use_profit_weight=self.use_profit_weight,
            profit_weight_base=self.profit_weight_base,
            profit_weight_alpha=self.profit_weight_alpha,
            profit_weight_min=self.profit_weight_min,
            profit_weight_max=self.profit_weight_max,
            profit_weight_mean_abs_diff=self.profit_weight_mean_abs_diff,
        )
        if self.use_gpu:
            x_data = self._cp.asarray(x_data)
            y_data = self._cp.asarray(y_data)
            weights = self._cp.asarray(weights)
        input_data(data=x_data, label=y_data, weight=weights)
        return 1

    def reset(self):
        self._batch_iter = None


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


def mean_abs_price_diff_array(price_path: Path) -> float:
    price_diffs = np.load(price_path)
    return float(np.abs(price_diffs).mean()) if price_diffs.size else 0.0


def make_sample_weights(
    labels: np.ndarray,
    *,
    price_diffs: np.ndarray | None,
    class_weights: Dict[int, float],
    use_profit_weight: bool,
    profit_weight_base: float,
    profit_weight_alpha: float,
    profit_weight_min: float,
    profit_weight_max: float,
    profit_weight_mean_abs_diff: float | None,
) -> np.ndarray:
    weights = np.ones_like(labels, dtype=np.float32)
    if use_profit_weight and price_diffs is not None:
        abs_diffs = np.abs(price_diffs).astype(np.float32)
        mean_diff = profit_weight_mean_abs_diff or float(np.mean(abs_diffs))
        mean_diff = float(mean_diff) + 1e-10
        profit_weights = profit_weight_base + profit_weight_alpha * (abs_diffs / mean_diff)
        weights *= np.clip(profit_weights, profit_weight_min, profit_weight_max).astype(np.float32)
    if class_weights:
        weights *= np.asarray([class_weights.get(int(label), 1.0) for label in labels], dtype=np.float32)
    return weights


def cleanup_label_cache(cache_dir: Path, label: str) -> Tuple[int, int]:
    patterns = [
        f"train_{label}.meta",
        f"val_{label}.meta",
        f"train_{label}.labels.npy",
        f"train_{label}.price.npy",
        f"train_{label}.sidecar.json",
        f"val_{label}.labels.npy",
        f"val_{label}.price.npy",
        f"val_{label}.sidecar.json",
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


def parse_float_list(value: str) -> List[float]:
    parsed = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed.append(float(item))
    if not parsed:
        raise ValueError("Expected at least one float value")
    return parsed


def make_ratio_values(ratio_min: float, ratio_max: float, ratio_step: float, ratio_values: str | None = None) -> List[float]:
    if ratio_values:
        values = parse_float_list(ratio_values)
    else:
        if ratio_step <= 0:
            raise ValueError("--ratio-step must be positive")
        count = int(round((ratio_max - ratio_min) / ratio_step))
        values = [ratio_min + idx * ratio_step for idx in range(count + 1)]
        if values[-1] < ratio_max - 1e-9:
            values.append(ratio_max)
    cleaned = sorted({round(float(value), 6) for value in values})
    if any(value < 0 for value in cleaned):
        raise ValueError("Ratio values must be non-negative")
    return cleaned


def apply_ratio_postprocess(pred_proba: np.ndarray, down_ratio: float, up_ratio: float) -> np.ndarray:
    adjusted = pred_proba.copy()
    adjusted[:, 0] *= down_ratio
    adjusted[:, 2] *= up_ratio
    return adjusted.argmax(axis=1).astype(np.int64)


def evaluate_predictions(
    pred: np.ndarray,
    y_true: np.ndarray,
    price_diff: np.ndarray,
    *,
    down_ratio: float | None = None,
    up_ratio: float | None = None,
) -> Dict[str, float]:
    signal_mask = (pred == 0) | (pred == 2)
    actual_moves = (y_true == 0) | (y_true == 2)
    signal_count = int(signal_mask.sum())
    total_actual_moves = max(int(actual_moves.sum()), 1)
    row = {
        "signal_pct": float(signal_count / len(y_true)) if len(y_true) else 0.0,
        "trades": signal_count,
        "precision": 0.0,
        "recall": 0.0,
        "f05": 0.0,
        "total_pnl": 0.0,
        "avg_pnl": 0.0,
    }
    if down_ratio is not None:
        row["down_ratio"] = float(down_ratio)
    if up_ratio is not None:
        row["up_ratio"] = float(up_ratio)
    if signal_count == 0:
        return row

    pred_signal = pred[signal_mask]
    true_signal = y_true[signal_mask]
    correct = int((pred_signal == true_signal).sum())
    precision = correct / signal_count
    recall = correct / total_actual_moves
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall + 1e-10)
    pnl = np.where(pred_signal == 2, price_diff[signal_mask], -price_diff[signal_mask])
    row.update({
        "precision": float(precision),
        "recall": float(recall),
        "f05": float(f05),
        "total_pnl": float(np.sum(pnl)),
        "avg_pnl": float(np.mean(pnl)),
    })
    return row


def evaluate_ratio_grid(
    pred_proba: np.ndarray,
    y_true: np.ndarray,
    price_diff: np.ndarray,
    ratio_values: Sequence[float],
) -> List[Dict[str, float]]:
    rows = []
    for down_ratio in ratio_values:
        for up_ratio in ratio_values:
            pred = apply_ratio_postprocess(pred_proba, down_ratio, up_ratio)
            rows.append(evaluate_predictions(
                pred,
                y_true,
                price_diff,
                down_ratio=down_ratio,
                up_ratio=up_ratio,
            ))
    return rows


def add_official_scores(rows: List[Dict[str, float]], pnl_baseline: float) -> List[Dict[str, float]]:
    for row in rows:
        delta = row["avg_pnl"] - pnl_baseline
        row["official_score_est"] = float(row["f05"] * delta * abs(delta) * 10000)
    return rows


def confusion_matrix_3(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=np.int64)
    for actual, pred in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= actual <= 2 and 0 <= pred <= 2:
            cm[actual, pred] += 1
    return cm


def classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    cm = confusion_matrix_3(y_true, y_pred)
    total = int(cm.sum())
    correct = int(np.trace(cm))
    rows: Dict[str, Dict[str, float]] = {}
    names = {0: "Down", 1: "Unchanged", 2: "Up"}

    precisions = []
    recalls = []
    f1s = []
    supports = []
    for label in [0, 1, 2]:
        tp = float(cm[label, label])
        support = float(cm[label, :].sum())
        predicted = float(cm[:, label].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows[names[label]] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": int(support),
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    support_arr = np.asarray(supports, dtype=np.float64)
    weight_den = float(support_arr.sum()) or 1.0
    rows["accuracy"] = correct / total if total else 0.0
    rows["macro avg"] = {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1-score": float(np.mean(f1s)),
        "support": total,
    }
    rows["weighted avg"] = {
        "precision": float(np.dot(precisions, support_arr) / weight_den),
        "recall": float(np.dot(recalls, support_arr) / weight_den),
        "f1-score": float(np.dot(f1s, support_arr) / weight_den),
        "support": total,
    }
    return rows


def classification_report_text(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    report = classification_report_dict(y_true, y_pred)
    lines = [f"{'':>12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"]
    for name in ["Down", "Unchanged", "Up"]:
        row = report[name]
        lines.append(
            f"{name:>12} {row['precision']:>10.4f} {row['recall']:>10.4f} "
            f"{row['f1-score']:>10.4f} {row['support']:>10}"
        )
    accuracy = report["accuracy"]
    support = report["weighted avg"]["support"]
    lines.append(f"{'accuracy':>12} {'':>10} {'':>10} {accuracy:>10.4f} {support:>10}")
    for name in ["macro avg", "weighted avg"]:
        row = report[name]
        lines.append(
            f"{name:>12} {row['precision']:>10.4f} {row['recall']:>10.4f} "
            f"{row['f1-score']:>10.4f} {row['support']:>10}"
        )
    return "\n".join(lines)


def direction_purity(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float | None]:
    cm = confusion_matrix_3(y_true, y_pred)
    down_den = int(cm[0, 0] + cm[2, 0])
    up_den = int(cm[2, 2] + cm[0, 2])
    return {
        "pred_down_purity_excluding_flat": float(cm[0, 0] / down_den) if down_den else None,
        "pred_up_purity_excluding_flat": float(cm[2, 2] / up_den) if up_den else None,
    }


def classification_summary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    cm = confusion_matrix_3(y_true, y_pred)
    report = classification_report_dict(y_true, y_pred)
    return {
        "classification_report": report,
        "confusion_matrix": cm.astype(int).tolist(),
        "direction_purity": direction_purity(y_true, y_pred),
    }


def write_legacy_report(
    *,
    metrics_dir: Path,
    label: str,
    param_preset: str,
    train_labels: np.ndarray,
    train_pred: np.ndarray,
    val_labels: np.ndarray,
    val_pred: np.ndarray,
) -> Dict[str, object]:
    train_summary = classification_summary(train_labels, train_pred)
    val_summary = classification_summary(val_labels, val_pred)
    payload = {
        "label": label,
        "param_preset": param_preset,
        "train": train_summary,
        "val": val_summary,
    }
    (metrics_dir / f"{label}_classification.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def format_block(split_name: str, labels: np.ndarray, pred: np.ndarray, summary: Dict[str, object]) -> str:
        purity = summary["direction_purity"]
        return "\n".join([
            f"{split_name}:",
            classification_report_text(labels, pred),
            str(np.asarray(summary["confusion_matrix"], dtype=int)),
            f"0 accu: {purity['pred_down_purity_excluding_flat']}",
            f"2 accu: {purity['pred_up_purity_excluding_flat']}",
        ])

    report_text = "\n\n".join([
        f"label={label}",
        f"param_preset={param_preset}",
        format_block("train", train_labels, train_pred, train_summary),
        format_block("val", val_labels, val_pred, val_summary),
    ])
    (metrics_dir / f"{label}_legacy_report.txt").write_text(report_text + "\n", encoding="utf-8")
    return payload


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
    feature_set: str,
    storage_mode: str,
    output_dir: Path,
    cache_dir: Path,
    batch_size: int,
    params: Dict[str, object],
    param_preset: str,
    num_boost_round: int,
    early_stopping_rounds: int,
    default_threshold: float,
    threshold_metric: str,
    pnl_baseline: float,
    ratio_values: Sequence[float],
    ratio_select_metric: str,
    drop_limit_samples: bool,
    max_bin: int | None,
    gpu_batch_mode: str,
    cache_host_ratio: float | None,
    max_quantile_batches: int | None,
    min_cache_page_bytes: int | None,
    reuse_buffers: bool,
    cleanup_cache: bool,
    use_class_weight: bool,
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
        train_buffers: List[str] = []
        val_buffers: List[str] = []
        train_sidecar: StreamSidecar | None = None
        val_sidecar: StreamSidecar | None = None

        if storage_mode == "buffer":
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
        elif storage_mode == "stream":
            if reuse_buffers:
                tprint(f"--reuse-buffers is ignored for stream storage mode on {label}")
            tprint(f"Preparing stream sidecars for {label}")
            train_sidecar = process_files_to_stream_sidecar(
                files=train_files,
                label=label,
                builder=builder,
                cache_dir=cache_dir,
                split_name="train",
                drop_limit_samples=drop_limit_samples,
            )
            val_sidecar = process_files_to_stream_sidecar(
                files=val_files,
                label=label,
                builder=builder,
                cache_dir=cache_dir,
                split_name="val",
                drop_limit_samples=drop_limit_samples,
            )
            train_counts = train_sidecar.label_counts
            train_samples = train_sidecar.samples
            val_samples = val_sidecar.samples
        else:
            raise ValueError(f"Unsupported storage mode: {storage_mode}")

        weights = class_weights_from_counts(train_counts) if use_class_weight else {}
        tprint(f"{label} train label counts: {train_counts}")
        tprint(f"{label} class weighting: {use_class_weight}")
        if weights:
            tprint(f"{label} class weights: {weights}")

        is_cuda_training = str(params.get("device", "")).startswith("cuda")
        use_gpu_data = is_cuda_training and gpu_batch_mode == "cupy"
        if is_cuda_training:
            tprint(f"{label} GPU training enabled; iterator batch mode={gpu_batch_mode}")
        if storage_mode == "buffer":
            train_mean_abs_diff = mean_abs_price_diff(train_buffers) if use_profit_weight else None
        else:
            assert train_sidecar is not None
            train_mean_abs_diff = mean_abs_price_diff_array(train_sidecar.price_path) if use_profit_weight else None
        if use_profit_weight:
            tprint(
                f"{label} profit weighting enabled: "
                f"base={profit_weight_base} alpha={profit_weight_alpha} "
                f"clip=[{profit_weight_min}, {profit_weight_max}] "
                f"mean_abs_diff={train_mean_abs_diff:.8f}"
            )
        if storage_mode == "buffer":
            train_iter = BufferDataIter(
                train_buffers,
                weights,
                cache_prefix=str(cache_dir / f"extmem_train_{label}"),
                min_cache_page_bytes=min_cache_page_bytes,
                use_gpu=use_gpu_data,
                use_profit_weight=use_profit_weight,
                profit_weight_base=profit_weight_base,
                profit_weight_alpha=profit_weight_alpha,
                profit_weight_min=profit_weight_min,
                profit_weight_max=profit_weight_max,
                profit_weight_mean_abs_diff=train_mean_abs_diff,
            )
        else:
            train_iter = SnapshotCSVDataIter(
                files=train_files,
                label=label,
                builder=builder,
                available_features=available_features,
                batch_size=batch_size,
                drop_limit_samples=drop_limit_samples,
                class_weights=weights,
                cache_prefix=str(cache_dir / f"extmem_train_{label}"),
                min_cache_page_bytes=min_cache_page_bytes,
                use_gpu=use_gpu_data,
                use_profit_weight=use_profit_weight,
                profit_weight_base=profit_weight_base,
                profit_weight_alpha=profit_weight_alpha,
                profit_weight_min=profit_weight_min,
                profit_weight_max=profit_weight_max,
                profit_weight_mean_abs_diff=train_mean_abs_diff,
                desc=f"stream-train:{label}",
            )
        extmem_kwargs: Dict[str, object] = {}
        if max_bin is not None:
            extmem_kwargs["max_bin"] = max_bin
        if cache_host_ratio is not None and use_gpu_data:
            extmem_kwargs["cache_host_ratio"] = cache_host_ratio
        if max_quantile_batches is not None:
            extmem_kwargs["max_quantile_batches"] = max_quantile_batches
        if cache_host_ratio is not None and not use_gpu_data:
            tprint(f"{label} cache_host_ratio ignored because iterator batch mode is {gpu_batch_mode}")
        tprint(
            f"{label} ExtMemQuantileDMatrix controls: "
            f"{extmem_kwargs if extmem_kwargs else 'xgboost defaults'}; "
            f"min_cache_page_bytes={min_cache_page_bytes if min_cache_page_bytes is not None else 'xgboost default'}"
        )
        dtrain = xgb.ExtMemQuantileDMatrix(train_iter, **extmem_kwargs)
        if storage_mode == "buffer":
            val_iter = BufferDataIter(
                val_buffers,
                weights,
                cache_prefix=str(cache_dir / f"extmem_val_{label}"),
                min_cache_page_bytes=min_cache_page_bytes,
                use_gpu=use_gpu_data,
                use_profit_weight=False,
            )
        else:
            val_iter = SnapshotCSVDataIter(
                files=val_files,
                label=label,
                builder=builder,
                available_features=available_features,
                batch_size=batch_size,
                drop_limit_samples=drop_limit_samples,
                class_weights=weights,
                cache_prefix=str(cache_dir / f"extmem_val_{label}"),
                min_cache_page_bytes=min_cache_page_bytes,
                use_gpu=use_gpu_data,
                use_profit_weight=False,
                desc=f"stream-val:{label}",
            )
        dval = xgb.ExtMemQuantileDMatrix(val_iter, ref=dtrain, **extmem_kwargs)

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

        if storage_mode == "buffer":
            train_pred_proba, train_y_true, _ = load_eval_arrays(train_buffers, booster)
            val_pred_proba, y_true, price_diff = load_eval_arrays(val_buffers, booster)
        else:
            assert train_sidecar is not None and val_sidecar is not None
            train_y_true = np.load(train_sidecar.labels_path)
            y_true = np.load(val_sidecar.labels_path)
            price_diff = np.load(val_sidecar.price_path)
            train_pred_proba = booster.predict(dtrain)
            val_pred_proba = booster.predict(dval)
        train_standard_pred = train_pred_proba.argmax(axis=1)
        val_standard_pred = val_pred_proba.argmax(axis=1)

        metrics = add_official_scores(
            evaluate_thresholds(val_pred_proba, y_true, price_diff, DEFAULT_THRESHOLDS),
            pnl_baseline,
        )
        selected_threshold = select_threshold(metrics, threshold_metric, default_threshold)
        tprint(f"{label} selected threshold={selected_threshold} by {threshold_metric}")
        metrics_dir = output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / f"{label}_thresholds.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        ratio_metrics = add_official_scores(
            evaluate_ratio_grid(val_pred_proba, y_true, price_diff, ratio_values),
            pnl_baseline,
        )
        ratio_best = max(ratio_metrics, key=lambda row: row[ratio_select_metric]) if ratio_metrics else None
        (metrics_dir / f"{label}_ratio_grid.json").write_text(json.dumps(ratio_metrics, indent=2), encoding="utf-8")
        if ratio_best is not None:
            (metrics_dir / f"{label}_ratio_best.json").write_text(json.dumps(ratio_best, indent=2), encoding="utf-8")
            tprint(
                f"{label} best ratio by {ratio_select_metric}: "
                f"down={ratio_best['down_ratio']} up={ratio_best['up_ratio']} "
                f"precision={ratio_best['precision']:.4f} recall={ratio_best['recall']:.4f} "
                f"f05={ratio_best['f05']:.4f} signal={ratio_best['signal_pct']:.4f}"
            )
        write_legacy_report(
            metrics_dir=metrics_dir,
            label=label,
            param_preset=param_preset,
            train_labels=train_y_true,
            train_pred=train_standard_pred,
            val_labels=y_true,
            val_pred=val_standard_pred,
        )

        best_iteration = getattr(booster, "best_iteration", None)
        result = LabelTrainResult(
            label=label,
            feature_set=feature_set,
            storage_mode=storage_mode,
            model_path=str(model_path),
            threshold=selected_threshold,
            train_samples=train_samples,
            val_samples=val_samples,
            train_label_counts={str(k): int(v) for k, v in train_counts.items()},
            best_iteration=int(best_iteration) if best_iteration is not None else None,
            param_preset=param_preset,
            max_bin=max_bin,
            gpu_batch_mode=gpu_batch_mode,
            cache_host_ratio=cache_host_ratio,
            max_quantile_batches=max_quantile_batches,
            min_cache_page_bytes=min_cache_page_bytes,
            class_weighting=use_class_weight,
            profit_weighting=use_profit_weight,
            profit_weight_base=profit_weight_base,
            profit_weight_alpha=profit_weight_alpha,
            profit_weight_min=profit_weight_min,
            profit_weight_max=profit_weight_max,
        )
        del dtrain, dval, train_iter, val_iter, booster
        gc.collect()
        return result
    finally:
        if cleanup_cache:
            removed_count, removed_bytes = cleanup_label_cache(cache_dir, label)
            if removed_count:
                tprint(f"Cleaned cache for {label}: files={removed_count} size={removed_bytes / (1024 ** 3):.2f} GiB")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train XGBoost models for reproducing the previous report")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/report_pdf_reproduction"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    parser.add_argument("--labels", nargs="*", default=LABEL_COLUMNS, choices=LABEL_COLUMNS)
    parser.add_argument("--split-mode", choices=["date", "index"], default="index")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation file ratio when --split-mode index")
    parser.add_argument("--val-start-date", type=int, default=63)
    parser.add_argument("--max-files", type=int, default=None, help="Optional smoke-test limit over sorted snapshot files")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--num-boost-round", type=int, default=2000)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--default-threshold", type=float, default=0.88)
    parser.add_argument(
        "--feature-set",
        choices=["pdf_report", "previous_code"],
        default="pdf_report",
        help="pdf_report follows only PDF-described features; previous_code uses the richer old-code feature set.",
    )
    parser.add_argument(
        "--pdf-levels",
        choices=["1-5", "1-3"],
        default="1-5",
        help="Unified i-level range for all PDF i-based features, including weighted_ab.",
    )
    parser.add_argument("--param-preset", choices=sorted(PARAM_PRESETS), default="report_pdf")
    parser.add_argument(
        "--min-history",
        type=int,
        default=None,
        help="Override the feature builder's minimum local row index. Use 0 for commit 8855d4b/a94734a behavior.",
    )
    parser.add_argument(
        "--storage-mode",
        choices=["buffer", "stream"],
        default="buffer",
        help="buffer writes intermediate DMatrix files; stream builds XGBoost external-memory data directly from CSV batches.",
    )
    parser.add_argument(
        "--class-weight-mode",
        choices=["preset", "on", "off"],
        default="preset",
        help="Use class weights according to the parameter preset, force on, or force off.",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=["official_score_est", "f05", "total_pnl", "avg_pnl", "precision", "default"],
        default="default",
        help="Metric used to select the saved threshold for each label",
    )
    parser.add_argument("--ratio-min", type=float, default=DEFAULT_RATIO_MIN)
    parser.add_argument("--ratio-max", type=float, default=DEFAULT_RATIO_MAX)
    parser.add_argument("--ratio-step", type=float, default=DEFAULT_RATIO_STEP)
    parser.add_argument(
        "--ratio-values",
        default=None,
        help="Optional comma-separated ratio values. Overrides --ratio-min/max/step.",
    )
    parser.add_argument(
        "--ratio-select-metric",
        choices=["official_score_est", "f05", "total_pnl", "avg_pnl", "precision", "recall", "signal_pct"],
        default="f05",
        help="Metric used to identify the best ratio pair in *_ratio_best.json.",
    )
    parser.add_argument("--pnl-baseline", type=float, default=DEFAULT_PNL_BASELINE)
    parser.add_argument("--nthread", type=int, default=None, help="Optional XGBoost nthread override")
    parser.add_argument("--max-bin", type=int, default=None, help="Optional XGBoost/QuantileDMatrix max_bin override")
    parser.add_argument(
        "--gpu-batch-mode",
        choices=["numpy", "cupy"],
        default="cupy",
        help="When --device cuda is used, feed DataIter batches as CPU NumPy arrays or pre-copy them to GPU CuPy arrays.",
    )
    parser.add_argument(
        "--cache-host-ratio",
        type=float,
        default=None,
        help="Optional ExtMemQuantileDMatrix cache_host_ratio override for GPU external memory.",
    )
    parser.add_argument(
        "--max-quantile-batches",
        type=int,
        default=None,
        help="Optional ExtMemQuantileDMatrix max_quantile_batches override.",
    )
    parser.add_argument(
        "--min-cache-page-bytes",
        type=int,
        default=None,
        help="Optional DataIter min_cache_page_bytes override; 0 disables GPU page concatenation.",
    )
    parser.add_argument("--tree-method", default=None, help="Optional XGBoost tree_method override")
    parser.add_argument("--device", default=None, help="Optional XGBoost device override, for example 'cpu' or 'cuda'")
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

    builder = build_feature_builder(args.feature_set, args.pdf_levels)
    if args.min_history is not None:
        builder.min_history = args.min_history
    available_features = determine_available_features(builder, train_files)
    spec_path = save_feature_spec(output_dir, builder, available_features)
    tprint(f"Feature spec saved: {spec_path}")
    tprint(
        f"Feature set={args.feature_set}, pdf_levels={getattr(builder, 'pdf_level_mode', None)}, "
        f"vector_mode={getattr(builder, 'vector_mode', 'unknown')}, "
        f"min_history={getattr(builder, 'min_history', 'unknown')}"
    )
    tprint(f"Available base features={len(available_features)}, final vector={len(builder.final_feature_names)}")

    preset = PARAM_PRESETS[args.param_preset]
    if args.class_weight_mode == "preset":
        use_class_weight = bool(preset["use_class_weight"])
    else:
        use_class_weight = args.class_weight_mode == "on"

    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
    }
    params.update(preset["params"])
    if args.nthread is not None:
        params["nthread"] = args.nthread
    if args.max_bin is not None:
        params["max_bin"] = args.max_bin
    if args.tree_method is not None:
        params["tree_method"] = args.tree_method
    if args.device is not None:
        params["device"] = args.device
    tprint(f"Parameter preset: {args.param_preset} - {preset['description']}")
    tprint(f"Storage mode: {args.storage_mode}")
    tprint(f"Class weighting resolved to: {use_class_weight}")
    tprint(f"XGBoost params: {params}")
    if str(params.get("device", "")).startswith("cuda"):
        tprint(f"GPU batch mode: {args.gpu_batch_mode}")
    else:
        tprint("GPU batch mode: inactive")
    extmem_controls = {
        key: value
        for key, value in {
            "max_bin": args.max_bin,
            "cache_host_ratio": args.cache_host_ratio,
            "max_quantile_batches": args.max_quantile_batches,
            "min_cache_page_bytes": args.min_cache_page_bytes,
        }.items()
        if value is not None
    }
    tprint(f"External-memory controls: {extmem_controls if extmem_controls else 'xgboost defaults'}")
    ratio_values = make_ratio_values(args.ratio_min, args.ratio_max, args.ratio_step, args.ratio_values)
    tprint(
        f"Ratio grid: {len(ratio_values)} values, {len(ratio_values) ** 2} pairs, "
        f"min={ratio_values[0]} max={ratio_values[-1]} select={args.ratio_select_metric}"
    )

    results = []
    thresholds = {}
    for label in args.labels:
        result = train_one_label(
            label=label,
            train_files=train_files,
            val_files=val_files,
            builder=builder,
            available_features=available_features,
            feature_set=args.feature_set,
            storage_mode=args.storage_mode,
            output_dir=output_dir,
            cache_dir=cache_dir,
            batch_size=args.batch_size,
            params=params,
            param_preset=args.param_preset,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
            default_threshold=args.default_threshold,
            threshold_metric=args.threshold_metric,
            pnl_baseline=args.pnl_baseline,
            ratio_values=ratio_values,
            ratio_select_metric=args.ratio_select_metric,
            drop_limit_samples=not args.keep_limit_samples,
            max_bin=args.max_bin,
            gpu_batch_mode=args.gpu_batch_mode,
            cache_host_ratio=args.cache_host_ratio,
            max_quantile_batches=args.max_quantile_batches,
            min_cache_page_bytes=args.min_cache_page_bytes,
            reuse_buffers=args.reuse_buffers,
            cleanup_cache=args.cleanup_cache,
            use_class_weight=use_class_weight,
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
