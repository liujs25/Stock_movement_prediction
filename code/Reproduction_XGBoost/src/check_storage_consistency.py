"""Check that buffer and stream storage modes build identical feature batches."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import xgboost as xgb

from feature_builder import LABEL_COLUMNS, build_feature_builder
from train import (
    DEFAULT_DATA_DIR,
    determine_available_features,
    iter_feature_batches,
    process_files_to_buffers,
    process_files_to_stream_sidecar,
    read_meta,
    scan_snapshot_files,
)


def dmatrix_to_dense(dmatrix: xgb.DMatrix) -> np.ndarray:
    data = dmatrix.get_data()
    if hasattr(data, "toarray"):
        data = data.toarray()
    return np.asarray(data, dtype=np.float32)


def load_buffer_payload(buffer_files: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = []
    labels = []
    prices = []
    for buffer_file in buffer_files:
        dmatrix = xgb.DMatrix(buffer_file)
        features.append(dmatrix_to_dense(dmatrix))
        labels.append(dmatrix.get_label().astype(np.int32))
        prices.append(np.load(buffer_file.replace(".buffer", ".price.npy")).astype(np.float32))
    return np.vstack(features), np.concatenate(labels), np.concatenate(prices)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare stream batches with saved buffer batches")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--feature-set", choices=["pdf_report", "previous_code"], default="pdf_report")
    parser.add_argument("--pdf-levels", choices=["1-5", "1-3"], default="1-3")
    parser.add_argument("--label", choices=LABEL_COLUMNS, default="label_5")
    parser.add_argument("--max-files", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--cache-parent", type=Path, default=Path("artifacts/consistency_cache"))
    parser.add_argument("--keep-limit-samples", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    data_dir = args.data_dir
    if not data_dir.is_absolute():
        cwd_candidate = Path.cwd() / data_dir
        project_candidate = project_root / data_dir
        data_dir = cwd_candidate if cwd_candidate.exists() else project_candidate
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    files = scan_snapshot_files(data_dir)[:args.max_files]
    if not files:
        raise ValueError(f"No snapshot files found in {data_dir}")

    builder = build_feature_builder(args.feature_set, args.pdf_levels)
    available_features = determine_available_features(builder, files)
    drop_limit_samples = not args.keep_limit_samples

    cache_parent = args.cache_parent if args.cache_parent.is_absolute() else project_root / args.cache_parent
    cache_parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(tempfile.mkdtemp(prefix="storage_consistency_", dir=cache_parent))
    try:
        stream_batch = next(iter_feature_batches(
            files=files,
            label=args.label,
            builder=builder,
            available_features=available_features,
            batch_size=args.batch_size,
            drop_limit_samples=drop_limit_samples,
            desc="stream-check",
        ))
        meta_path, buffer_counts, buffer_samples = process_files_to_buffers(
            files=files,
            label=args.label,
            builder=builder,
            available_features=available_features,
            cache_dir=cache_dir,
            split_name="check",
            batch_size=args.batch_size,
            drop_limit_samples=drop_limit_samples,
        )
        buffer_features, buffer_labels, buffer_prices = load_buffer_payload(read_meta(meta_path))
        sidecar = process_files_to_stream_sidecar(
            files=files,
            label=args.label,
            builder=builder,
            cache_dir=cache_dir,
            split_name="sidecar_check",
            drop_limit_samples=drop_limit_samples,
        )
        sidecar_labels = np.load(sidecar.labels_path)
        sidecar_prices = np.load(sidecar.price_path)

        first_n = int(stream_batch.labels.size)
        checks = {
            "first_batch_features_equal": bool(np.array_equal(stream_batch.features, buffer_features[:first_n])),
            "first_batch_labels_equal": bool(np.array_equal(stream_batch.labels, buffer_labels[:first_n])),
            "first_batch_prices_equal": bool(np.array_equal(stream_batch.price_diffs, buffer_prices[:first_n])),
            "sidecar_labels_equal": bool(np.array_equal(sidecar_labels, buffer_labels)),
            "sidecar_prices_equal": bool(np.array_equal(sidecar_prices, buffer_prices)),
            "buffer_samples": int(buffer_samples),
            "sidecar_samples": int(sidecar.samples),
            "buffer_label_counts": {str(key): int(value) for key, value in buffer_counts.items()},
            "sidecar_label_counts": {str(key): int(value) for key, value in sidecar.label_counts.items()},
            "features": int(buffer_features.shape[1]),
        }
        checks["passed"] = all(
            value for key, value in checks.items()
            if key.endswith("_equal")
        ) and checks["buffer_samples"] == checks["sidecar_samples"]
        print(json.dumps(checks, indent=2))
        if not checks["passed"]:
            raise SystemExit(1)
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
