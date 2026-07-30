"""Smoke test a built flat submission package against local raw data."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd


DEFAULT_DATA_DIR = Path("../EDA/raw data/FBDQA2021A_MMP_Challenge/data")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test the built submission package")
    parser.add_argument("--package-dir", type=Path, default=Path("artifacts/submission_package"))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--batch", type=int, default=32)
    return parser


def load_predictor(package_dir: Path):
    sys.path.insert(0, str(package_dir))
    predictor_path = package_dir / "Predictor.py"
    spec = importlib.util.spec_from_file_location("submission_predictor", predictor_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Predictor from {predictor_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Predictor()


def collect_windows(data_dir: Path, batch: int):
    windows = []
    for csv_path in sorted(data_dir.glob("snapshot_*.csv")):
        df = pd.read_csv(csv_path)
        if len(df) >= 100:
            windows.append(df.iloc[:100].copy())
        if len(windows) >= batch:
            break
    if len(windows) < batch:
        raise ValueError(f"Only collected {len(windows)} windows; requested {batch}")
    return windows


def main() -> None:
    args = build_arg_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    package_dir = args.package_dir if args.package_dir.is_absolute() else project_root / args.package_dir
    data_dir = args.data_dir
    if not data_dir.is_absolute():
        cwd_candidate = Path.cwd() / data_dir
        project_candidate = project_root / data_dir
        data_dir = cwd_candidate if cwd_candidate.exists() else project_candidate

    predictor = load_predictor(package_dir)
    windows = collect_windows(data_dir, args.batch)
    predictions = predictor.predict(windows)

    assert isinstance(predictions, list)
    assert len(predictions) == args.batch
    for row in predictions:
        assert isinstance(row, list)
        assert len(row) == 5
        assert all(value in {0, 1, 2} for value in row)
    print(f"Smoke test passed for batch={args.batch}")


if __name__ == "__main__":
    main()
