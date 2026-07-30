"""Compare reproduced classification outputs with Final_Report.pdf matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare reproduced metrics with the PDF reference matrices")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/report_pdf_reproduction"))
    parser.add_argument("--reference", type=Path, default=Path("reference/final_report_confusion_matrices.json"))
    return parser


def accuracy(cm: np.ndarray) -> float:
    total = cm.sum()
    return float(np.trace(cm) / total) if total else 0.0


def main() -> None:
    args = build_arg_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    artifacts_dir = args.artifacts_dir if args.artifacts_dir.is_absolute() else project_root / args.artifacts_dir
    reference_path = args.reference if args.reference.is_absolute() else project_root / args.reference

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    metrics_dir = artifacts_dir / "metrics"
    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics directory not found: {metrics_dir}")

    for label, expected_splits in reference.items():
        actual_path = metrics_dir / f"{label}_classification.json"
        if not actual_path.exists():
            print(f"{label}: missing {actual_path.name}")
            continue
        actual = json.loads(actual_path.read_text(encoding="utf-8"))
        print(f"\n{label}")
        for split_name, expected_matrix in expected_splits.items():
            expected_cm = np.asarray(expected_matrix, dtype=np.int64)
            actual_cm = np.asarray(actual[split_name]["confusion_matrix"], dtype=np.int64)
            diff = actual_cm - expected_cm
            print(
                f"  {split_name}: "
                f"expected_acc={accuracy(expected_cm):.6f} "
                f"actual_acc={accuracy(actual_cm):.6f} "
                f"sample_diff={int(actual_cm.sum() - expected_cm.sum())} "
                f"abs_matrix_diff={int(np.abs(diff).sum())}"
            )


if __name__ == "__main__":
    main()

