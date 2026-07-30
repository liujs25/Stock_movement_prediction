"""Build a flat platform submission zip from trained XGBoost artifacts."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from feature_builder import LABEL_COLUMNS


REQUIRED_ARTIFACTS = [
    "feature_spec.json",
    "thresholds.json",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build stock movement submission zip")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/submission_package"))
    parser.add_argument("--zip-path", type=Path, default=Path("artifacts/submission_xgboost.zip"))
    return parser


def copy_required_files(project_root: Path, artifacts_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    copies = [
        (project_root / "src" / "Predictor.py", output_dir / "Predictor.py"),
        (project_root / "src" / "feature_builder.py", output_dir / "feature_builder.py"),
        (project_root / "config.json", output_dir / "config.json"),
        (project_root / "requirements_submission.txt", output_dir / "requirements.txt"),
    ]
    for source, target in copies:
        if not source.exists():
            raise FileNotFoundError(f"Missing required source file: {source}")
        shutil.copy2(source, target)

    for artifact_name in REQUIRED_ARTIFACTS:
        source = artifacts_dir / artifact_name
        if not source.exists():
            raise FileNotFoundError(f"Missing trained artifact: {source}")
        shutil.copy2(source, output_dir / artifact_name)

    models_dir = artifacts_dir / "models"
    for label in LABEL_COLUMNS:
        source = models_dir / f"model_{label}.json"
        if not source.exists():
            raise FileNotFoundError(f"Missing trained model for {label}: {source}")
        shutil.copy2(source, output_dir / source.name)


def write_zip(output_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(output_dir.iterdir()):
            if path.is_file():
                zf.write(path, arcname=path.name)


def validate_flat_package(output_dir: Path) -> None:
    nested = [path for path in output_dir.iterdir() if path.is_dir()]
    if nested:
        raise ValueError(f"Submission package must be flat; found directories: {nested}")

    required = {
        "Predictor.py",
        "feature_builder.py",
        "config.json",
        "requirements.txt",
        "feature_spec.json",
        "thresholds.json",
        *{f"model_{label}.json" for label in LABEL_COLUMNS},
    }
    present = {path.name for path in output_dir.iterdir() if path.is_file()}
    missing = sorted(required - present)
    if missing:
        raise FileNotFoundError(f"Submission package missing files: {missing}")


def main() -> None:
    args = build_arg_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    artifacts_dir = args.artifacts_dir
    if not artifacts_dir.is_absolute():
        artifacts_dir = project_root / artifacts_dir
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    zip_path = args.zip_path
    if not zip_path.is_absolute():
        zip_path = project_root / zip_path

    if output_dir.exists():
        shutil.rmtree(output_dir)
    copy_required_files(project_root, artifacts_dir, output_dir)
    validate_flat_package(output_dir)
    write_zip(output_dir, zip_path)
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Built {zip_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
