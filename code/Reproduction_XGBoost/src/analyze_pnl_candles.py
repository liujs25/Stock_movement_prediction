"""Analyze platform-style PnL and draw candlestick charts for submissions.

The script can run in three modes:
- model: load a flat submission package and call Predictor.predict on rolling windows
- label: use the ground-truth labels as oracle trade signals
- signals: recompute summaries and charts from an existing signals CSV
"""

from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import json
import math
import re
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


LABELS = ["label_5", "label_10", "label_20", "label_40", "label_60"]
DEFAULT_DATA_DIR = Path("../EDA/raw data/FBDQA2021A_MMP_Challenge/data")


@dataclass(frozen=True)
class SnapshotFile:
    path: Path
    sym: int
    date: int
    session: str


def parse_snapshot_path(path: Path) -> SnapshotFile | None:
    match = re.match(r"snapshot_sym(\d+)_date(\d+)_(am|pm)\.csv$", path.name)
    if not match:
        return None
    return SnapshotFile(path=path, sym=int(match.group(1)), date=int(match.group(2)), session=match.group(3))


def resolve_path(path: Path, project_root: Path) -> Path:
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return project_root / path


def scan_files(
    data_dir: Path,
    csv_files: Sequence[Path],
    sym: int | None,
    date_start: int | None,
    date_end: int | None,
    session: str | None,
    max_files: int | None,
) -> list[Path]:
    if csv_files:
        files = [path if path.is_absolute() else Path.cwd() / path for path in csv_files]
        missing = [str(path) for path in files if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing CSV files: {missing}")
        return files[:max_files] if max_files else files

    parsed = []
    for path in data_dir.glob("snapshot_sym*_date*_*.csv"):
        item = parse_snapshot_path(path)
        if item is None:
            continue
        if sym is not None and item.sym != sym:
            continue
        if date_start is not None and item.date < date_start:
            continue
        if date_end is not None and item.date > date_end:
            continue
        if session is not None and item.session != session:
            continue
        parsed.append(item)
    parsed.sort(key=lambda item: (item.date, 0 if item.session == "am" else 1, item.sym))
    files = [item.path for item in parsed]
    return files[:max_files] if max_files else files


def load_predictor(package_dir: Path | None, zip_path: Path | None):
    if package_dir is None and zip_path is None:
        raise ValueError("--package-dir or --zip-path is required in model mode")

    temp_dir = None
    if zip_path is not None:
        temp_dir = tempfile.TemporaryDirectory(prefix="submission_pkg_")
        package_dir = Path(temp_dir.name)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(package_dir)
    assert package_dir is not None

    predictor_path = package_dir / "Predictor.py"
    if not predictor_path.exists():
        raise FileNotFoundError(f"Cannot find Predictor.py in {package_dir}")

    sys.path.insert(0, str(package_dir))
    spec = importlib.util.spec_from_file_location("submission_predictor_for_pnl", predictor_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import Predictor from {predictor_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        if exc.name == "xgboost":
            raise RuntimeError(
                "xgboost is not installed in this Python environment. "
                "Use --mode label locally, or run --mode model on the training server/venv."
            ) from exc
        raise
    return module.Predictor(), temp_dir


def label_horizon(label: str) -> int:
    return int(label.split("_")[1])


def time_to_seconds(value) -> int:
    try:
        hh, mm, ss = str(value).split(":")
        return int(hh) * 3600 + int(mm) * 60 + int(float(ss))
    except Exception:
        return 0


def session_from_path(path: Path) -> str:
    parsed = parse_snapshot_path(path)
    return parsed.session if parsed is not None else ""


def endpoint_indices(length: int, window_size: int, max_horizon: int, stride: int, max_windows: int | None) -> list[int]:
    start = max(0, window_size - 1)
    stop = length - max_horizon
    if stop <= start:
        return []
    indices = list(range(start, stop, stride))
    return indices[:max_windows] if max_windows else indices


def model_predictions(
    predictor,
    group: pd.DataFrame,
    labels: Sequence[str],
    indices: Sequence[int],
    window_size: int,
    batch_size: int,
) -> dict[str, list[int]]:
    label_positions = {label: LABELS.index(label) for label in labels}
    out = {label: [] for label in labels}
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        windows = [group.iloc[max(0, idx - window_size + 1) : idx + 1].copy() for idx in batch_indices]
        rows = predictor.predict(windows)
        if len(rows) != len(batch_indices):
            raise ValueError(f"Predictor returned {len(rows)} rows for {len(batch_indices)} windows")
        for row in rows:
            for label, pos in label_positions.items():
                out[label].append(int(row[pos]))
    return out


def label_predictions(group: pd.DataFrame, labels: Sequence[str], indices: Sequence[int]) -> dict[str, list[int]]:
    out = {}
    for label in labels:
        out[label] = [int(group.iloc[idx][label]) if pd.notna(group.iloc[idx][label]) else 1 for idx in indices]
    return out


def analyze_file(
    path: Path,
    mode: str,
    labels: Sequence[str],
    predictor,
    window_size: int,
    stride: int,
    max_windows: int | None,
    prediction_batch: int,
) -> list[dict[str, object]]:
    df = pd.read_csv(path)
    df = df.sort_values(["sym", "date", "time"], kind="mergesort").reset_index(drop=True)
    max_horizon = max(label_horizon(label) for label in labels)
    rows: list[dict[str, object]] = []

    for (sym, date), group in df.groupby(["sym", "date"], sort=False):
        group = group.reset_index(drop=True)
        indices = endpoint_indices(len(group), window_size, max_horizon, stride, max_windows)
        if not indices:
            continue
        if mode == "model":
            pred_by_label = model_predictions(predictor, group, labels, indices, window_size, prediction_batch)
        elif mode == "label":
            pred_by_label = label_predictions(group, labels, indices)
        else:
            raise ValueError(f"Unsupported mode for direct analysis: {mode}")

        for label in labels:
            horizon = label_horizon(label)
            future_mid = group["n_midprice"].shift(-horizon)
            for pos, idx in enumerate(indices):
                pred = int(pred_by_label[label][pos])
                actual_raw = group.iloc[idx][label] if label in group.columns else 1
                actual = int(actual_raw) if pd.notna(actual_raw) else 1
                entry_mid = float(group.iloc[idx]["n_midprice"])
                exit_mid = float(future_mid.iloc[idx])
                price_diff = exit_mid - entry_mid
                pnl = price_diff if pred == 2 else -price_diff if pred == 0 else 0.0
                rows.append(
                    {
                        "source_file": path.name,
                        "sym": int(sym),
                        "date": int(date),
                        "session": session_from_path(path),
                        "row_idx": int(idx),
                        "time": group.iloc[idx]["time"] if "time" in group.columns else "",
                        "label": label,
                        "horizon": horizon,
                        "pred": pred,
                        "actual": actual,
                        "entry_mid": entry_mid,
                        "exit_mid": exit_mid,
                        "price_diff": float(price_diff),
                        "pnl": float(pnl),
                        "correct": bool(pred in {0, 2} and pred == actual),
                    }
                )
    return rows


def summarize_signals(rows: Sequence[dict[str, object]], labels: Sequence[str], pnl_baseline: float) -> list[dict[str, object]]:
    summary = []
    for label in labels:
        label_rows = [row for row in rows if row["label"] == label]
        signal_rows = [row for row in label_rows if int(row["pred"]) in {0, 2}]
        actual_moves = [row for row in label_rows if int(row["actual"]) in {0, 2}]
        correct = [row for row in signal_rows if bool(row["correct"])]
        pnls = [float(row["pnl"]) for row in signal_rows]
        positive = [pnl for pnl in pnls if pnl > 0]
        negative = [pnl for pnl in pnls if pnl <= 0]

        evaluated = len(label_rows)
        signals = len(signal_rows)
        precision = len(correct) / signals if signals else 0.0
        recall = len(correct) / len(actual_moves) if actual_moves else 0.0
        f05 = (1.25 * precision * recall) / (0.25 * precision + recall + 1e-10) if precision or recall else 0.0
        total_pnl = sum(pnls)
        avg_pnl = total_pnl / signals if signals else 0.0
        delta = avg_pnl - pnl_baseline
        summary.append(
            {
                "label": label,
                "evaluated": evaluated,
                "actual_moves": len(actual_moves),
                "signals": signals,
                "signal_pct": signals / evaluated if evaluated else 0.0,
                "precision": precision,
                "recall": recall,
                "f05": f05,
                "total_pnl": total_pnl,
                "avg_pnl": avg_pnl,
                "avg_pnl_x10000": avg_pnl * 10000,
                "official_score_est": f05 * delta * abs(delta) * 10000,
                "win_rate": len(positive) / signals if signals else 0.0,
                "avg_win": sum(positive) / len(positive) if positive else 0.0,
                "avg_loss": sum(negative) / len(negative) if negative else 0.0,
                "max_drawdown": max_drawdown(pnls),
            }
        )
    return summary


def max_drawdown(pnls: Sequence[float]) -> float:
    equity = 0.0
    peak = 0.0
    worst = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        worst = min(worst, equity - peak)
    return worst


def write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_signals_csv(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, object]] = []
        for row in reader:
            parsed = dict(row)
            for key in ["sym", "date", "row_idx", "horizon", "pred", "actual"]:
                parsed[key] = int(float(parsed[key])) if parsed.get(key, "") != "" else 0
            for key in ["entry_mid", "exit_mid", "price_diff", "pnl"]:
                parsed[key] = float(parsed[key]) if parsed.get(key, "") != "" else 0.0
            parsed["correct"] = str(parsed.get("correct", "")).lower() in {"1", "true", "yes"}
            rows.append(parsed)
        return rows


def make_candles(df: pd.DataFrame, interval_seconds: int) -> pd.DataFrame:
    frame = df.copy().reset_index(drop=True)
    seconds = frame["time"].map(time_to_seconds) if "time" in frame.columns else pd.Series(range(len(frame)))
    if seconds.eq(0).all():
        bucket = frame.index // max(1, interval_seconds)
    else:
        bucket = seconds // interval_seconds
    frame["_bucket"] = bucket
    frame["_row_idx"] = frame.index
    candles = (
        frame.groupby("_bucket", sort=True)
        .agg(
            open=("n_midprice", "first"),
            high=("n_midprice", "max"),
            low=("n_midprice", "min"),
            close=("n_midprice", "last"),
            volume=("amount_delta", "sum"),
            time=("time", "first"),
            row_start=("_row_idx", "first"),
            row_end=("_row_idx", "last"),
        )
        .reset_index(drop=True)
    )
    candles["candle_idx"] = candles.index
    return candles


def draw_svg_candles(
    raw_file: Path,
    signal_rows: Sequence[dict[str, object]],
    label: str,
    interval_seconds: int,
    max_markers: int,
    output_path: Path,
    title: str,
) -> None:
    df = pd.read_csv(raw_file)
    df = df.sort_values(["sym", "date", "time"], kind="mergesort").reset_index(drop=True)
    candles = make_candles(df, interval_seconds)
    if candles.empty:
        return

    plot_rows = [row for row in signal_rows if row["source_file"] == raw_file.name and row["label"] == label and int(row["pred"]) in {0, 2}]
    if len(plot_rows) > max_markers:
        step = max(1, math.ceil(len(plot_rows) / max_markers))
        plot_rows = plot_rows[::step][:max_markers]

    width, height = 1400, 760
    left, right, top, bottom = 72, 36, 54, 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    low = float(candles["low"].min())
    high = float(candles["high"].max())
    pad = max((high - low) * 0.08, 1e-6)
    low -= pad
    high += pad

    def x_for(i: int) -> float:
        if len(candles) == 1:
            return left + plot_w / 2
        return left + (i / (len(candles) - 1)) * plot_w

    def y_for(value: float) -> float:
        return top + (high - value) / (high - low) * plot_h

    row_to_candle = {}
    for _, candle in candles.iterrows():
        for row_idx in range(int(candle["row_start"]), int(candle["row_end"]) + 1):
            row_to_candle[row_idx] = int(candle["candle_idx"])

    candle_width = max(2.0, min(12.0, plot_w / max(len(candles), 1) * 0.62))
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="30" font-family="Arial" font-size="18" fill="#111827">{html.escape(title)}</text>',
    ]

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        value = low + (high - low) * frac
        y = y_for(value)
        svg.append(f'<line x1="{left}" x2="{width - right}" y1="{y:.2f}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>')
        svg.append(
            f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="Arial" font-size="11" fill="#4b5563">{value:.6f}</text>'
        )

    for i, candle in candles.iterrows():
        x = x_for(i)
        open_price = float(candle["open"])
        close_price = float(candle["close"])
        high_price = float(candle["high"])
        low_price = float(candle["low"])
        color = "#157f5b" if close_price >= open_price else "#c2413c"
        y_high = y_for(high_price)
        y_low = y_for(low_price)
        y_open = y_for(open_price)
        y_close = y_for(close_price)
        body_top = min(y_open, y_close)
        body_h = max(abs(y_close - y_open), 1.0)
        svg.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{y_high:.2f}" y2="{y_low:.2f}" stroke="#111827" stroke-width="1"/>')
        svg.append(
            f'<rect x="{x - candle_width / 2:.2f}" y="{body_top:.2f}" width="{candle_width:.2f}" '
            f'height="{body_h:.2f}" fill="{color}" stroke="#111827" stroke-width="0.6"/>'
        )

    for tick in range(0, len(candles), max(1, len(candles) // 8)):
        x = x_for(tick)
        label_text = str(candles.iloc[tick]["time"])
        svg.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{height - bottom}" y2="{height - bottom + 5}" stroke="#6b7280"/>')
        svg.append(
            f'<text x="{x:.2f}" y="{height - bottom + 22}" text-anchor="middle" '
            f'font-family="Arial" font-size="11" fill="#4b5563">{html.escape(label_text)}</text>'
        )

    for row in plot_rows:
        candle_idx = row_to_candle.get(int(row["row_idx"]))
        if candle_idx is None or candle_idx >= len(candles):
            continue
        candle = candles.iloc[candle_idx]
        x = x_for(candle_idx)
        pred = int(row["pred"])
        pnl = float(row["pnl"])
        if pred == 2:
            y = y_for(float(candle["low"])) + 18
            points = [(x, y - 12), (x - 8, y + 4), (x + 8, y + 4)]
            fill = "#2563eb"
        else:
            y = y_for(float(candle["high"])) - 18
            points = [(x, y + 12), (x - 8, y - 4), (x + 8, y - 4)]
            fill = "#dc2626"
        point_text = " ".join(f"{px:.2f},{py:.2f}" for px, py in points)
        svg.append(f'<polygon points="{point_text}" fill="{fill}" stroke="#111827" stroke-width="0.8"/>')
        svg.append(
            f'<text x="{x:.2f}" y="{y + (24 if pred == 2 else -18):.2f}" text-anchor="middle" '
            f'font-family="Arial" font-size="10" fill="{fill}">{pnl * 10000:.1f}</text>'
        )

    svg.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#111827" stroke-width="1"/>')
    svg.append(
        f'<text x="{width / 2:.2f}" y="{height - 18}" text-anchor="middle" '
        f'font-family="Arial" font-size="12" fill="#4b5563">marker text = trade PnL x 10000</text>'
    )
    svg.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze PnL and draw SVG candlestick charts")
    parser.add_argument("--mode", choices=["model", "label", "signals"], default="label")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--csv-file", type=Path, action="append", default=[])
    parser.add_argument("--signals-csv", type=Path, default=None, help="Input signals CSV for --mode signals")
    parser.add_argument("--package-dir", type=Path, default=None)
    parser.add_argument("--zip-path", type=Path, default=None)
    parser.add_argument("--labels", nargs="*", default=LABELS, choices=LABELS)
    parser.add_argument("--plot-label", default="label_20", choices=LABELS)
    parser.add_argument("--sym", type=int, default=None)
    parser.add_argument("--date-start", type=int, default=None)
    parser.add_argument("--date-end", type=int, default=None)
    parser.add_argument("--session", choices=["am", "pm"], default=None)
    parser.add_argument("--max-files", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--prediction-batch", type=int, default=256)
    parser.add_argument("--candle-interval-seconds", type=int, default=30)
    parser.add_argument("--max-markers", type=int, default=250)
    parser.add_argument("--pnl-baseline", type=float, default=0.0004)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/pnl_candles"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    data_dir = resolve_path(args.data_dir, project_root)
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir

    package_dir = resolve_path(args.package_dir, project_root) if args.package_dir else None
    zip_path = resolve_path(args.zip_path, project_root) if args.zip_path else None
    predictor = None
    temp_dir = None
    if args.mode == "model":
        predictor, temp_dir = load_predictor(package_dir, zip_path)

    if args.mode == "signals":
        if args.signals_csv is None:
            raise ValueError("--signals-csv is required in signals mode")
        rows = read_signals_csv(resolve_path(args.signals_csv, project_root))
        files = scan_files(data_dir, args.csv_file, args.sym, args.date_start, args.date_end, args.session, args.max_files)
    else:
        files = scan_files(data_dir, args.csv_file, args.sym, args.date_start, args.date_end, args.session, args.max_files)
        if not files:
            raise FileNotFoundError(f"No snapshot CSV files found under {data_dir}")
        rows = []
        for path in files:
            print(f"Analyzing {path}")
            rows.extend(
                analyze_file(
                    path=path,
                    mode=args.mode,
                    labels=args.labels,
                    predictor=predictor,
                    window_size=args.window_size,
                    stride=args.stride,
                    max_windows=args.max_windows,
                    prediction_batch=args.prediction_batch,
                )
            )

    summary = summarize_signals(rows, args.labels, args.pnl_baseline)
    signals_path = output_dir / "signals.csv"
    trades_path = output_dir / "trades.csv"
    summary_path = output_dir / "summary.csv"
    json_path = output_dir / "summary.json"
    trade_rows = [row for row in rows if int(row["pred"]) in {0, 2}]
    write_rows(signals_path, rows)
    write_rows(trades_path, trade_rows)
    write_rows(summary_path, summary)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    chart_path = None
    if files:
        first_file = files[0]
        plot_summary = next((row for row in summary if row["label"] == args.plot_label), None)
        title = f"{args.mode} {args.plot_label} {first_file.name}"
        if plot_summary:
            title += f" avg_pnl_x10000={float(plot_summary['avg_pnl_x10000']):.2f} signals={plot_summary['signals']}"
        chart_path = output_dir / f"candles_{first_file.stem}_{args.plot_label}.svg"
        draw_svg_candles(
            raw_file=first_file,
            signal_rows=rows,
            label=args.plot_label,
            interval_seconds=args.candle_interval_seconds,
            max_markers=args.max_markers,
            output_path=chart_path,
            title=title,
        )

    print(f"Wrote {signals_path}")
    print(f"Wrote {trades_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {json_path}")
    if chart_path:
        print(f"Wrote {chart_path}")
    if temp_dir is not None:
        temp_dir.cleanup()


if __name__ == "__main__":
    main()
