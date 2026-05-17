"""
Feature Engineering Pipeline for Stock Movement Prediction.

This script reads raw snapshot CSV files from data/raw/, constructs per-file
features without crossing symbols, dates, or sessions, and writes:
- data/processed/tabular_features.parquet
- data/processed/feature_columns.json
- reports/feature_summary.md
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("ARROW_NUM_THREADS", "1")
import numpy as np
import pandas as pd
# tqdm removed to control custom progress printing

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")

OUTPUT_PARQUET = PROCESSED_DIR / "tabular_features.parquet"
OUTPUT_FEATURE_COLUMNS = PROCESSED_DIR / "feature_columns.json"
OUTPUT_SUMMARY = REPORTS_DIR / "feature_summary.md"
OUTPUT_FILE_ROW_COUNTS = REPORTS_DIR / "file_feature_row_counts.csv"

INCLUDE_ID_FEATURES = True
LABEL_COLS = ["label_5", "label_10", "label_20", "label_40", "label_60"]

# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS
# -----------------------------------------------------------------------------

def ensure_dirs(directories: List[Path]) -> None:
    """Create required directories."""
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def parse_snapshot_filename(filename: str) -> Optional[Dict[str, int]]:
    """Parse snapshot filename to extract metadata."""
    pattern = r"snapshot_sym(\d+)_date(\d+)_(am|pm)\.csv"
    match = re.match(pattern, filename)
    if not match:
        return None
    session_name = match.group(3)
    return {
        "sym_id": int(match.group(1)),
        "date_id": int(match.group(2)),
        "session": 0 if match.group(3) == "am" else 1,
        "session_name": session_name
    }


def scan_csv_files(data_dir: Path) -> List[Tuple[Path, Dict[str, int]]]:
    """Find all raw CSV files in the input directory."""
    result: List[Tuple[Path, Dict[str, int]]] = []
    if not data_dir.exists():
        return result

    for csv_file in sorted(data_dir.glob("snapshot_*.csv")):
        metadata = parse_snapshot_filename(csv_file.name)
        if metadata is not None:
            result.append((csv_file, metadata))
    return result


def normalize_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize label5 style columns to label_5 style."""
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        if col.startswith("label") and "_" not in col:
            match = re.match(r"label(\d+)$", col)
            if match:
                rename_map[col] = f"label_{match.group(1)}"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def get_label_columns(df: pd.DataFrame) -> List[str]:
    """Return label columns in the fixed project order."""
    return [col for col in LABEL_COLS if col in df.columns]

def read_csv_safe(filepath: Path) -> Optional[pd.DataFrame]:
    """Read CSV safely, returning None on failure."""
    try:
        return pd.read_csv(filepath, low_memory=False)
    except Exception:
        return None


def is_numeric_series(series: pd.Series) -> bool:
    """Check whether a pandas series is numeric."""
    try:
        return series.dtype.kind in "biufc"
    except Exception:
        return False


def collect_feature_columns(df: pd.DataFrame, include_id_features: bool = True) -> List[str]:
    """Select feature columns for training according to project rules.

    Rules enforced:
    - Always exclude tracking fields: source_file, date, time, sym, session_name, date_id
    - Exclude label_* columns
    - Keep `sym_id` and `session` as model input features when `include_id_features` is True
    """
    exclude_columns = {"source_file", "time", "sym", "date", "session_name", "date_id"}
    # exclude label columns
    exclude_columns.update(get_label_columns(df))

    # If id features aren't requested, exclude them
    if not include_id_features:
        exclude_columns.update({"sym_id", "session"})

    # Build feature list: ensure sym_id and session appear first (if included)
    features: List[str] = []
    if include_id_features:
        for key in ("sym_id", "session"):
            if key in df.columns and is_numeric_series(df[key]):
                features.append(key)

    other_features = [
        col for col in df.columns
        if col not in exclude_columns and col not in features and is_numeric_series(df[col])
    ]
    other_features_sorted = sorted(other_features)
    features.extend(other_features_sorted)
    return features


def check_data_quality(df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Any]:
    """Check final dataset quality."""
    numeric_features = [col for col in feature_columns if is_numeric_series(df[col])]
    nan_in_features = int(df[numeric_features].isna().sum().sum()) if numeric_features else 0
    inf_in_features = int(np.isinf(df[numeric_features]).sum().sum()) if numeric_features else 0

    return {
        "total_rows": int(df.shape[0]),
        "total_cols": int(df.shape[1]),
        "feature_cols": int(len(feature_columns)),
        "nan_in_features": nan_in_features,
        "inf_in_features": inf_in_features,
        "label_columns": get_label_columns(df),
        "feature_columns_contains_label": any(col.startswith("label_") for col in feature_columns),
        "feature_columns_contains_source_file": "source_file" in feature_columns,
        "feature_columns_contains_time": "time" in feature_columns,
    }


def update_missing_field_counts(df: pd.DataFrame, missing_counts: Dict[str, int]) -> None:
    """Track which important raw field groups are missing."""
    groups = {
        "amount_delta": ["amount_delta"],
        "close_or_n_close": ["n_close", "close"],
        "n_midprice": ["n_midprice"],
        "bid1_ask1": ["n_bid1", "n_ask1"],
        "bid_size_levels": [f"n_bsize{i}" for i in range(1, 6)],
        "ask_size_levels": [f"n_asize{i}" for i in range(1, 6)],
        "time": ["time"],
    }
    for label, fields in groups.items():
        if not any(field in df.columns for field in fields):
            missing_counts[label] += 1


def construct_features_for_file(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    missing_counts: Dict[str, int],
) -> Optional[pd.DataFrame]:
    """Construct features for one CSV file without crossing file boundaries."""
    if df.shape[0] == 0:
        return None

    df = normalize_label_columns(df).copy()
    if "time" in df.columns:
        df = df.sort_values("time", kind="mergesort").reset_index(drop=True)

    df["source_file"] = metadata["source_file"]
    df["sym_id"] = metadata["sym_id"]
    df["date_id"] = metadata["date_id"]
    df["session"] = metadata["session"]
    df["session_name"] = metadata.get(
        "session_name",
        "am" if metadata["session"] == 0 else "pm"
        )

    update_missing_field_counts(df, missing_counts)

    raw_numeric_candidates = [
        "n_close", "close", "amount_delta", "n_midprice",
    ] + [f"n_bid{i}" for i in range(1, 6)] + [f"n_ask{i}" for i in range(1, 6)]
    raw_numeric_candidates += [f"n_bsize{i}" for i in range(1, 6)] + [f"n_asize{i}" for i in range(1, 6)]
    for column in raw_numeric_candidates:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "amount_delta" in df.columns:
        df["log_amount_delta"] = np.log1p(np.maximum(df["amount_delta"].fillna(0), 0))

    if "n_bid1" in df.columns and "n_ask1" in df.columns:
        df["is_crossed_book"] = (
            (df["n_bid1"] > df["n_ask1"]) & df["n_bid1"].notna() & df["n_ask1"].notna()
        ).astype(int)

    if all(col in df.columns for col in ["n_bid1", "n_ask1", "n_midprice"]):
        bid1 = df["n_bid1"]
        ask1 = df["n_ask1"]
        calc_midprice = pd.Series(np.nan, index=df.index, dtype=float)
        both_nonzero = (bid1 != 0) & (ask1 != 0)
        only_bid = (bid1 != 0) & (ask1 == 0)
        only_ask = (ask1 != 0) & (bid1 == 0)
        calc_midprice = np.where(both_nonzero, (bid1 + ask1) / 2, calc_midprice)
        calc_midprice = np.where(only_bid, bid1, calc_midprice)
        calc_midprice = np.where(only_ask, ask1, calc_midprice)
        df["calc_midprice"] = calc_midprice
        df["midprice_diff"] = df["n_midprice"] - df["calc_midprice"]
        df["abs_midprice_diff"] = df["midprice_diff"].abs()
        df["is_midprice_mismatch"] = (df["abs_midprice_diff"] > 1e-8).astype(int)

    for i in range(1, 6):
        bid_col = f"n_bid{i}"
        ask_col = f"n_ask{i}"
        if bid_col in df.columns and ask_col in df.columns:
            df[f"spread_{i}"] = df[ask_col] - df[bid_col]

    if "spread_1" in df.columns and "n_midprice" in df.columns:
        df["spread_pct"] = df["spread_1"] / (df["n_midprice"].abs() + 1e-8)

    bid_size_cols = [f"n_bsize{i}" for i in range(1, 6) if f"n_bsize{i}" in df.columns]
    ask_size_cols = [f"n_asize{i}" for i in range(1, 6) if f"n_asize{i}" in df.columns]
    if bid_size_cols:
        df["bid_depth"] = df[bid_size_cols].sum(axis=1)
    if ask_size_cols:
        df["ask_depth"] = df[ask_size_cols].sum(axis=1)
    if "bid_depth" in df.columns and "ask_depth" in df.columns:
        df["total_depth"] = df["bid_depth"] + df["ask_depth"]
        df["depth_diff"] = df["bid_depth"] - df["ask_depth"]

    if "bid_depth" in df.columns and "ask_depth" in df.columns:
        df["imbalance_total"] = (
            df["bid_depth"] - df["ask_depth"]
        ) / (df["bid_depth"] + df["ask_depth"] + 1e-8)

    for i in range(1, 6):
        bsize_col = f"n_bsize{i}"
        asize_col = f"n_asize{i}"
        if bsize_col in df.columns and asize_col in df.columns:
            df[f"imbalance_{i}"] = (
                df[bsize_col] - df[asize_col]
            ) / (df[bsize_col] + df[asize_col] + 1e-8)

    if all(col in df.columns for col in ["n_ask1", "n_bid1", "n_bsize1", "n_asize1"]):
        df["weighted_midprice_1"] = (
            df["n_ask1"] * df["n_bsize1"] + df["n_bid1"] * df["n_asize1"]
        ) / (df["n_bsize1"] + df["n_asize1"] + 1e-8)

    if "n_midprice" in df.columns:
        for lag in [1, 5, 10, 20]:
            df[f"mid_diff_{lag}"] = df["n_midprice"] - df["n_midprice"].shift(lag)

    rolling_vars = [
        "n_midprice", "spread_1", "imbalance_total", "bid_depth", "ask_depth", "log_amount_delta"
    ]
    rolling_windows = [5, 10, 20, 50, 100]
    rolling_feature_data = {}
    for var in rolling_vars:
        if var not in df.columns:
            continue
        numeric_var = pd.to_numeric(df[var], errors="coerce")
        for window in rolling_windows:
            rolling_obj = numeric_var.rolling(window=window, min_periods=window)
            rolling_feature_data[f"{var}_roll_{window}_mean"] = rolling_obj.mean()
            rolling_feature_data[f"{var}_roll_{window}_std"] = rolling_obj.std()
            rolling_feature_data[f"{var}_roll_{window}_min"] = rolling_obj.min()
            rolling_feature_data[f"{var}_roll_{window}_max"] = rolling_obj.max()
            rolling_feature_data[f"{var}_roll_{window}_change"] = numeric_var - numeric_var.shift(window - 1)
            if var == "log_amount_delta":
                rolling_feature_data[f"{var}_roll_{window}_sum"] = rolling_obj.sum()

    if rolling_feature_data:
        rolling_df = pd.DataFrame(rolling_feature_data, index=df.index)
        df = pd.concat([df, rolling_df], axis=1)

    return df


def delete_rows_with_rolling_nan(df: pd.DataFrame, preserve_last_n: int = 60) -> Tuple[pd.DataFrame, int]:
    """Remove rows where any rolling feature is NaN, keeping the final rows intact."""
    rolling_cols = [col for col in df.columns if re.search(r"_roll_\d+_", col)]
    if not rolling_cols:
        return df.reset_index(drop=True), 0

    nan_mask = df[rolling_cols].isna().any(axis=1)
    preserve_mask = pd.Series(False, index=df.index)
    preserve_mask.iloc[-preserve_last_n:] = True
    delete_mask = nan_mask & ~preserve_mask
    deleted_rows = int(delete_mask.sum())
    df_cleaned = df.loc[~delete_mask].reset_index(drop=True)
    return df_cleaned, deleted_rows


def generate_summary_report(
    stats: Dict[str, Any],
    quality: Dict[str, Any],
    labels: List[str],
    feature_columns: List[str],
    missing_counts: Dict[str, int],
    file_records: List[Dict[str, Any]],
) -> List[str]:
    """Generate the markdown summary document."""
    lines = [
        "# Feature Engineering Summary",
        "",
        "## 1. 特征工程目标",
        "本脚本将 data/raw/ 下的原始股票快照 CSV 文件处理成建模可用的表格特征数据。",
        "",
        "## 2. 输入数据",
        f"- 原始文件路径: {RAW_DATA_DIR}",
        f"- 扫描到的 CSV 文件数量: {stats['total_files']}",
        f"- 成功读取文件数: {stats['success_files']}",
        f"- 读取失败文件数: {stats['failed_files']}",
        f"- 跳过文件数: {stats['skipped_files']}",
        "",
        "## 3. 输出数据",
        f"- tabular_features.parquet 路径: {OUTPUT_PARQUET}",
        f"- feature_columns.json 路径: {OUTPUT_FEATURE_COLUMNS}",
        f"- 最终样本数: {quality['total_rows']}",
        f"- 最终总列数: {quality['total_cols']}",
        f"- 最终模型特征数: {quality['feature_cols']}",
        "",
        "## 4. 构造的特征类别",
        "- 原始基础特征: n_close 或 close, amount_delta, n_midprice, n_bid1~n_bid5, n_ask1~n_ask5, n_bsize1~n_bsize5, n_asize1~n_asize5",
        "- amount_delta 变换特征: log_amount_delta",
        "- crossed book 特征: is_crossed_book",
        "- midprice 质量检查特征: calc_midprice, midprice_diff, abs_midprice_diff, is_midprice_mismatch",
        "- spread 特征: spread_1~spread_5, spread_pct",
        "- depth 特征: bid_depth, ask_depth, total_depth, depth_diff",
        "- imbalance 特征: imbalance_total, imbalance_1~imbalance_5",
        "- weighted_midprice 特征: weighted_midprice_1",
        "- price diff 特征: mid_diff_1, mid_diff_5, mid_diff_10, mid_diff_20",
        "- rolling 特征说明: 对以下变量计算 rolling mean/std/min/max/change： n_midprice、spread_1、imbalance_total、bid_depth、ask_depth、log_amount_delta；\n  另外对 log_amount_delta 还计算 rolling sum（窗口：5/10/20/50/100）",
        "",
        "## 5. 清洗策略",
        "- 是否删除 crossed book: 否",
        "- 是否保留 is_crossed_book: 是",
        "- 是否覆盖 n_midprice: 否",
        "- 是否删除 rolling NaN: 是",
        "- 是否补齐缺失文件: 否",
        "- 是否删除最后 60 行: 否",
        "",
        "## 6. 标签列",
    ]
    if labels:
        for label in labels:
            lines.append(f"- {label} (目标变量，不进入 feature_columns.json)")
    else:
        lines.append("- 未检测到标签列，请检查原始数据是否包含 label_5/label_10/label_20/label_40/label_60。")

    lines.extend([
        "",
        "## 7. 数据质量检查",
        f"- 原始总行数: {stats['original_rows']}",
        f"- Rolling 删除行数: {stats['deleted_rolling_nan']}",
        f"- 最终样本数: {quality['total_rows']}",
        f"- 期望的 rolling 删除后样本数 (原始总行数 - success_files * 99): {stats.get('expected_rows_after_rolling_drop', 'N/A')}",
        f"- 实际保存到 parquet 的行数: {stats.get('actual_saved_rows', 'N/A')}",
        f"- feature_columns.json 包含 sym_id: {'是' if 'sym_id' in feature_columns else '否'}",
        f"- feature_columns.json 包含 session: {'是' if 'session' in feature_columns else '否'}",
        f"- feature_columns.json 不包含 date_id: {'是' if 'date_id' not in feature_columns else '否'}",
        f"- feature_columns.json 不包含 source_file/date/time/sym/session_name/标签列: {'是' if not any(field in feature_columns for field in ['source_file', 'date', 'time', 'sym', 'session_name']) and not quality['feature_columns_contains_label'] else '否'}",
        f"- 特征中是否仍有 NaN: {'是' if quality['nan_in_features'] > 0 else '否'} ({quality['nan_in_features']} 个)",
        f"- 特征中是否仍有 Inf: {'是' if quality['inf_in_features'] > 0 else '否'} ({quality['inf_in_features']} 个)",
        f"- 标签列是否存在: {'，'.join(quality['label_columns']) if quality['label_columns'] else '否'}",
        f"- 是否有文件因为行数少于 100 而全部被删除: {'是' if stats['files_deleted_due_to_short_rows'] > 0 else '否'}",
        "",
        "## 8. 每个文件处理后的行数",
        f"- 共处理文件数: {stats['total_files']}",
        f"- 详细逐文件结果见: {OUTPUT_FILE_ROW_COUNTS}",
        "- 以下为前 10 个文件示例：",
    ])
    for record in file_records[:10]:
        lines.append(
            f"- {record['file_name']}: 原始 {record['original_rows']} 行，保留 {record['final_rows']} 行，删除 {record['deleted_rows']} 行，状态 {record['status']}"
        )

    lines.extend([
        "",
        "## 9. 数据泄露检查",
        "- 每个 CSV 文件单独构造特征，不跨股票、不跨日期、不跨 am/pm session。",
        "- 在构造 mid_diff 和 rolling 特征前，已在单个文件内部按 time 升序排序。",
        "- rolling 特征只使用当前 tick 及过去 tick，不使用未来信息。",
        "- mid_diff 特征只使用过去 tick，不使用未来价格。",
        "- label_5、label_10、label_20、label_40、label_60 只作为目标变量，不进入 feature_columns.json。",
        "- date_id 只用于训练/验证/测试集的时间顺序划分，不作为模型输入特征。",
        "- 最后 60 行未被额外删除，因为原始数据已提供标签。",
    ])

    missing_fields = [
        f"{key}: {count} 个文件缺失"
        for key, count in missing_counts.items()
        if count > 0
    ]
    if missing_fields:
        lines.extend([
            "",
            "## 10. 缺失字段说明",
            "以下原始字段在部分文件中缺失，已按实际存在情况跳过相关特征：",
        ])
        lines.extend([f"- {item}" for item in missing_fields])

    lines.extend([
        "",
        "## 11. 给建模同学 B 的说明",
        "### 读取数据",
        "```python",
        "import pandas as pd",
        "import json",
        "",
        "df = pd.read_parquet('data/processed/tabular_features.parquet')",
        "with open('data/processed/feature_columns.json', 'r', encoding='utf-8') as f:",
        "    feature_columns = json.load(f)",
        "",
        "X = df[feature_columns]",
        "y = df['label_5']",
        "```",
        "",
        "### 重要提示",
        "- 训练 label_5、label_10、label_20、label_40、label_60 时应分别建模。",
        "- `date_id` 仅用于按时间顺序划分训练/验证/测试集，不作为模型输入特征。",
        "- `sym_id` 和 `session` 当前被保留并作为模型输入特征。",
        "- 训练/验证/测试集必须严格按 `date_id` 的时间顺序划分，不能随机划分。",
        "- feature_columns.json 中不包含标签列和追踪字段（date/time/source_file/sym/session_name/date_id）。",
        "- source_file 和 time 仅用于追踪，不用于训练。",
    ])
    return lines


def main() -> None:
    ensure_dirs([PROCESSED_DIR, REPORTS_DIR])

    print("=" * 80)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 80)

    files = scan_csv_files(RAW_DATA_DIR)
    print(f"\nScanned {len(files)} CSV files from {RAW_DATA_DIR}")
    if not files:
        print("No CSV files found. Exiting.")
        return

    stats = {
        "total_files": len(files),
        "success_files": 0,
        "failed_files": 0,
        "skipped_files": 0,
        "original_rows": 0,
        "deleted_rolling_nan": 0,
        "final_rows": 0,
        "files_deleted_due_to_short_rows": 0,
    }
    missing_counts: Dict[str, int] = defaultdict(int)
    file_records: List[Dict[str, Any]] = []
    all_dfs: List[pd.DataFrame] = []

    # iterate with index and print progress every 10 files
    for idx, (filepath, metadata) in enumerate(files, start=1):
        if idx % 10 == 0:
            print(f"Processing file {idx} / {len(files)}")
        file_metadata = dict(metadata)
        file_metadata["source_file"] = filepath.name
        df_raw = read_csv_safe(filepath)
        if df_raw is None or df_raw.shape[0] == 0:
            stats["failed_files"] += 1
            file_records.append({
                "file_name": filepath.name,
                "original_rows": 0,
                "final_rows": 0,
                "deleted_rows": 0,
                "status": "读取失败或空文件",
            })
            continue

        df_processed = construct_features_for_file(df_raw, file_metadata, missing_counts)
        if df_processed is None or df_processed.shape[0] == 0:
            stats["skipped_files"] += 1
            file_records.append({
                "file_name": filepath.name,
                "original_rows": int(df_raw.shape[0]),
                "final_rows": 0,
                "deleted_rows": 0,
                "status": "构造特征后无数据",
            })
            continue

        original_rows = int(df_processed.shape[0])
        stats["original_rows"] += original_rows

        df_cleaned, deleted_rows = delete_rows_with_rolling_nan(df_processed, preserve_last_n=60)
        final_rows = int(df_cleaned.shape[0])
        stats["deleted_rolling_nan"] += deleted_rows

        if final_rows == 0:
            stats["skipped_files"] += 1
            if original_rows < 100:
                stats["files_deleted_due_to_short_rows"] += 1
            file_records.append({
                "file_name": filepath.name,
                "original_rows": original_rows,
                "final_rows": final_rows,
                "deleted_rows": deleted_rows,
                "status": "Rolling NaN 删除后无保留行",
            })
            continue

        stats["success_files"] += 1
        stats["final_rows"] += final_rows
        file_records.append({
            "file_name": filepath.name,
            "original_rows": original_rows,
            "final_rows": final_rows,
            "deleted_rows": deleted_rows,
            "status": "成功",
        })
        all_dfs.append(df_cleaned)

    final_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    final_df = final_df.copy()

    # Save parquet
    final_df.to_parquet(OUTPUT_PARQUET, index=False)

    # Re-read parquet to be sure what was actually written
    try:
        final_df_reloaded = pd.read_parquet(OUTPUT_PARQUET)
        actual_saved_rows = int(final_df_reloaded.shape[0])
    except Exception:
        final_df_reloaded = final_df
        actual_saved_rows = int(final_df_reloaded.shape[0])

    # Expected rows after rolling drop (heuristic per request)
    stats['expected_rows_after_rolling_drop'] = int(stats.get('original_rows', 0) - stats.get('success_files', 0) * 99)
    stats['actual_saved_rows'] = actual_saved_rows

    # Recompute feature columns from reloaded dataframe according to rules
    feature_columns = collect_feature_columns(final_df_reloaded, include_id_features=INCLUDE_ID_FEATURES)
    quality = check_data_quality(final_df_reloaded, feature_columns)

    # Save per-file row counts CSV
    file_counts_df = pd.DataFrame([
        {
            "source_file": rec["file_name"],
            "raw_rows": rec["original_rows"],
            "kept_rows": rec["final_rows"],
            "dropped_rows": rec["deleted_rows"],
            "status": rec["status"],
        }
        for rec in file_records
    ])
    file_counts_df.to_csv(OUTPUT_FILE_ROW_COUNTS, index=False, encoding="utf-8")

    summary_lines = generate_summary_report(
        stats=stats,
        quality=quality,
        labels=get_label_columns(final_df_reloaded),
        feature_columns=feature_columns,
        missing_counts=missing_counts,
        file_records=file_records,
    )

    # Write feature columns and summary
    with open(OUTPUT_FEATURE_COLUMNS, "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    # Final concise prints
    print("Feature engineering completed")
    print(f"Output parquet: {OUTPUT_PARQUET}")
    print(f"Feature columns: {OUTPUT_FEATURE_COLUMNS}")
    print(f"Summary report: {OUTPUT_SUMMARY}")
    print(f"File row counts: {OUTPUT_FILE_ROW_COUNTS}")
    print(f"Actual saved rows: {actual_saved_rows}")
    print(f"Final feature count: {len(feature_columns)}")


if __name__ == "__main__":
    main()
