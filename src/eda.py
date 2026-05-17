"""
EDA script for stock snapshot movement prediction project.

Usage:
    python src/eda.py

功能：
- 扫描 data/raw/ 下所有 csv 文件
- 统计并读取文件，合并数据，添加文件级元信息
- 生成基础统计、字段完整性检查、数据质量检查、订单簿异常检查、标签分布分析
- 保存图表到 reports/figures/，并输出 reports/eda_summary.md

要求：新手友好，函数化，包含 main()
"""
import os
import re
import glob
import math
import json
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

GENERAL_EXPECTED_FIELDS = [
    "date",
    "time",
    "sym",
    "n_close",
    "amount_delta",
    "n_midprice",
]
for i in range(1, 6):
    GENERAL_EXPECTED_FIELDS += [f"n_bid{i}", f"n_bsize{i}", f"n_ask{i}", f"n_asize{i}"]
LABEL_FIELD_GROUPS = [
    ("label5", "label_5"),
    ("label10", "label_10"),
    ("label20", "label_20"),
    ("label40", "label_40"),
    ("label60", "label_60"),
]
ALLOWED_LABEL_FIELDS = [x for pair in LABEL_FIELD_GROUPS for x in pair]
ALLOWED_FIELDS = GENERAL_EXPECTED_FIELDS + ALLOWED_LABEL_FIELDS


def ensure_dirs(paths: List[str]):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def parse_filename(fname: str) -> Dict[str, Optional[int]]:
    """从文件名解析 sym_id, date_id, session, source_file
    期望格式: snapshot_sym<xx>_date<yy>_am.csv
    返回 dict
    """
    base = os.path.basename(fname)
    m = re.match(r"snapshot_sym(?P<sym>\d+)_date(?P<date>\d+)_(?P<session>am|pm)\.csv", base)
    if not m:
        return {"sym_id": None, "date_id": None, "session": None, "source_file": base}
    return {
        "sym_id": int(m.group("sym")),
        "date_id": int(m.group("date")),
        "session": m.group("session"),
        "source_file": base,
    }


def scan_csv_files(data_dir: str) -> List[str]:
    pattern = os.path.join(data_dir, "*.csv")
    files = sorted(glob.glob(pattern))
    return files


def get_label_columns(df: pd.DataFrame) -> List[str]:
    """自动检测标签列名，优先使用 label_# 形式。"""
    labels = []
    has_underscore = any(underscore in df.columns for _, underscore in LABEL_FIELD_GROUPS)
    if has_underscore:
        for _, underscore in LABEL_FIELD_GROUPS:
            if underscore in df.columns:
                labels.append(underscore)
    else:
        for no_underscore, _ in LABEL_FIELD_GROUPS:
            if no_underscore in df.columns:
                labels.append(no_underscore)
    return labels


def normalize_label_name(label: str) -> str:
    if label.startswith("label_"):
        return label
    if label.startswith("label"):
        return f"label_{label[5:]}"
    return label


def detect_missing_files(files: List[str], missing_output_path: str, unexpected_output_path: str) -> Dict[str, any]:
    observed_valid = set()
    unexpected_rows = []
    for f in files:
        meta = parse_filename(f)
        if meta["sym_id"] is None or meta["date_id"] is None or meta["session"] is None:
            unexpected_rows.append({
                "source_file": meta["source_file"],
                "sym_id": meta.get("sym_id"),
                "date_id": meta.get("date_id"),
                "session": meta.get("session"),
                "reason": "parse_failed",
            })
            continue
        if 0 <= meta["sym_id"] <= 9 and 0 <= meta["date_id"] <= 78 and meta["session"] in {"am", "pm"}:
            observed_valid.add((meta["sym_id"], meta["date_id"], meta["session"]))
        else:
            unexpected_rows.append({
                "source_file": meta["source_file"],
                "sym_id": meta["sym_id"],
                "date_id": meta["date_id"],
                "session": meta["session"],
                "reason": "out_of_theoretical_range",
            })

    expected_syms = range(0, 10)
    expected_dates = range(0, 79)
    expected_sessions = ["am", "pm"]
    missing_rows = []
    for sym_id in expected_syms:
        for date_id in expected_dates:
            for session in expected_sessions:
                if (sym_id, date_id, session) not in observed_valid:
                    missing_rows.append({
                        "sym_id": sym_id,
                        "date_id": date_id,
                        "session": session,
                        "expected_file_name": f"snapshot_sym{sym_id}_date{date_id}_{session}.csv",
                    })

    missing_df = pd.DataFrame(missing_rows)
    unexpected_df = pd.DataFrame(unexpected_rows)
    ensure_dirs([os.path.dirname(missing_output_path), os.path.dirname(unexpected_output_path)])
    missing_df.to_csv(missing_output_path, index=False, encoding="utf-8")
    unexpected_df.to_csv(unexpected_output_path, index=False, encoding="utf-8")

    return {
        "expected_file_count": len(expected_syms) * len(expected_dates) * len(expected_sessions),
        "actual_valid_combo_count": len(observed_valid),
        "actual_file_count": len(files),
        "extra_unexpected_count": len(unexpected_rows),
        "missing_file_count": len(missing_rows),
        "missing_file_path": missing_output_path,
        "unexpected_file_path": unexpected_output_path,
        "missing_df": missing_df,
        "unexpected_df": unexpected_df,
    }


def safe_read_csv(path: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        if os.path.getsize(path) == 0:
            return pd.DataFrame(), None
        df = pd.read_csv(path)
        return df, None
    except Exception as e:
        return None, str(e)


def read_and_merge(files: List[str]) -> Tuple[pd.DataFrame, Dict]:
    records = []
    stats = {
        "total_files": len(files),
        "success": 0,
        "failed": 0,
        "failed_files": [],
        "empty_files": [],
        "file_row_counts": {},
        "file_columns": {},
    }
    dfs = []
    for f in files:
        meta = parse_filename(f)
        df, err = safe_read_csv(f)
        if df is None:
            stats["failed"] += 1
            stats["failed_files"].append({"file": meta["source_file"], "error": err})
            continue
        # empty dataframe (file exists but no rows or only header?)
        if df.shape[0] == 0:
            stats["empty_files"].append(meta["source_file"])
            stats["success"] += 1
            stats["file_row_counts"][meta["source_file"]] = 0
            stats["file_columns"][meta["source_file"]] = list(df.columns)
            # still append so field checks include it
            df = df.copy()
            df["source_file"] = meta["source_file"]
            df["sym_id"] = meta["sym_id"]
            df["date_id"] = meta["date_id"]
            df["session"] = meta["session"]
            dfs.append(df)
            continue
        # normal case
        df = df.copy()
        df["source_file"] = meta["source_file"]
        df["sym_id"] = meta["sym_id"]
        df["date_id"] = meta["date_id"]
        df["session"] = meta["session"]
        stats["success"] += 1
        stats["file_row_counts"][meta["source_file"]] = df.shape[0]
        stats["file_columns"][meta["source_file"]] = list(df.columns)
        dfs.append(df)

    if len(dfs) == 0:
        merged = pd.DataFrame()
    else:
        merged = pd.concat(dfs, axis=0, ignore_index=True, sort=False)

    return merged, stats


def summarize_basic_stats(df: pd.DataFrame, stats: Dict) -> Dict:
    res = {}
    res["total_files"] = stats.get("total_files", 0)
    res["success_files"] = stats.get("success", 0)
    res["failed_files"] = stats.get("failed", 0)
    res["failed_file_list"] = stats.get("failed_files", [])
    res["empty_files"] = stats.get("empty_files", [])
    res["total_rows"] = int(df.shape[0])
    res["total_columns"] = int(df.shape[1])
    # per sym, date, session
    if df.shape[0] > 0:
        res["per_sym_count"] = df["sym_id"].value_counts().to_dict()
        res["per_date_count"] = df["date_id"].value_counts().to_dict()
        res["per_session_count"] = df["session"].value_counts().to_dict()
        res["per_file_count"] = stats.get("file_row_counts", {})
    else:
        res["per_sym_count"] = {}
        res["per_date_count"] = {}
        res["per_session_count"] = {}
        res["per_file_count"] = {}
    return res


def field_completeness(stats: Dict) -> Dict:
    file_columns = stats.get("file_columns", {})
    all_columns = set()
    for cols in file_columns.values():
        all_columns.update(cols)

    missing_expected = []
    for field in GENERAL_EXPECTED_FIELDS:
        if field not in all_columns:
            missing_expected.append(field)

    missing_labels = []
    for no_underscore, underscore in LABEL_FIELD_GROUPS:
        if no_underscore not in all_columns and underscore not in all_columns:
            missing_labels.append(f"{no_underscore}/{underscore}")

    extra_fields = sorted(list(all_columns - set(ALLOWED_FIELDS) - set(["source_file", "sym_id", "date_id", "session"])))
    per_file_consistency = {}
    for fname, cols in file_columns.items():
        missing_in_file = [field for field in GENERAL_EXPECTED_FIELDS if field not in cols]
        for no_underscore, underscore in LABEL_FIELD_GROUPS:
            if no_underscore not in cols and underscore not in cols:
                missing_in_file.append(f"{no_underscore}/{underscore}")
        per_file_consistency[fname] = {
            "missing_expected_in_file": missing_in_file,
            "extra_in_file": [c for c in cols if c not in ALLOWED_FIELDS and c not in ["source_file", "sym_id", "date_id", "session"]],
        }
    return {
        "all_columns": sorted(list(all_columns)),
        "missing_expected": missing_expected + missing_labels,
        "extra_fields": extra_fields,
        "per_file": per_file_consistency,
    }


def data_quality_checks(df: pd.DataFrame) -> Dict:
    out = {}
    if df.shape[0] == 0:
        return out
    # missing values
    missing_counts = df.isnull().sum()
    missing_ratio = (missing_counts / len(df)).round(4)
    out["missing_counts"] = missing_counts.to_dict()
    out["missing_ratio"] = missing_ratio.to_dict()
    # duplicates
    out["duplicate_rows"] = int(df.duplicated().sum())
    # inf values
    inf_counts = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        s = df[col]
        inf_counts[col] = int(np.isposinf(s).sum() + np.isneginf(s).sum())
    out["inf_counts"] = inf_counts
    # time monotonic per file
    time_order = {}
    if "time" in df.columns:
        for fname, g in df.groupby("source_file"):
            try:
                # attempt numeric or string compare
                series = g["time"].dropna()
                # coerce to numeric if possible
                ser_num = pd.to_numeric(series, errors="coerce")
                if ser_num.notnull().all():
                    time_order[fname] = bool(ser_num.is_monotonic_increasing)
                else:
                    time_order[fname] = bool(series.is_monotonic_increasing)
            except Exception:
                time_order[fname] = None
    out["time_order_per_file"] = time_order
    return out


def orderbook_checks(df: pd.DataFrame) -> Dict:
    out = {}
    if df.shape[0] == 0:
        return out
    # count anomalies
    def safe_count_and_ratio(cond):
        try:
            count = int(cond.sum())
            denom = int(cond.notnull().sum())
            ratio = float(count / denom) if denom > 0 else 0.0
            return count, round(ratio, 4)
        except Exception:
            return 0, 0.0

    # bid vs ask
    for i in range(1, 6):
        b = f"n_bid{i}"
        a = f"n_ask{i}"
        if b in df.columns and a in df.columns:
            count, ratio = safe_count_and_ratio(df[b] > df[a])
            out[f"n_bid{i}_gt_n_ask{i}_count"] = count
            out[f"n_bid{i}_gt_n_ask{i}_ratio"] = ratio

    # bid levels monotonic decreasing? we check anomalies where deeper bid > shallower
    for i in range(2, 6):
        prev = f"n_bid{i-1}"
        cur = f"n_bid{i}"
        if prev in df.columns and cur in df.columns:
            count, ratio = safe_count_and_ratio(df[cur] > df[prev])
            out[f"n_bid{i}_gt_n_bid{i-1}_count"] = count
            out[f"n_bid{i}_gt_n_bid{i-1}_ratio"] = ratio

    # ask levels monotonic increasing? anomaly where deeper ask < shallower
    for i in range(2, 6):
        prev = f"n_ask{i-1}"
        cur = f"n_ask{i}"
        if prev in df.columns and cur in df.columns:
            count, ratio = safe_count_and_ratio(df[cur] < df[prev])
            out[f"n_ask{i}_lt_n_ask{i-1}_count"] = count
            out[f"n_ask{i}_lt_n_ask{i-1}_ratio"] = ratio

    # size negative
    size_neg = {}
    size_neg_ratio = {}
    for i in range(1, 6):
        for prefix in ["n_bsize", "n_asize"]:
            col = f"{prefix}{i}"
            if col in df.columns:
                count = int((df[col] < 0).sum())
                denom = int(df[col].notnull().sum())
                ratio = round(float(count / denom), 4) if denom > 0 else 0.0
                size_neg[col] = count
                size_neg_ratio[col] = ratio
    out["size_negative_counts"] = size_neg
    out["size_negative_ratio"] = size_neg_ratio

    # n_midprice missing
    if "n_midprice" in df.columns:
        out["n_midprice_missing"] = int(df["n_midprice"].isnull().sum())

    # amount_delta extreme values using z-score
    if "amount_delta" in df.columns:
        ser = pd.to_numeric(df["amount_delta"], errors="coerce").dropna()
        if len(ser) > 0:
            z = (ser - ser.mean()) / ser.std(ddof=0)
            out["amount_delta_extreme_count_absz_gt5"] = int((z.abs() > 5).sum())
            out["amount_delta_min"] = float(ser.min())
            out["amount_delta_max"] = float(ser.max())

    # spread distribution
    if "n_ask1" in df.columns and "n_bid1" in df.columns:
        spread = pd.to_numeric(df["n_ask1"], errors="coerce") - pd.to_numeric(df["n_bid1"], errors="coerce")
        out["spread_stats"] = {
            "count": int(spread.count()),
            "mean": float(spread.mean()) if spread.count() > 0 else None,
            "median": float(spread.median()) if spread.count() > 0 else None,
            "std": float(spread.std()) if spread.count() > 0 else None,
            "min": float(spread.min()) if spread.count() > 0 else None,
            "max": float(spread.max()) if spread.count() > 0 else None,
        }
    return out


def midprice_rule_check(df: pd.DataFrame, output_path: str) -> Dict[str, any]:
    if df.shape[0] == 0:
        pd.DataFrame([], columns=["source_file","sym_id","date_id","session","n_bid1","n_ask1","n_midprice","calc_midprice","abs_diff","mismatch"]).to_csv(output_path, index=False, encoding="utf-8")
        return {
            "midprice_mismatch_count": 0,
            "midprice_mismatch_ratio": 0.0,
            "max_abs_midprice_diff": 0.0,
            "mean_abs_midprice_diff": 0.0,
        }

    bid = pd.to_numeric(df.get("n_bid1", pd.Series(dtype="float64")), errors="coerce").fillna(0.0)
    ask = pd.to_numeric(df.get("n_ask1", pd.Series(dtype="float64")), errors="coerce").fillna(0.0)
    actual_midprice = pd.to_numeric(df.get("n_midprice", pd.Series(dtype="float64")), errors="coerce")

    calc_midprice = pd.Series(index=df.index, dtype="float64")
    both_nonzero = (bid != 0) & (ask != 0)
    bid_only = (bid != 0) & (ask == 0)
    ask_only = (ask != 0) & (bid == 0)
    calc_midprice[both_nonzero] = (bid[both_nonzero] + ask[both_nonzero]) / 2.0
    calc_midprice[bid_only] = bid[bid_only]
    calc_midprice[ask_only] = ask[ask_only]
    calc_midprice[~(both_nonzero | bid_only | ask_only)] = np.nan

    abs_diff = (actual_midprice - calc_midprice).abs()
    mismatch = (~actual_midprice.isna() | ~calc_midprice.isna()) & ~((actual_midprice.isna()) & (calc_midprice.isna())) & (abs_diff > 1e-8)

    report_df = pd.DataFrame({
        "source_file": df["source_file"],
        "sym_id": df["sym_id"],
        "date_id": df["date_id"],
        "session": df["session"],
        "n_bid1": bid,
        "n_ask1": ask,
        "n_midprice": actual_midprice,
        "calc_midprice": calc_midprice,
        "abs_diff": abs_diff,
        "mismatch": mismatch,
    })
    ensure_dirs([os.path.dirname(output_path)])
    report_df.to_csv(output_path, index=False, encoding="utf-8")

    valid_mask = (~actual_midprice.isna()) | (~calc_midprice.isna())
    mismatch_count = int(mismatch.sum())
    valid_count = int(valid_mask.sum())
    mismatch_ratio = round(float(mismatch_count / valid_count) if valid_count > 0 else 0.0, 4)
    max_diff = float(abs_diff.max()) if len(abs_diff) > 0 else 0.0
    mean_diff = float(abs_diff.mean()) if len(abs_diff) > 0 else 0.0
    return {
        "midprice_mismatch_count": mismatch_count,
        "midprice_mismatch_ratio": mismatch_ratio,
        "max_abs_midprice_diff": max_diff,
        "mean_abs_midprice_diff": mean_diff,
        "midprice_check_path": output_path,
    }


def label_direction_check(df: pd.DataFrame, output_path: str) -> Dict[str, any]:
    label_cols = get_label_columns(df)
    if df.shape[0] == 0 or not label_cols or "n_midprice" not in df.columns:
        pd.DataFrame([], columns=["source_file","label","direction","match_count","total_count","match_ratio"]).to_csv(output_path, index=False, encoding="utf-8")
        return {"label_direction_summary": {}}

    rows = []
    summary = {}

    def direction_label(x: float, alpha: float) -> Optional[int]:
        if pd.isna(x):
            return None
        if x < -alpha:
            return 0
        if x > alpha:
            return 2
        return 1

    alpha_map = {5: 0.0005, 10: 0.0005, 20: 0.001, 40: 0.001, 60: 0.001}
    for source_file, group in df.groupby("source_file", sort=False):
        midprices = pd.to_numeric(group["n_midprice"], errors="coerce")
        for lab in label_cols:
            canonical = normalize_label_name(lab)
            actual = pd.to_numeric(group[lab], errors="coerce")
            N = int(canonical.split("_")[-1])
            alpha = alpha_map.get(N, 0.001)
            future_diff = midprices.shift(-N) - midprices
            past_diff = midprices - midprices.shift(N)
            future_labels = future_diff.apply(lambda x: direction_label(x, alpha))
            past_labels = past_diff.apply(lambda x: direction_label(x, alpha))
            for direction, predicted in [("future", future_labels), ("past", past_labels)]:
                valid = ~actual.isna() & ~predicted.isna()
                total = int(valid.sum())
                match = int((actual[valid] == predicted[valid]).sum()) if total > 0 else 0
                ratio = round(float(match / total), 4) if total > 0 else 0.0
                rows.append({
                    "source_file": source_file,
                    "label": canonical,
                    "direction": direction,
                    "match_count": match,
                    "total_count": total,
                    "match_ratio": ratio,
                })
                summary.setdefault(canonical, {"future_match": 0, "future_total": 0, "past_match": 0, "past_total": 0})
                if direction == "future":
                    summary[canonical]["future_match"] += match
                    summary[canonical]["future_total"] += total
                else:
                    summary[canonical]["past_match"] += match
                    summary[canonical]["past_total"] += total

    result_rows = []
    for label, stats in summary.items():
        future_ratio = round(float(stats["future_match"] / stats["future_total"]), 4) if stats["future_total"] > 0 else 0.0
        past_ratio = round(float(stats["past_match"] / stats["past_total"]), 4) if stats["past_total"] > 0 else 0.0
        if future_ratio > past_ratio:
            better = "future"
        elif past_ratio > future_ratio:
            better = "past"
        else:
            better = "tie"
        result_rows.append({
            "label": label,
            "future_match_count": stats["future_match"],
            "future_total_count": stats["future_total"],
            "future_match_ratio": future_ratio,
            "past_match_count": stats["past_match"],
            "past_total_count": stats["past_total"],
            "past_match_ratio": past_ratio,
            "better_direction": better,
        })

    direction_df = pd.DataFrame(rows)
    ensure_dirs([os.path.dirname(output_path)])
    direction_df.to_csv(output_path, index=False, encoding="utf-8")
    return {
        "label_direction_summary": result_rows,
        "label_direction_path": output_path,
    }


def label_analysis(df: pd.DataFrame) -> Dict:
    out = {}
    if df.shape[0] == 0:
        return out
    labels = get_label_columns(df)
    for lab in labels:
        counts = df[lab].value_counts(dropna=False)
        props = (counts / counts.sum()).round(4)
        canonical = normalize_label_name(lab)
        out[f"{canonical}_counts"] = counts.to_dict()
        out[f"{canonical}_props"] = props.to_dict()
        out[f"{canonical}_per_sym"] = df.groupby("sym_id")[lab].value_counts().unstack(fill_value=0).to_dict()
        out[f"{canonical}_per_session"] = df.groupby("session")[lab].value_counts().unstack(fill_value=0).to_dict()
    return out


def generate_plots(df: pd.DataFrame, figures_dir: str) -> Dict[str, str]:
    ensure_dirs([figures_dir])
    saved = {}
    sns.set(style="whitegrid")
    if df.shape[0] == 0:
        return saved

    # label distributions
    label_cols = get_label_columns(df)
    for lab in label_cols:
        canonical = normalize_label_name(lab)
        plt.figure(figsize=(6, 4))
        ax = sns.countplot(x=lab, data=df)
        plt.title(f"Distribution of {lab}")
        plt.tight_layout()
        p = os.path.join(figures_dir, f"{canonical}_distribution.png")
        plt.savefig(p)
        plt.close()
        saved[canonical] = p

    # per stock sample count
    if "sym_id" in df.columns:
        plt.figure(figsize=(8, 4))
        vc = df["sym_id"].value_counts().sort_index()
        sns.barplot(x=vc.index.astype(str), y=vc.values)
        plt.xlabel("sym_id")
        plt.ylabel("count")
        plt.title("Samples per sym_id")
        plt.tight_layout()
        p = os.path.join(figures_dir, "samples_per_sym.png")
        plt.savefig(p)
        plt.close()
        saved["samples_per_sym"] = p

    # per date sample count
    if "date_id" in df.columns:
        plt.figure(figsize=(10, 4))
        vc = df["date_id"].value_counts().sort_index()
        sns.barplot(x=vc.index.astype(str), y=vc.values)
        plt.xlabel("date_id")
        plt.ylabel("count")
        plt.title("Samples per date_id")
        plt.tight_layout()
        p = os.path.join(figures_dir, "samples_per_date.png")
        plt.savefig(p)
        plt.close()
        saved["samples_per_date"] = p

    # session counts
    if "session" in df.columns:
        plt.figure(figsize=(4, 4))
        vc = df["session"].value_counts()
        sns.barplot(x=vc.index.astype(str), y=vc.values)
        plt.title("Samples per session")
        plt.tight_layout()
        p = os.path.join(figures_dir, "samples_per_session.png")
        plt.savefig(p)
        plt.close()
        saved["samples_per_session"] = p

    # n_midprice distribution
    if "n_midprice" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(pd.to_numeric(df["n_midprice"], errors="coerce").dropna(), kde=False, bins=80)
        plt.title("n_midprice distribution")
        plt.tight_layout()
        p = os.path.join(figures_dir, "n_midprice_hist.png")
        plt.savefig(p)
        plt.close()
        saved["n_midprice_hist"] = p

    # amount_delta distribution
    if "amount_delta" in df.columns:
        amount = pd.to_numeric(df["amount_delta"], errors="coerce").dropna()
        plt.figure(figsize=(6, 4))
        sns.histplot(amount, kde=False, bins=80)
        plt.title("amount_delta distribution")
        plt.tight_layout()
        p = os.path.join(figures_dir, "amount_delta_hist.png")
        plt.savefig(p)
        plt.close()
        saved["amount_delta_hist"] = p

        positive_amount = amount[amount >= 0]
        if len(positive_amount) > 0:
            log_amount = np.log1p(positive_amount)
            plt.figure(figsize=(6, 4))
            sns.histplot(log_amount, kde=False, bins=80)
            plt.title("log1p(amount_delta >= 0) distribution")
            plt.tight_layout()
            p = os.path.join(figures_dir, "log_amount_delta_hist.png")
            plt.savefig(p)
            plt.close()
            saved["log_amount_delta_hist"] = p

    # spread distribution
    if "n_ask1" in df.columns and "n_bid1" in df.columns:
        spread = pd.to_numeric(df["n_ask1"], errors="coerce") - pd.to_numeric(df["n_bid1"], errors="coerce")
        plt.figure(figsize=(6, 4))
        sns.histplot(spread.dropna(), bins=80)
        plt.title("bid-ask spread distribution (n_ask1 - n_bid1)")
        plt.tight_layout()
        p = os.path.join(figures_dir, "spread_hist.png")
        plt.savefig(p)
        plt.close()
        saved["spread_hist"] = p

        clipped = spread.dropna()
        if len(clipped) > 0:
            low = np.nanpercentile(clipped, 1)
            high = np.nanpercentile(clipped, 99)
            clipped = clipped[(clipped >= low) & (clipped <= high)]
            if len(clipped) > 0:
                plt.figure(figsize=(6, 4))
                sns.histplot(clipped, bins=80)
                plt.title("bid-ask spread clipped 1%-99% distribution")
                plt.tight_layout()
                p = os.path.join(figures_dir, "spread_clipped_hist.png")
                plt.savefig(p)
                plt.close()
                saved["spread_clipped_hist"] = p

    # missing ratio heatmap (columns with some missing)
    miss = df.isnull().mean()
    plt.figure(figsize=(8, 2))
    miss = miss[miss > 0]
    if not miss.empty:
        sns.barplot(x=miss.index.astype(str), y=miss.values)
        plt.xticks(rotation=45, ha='right')
        plt.title("Missing ratio per column (only >0)")
        plt.tight_layout()
        p = os.path.join(figures_dir, "missing_ratio.png")
        plt.savefig(p)
        plt.close()
        saved["missing_ratio"] = p

    # correlation heatmap for numeric fields
    num = df.select_dtypes(include=[np.number]).copy()
    if num.shape[1] >= 2:
        corr = num.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
        plt.title("Numeric fields correlation")
        plt.tight_layout()
        p = os.path.join(figures_dir, "correlation_heatmap.png")
        plt.savefig(p)
        plt.close()
        saved["correlation_heatmap"] = p

    return saved


def write_report(report_path: str, summary: Dict, field_check: Dict, dq: Dict, ob: Dict, label_res: Dict, figures: Dict, missing_stats: Dict):
    lines = []
    lines.append("# EDA Summary")
    lines.append("")
    lines.append("## 1. EDA 目标")
    lines.append("- 了解数据文件、字段、数据质量、订单簿异常与标签分布，为后续建模提供依据。")
    lines.append("")
    lines.append("## 2. 数据概览")
    lines.append(f"- 总文件数: {summary.get('total_files')}")
    lines.append(f"- 成功读取文件数: {summary.get('success_files')}")
    lines.append(f"- 失败文件数: {summary.get('failed_files')}")
    lines.append(f"- 总行数: {summary.get('total_rows')}")
    lines.append(f"- 总列数: {summary.get('total_columns')}")
    lines.append("")
    lines.append("## 3. 字段说明")
    lines.append("- 参考预期字段列表（部分字段可能缺失或额外存在，请见下文字段检查结果）。")
    lines.append("- 关键字段：date, time, sym, n_close, amount_delta, n_midprice, n_bid*, n_ask*, n_bsize*, n_asize*, label_*. ")
    lines.append("")
    lines.append("## 4. 文件检查结果")
    lines.append(f"- 读取失败文件: {json.dumps(summary.get('failed_file_list', []), ensure_ascii=False)}")
    lines.append(f"- 空文件: {summary.get('empty_files')}")
    lines.append("")
    lines.append("## 4.1 文件完整性")
    lines.append(f"- 理论文件数: {missing_stats.get('expected_file_count')}")
    lines.append(f"- 实际有效组合数: {missing_stats.get('actual_valid_combo_count')}")
    lines.append(f"- 实际文件数: {missing_stats.get('actual_file_count')}")
    lines.append(f"- 额外异常组合数: {missing_stats.get('extra_unexpected_count')}")
    lines.append(f"- 缺失组合数: {missing_stats.get('missing_file_count')}")
    lines.append(f"- missing_files.csv 保存路径: {missing_stats.get('missing_file_path')}")
    lines.append(f"- unexpected_files.csv 保存路径: {missing_stats.get('unexpected_file_path')}")
    lines.append("- 关系说明: 理论组合数 = 实际有效组合数 + 缺失组合数。实际文件数可能大于有效组合数，说明存在异常文件名或额外组合。")
    lines.append("")
    lines.append("## 5. 数据质量检查")
    lines.append(f"- 缺失值概览（部分）： { {k: v for k, v in list(dq.get('missing_counts', {}).items()) if v>0} }")
    lines.append(f"- 重复行数: {dq.get('duplicate_rows')}")
    lines.append(f"- Inf 值计数（部分）: { {k: v for k, v in list(dq.get('inf_counts', {}).items()) if v>0} }")
    lines.append("")
    lines.append("## 6. 订单簿异常检查")
    lines.append(f"- 订单簿异常统计（部分）: { {k:v for k,v in ob.items() if k.endswith('_count') or k.endswith('_ratio') or k=='size_negative_counts' or k=='size_negative_ratio'} }")
    lines.append("")
    lines.append("## 6. Midprice 规则检查")
    lines.append(f"- midprice_mismatch_count: {missing_stats.get('midprice_mismatch_count')}")
    lines.append(f"- midprice_mismatch_ratio: {missing_stats.get('midprice_mismatch_ratio')}")
    lines.append(f"- max_abs_midprice_diff: {missing_stats.get('max_abs_midprice_diff')}")
    lines.append(f"- mean_abs_midprice_diff: {missing_stats.get('mean_abs_midprice_diff')}")
    lines.append(f"- midprice_check.csv 保存路径: {missing_stats.get('midprice_check_path')}")
    lines.append("")
    lines.append("## 7. 标签方向统计")
    lines.append(f"- label_direction_check.csv 保存路径: {missing_stats.get('label_direction_path')}")
    if missing_stats.get('label_direction_summary'):
        for row in missing_stats['label_direction_summary']:
            lines.append(f"- {row['label']}: future_ratio={row['future_match_ratio']}, past_ratio={row['past_match_ratio']} -> better={row['better_direction']}")
    lines.append("")
    lines.append("## 8. 标签分布分析")
    if label_res:
        for key, value in label_res.items():
            if key.endswith("_counts"):
                lab = key.replace("_counts", "")
                lines.append(f"### {lab}")
                counts = value
                props = label_res.get(f"{lab}_props", {})
                for label_value, count in counts.items():
                    ratio = props.get(label_value, 0)
                    lines.append(f"- {label_value}: {count} (ratio={ratio})")
                lines.append("")
    else:
        lines.append("- 未检测到任何标签列。")
        lines.append("")
    lines.append("## 8. 可视化结果说明")
    lines.append("- 图表已生成并保存到 reports/figures/，包括标签分布、样本量、n_midprice、amount_delta、log_amount_delta、spread、spread_clipped、缺失比例、相关性热力图等。")
    lines.append("")
    lines.append("## 9. 初步结论")
    if missing_stats.get('extra_unexpected_count', 0) > 0:
        lines.append(f"- 数据存在 {missing_stats.get('extra_unexpected_count')} 个异常组合，建议优先检查 unexpected_files.csv。")
    if missing_stats.get('missing_file_count', 0) > 0:
        lines.append(f"- 共有 {missing_stats.get('missing_file_count')} 个理论组合缺失，缺失详情见 missing_files.csv。")
    if missing_stats.get('midprice_mismatch_ratio', 0) > 0.01:
        lines.append(f"- midprice 计算与记录存在较多不一致，差异比率为 {missing_stats.get('midprice_mismatch_ratio')}，请进一步校验 n_bid1/n_ask1 和 n_midprice 的来源。")
    else:
        lines.append(f"- midprice 规则总体一致， mismatch ratio={missing_stats.get('midprice_mismatch_ratio')}。")
    if missing_stats.get('label_direction_summary'):
        for row in missing_stats['label_direction_summary']:
            if row['better_direction'] == 'future':
                lines.append(f"- {row['label']} 更符合未来方向（future_ratio={row['future_match_ratio']} > past_ratio={row['past_match_ratio']}）。")
            elif row['better_direction'] == 'past':
                lines.append(f"- {row['label']} 更符合过去方向（past_ratio={row['past_match_ratio']} > future_ratio={row['future_match_ratio']}）。")
            else:
                lines.append(f"- {row['label']} 未来/过去方向匹配率相近。")
    lines.append("")
    lines.append("## 10. 后续建议")
    if missing_stats.get('extra_unexpected_count', 0) > 0:
        lines.append("- 对 unexpected_files.csv 中的异常文件做逐一排查，确认是否录入错误或文件名异常。")
    if missing_stats.get('missing_file_count', 0) > 0:
        lines.append("- missing_files.csv 列出缺失组合，可以用来补齐数据或排除缺失日期/品种。")
    if missing_stats.get('midprice_mismatch_ratio', 0) > 0.01:
        lines.append("- 若 midprice 误差较大，建议根据 n_bid1/n_ask1 重新计算 midprice 并比对标签生成逻辑。")
    lines.append("- 结合标签方向检查结果，优先采用更符合的方向特征进行后续建模。")
    lines.append("")
    lines.append("---")
    lines.append("### 生成的图表文件")
    for k, v in figures.items():
        lines.append(f"- {k}: {v}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data", "raw")
    figures_dir = os.path.join(base_dir, "reports", "figures")
    report_path = os.path.join(base_dir, "reports", "eda_summary.md")

    ensure_dirs([os.path.join(base_dir, "reports"), figures_dir])

    files = scan_csv_files(data_dir)
    print(f"Found {len(files)} csv files under {data_dir}")

    merged, stats = read_and_merge(files)

    summary = summarize_basic_stats(merged, stats)
    field_check = field_completeness(stats)
    dq = data_quality_checks(merged)
    ob = orderbook_checks(merged)
    label_res = label_analysis(merged)
    missing_stats = detect_missing_files(
        files,
        os.path.join(base_dir, "reports", "missing_files.csv"),
        os.path.join(base_dir, "reports", "unexpected_files.csv"),
    )
    midprice_stats = midprice_rule_check(merged, os.path.join(base_dir, "reports", "midprice_check.csv"))
    label_direction_stats = label_direction_check(merged, os.path.join(base_dir, "reports", "label_direction_check.csv"))
    missing_stats.update(midprice_stats)
    missing_stats.update(label_direction_stats)

    figures = generate_plots(merged, figures_dir)
    write_report(report_path, summary, field_check, dq, ob, label_res, figures, missing_stats)

    # final prints
    print("EDA completed")
    print(f"Report saved to: {report_path}")
    print(f"Figures saved to: {figures_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("EDA failed with exception:", e)
        raise
