import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm


FEATURE_COLUMNS = [
    "n_close", "amount_delta", "n_midprice",
    "n_bid1", "n_bsize1", "n_bid2", "n_bsize2", "n_bid3", "n_bsize3",
    "n_bid4", "n_bsize4", "n_bid5", "n_bsize5",
    "n_ask1", "n_asize1", "n_ask2", "n_asize2", "n_ask3", "n_asize3",
    "n_ask4", "n_asize4", "n_ask5", "n_asize5",
]


def parse_file_order(file_path):
    name = os.path.basename(file_path)
    try:
        sym_part = name.split("_")[1]
        date_part = name.split("_")[2]
        session_part = name.split("_")[3].split(".")[0]
        sym_idx = int(sym_part.replace("sym", "").replace("sym", ""))
        date_idx = int(date_part.replace("date", ""))
        session_idx = 0 if "am" in session_part else 1
        return date_idx, session_idx, sym_idx
    except Exception:
        return 0, 0, 0


def validation_files(data_dir, test_size):
    pattern = os.path.join(data_dir, "snapshot_sym*_date*_*.csv")
    csv_files = sorted(glob.glob(pattern), key=parse_file_order)
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")
    split_idx = int(len(csv_files) * (1 - test_size))
    return csv_files[split_idx:]


def make_window_features_for_file(file_path, label_name, window):
    df = pd.read_csv(file_path)
    df = df.sort_values(["sym", "date", "time"]).reset_index(drop=True)
    n_shift = int(label_name.split("_")[1])
    df["price_diff_raw"] = df.groupby(["sym", "date"])["n_midprice"].shift(-n_shift) - df["n_midprice"]

    available = [col for col in FEATURE_COLUMNS if col in df.columns]
    feature_df = df[available].copy()
    for col in feature_df.columns:
        feature_df[col] = feature_df[col].fillna(0)

    features, labels, diffs = [], [], []
    for _, group in df.groupby(["sym", "date"]):
        group_features = feature_df.loc[group.index].values
        group_labels = group[label_name].fillna(1).astype(int).values
        group_diffs = group["price_diff_raw"].fillna(0).values
        for i in range(len(group)):
            if i < window - 1:
                hist_features = group_features[: i + 1]
                padding = np.zeros((window - i - 1, len(available)))
                hist_features = np.vstack([padding, hist_features])
            else:
                hist_features = group_features[i - window + 1 : i + 1]
            features.append(hist_features.flatten())
            labels.append(int(group_labels[i]))
            diffs.append(float(group_diffs[i]))

    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int32), np.asarray(diffs, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--label", default="label_20")
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    with open(args.model_path, "rb") as f:
        bst = pickle.load(f)

    all_y, all_diffs, all_pred = [], [], []
    files = validation_files(args.data_dir, args.test_size)
    print(f"Validation files: {len(files)}")
    print(f"Model features: {bst.num_features()}, rounds: {bst.num_boosted_rounds()}, attrs: {bst.attributes()}")

    for file_path in tqdm(files, desc="Evaluating files", ncols=80):
        x_batch, y_batch, diffs_batch = make_window_features_for_file(file_path, args.label, args.window)
        pred_batch = bst.predict(xgb.DMatrix(x_batch))
        all_y.append(y_batch)
        all_diffs.append(diffs_batch)
        all_pred.append(pred_batch)

    y_true = np.concatenate(all_y).astype(int)
    p_diff_raw = np.concatenate(all_diffs)
    pred_proba = np.vstack(all_pred)
    print(f"Total samples for testing: {len(y_true)}")
    print(f"True counts [down, unchanged, up]: {np.bincount(y_true, minlength=3).tolist()}")

    actual_moves_mask = (y_true == 0) | (y_true == 2)
    total_actual_moves = np.sum(actual_moves_mask)
    thresholds = [0.0, 0.35, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.88, 0.90, 0.95]

    print("Threshold | Signal % | Trades | Prec | Recall | Total PnL | Avg PnL | F0.5 | Pred counts [d,u0,u]")
    for thr in thresholds:
        mask_down = pred_proba[:, 0] > thr
        mask_up = pred_proba[:, 2] > thr
        if thr == 0:
            final_pred_tmp = pred_proba.argmax(axis=1)
            mask_signal = (final_pred_tmp == 0) | (final_pred_tmp == 2)
        else:
            final_pred_tmp = np.ones(len(y_true), dtype=int)
            final_pred_tmp[mask_down] = 0
            final_pred_tmp[mask_up] = 2
            mask_signal = mask_down | mask_up

        pred_counts = np.bincount(final_pred_tmp, minlength=3)
        num_signals = int(np.sum(mask_signal))
        if num_signals == 0:
            print(f"{thr:.2f} | 0.00 | 0 | NA | NA | 0.000000 | 0.000000 | 0.0000 | {pred_counts.tolist()}")
            continue

        signal_pct = 100 * num_signals / len(y_true)
        preds_in_mask = pred_proba[mask_signal].argmax(axis=1)
        labels_in_mask = y_true[mask_signal]
        diffs_in_mask = p_diff_raw[mask_signal]
        correct = np.sum(preds_in_mask == labels_in_mask)
        precision = correct / num_signals
        recall = correct / (total_actual_moves + 1e-10)
        trade_pnls = np.where(preds_in_mask == 2, diffs_in_mask, -diffs_in_mask)
        total_pnl = float(np.sum(trade_pnls))
        avg_pnl = total_pnl / num_signals
        f05 = (1.25 * precision * recall) / (0.25 * precision + recall + 1e-10)
        print(f"{thr:.2f} | {signal_pct:.2f} | {num_signals} | {precision:.4f} | {recall:.4f} | {total_pnl:.6f} | {avg_pnl:.6f} | {f05:.4f} | {pred_counts.tolist()}")


if __name__ == "__main__":
    main()
