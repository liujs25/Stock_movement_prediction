import argparse
import os
import pickle

import numpy as np
import scipy.sparse as sp
import xgboost as xgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--label", default="label_20")
    parser.add_argument("--threshold", type=float, default=0.88)
    args = parser.parse_args()

    meta_path = os.path.join(args.cache_dir, f"val_{args.label}.meta")
    with open(args.model_path, "rb") as f:
        bst = pickle.load(f)

    with open(meta_path, "r", encoding="utf-8") as f:
        buffer_files = [line.strip() for line in f if line.strip()]

    xs, ys, price_deltas = [], [], []
    for bf in buffer_files:
        dm = xgb.DMatrix(bf)
        xs.append(dm.get_data())
        ys.append(dm.get_label())
        base_name, _ = os.path.splitext(bf)
        pf = base_name + ".price.npy"
        if os.path.exists(pf):
            price_deltas.append(np.load(pf))
        else:
            price_deltas.append(np.zeros(int(dm.num_row()), dtype=np.float32))

    x_all = sp.vstack(xs) if sp.issparse(xs[0]) else np.vstack(xs)
    y_true = np.concatenate(ys).astype(int)
    p_diff_raw = np.concatenate(price_deltas)
    dtest = xgb.DMatrix(x_all, label=y_true)
    pred_proba = bst.predict(dtest)

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
            pred_counts = np.bincount(final_pred_tmp, minlength=3)
        else:
            mask_signal = mask_down | mask_up
            final_pred_tmp = np.ones(len(y_true), dtype=int)
            final_pred_tmp[mask_down] = 0
            final_pred_tmp[mask_up] = 2
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

    final_pred = np.ones(len(y_true), dtype=int)
    final_pred[pred_proba[:, 0] > args.threshold] = 0
    final_pred[pred_proba[:, 2] > args.threshold] = 2
    print(f"Final threshold {args.threshold} pred counts [down, unchanged, up]: {np.bincount(final_pred, minlength=3).tolist()}")


if __name__ == "__main__":
    main()
