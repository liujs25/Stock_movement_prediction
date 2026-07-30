import argparse
import os
import sys
import numpy as np
import xgboost as xgb
import joblib
import scipy.sparse as sp
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.data_process import DataProcessor

def tprint(*args, **kwargs):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{timestamp}]', *args, **kwargs)

def evaluate_model(model_path, meta_path, data_dir, eval_threshold=0.88):
    tprint(f"Starting evaluation...")
    tprint(f"Model: {model_path}")
    tprint(f"Final Report Threshold: {eval_threshold}")

    # ==========================================
    # 🚀 1. 智能模型加载 (核心修改)
    # ==========================================
    try:
        # 判断后缀是否为 XGBoost 原生格式
        if model_path.endswith('.ubj') or model_path.endswith('.json') or model_path.endswith('.model'):
            tprint("Detected XGBoost native model format. Loading via xgb.Booster...")
            bst = xgb.Booster()
            bst.load_model(model_path)
        else:
            # 默认为 joblib 格式 (.pth/.pkl)
            tprint("Loading via joblib (.pth)...")
            loaded_obj = joblib.load(model_path)
            # 兼容 XGBoostModel 包装类或直接的 Booster 对象
            bst = loaded_obj.model if hasattr(loaded_obj, 'model') else loaded_obj
            
        tprint("Model loaded successfully.")
    except Exception as e:
        tprint(f"Error loading model: {e}")
        return

    # ==========================================
    # 2. 流式加载数据并分批预测（内存优化）
    # ==========================================
    if not os.path.exists(meta_path):
        tprint(f"Error: Meta file {meta_path} not found."); return
        
    with open(meta_path, 'r') as f:
        buffer_files = [line.strip() for line in f if line.strip()]
    
    tprint(f"Processing {len(buffer_files)} blocks in streaming mode...")
    
    pred_proba_list = []
    y_true_list = []
    p_diff_raw_list = []
    
    for idx, bf in enumerate(buffer_files):
        try:
            if (idx + 1) % 10 == 0:
                tprint(f"Processing block {idx + 1}/{len(buffer_files)}...")
            
            dm = xgb.DMatrix(bf)
            y_block = dm.get_label().astype(int)
            
            base_name, _ = os.path.splitext(bf)
            pf = base_name + '.price.npy'
            
            if os.path.exists(pf):
                p_diff_block = np.load(pf)
            else:
                pf_fallback = bf.replace('.buffer', '.price.npy')
                if os.path.exists(pf_fallback):
                    p_diff_block = np.load(pf_fallback)
                else:
                    p_diff_block = np.zeros(int(dm.num_row()))
            
            pred_proba_block = bst.predict(dm)
            
            pred_proba_list.append(pred_proba_block)
            y_true_list.append(y_block)
            p_diff_raw_list.append(p_diff_block)
                    
        except Exception as e:
            tprint(f"Error processing block {bf}: {e}")
            continue
    
    if not pred_proba_list:
        tprint("No valid data loaded.")
        return

    pred_proba = np.vstack(pred_proba_list)
    y_true = np.concatenate(y_true_list)
    p_diff_raw = np.concatenate(p_diff_raw_list)
    
    tprint(f"Total samples for testing: {len(y_true)}") 

    # 4. 置信度阈值扫描
    tprint("\n" + "="*135)
    tprint(f"{'Threshold':<8} | {'Signal %':<9} | {'Trades':<8} | {'Prec':<8} | {'Recall':<8} | {'Total PnL':<14} | {'Avg PnL':<12} | {'F0.5'}")
    tprint("-" * 135)

    actual_moves_mask = (y_true == 0) | (y_true == 2)
    total_actual_moves = np.sum(actual_moves_mask)
    thresholds = [0.0, 0.60, 0.65, 0.70, 0.72, 0.73, 0.74, 0.75, 0.78, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.88, 0.90]
    
    for thr in thresholds:
        mask_down = pred_proba[:, 0] > thr
        mask_up = pred_proba[:, 2] > thr
        
        if thr == 0:
            final_pred_tmp = pred_proba.argmax(axis=1)
            mask_signal = (final_pred_tmp == 0) | (final_pred_tmp == 2)
        else:
            mask_signal = mask_down | mask_up
            
        num_signals = np.sum(mask_signal)
        
        if num_signals > 0:
            signal_pct = 100 * num_signals / len(y_true)
            preds_in_mask = pred_proba[mask_signal].argmax(axis=1)
            labels_in_mask = y_true[mask_signal]
            diffs_in_mask = p_diff_raw[mask_signal]
            
            # 精准率
            correct = np.sum(preds_in_mask == labels_in_mask)
            precision = correct / num_signals
            recall = correct / (total_actual_moves + 1e-10)
            
            # --- 收益率计算逻辑 ---
            trade_pnls = np.where(preds_in_mask == 2, diffs_in_mask, -diffs_in_mask)
            total_pnl = np.sum(trade_pnls)
            avg_pnl = total_pnl / num_signals
            
            f05 = (1.25 * precision * recall) / (0.25 * precision + recall + 1e-10)
            
            tprint(f"{thr:<8.2f} | {signal_pct:<8.2f}% | {num_signals:<8} | {precision:<8.4f} | {recall:<8.4f} | {total_pnl:<14.6f} | {avg_pnl:<12.6f} | {f05:.4f}")
        else:
            tprint(f"{thr:<8.2f} | 0.00%     | 0        | N/A      | N/A      | 0.000000       | 0.000000     | 0.0000")

    tprint("="*135)

    # 5. 指定阈值下的最终报告
    tprint(f"\nFinal Summary Report at Threshold {eval_threshold}:")
    y_pred_final = np.ones(len(y_true))
    y_pred_final[pred_proba[:, 0] > eval_threshold] = 0
    y_pred_final[pred_proba[:, 2] > eval_threshold] = 2
    
    print(classification_report(y_true, y_pred_final, target_names=['Down', 'Unchanged', 'Up'], digits=4))
    
    final_mask = (y_pred_final == 0) | (y_pred_final == 2)
    if np.sum(final_mask) > 0:
        f_preds = y_pred_final[final_mask]
        f_diffs = p_diff_raw[final_mask]
        f_pnls = np.where(f_preds == 2, f_diffs, -f_diffs)
        tprint(f"Results for Threshold {eval_threshold}:")
        tprint(f"  - Total PnL: {np.sum(f_pnls):.6f}")
        tprint(f"  - Avg PnL:   {np.mean(f_pnls):.6f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default='/mnt/data/cache')
    parser.add_argument('--label', type=str, default='label_5')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=0.88)
    args = parser.parse_args()
    
    val_meta_path = os.path.join(args.cache_dir, f'val_{args.label}.meta')
    evaluate_model(args.model_path, val_meta_path, args.data_dir, eval_threshold=args.threshold)