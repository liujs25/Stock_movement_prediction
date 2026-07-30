import argparse
import os
import sys
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.data_process import DataProcessor

def tprint(*args, **kwargs):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{timestamp}]', *args, **kwargs)

def count_max_pnl(label_name, cache_dir='./cache', data_dir=None, top_percent=5, use_train=True, use_val=True):
    tprint(f"Analyzing PnL for {label_name}")
    tprint(f"Cache directory: {cache_dir}")
    tprint(f"Top {top_percent}% analysis")
    
    all_labels = []
    all_price_diffs = []
    
    if use_train:
        train_meta = os.path.join(cache_dir, f'train_{label_name}.meta')
        if os.path.exists(train_meta):
            tprint(f"Loading training data from {train_meta}...")
            with open(train_meta, 'r') as f:
                buffer_files = [line.strip() for line in f if line.strip()]
            
            for bf in tqdm(buffer_files, desc="Loading train buffers", ncols=80):
                if os.path.exists(bf):
                    dm = xgb.DMatrix(bf)
                    labels = dm.get_label().astype(int)
                    
                    pf = bf.replace('.buffer', '.price.npy')
                    if os.path.exists(pf):
                        price_diffs = np.load(pf)
                    else:
                        tprint(f"Warning: Price file {pf} missing! Skipping.")
                        continue
                    
                    all_labels.append(labels)
                    all_price_diffs.append(price_diffs)
                    del dm
        else:
            tprint(f"Warning: Train meta file {train_meta} not found.")
    
    if use_val:
        val_meta = os.path.join(cache_dir, f'val_{label_name}.meta')
        if os.path.exists(val_meta):
            tprint(f"Loading validation data from {val_meta}...")
            with open(val_meta, 'r') as f:
                buffer_files = [line.strip() for line in f if line.strip()]
            
            for bf in tqdm(buffer_files, desc="Loading val buffers", ncols=80):
                if os.path.exists(bf):
                    dm = xgb.DMatrix(bf)
                    labels = dm.get_label().astype(int)
                    
                    pf = bf.replace('.buffer', '.price.npy')
                    if os.path.exists(pf):
                        price_diffs = np.load(pf)
                    else:
                        tprint(f"Warning: Price file {pf} missing! Skipping.")
                        continue
                    
                    all_labels.append(labels)
                    all_price_diffs.append(price_diffs)
                    del dm
        else:
            tprint(f"Warning: Val meta file {val_meta} not found.")
    
    if not all_labels:
        tprint("Error: No data loaded!")
        return
    
    y_all = np.concatenate(all_labels).astype(int)
    p_diff_all = np.concatenate(all_price_diffs)
    
    tprint(f"\nTotal samples: {len(y_all)}")
    tprint(f"  Down (0): {np.sum(y_all == 0)}")
    tprint(f"  Unchanged (1): {np.sum(y_all == 1)}")
    tprint(f"  Up (2): {np.sum(y_all == 2)}")
    
    signal_mask = (y_all == 0) | (y_all == 2)
    signal_labels = y_all[signal_mask]
    signal_price_diffs = p_diff_all[signal_mask]
    
    tprint(f"\nTotal signals (Up + Down): {len(signal_labels)}")
    
    pnls = np.where(signal_labels == 2, signal_price_diffs, -signal_price_diffs)
    
    tprint(f"\nPnL Statistics:")
    tprint(f"  Min PnL: {np.min(pnls):.6f}")
    tprint(f"  Max PnL: {np.max(pnls):.6f}")
    tprint(f"  Mean PnL: {np.mean(pnls):.6f}")
    tprint(f"  Median PnL: {np.median(pnls):.6f}")
    tprint(f"  Std PnL: {np.std(pnls):.6f}")
    
    sorted_indices = np.argsort(pnls)[::-1]
    sorted_pnls = pnls[sorted_indices]
    
    top_n = max(1, int(len(sorted_pnls) * top_percent / 100))
    top_pnls = sorted_pnls[:top_n]
    
    tprint(f"\n{'='*80}")
    tprint(f"Top {top_percent}% Analysis:")
    tprint(f"  Number of trades: {top_n}")
    tprint(f"  Average PnL: {np.mean(top_pnls):.6f}")
    tprint(f"  Total PnL: {np.sum(top_pnls):.6f}")
    tprint(f"  Min PnL in top {top_percent}%: {np.min(top_pnls):.6f}")
    tprint(f"  Max PnL in top {top_percent}%: {np.max(top_pnls):.6f}")
    tprint(f"{'='*80}")
    
    up_signals = signal_labels == 2
    down_signals = signal_labels == 0
    
    up_pnls = pnls[up_signals]
    down_pnls = pnls[down_signals]
    
    if len(up_pnls) > 0:
        sorted_up_indices = np.argsort(up_pnls)[::-1]
        top_up_n = max(1, int(len(up_pnls) * top_percent / 100))
        top_up_pnls = up_pnls[sorted_up_indices[:top_up_n]]
        tprint(f"\nUp Signals (label=2) - Top {top_percent}%:")
        tprint(f"  Number of trades: {top_up_n}")
        tprint(f"  Average PnL: {np.mean(top_up_pnls):.6f}")
        tprint(f"  Total PnL: {np.sum(top_up_pnls):.6f}")
    
    if len(down_pnls) > 0:
        sorted_down_indices = np.argsort(down_pnls)[::-1]
        top_down_n = max(1, int(len(down_pnls) * top_percent / 100))
        top_down_pnls = down_pnls[sorted_down_indices[:top_down_n]]
        tprint(f"\nDown Signals (label=0) - Top {top_percent}%:")
        tprint(f"  Number of trades: {top_down_n}")
        tprint(f"  Average PnL: {np.mean(top_down_pnls):.6f}")
        tprint(f"  Total PnL: {np.sum(top_down_pnls):.6f}")
    
    return {
        'total_signals': len(signal_labels),
        'top_percent': top_percent,
        'top_n': top_n,
        'avg_pnl_top': np.mean(top_pnls),
        'total_pnl_top': np.sum(top_pnls)
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Count max PnL for top percentage of signals')
    parser.add_argument('--label', type=str, required=True, 
                       choices=['label_5', 'label_10', 'label_20', 'label_40', 'label_60'],
                       help='Label name (e.g., label_5, label_20)')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                       help='Directory containing cache files')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Data directory (not used if cache exists)')
    parser.add_argument('--top_percent', type=float, default=5.0,
                       help='Top percentage to analyze (default: 5.0)')
    parser.add_argument('--train_only', action='store_true',
                       help='Only analyze training data')
    parser.add_argument('--val_only', action='store_true',
                       help='Only analyze validation data')
    
    args = parser.parse_args()
    
    use_train = not args.val_only
    use_val = not args.train_only
    
    count_max_pnl(
        label_name=args.label,
        cache_dir=args.cache_dir,
        data_dir=args.data_dir,
        top_percent=args.top_percent,
        use_train=use_train,
        use_val=use_val
    )

