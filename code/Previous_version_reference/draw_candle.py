import argparse
import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.data_process import DataProcessor

def load_model(model_path):
    try:
        if model_path.endswith('.ubj') or model_path.endswith('.json') or model_path.endswith('.model'):
            bst = xgb.Booster()
            bst.load_model(model_path)
        else:
            loaded_obj = joblib.load(model_path)
            bst = loaded_obj.model if hasattr(loaded_obj, 'model') else loaded_obj
        return bst
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def create_candle_data(df, interval_seconds=30):
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S', errors='coerce')
    df = df.sort_values('time').reset_index(drop=True)
    
    df['time_bucket'] = (df['time'].astype('int64') // (interval_seconds * 1e9)).astype(int)
    
    ohlc = df.groupby('time_bucket').agg({
        'n_midprice': ['first', 'max', 'min', 'last'],
        'amount_delta': 'sum',
        'time': 'first'
    }).reset_index()
    
    ohlc.columns = ['time_bucket', 'open', 'high', 'low', 'close', 'volume', 'time']
    
    return ohlc

def predict_and_get_signals(processor, bst, group, threshold, window=100):
    group = group.copy().reset_index(drop=True)
    
    if len(group) < window + 10:
        return []
    
    df_features = processor._add_derived_features(group)
    available_features = [col for col in processor.feature_columns if col in df_features.columns]
    
    feature_df = df_features[available_features].copy()
    for col in feature_df.columns:
        feature_df[col] = feature_df[col].replace([np.inf, -np.inf], 0).fillna(0).astype(np.float32)
    
    mid_idx = available_features.index('n_midprice') if 'n_midprice' in available_features else 0
    imb_idx = available_features.index('total_imbalance') if 'total_imbalance' in available_features else 0
    
    signals = []
    feature_matrix = feature_df.values
    
    for i in range(window, len(group)):
        pyramid_vec = processor._assemble_pyramid_vector(feature_matrix, i, mid_idx, imb_idx)
        pyramid_vec = pyramid_vec.reshape(1, -1)
        
        dm = xgb.DMatrix(pyramid_vec)
        pred_proba = bst.predict(dm)[0]
        
        if pred_proba[0] > threshold:
            signals.append((i, 'sell', pred_proba[0]))
        elif pred_proba[2] > threshold:
            signals.append((i, 'buy', pred_proba[2]))
    
    return signals

def draw_candle_with_signals(data_dir, model_path, threshold=0.88, interval_seconds=30, output_dir='./candle_plots', window=100, date_start=60, date_end=75):
    print("Loading model...")
    bst = load_model(model_path)
    if bst is None:
        print("Failed to load model")
        return
    
    print("Loading raw data files...")
    pattern = os.path.join(data_dir, 'snapshot_sym*_date*_*.csv')
    csv_files = sorted(glob.glob(pattern))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return
    
    processor = DataProcessor(data_dir)
    
    sym_date_files = {}
    for csv_file in csv_files:
        basename = os.path.basename(csv_file)
        parts = basename.replace('.csv', '').split('_')
        if len(parts) >= 3:
            try:
                sym_val = int(parts[1].replace('sym', ''))
                date_val = int(parts[2].replace('date', ''))
                if date_start <= date_val <= date_end:
                    if sym_val not in sym_date_files:
                        sym_date_files[sym_val] = []
                    sym_date_files[sym_val].append((date_val, csv_file))
            except:
                continue
    
    print(f"Found {len(sym_date_files)} symbols with dates in range [{date_start}, {date_end}]")
    
    for sym_val in sorted(sym_date_files.keys()):
        sym_output_dir = os.path.join(output_dir, f'sym{sym_val}')
        os.makedirs(sym_output_dir, exist_ok=True)
        
        print(f"\nProcessing sym={sym_val}...")
        date_files = sorted(sym_date_files[sym_val], key=lambda x: x[0])
        
        for date_val, csv_file in tqdm(date_files, desc=f"Sym {sym_val}", ncols=80):
            try:
                df = pd.read_csv(csv_file)
                df = df.sort_values(['sym', 'date', 'time']).reset_index(drop=True)
                
                if len(df) == 0:
                    continue
                
                grouped = df.groupby(['sym', 'date'])
                for (sym_group, date_group), group in grouped:
                    if sym_group != sym_val or date_group != date_val:
                        continue
                    
                    if len(group) < window + 10:
                        continue
                    
                    group = group.reset_index(drop=True)
                    
                    signals = predict_and_get_signals(processor, bst, group, threshold, window)
                    
                    candle_df = create_candle_data(group, interval_seconds)
                    
                    if len(candle_df) < 5:
                        continue
                    
                    signal_time_indices = [s[0] for s in signals]
                    signal_types = [s[1] for s in signals]
                    signal_confidences = [s[2] for s in signals]
                    
                    fig, ax = plt.subplots(figsize=(16, 8))
                    
                    candle_width = 0.6
                    
                    for i, row in candle_df.iterrows():
                        open_price = row['open']
                        high_price = row['high']
                        low_price = row['low']
                        close_price = row['close']
                        
                        is_up = close_price >= open_price
                        color = '#26a69a' if is_up else '#ef5350'
                        
                        body_top = max(open_price, close_price)
                        body_bottom = min(open_price, close_price)
                        body_height = body_top - body_bottom
                        
                        if body_height == 0:
                            body_height = max((high_price - low_price) * 0.1, close_price * 0.001)
                            body_bottom = close_price - body_height / 2
                            body_top = close_price + body_height / 2
                        
                        ax.plot([i, i], [low_price, high_price], color='black', linewidth=0.8, zorder=1)
                        
                        rect = patches.Rectangle((i - candle_width/2, body_bottom), 
                                                candle_width, body_height,
                                                facecolor=color, edgecolor='black', linewidth=0.8, zorder=2)
                        ax.add_patch(rect)
                    
                    for sig_idx, sig_type, conf in zip(signal_time_indices, signal_types, signal_confidences):
                        tick_idx = sig_idx
                        ticks_per_candle = max(1, interval_seconds // 3)
                        candle_idx = tick_idx // ticks_per_candle
                        if candle_idx < len(candle_df):
                            candle_row = candle_df.iloc[candle_idx]
                            high_price = candle_row['high']
                            low_price = candle_row['low']
                            close_price = candle_row['close']
                            
                            offset = high_price - low_price
                            if offset == 0:
                                offset = close_price * 0.02
                            
                            if sig_type == 'buy':
                                marker = '^'
                                color = 'blue'
                                arrow_y = low_price - offset * 0.3
                                text_y = low_price - offset * 0.5
                            else:
                                marker = 'v'
                                color = 'red'
                                arrow_y = high_price + offset * 0.3
                                text_y = high_price + offset * 0.5
                            
                            ax.scatter(candle_idx, arrow_y, s=200, marker=marker, 
                                     color=color, edgecolors='black', linewidths=2, zorder=5)
                            ax.text(candle_idx, text_y, f'{conf:.2f}', 
                                  fontsize=8, ha='center', color=color, weight='bold')
                    
                    ax.set_xlabel('Time (Candle Index)', fontsize=12)
                    ax.set_ylabel('Price', fontsize=12)
                    ax.set_title(f'Sym: {sym_val}, Date: {date_val}, Threshold: {threshold}\n'
                               f'Buy signals: {sum(1 for s in signal_types if s == "buy")}, '
                               f'Sell signals: {sum(1 for s in signal_types if s == "sell")}',
                               fontsize=14)
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    
                    output_file = os.path.join(sym_output_dir, f'date{date_val}.png')
                    plt.savefig(output_file, dpi=150, bbox_inches='tight')
                    plt.close()
            
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\nAll plots saved to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw candlestick charts with buy/sell signals')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing CSV files')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--threshold', type=float, default=0.88, help='Confidence threshold for signals')
    parser.add_argument('--interval_seconds', type=int, default=30, help='Candle interval in seconds')
    parser.add_argument('--output_dir', type=str, default='./candle_plots', help='Output directory for plots')
    parser.add_argument('--window', type=int, default=100, help='Window size for feature assembly')
    
    args = parser.parse_args()
    
    draw_candle_with_signals(
        data_dir=args.data_dir,
        model_path=args.model_path,
        threshold=args.threshold,
        interval_seconds=args.interval_seconds,
        output_dir=args.output_dir,
        window=args.window,
        date_start=60,
        date_end=75
    )
