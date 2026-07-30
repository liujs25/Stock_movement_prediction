import argparse
import os
import sys
import glob
import re
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_process import DataProcessor
from model.model import XGBoostModel
from tqdm import tqdm

def tprint(*args, **kwargs):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{timestamp}]', *args, **kwargs)


# 1. 修改函数签名，增加 resume 参数
def train_model(data_dir, label_name, model_save_dir, window=100, test_size=0.2, random_state=42, cache_dir='./cache', batch_size=5000, use_cache=False, resume=False):
    import numpy as np
    import xgboost as xgb
    import gc
    
    num_boost_round = 2000
    
    tprint(f"Training model for {label_name}")
    tprint(f"Data directory: {data_dir}")
    tprint(f"Window size: {window}")
    tprint(f"Batch size: {batch_size}")
    tprint(f"Use cache: {use_cache}")
    tprint(f"Resume training: {resume}") # 打印是否续训
    
    train_meta = os.path.join(cache_dir, f'train_{label_name}.meta')
    val_meta = os.path.join(cache_dir, f'val_{label_name}.meta')
    
    if use_cache:
        if not os.path.exists(train_meta) or not os.path.exists(val_meta):
            raise FileNotFoundError(f"Cache files not found. train_meta: {train_meta}, val_meta: {val_meta}")
        
        tprint("Using existing cache files...")
        
        with open(train_meta, 'r') as f:
            buffer_files = [line.strip() for line in f if line.strip()]
        
        label_counts = {0: 0, 1: 0, 2: 0}
        for buffer_file in tqdm(buffer_files, desc="Calculating class weights", ncols=80):
            if os.path.exists(buffer_file):
                dm = xgb.DMatrix(buffer_file)
                labels = dm.get_label().astype(int)
                unique, counts = np.unique(labels, return_counts=True)
                for label, count in zip(unique, counts):
                    label_counts[int(label)] = label_counts.get(int(label), 0) + int(count)
                del dm
                gc.collect()
        
        train_label_counts = label_counts
    else:
        processor = DataProcessor(data_dir)
        
        tprint("Processing data and saving to binary DMatrix blocks...")
        train_meta, val_meta, train_label_counts = processor.get_train_test_split(
            label_name=label_name,
            test_size=test_size,
            random_state=random_state,
            window=window,
            cache_dir=cache_dir,
            batch_size=batch_size
        )
    
    tprint("Calculating class weights...")
    total = sum(train_label_counts.values())
    num_classes = len([v for v in train_label_counts.values() if v > 0])
    class_weights = {}
    for label, count in train_label_counts.items():
        class_weights[label] = total / (num_classes * count) if count > 0 else 1.0
    
    tprint(f"Class distribution - Down: {train_label_counts.get(0, 0)}, Unchanged: {train_label_counts.get(1, 0)}, Up: {train_label_counts.get(2, 0)}")
    tprint(f"Class weights - Down: {class_weights.get(0, 1.0):.3f}, Unchanged: {class_weights.get(1, 1.0):.3f}, Up: {class_weights.get(2, 1.0):.3f}")
    
    tprint("Loading buffer file lists...")
    with open(train_meta, 'r') as f:
        buffer_files = [line.strip() for line in f if line.strip()]
    
    with open(val_meta, 'r') as f:
        val_buffer_files = [line.strip() for line in f if line.strip()]
    
    tprint(f"Using external memory training with {len(buffer_files)} train buffers and {len(val_buffer_files)} val buffers")
    
    class DMatrixIterator(xgb.DataIter):
        def __init__(self, buffer_files, class_weights=None, desc="Loading", cache_prefix=None, use_profit_weight=True):
            self.buffer_files = buffer_files
            self.class_weights = class_weights
            self.use_profit_weight = use_profit_weight
            self.current_idx = 0
            self.desc = desc
            self.pbar = None
            self._cache_prefix = cache_prefix
            super().__init__(cache_prefix=cache_prefix)
        
        def next(self, input_data):
            if self.current_idx >= len(self.buffer_files):
                if self.pbar:
                    self.pbar.close()
                    self.pbar = None
                return 0
            if self.pbar is None:
                self.pbar = tqdm(total=len(self.buffer_files), desc=self.desc, ncols=80)
            
            buffer_file = self.buffer_files[self.current_idx]
            dm = xgb.DMatrix(buffer_file)
            X = dm.get_data()
            y = dm.get_label()
            
            w = np.ones_like(y, dtype=np.float32)
            
            if self.use_profit_weight:
                price_file = buffer_file.replace('.buffer', '.price.npy')
                if os.path.exists(price_file):
                    price_diffs = np.load(price_file)
                    abs_diffs = np.abs(price_diffs)
                    mean_diff = np.mean(abs_diffs) + 1e-10
                    
                    profit_w = 0.5 + 1.5 * (abs_diffs / mean_diff)
                    w = w * profit_w
            
            if self.class_weights:
                cw = np.array([self.class_weights.get(int(label), 1.0) for label in y])
                w = w * cw
            
            if self.use_profit_weight:
                w = np.clip(w, 0.1, 10.0)
            
            input_data(data=X, label=y, weight=w)
            
            del dm, X, y, w
            gc.collect()
            
            self.current_idx += 1
            self.pbar.update(1)
            return 1
        
        def reset(self):
            self.current_idx = 0
            if self.pbar:
                self.pbar.close()
                self.pbar = None
    
    tprint("Creating ExtMemQuantileDMatrix with Profit Weighting (train only)...")
    train_cache = os.path.join(cache_dir, f"extmem_train_{label_name}")
    val_cache = os.path.join(cache_dir, f"extmem_val_{label_name}")
    
    train_iter = DMatrixIterator(buffer_files, class_weights, desc="Loading train data", cache_prefix=train_cache, use_profit_weight=False)
    dtrain = xgb.ExtMemQuantileDMatrix(train_iter, max_bin=256)
    gc.collect()
    tprint("Training data loaded, memory cleaned")
    
    val_iter = DMatrixIterator(val_buffer_files, class_weights, desc="Loading val data", cache_prefix=val_cache, use_profit_weight=False)
    dval = xgb.ExtMemQuantileDMatrix(val_iter, max_bin=256, ref=dtrain)
    gc.collect()
    tprint("Validation data loaded, memory cleaned (no profit weighting for unbiased evaluation)")
    
    tprint(f"Train samples: {dtrain.num_row()}, Final features (with time lags): {dtrain.num_col()}")
    tprint(f"Validation samples: {dval.num_row()}, Final features (with time lags): {dval.num_col()}")
    
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.6,
        'colsample_bytree': 0.4,
        'min_child_weight': 50,
        'reg_alpha': 10.0,
        'reg_lambda': 2.0,
        'gamma': 0.5,
        'random_state': random_state,
        'nthread': -1,
        'tree_method': 'hist'
    }
    
    tprint("Training parameters:")
    tprint(f"  batch_size: {batch_size}")
    tprint(f"  num_boost_round: {num_boost_round}")
    for key, value in params.items():
        tprint(f"  {key}: {value}")
    
    # ==========================================
    # 🚀 Checkpoint 与 续训逻辑 (核心修改)
    # ==========================================
    
    checkpoint_dir = os.path.join(model_save_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    xgb_model_path = None
    
    # 2. 如果开启了 resume，则搜索 Checkpoint
    if resume:
        ckpt_pattern = os.path.join(checkpoint_dir, f"ckpt_{label_name}_*.json")
        existing_ckpts = glob.glob(ckpt_pattern)
        
        if existing_ckpts:
            latest_ckpt = None
            max_iter = -1
            
            for f in existing_ckpts:
                # 匹配文件名中的迭代数
                match = re.search(rf"ckpt_{label_name}_(\d+)\.json", f)
                if match:
                    iteration = int(match.group(1))
                    if iteration > max_iter:
                        max_iter = iteration
                        latest_ckpt = f
            
            if latest_ckpt:
                tprint(f"🔄 Found checkpoint! Resuming training from: {latest_ckpt} (Iteration {max_iter})")
                xgb_model_path = latest_ckpt
            else:
                tprint("⚠️ No valid checkpoint found matching pattern. Starting fresh training...")
        else:
            tprint("⚠️ No checkpoints found in directory. Starting fresh training...")
    else:
        tprint("🚀 Starting FRESH training (ignoring any existing checkpoints)...")

    # 3. 配置每250轮保存一次的回调
    ckpt_callback = xgb.callback.TrainingCheckPoint(
        directory=checkpoint_dir,
        name=f"ckpt_{label_name}",
        interval=250 # 每隔250个 tree 存一次
    )

    tprint("Training model with external memory...")
    xgb_model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dval, 'eval')] if dval else None,
        early_stopping_rounds=100,
        verbose_eval=10,
        xgb_model=xgb_model_path,   # 传入断点路径（如果不需要续训则为 None）
        callbacks=[ckpt_callback]   # 传入自动保存回调
    )
    
    model = XGBoostModel(**params)
    model.model = xgb_model
    
    os.makedirs(model_save_dir, exist_ok=True)
    model_path = os.path.join(model_save_dir, f"model_{label_name}.pth")
    model.save_model(model_path)
    tprint(f"Final Model saved to: {model_path}")
    
    tprint("Evaluating on validation set with Confidence Filter...")
    pred_proba = xgb_model.predict(dval)
    
    standard_pred = pred_proba.argmax(axis=1)
    val_labels = dval.get_label().astype(int)

    thresholds = [0.0, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.84, 0.90]
    tprint(f"{'Threshold':<10} | {'Signals %':<12} | {'Precision (Up/Down)':<20} | {'Recall (Up/Down)':<15}")
    tprint("-" * 65)

    for thr in thresholds:
        mask_down = pred_proba[:, 0] > thr
        mask_up = pred_proba[:, 2] > thr
        
        total_signals = np.sum(mask_down) + np.sum(mask_up)
        signal_pct = 100 * total_signals / len(val_labels)
        
        if total_signals > 0:
            correct_down = np.sum((val_labels[mask_down] == 0))
            correct_up = np.sum((val_labels[mask_up] == 2))
            
            precision = (correct_down + correct_up) / total_signals
            
            actual_moves = np.sum((val_labels == 0) | (val_labels == 2))
            recall = (correct_down + correct_up) / actual_moves
            
            tprint(f"{thr:<10.2f} | {signal_pct:<12.2f}% | {precision:<20.4f} | {recall:<15.4f}")
        else:
            tprint(f"{thr:<10.2f} | 0.00%         | N/A                  | 0.0000")

    from sklearn.metrics import classification_report
    tprint("\nFull Classification Report (Standard Argmax):")
    print(classification_report(val_labels, standard_pred, target_names=['Down', 'Unchanged', 'Up']))
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost models for stock price direction prediction')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing CSV files')
    parser.add_argument('--model_save_dir', type=str, default='./Experiments', help='Directory to save trained models')
    parser.add_argument('--cache_dir', type=str, default='/mnt/data/cache', help='Directory for binary DMatrix cache files')
    parser.add_argument('--label', type=str, default='all', 
                        choices=['label_5', 'label_10', 'label_20', 'label_40', 'label_60', 'all'],
                        help='Label to train (default: all)')
    parser.add_argument('--window', type=int, default=100, help='Window size for historical features')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size ratio')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--batch_size', type=int, default=5000, help='Batch size for processing data blocks')
    parser.add_argument('--use_cache', action='store_true', default=False, help='Use existing cache files instead of processing data')
    
    # 4. 在 main 函数增加 --resume 参数
    parser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint if available')
    
    args = parser.parse_args()
    
    labels = ['label_5', 'label_10', 'label_20', 'label_40', 'label_60'] if args.label == 'all' else [args.label]
    
    for label in labels:
        tprint(f"\n{'='*60}")
        train_model(
            data_dir=args.data_dir,
            label_name=label,
            model_save_dir=args.model_save_dir,
            window=args.window,
            test_size=args.test_size,
            random_state=args.random_state,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
            use_cache=args.use_cache,
            resume=args.resume  # 传入参数
        )
        tprint(f"{'='*60}\n")
    
    tprint("All models trained successfully!")


if __name__ == '__main__':
    main()