import os
import pandas as pd
import glob
import xgboost as xgb
import numpy as np
from tqdm import tqdm


class DataProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.base_feature_names = [
            'n_close', 'amount_delta', 'n_midprice',
            'n_bid1', 'n_bsize1', 'n_bid2', 'n_bsize2', 'n_bid3', 'n_bsize3',
            'n_bid4', 'n_bsize4', 'n_bid5', 'n_bsize5',
            'n_ask1', 'n_asize1', 'n_ask2', 'n_asize2', 'n_ask3', 'n_asize3',
            'n_ask4', 'n_asize4', 'n_ask5', 'n_asize5',
            'spread_1', 'spread_3', 'spread_5',
            'mid_price_1', 'mid_price_3', 'mid_price_5',
            'relative_bid_density_1', 'relative_ask_density_1',
            'relative_bid_density_3', 'relative_ask_density_3',
            'weighted_ab_1', 'weighted_ab_3',
            'vol1_rel_diff', 'vol3_rel_diff', 'vol5_rel_diff',
            'amount_normalized',
            'log_bsize1', 'log_asize1', 'log_bsize3', 'log_asize3', 'log_bsize5', 'log_asize5',
            'close_delta', 'bid1_delta', 'ask1_delta', 'midprice_delta',
            'close_mean', 'close_std', 'close_vs_mean',
            'bid1_mean', 'bid1_std', 'bid1_vs_mean',
            'bid3_mean', 'bid3_std', 'bid3_vs_mean',
            'bid5_mean', 'bid5_std', 'bid5_vs_mean',
            'ask1_mean', 'ask1_std', 'ask1_vs_mean',
            'ask3_mean', 'ask3_std', 'ask3_vs_mean',
            'ask5_mean', 'ask5_std', 'ask5_vs_mean',
            'bsize1_mean', 'bsize1_std', 'bsize1_vs_mean',
            'bsize3_mean', 'bsize3_std', 'bsize3_vs_mean',
            'bsize5_mean', 'bsize5_std', 'bsize5_vs_mean',
            'asize1_mean', 'asize1_std', 'asize1_vs_mean',
            'asize3_mean', 'asize3_std', 'asize3_vs_mean',
            'asize5_mean', 'asize5_std', 'asize5_vs_mean',
            'midprice_mean', 'midprice_std',
            'mid_price_1_mean', 'mid_price_1_std',
            'mid_price_3_mean', 'mid_price_3_std',
            'mid_price_5_mean', 'mid_price_5_std',
            'time_seconds', 'time_interval',
            'bid1_plus1', 'bid3_plus1', 'bid5_plus1',
            'ask1_plus1', 'ask3_plus1', 'ask5_plus1',
            'cross_weighted_1', 'cross_weighted_2',
            'midprice_ma5',
            'volatility_5', 'volatility_10', 'volatility_20', 'volatility_40', 'volatility_60',
            'macd_dif', 'macd_dea', 'macd_bar',
            'kdj_k', 'kdj_d', 'kdj_j',
            'roc_1', 'roc_5', 'roc_10', 'roc_30', 'roc_60', 'roc_100',
            'vol1_rel_diff_mean_5', 'vol1_rel_diff_mean_20',
            'price_zscore_20', 'price_zscore_100', 'price_zscore_300',
            'price_slope_20', 'price_slope_100', 'price_slope_300',
            'price_percentile_100',
            'amount_zscore_20', 'amount_zscore_100', 'amount_zscore_300',
            'amount_slope_20', 'amount_slope_100', 'amount_slope_300',
            'lag_mid_1', 'lag_mid_5', 'lag_mid_20',
            'lag_bid1_1', 'lag_bid1_5', 'lag_ask1_1', 'lag_ask1_5',
            'lag_bsize1_1', 'lag_bsize1_5', 'lag_asize1_1', 'lag_asize1_5',
            'volume_flow_5', 'volume_flow_20', 'volume_flow_60',
            'total_imbalance', 'total_imbalance_weighted',
            'bid_slope', 'ask_slope',
            'price_elasticity_10', 'orderbook_pressure',
            'ofi_1', 'ofi_avg_3', 'ofi_1_rolling_5', 'ofi_spread_ratio',
            'midprice_accel', 'imb_velocity', 'imb_accel', 'energy_burst',
            'micro_price', 'micro_price_diff', 'weighted_depth_imbalance',
            'ofi_momentum_sync', 'vov_10', 'bid_convexity', 'ask_convexity',
            'buy_intensity', 'sell_intensity', 'ofi_ema_5', 'ofi_ema_10',
            'vpin_5', 'vpin_20', 'book_curvature', 'depth_pressure'
        ]
        self.raw_snapshot_cols = [
            'n_midprice', 'n_bid1', 'n_ask1', 'n_bsize1', 'n_asize1',
            'vol1_rel_diff', 'spread_1', 'amount_delta'
        ]
        self.core_snapshot_cols = [
            'n_midprice', 'n_bid1', 'n_ask1', 'n_bsize1', 'n_asize1', 
            'vol1_rel_diff', 'spread_1', 'amount_delta'
        ]
        self.feature_columns = self.base_feature_names
        self.final_feature_names = self._generate_final_names()
        self.label_columns = ['label_5', 'label_10', 'label_20', 'label_40', 'label_60']
    
    def _generate_final_names(self):
        names = []
        for lag in [0, 1, 2, 3, 4]:
            names += [f"{c}_t{lag}" for c in self.base_feature_names]
        for lag in [5, 10, 20, 40, 80, 100]:
            names += [f"mid_lag{lag}", f"imb_lag{lag}"]
        return names
    
    def _assemble_pyramid_vector(self, f_matrix, i, mid_idx, imb_idx):
        full_frames = []
        for lag in [0, 1, 2, 3, 4]:
            idx = max(0, i - lag)
            full_frames.append(f_matrix[idx])
        
        history_samples = []
        for lag in [5, 10, 20, 40, 80, 100]:
            idx = max(0, i - lag)
            history_samples.append([f_matrix[idx][mid_idx], f_matrix[idx][imb_idx]])
        
        return np.concatenate(full_frames + history_samples)
    
    def _add_derived_features(self, df):
        feats = {}
        
        def time_to_seconds(time_str):
            parts = str(time_str).split(':')
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        
        t_seconds = df['time'].apply(time_to_seconds)
        feats['time_seconds'] = t_seconds
        feats['time_interval'] = t_seconds.apply(lambda x: min(int((x - 34200) / 1800), 7) if x >= 34200 else 0)
        
        for i in [1, 3, 5]:
            feats[f'spread_{i}'] = df[f'n_ask{i}'] - df[f'n_bid{i}']
            feats[f'mid_price_{i}'] = (df[f'n_ask{i}'] + df[f'n_bid{i}']) / 2
            total_size = df[f'n_bsize{i}'] + df[f'n_asize{i}']
            feats[f'relative_bid_density_{i}'] = df[f'n_bsize{i}'] / (total_size + 1e-10)
            feats[f'relative_ask_density_{i}'] = df[f'n_asize{i}'] / (total_size + 1e-10)
        
        for i in [1, 3]:
            feats[f'weighted_ab_{i}'] = (df[f'n_bid{i}'] * df[f'n_asize{i}'] + df[f'n_ask{i}'] * df[f'n_bsize{i}']) / (df[f'n_bsize{i}'] + df[f'n_asize{i}'] + 1e-10)
        
        feats['vol1_rel_diff'] = (df['n_bsize1'] - df['n_asize1']) / (df['n_bsize1'] + df['n_asize1'] + 1e-10)
        feats['vol3_rel_diff'] = (df['n_bsize1'] + df['n_bsize2'] + df['n_bsize3'] - df['n_asize1'] - df['n_asize2'] - df['n_asize3']) / (df['n_bsize1'] + df['n_bsize2'] + df['n_bsize3'] + df['n_asize1'] + df['n_asize2'] + df['n_asize3'] + 1e-10)
        feats['vol5_rel_diff'] = (df['n_bsize1'] + df['n_bsize2'] + df['n_bsize3'] + df['n_bsize4'] + df['n_bsize5'] - df['n_asize1'] - df['n_asize2'] - df['n_asize3'] - df['n_asize4'] - df['n_asize5']) / (df['n_bsize1'] + df['n_bsize2'] + df['n_bsize3'] + df['n_bsize4'] + df['n_bsize5'] + df['n_asize1'] + df['n_asize2'] + df['n_asize3'] + df['n_asize4'] + df['n_asize5'] + 1e-10)
        
        feats['amount_normalized'] = np.log1p(df['amount_delta'] / (1 + df['n_midprice']))
        
        for i in [1, 3, 5]:
            feats[f'log_bsize{i}'] = np.log1p(df[f'n_bsize{i}'])
            feats[f'log_asize{i}'] = np.log1p(df[f'n_asize{i}'])
        
        grouped = df.groupby(['sym', 'date'])
        feats['close_delta'] = grouped['n_close'].diff()
        feats['bid1_delta'] = grouped['n_bid1'].diff()
        feats['ask1_delta'] = grouped['n_ask1'].diff()
        feats['midprice_delta'] = grouped['n_midprice'].diff()
        
        feats['close_mean'] = grouped['n_close'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
        feats['close_std'] = grouped['n_close'].transform(lambda x: x.rolling(window=10, min_periods=1).std())
        feats['close_vs_mean'] = df['n_close'] / (feats['close_mean'] + 1e-10)
        
        for i in [1, 3, 5]:
            feats[f'bid{i}_mean'] = grouped[f'n_bid{i}'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
            feats[f'bid{i}_std'] = grouped[f'n_bid{i}'].transform(lambda x: x.rolling(window=10, min_periods=1).std())
            feats[f'bid{i}_vs_mean'] = df[f'n_bid{i}'] / (feats[f'bid{i}_mean'] + 1e-10)
            
            feats[f'ask{i}_mean'] = grouped[f'n_ask{i}'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
            feats[f'ask{i}_std'] = grouped[f'n_ask{i}'].transform(lambda x: x.rolling(window=10, min_periods=1).std())
            feats[f'ask{i}_vs_mean'] = df[f'n_ask{i}'] / (feats[f'ask{i}_mean'] + 1e-10)
            
            feats[f'bsize{i}_mean'] = grouped[f'n_bsize{i}'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
            feats[f'bsize{i}_std'] = grouped[f'n_bsize{i}'].transform(lambda x: x.rolling(window=10, min_periods=1).std())
            feats[f'bsize{i}_vs_mean'] = df[f'n_bsize{i}'] / (feats[f'bsize{i}_mean'] + 1e-10)
            
            feats[f'asize{i}_mean'] = grouped[f'n_asize{i}'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
            feats[f'asize{i}_std'] = grouped[f'n_asize{i}'].transform(lambda x: x.rolling(window=10, min_periods=1).std())
            feats[f'asize{i}_vs_mean'] = df[f'n_asize{i}'] / (feats[f'asize{i}_mean'] + 1e-10)
            
            temp_mid_price_df = pd.DataFrame({'sym': df['sym'], 'date': df['date'], 'mid_price': feats[f'mid_price_{i}']}, index=df.index)
            feats[f'mid_price_{i}_mean'] = temp_mid_price_df.groupby(['sym', 'date'])['mid_price'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
            feats[f'mid_price_{i}_std'] = temp_mid_price_df.groupby(['sym', 'date'])['mid_price'].transform(lambda x: x.rolling(window=10, min_periods=1).std())
        
        feats['midprice_mean'] = grouped['n_midprice'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
        feats['midprice_std'] = grouped['n_midprice'].transform(lambda x: x.rolling(window=10, min_periods=1).std())
        
        for i in [1, 3, 5]:
            feats[f'bid{i}_plus1'] = df[f'n_bid{i}'] + 1
            feats[f'ask{i}_plus1'] = df[f'n_ask{i}'] + 1
        
        feats['cross_weighted_1'] = (df['n_ask1'] * df['n_bsize2'] + df['n_ask2'] * df['n_bsize1']) / (df['n_bsize1'] + df['n_bsize2'] + 1e-10)
        feats['cross_weighted_2'] = (df['n_bid1'] * df['n_asize2'] + df['n_bid2'] * df['n_asize1']) / (df['n_asize1'] + df['n_asize2'] + 1e-10)
        
        feats['midprice_ma5'] = grouped['n_midprice'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        
        temp_mid = 2 + df['n_ask1'] + df['n_bid1']
        temp_mid_df = pd.DataFrame({'sym': df['sym'], 'date': df['date'], 'temp_mid': temp_mid}, index=df.index)
        for period in [5, 10, 20, 40, 60]:
            feats[f'volatility_{period}'] = temp_mid_df.groupby(['sym', 'date'])['temp_mid'].transform(
                lambda x: (x / (x.shift(period) + 1e-10) - 1).fillna(0)
            )
        
        ema12 = grouped['n_midprice'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
        ema26 = grouped['n_midprice'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
        feats['macd_dif'] = ema12 - ema26
        temp_df = pd.DataFrame({'sym': df['sym'], 'date': df['date'], 'dif': feats['macd_dif']}, index=df.index)
        feats['macd_dea'] = temp_df.groupby(['sym', 'date'])['dif'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
        feats['macd_bar'] = feats['macd_dif'] - feats['macd_dea']
        
        low_9 = grouped['n_bid1'].transform(lambda x: x.rolling(window=9, min_periods=1).min())
        high_9 = grouped['n_ask1'].transform(lambda x: x.rolling(window=9, min_periods=1).max())
        rsv = 100 * (df['n_midprice'] - low_9) / (high_9 - low_9 + 1e-10)
        
        rsv_df = pd.DataFrame({'sym': df['sym'], 'date': df['date'], 'rsv': rsv}, index=df.index)
        feats['kdj_k'] = rsv_df.groupby(['sym', 'date'])['rsv'].transform(lambda x: x.ewm(alpha=1/3, adjust=False).mean())
        
        k_df = pd.DataFrame({'sym': df['sym'], 'date': df['date'], 'k': feats['kdj_k']}, index=df.index)
        feats['kdj_d'] = k_df.groupby(['sym', 'date'])['k'].transform(lambda x: x.ewm(alpha=1/3, adjust=False).mean())
        feats['kdj_j'] = 3 * feats['kdj_k'] - 2 * feats['kdj_d']
        
        new_df = pd.concat([df, pd.DataFrame(feats, index=df.index)], axis=1)
        new_df = self._add_advanced_features(new_df)
        return new_df.copy()
    
    def _add_advanced_features(self, df):
        grouped = df.groupby(['sym', 'date'])
        feats = {}
        
        mid = df['n_midprice']
        
        for w in [1, 5, 10, 30, 60, 100]:
            feats[f'roc_{w}'] = grouped['n_midprice'].transform(lambda x: x / x.shift(w) - 1).fillna(0)
        
        vol1_rel_diff = df['vol1_rel_diff'] if 'vol1_rel_diff' in df.columns else (df['n_bsize1'] - df['n_asize1']) / (df['n_bsize1'] + df['n_asize1'] + 1e-10)
        vol1_df = pd.DataFrame({'sym': df['sym'], 'date': df['date'], 'vol1': vol1_rel_diff}, index=df.index)
        feats['vol1_rel_diff_mean_5'] = vol1_df.groupby(['sym', 'date'])['vol1'].transform(lambda x: x.rolling(window=5, min_periods=1).mean()).fillna(0)
        feats['vol1_rel_diff_mean_20'] = vol1_df.groupby(['sym', 'date'])['vol1'].transform(lambda x: x.rolling(window=20, min_periods=1).mean()).fillna(0)
        
        for window in [20, 100, 300]:
            roll_mid = grouped['n_midprice'].rolling(window=window, min_periods=1)
            mid_mean = roll_mid.mean().reset_index(level=[0,1], drop=True)
            mid_std = roll_mid.std().reset_index(level=[0,1], drop=True)
            feats[f'price_zscore_{window}'] = (mid - mid_mean) / (mid_std + 1e-10)
            
            mid_recent_mean = grouped['n_midprice'].transform(lambda x: x.rolling(window=window//3, min_periods=1).mean())
            mid_early_mean = grouped['n_midprice'].transform(lambda x: x.shift(window*2//3).rolling(window=window//3, min_periods=1).mean())
            feats[f'price_slope_{window}'] = (mid_recent_mean - mid_early_mean) / (window * 2 // 3 + 1e-10)
            
            roll_amount = grouped['amount_delta'].rolling(window=window, min_periods=1)
            amount_mean = roll_amount.mean().reset_index(level=[0,1], drop=True)
            amount_std = roll_amount.std().reset_index(level=[0,1], drop=True)
            feats[f'amount_zscore_{window}'] = (df['amount_delta'] - amount_mean) / (amount_std + 1e-10)
            
            amount_recent_mean = grouped['amount_delta'].transform(lambda x: x.rolling(window=window//3, min_periods=1).mean())
            amount_early_mean = grouped['amount_delta'].transform(lambda x: x.shift(window*2//3).rolling(window=window//3, min_periods=1).mean())
            feats[f'amount_slope_{window}'] = (amount_recent_mean - amount_early_mean) / (window * 2 // 3 + 1e-10)
        
        roll100_max = grouped['n_midprice'].rolling(window=100, min_periods=1).max().reset_index(level=[0,1], drop=True)
        roll100_min = grouped['n_midprice'].rolling(window=100, min_periods=1).min().reset_index(level=[0,1], drop=True)
        feats['price_percentile_100'] = (mid - roll100_min) / (roll100_max - roll100_min + 1e-10)
        
        total_bid_size = df['n_bsize1'] + df['n_bsize2'] + df['n_bsize3'] + df['n_bsize4'] + df['n_bsize5']
        total_ask_size = df['n_asize1'] + df['n_asize2'] + df['n_asize3'] + df['n_asize4'] + df['n_asize5']
        feats['total_imbalance'] = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size + 1e-10)
        
        weighted_bid = (df['n_bid1'] * df['n_bsize1'] + df['n_bid2'] * df['n_bsize2'] + 
                       df['n_bid3'] * df['n_bsize3'] + df['n_bid4'] * df['n_bsize4'] + 
                       df['n_bid5'] * df['n_bsize5']) / (total_bid_size + 1e-10)
        weighted_ask = (df['n_ask1'] * df['n_asize1'] + df['n_ask2'] * df['n_asize2'] + 
                       df['n_ask3'] * df['n_asize3'] + df['n_ask4'] * df['n_asize4'] + 
                       df['n_ask5'] * df['n_asize5']) / (total_ask_size + 1e-10)
        feats['total_imbalance_weighted'] = (weighted_bid - weighted_ask) / (weighted_bid + weighted_ask + 1e-10)
        
        bid_price_diff = (df['n_bid1'] - df['n_bid5']) / (df['n_bid1'] + 1e-10)
        ask_price_diff = (df['n_ask5'] - df['n_ask1']) / (df['n_ask1'] + 1e-10)
        feats['bid_slope'] = bid_price_diff
        feats['ask_slope'] = ask_price_diff
        
        price_change_10 = grouped['n_midprice'].transform(lambda x: x.diff(10).fillna(0))
        volume_sum_10 = grouped['amount_delta'].transform(lambda x: x.rolling(window=10, min_periods=1).sum())
        feats['price_elasticity_10'] = price_change_10 / (volume_sum_10 + 1e-10)
        
        orderbook_pressure = (total_bid_size - total_ask_size) * (df['n_midprice'] - df['n_bid1']) / (df['n_ask1'] - df['n_bid1'] + 1e-10)
        feats['orderbook_pressure'] = orderbook_pressure
        
        def calc_ofi_series(bid_p, bid_v, ask_p, ask_v):
            prev_bid_p = bid_p.shift(1)
            prev_bid_v = bid_v.shift(1)
            prev_ask_p = ask_p.shift(1)
            prev_ask_v = ask_v.shift(1)
            
            ofi_bid = np.where(bid_p > prev_bid_p, bid_v,
                      np.where(bid_p == prev_bid_p, bid_v - prev_bid_v, -prev_bid_v))
            ofi_ask = np.where(ask_p < prev_ask_p, ask_v,
                      np.where(ask_p == prev_ask_p, ask_v - prev_ask_v, -prev_ask_v))
            
            return pd.Series(ofi_bid - ofi_ask, index=bid_p.index)
        
        ofi1_list = []
        ofi2_list = []
        ofi3_list = []
        for (sym, date), group in grouped:
            ofi1_list.append(calc_ofi_series(group['n_bid1'], group['n_bsize1'], group['n_ask1'], group['n_asize1']).fillna(0))
            ofi2_list.append(calc_ofi_series(group['n_bid2'], group['n_bsize2'], group['n_ask2'], group['n_asize2']).fillna(0))
            ofi3_list.append(calc_ofi_series(group['n_bid3'], group['n_bsize3'], group['n_ask3'], group['n_asize3']).fillna(0))
        
        feats['ofi_1'] = pd.concat(ofi1_list).reindex(df.index).fillna(0)
        ofi2_series = pd.concat(ofi2_list).reindex(df.index).fillna(0)
        ofi3_series = pd.concat(ofi3_list).reindex(df.index).fillna(0)
        feats['ofi_avg_3'] = (feats['ofi_1'] + ofi2_series + ofi3_series) / 3
        
        temp_ofi_df = pd.DataFrame({'sym': df['sym'], 'date': df['date'], 'ofi_1': feats['ofi_1']}, index=df.index)
        temp_ofi_grouped = temp_ofi_df.groupby(['sym', 'date'])
        feats['ofi_1_rolling_5'] = temp_ofi_grouped['ofi_1'].transform(lambda x: x.rolling(window=5, min_periods=1).sum()).fillna(0)
        
        spread_1 = df['spread_1'] if 'spread_1' in df.columns else (df['n_ask1'] - df['n_bid1'])
        feats['ofi_spread_ratio'] = feats['ofi_1'] / (spread_1 + 1e-10)
        
        if 'midprice_delta' in df.columns:
            feats['midprice_accel'] = grouped['midprice_delta'].diff().fillna(0)
        else:
            feats['midprice_accel'] = grouped['n_midprice'].transform(lambda x: x.diff().diff()).fillna(0)
        
        feats['energy_burst'] = df['amount_delta'] * feats['midprice_accel']
        
        imb_vel_list = []
        imb_accel_list = []
        for (sym, date), group in grouped:
            group_imb = feats['total_imbalance'].loc[group.index]
            group_vel = group_imb.diff().fillna(0)
            group_accel = group_vel.diff().fillna(0)
            imb_vel_list.append(group_vel)
            imb_accel_list.append(group_accel)
        
        feats['imb_velocity'] = pd.concat(imb_vel_list).reindex(df.index).fillna(0)
        feats['imb_accel'] = pd.concat(imb_accel_list).reindex(df.index).fillna(0)
        
        for lag in [1, 5, 20]:
            feats[f'lag_mid_{lag}'] = grouped['n_midprice'].shift(lag).fillna(0)
        
        for lag in [1, 5]:
            feats[f'lag_bid1_{lag}'] = grouped['n_bid1'].shift(lag).fillna(0)
            feats[f'lag_ask1_{lag}'] = grouped['n_ask1'].shift(lag).fillna(0)
            feats[f'lag_bsize1_{lag}'] = grouped['n_bsize1'].shift(lag).fillna(0)
            feats[f'lag_asize1_{lag}'] = grouped['n_asize1'].shift(lag).fillna(0)
        
        amount_mean_100_val = grouped['amount_delta'].transform(lambda x: x.rolling(window=100, min_periods=1).mean()).fillna(1e-10)
        for period in [5, 20, 60]:
            amount_sum = grouped['amount_delta'].transform(lambda x: x.rolling(window=period, min_periods=1).sum()).fillna(0)
            feats[f'volume_flow_{period}'] = amount_sum / (amount_mean_100_val + 1e-10)
        
        feats['micro_price'] = (df['n_bid1'] * df['n_asize1'] + df['n_ask1'] * df['n_bsize1']) / (df['n_bsize1'] + df['n_asize1'] + 1e-10)
        micro_price_series = pd.Series(feats['micro_price'], index=df.index)
        micro_price_df = pd.DataFrame({'sym': df['sym'], 'date': df['date'], 'micro_price': micro_price_series}, index=df.index)
        micro_price_grouped = micro_price_df.groupby(['sym', 'date'])
        feats['micro_price_diff'] = micro_price_grouped['micro_price'].transform(lambda x: x.diff()).fillna(0)
        
        w = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        bid_depth = (df['n_bsize1']*w[0] + df['n_bsize2']*w[1] + df['n_bsize3']*w[2] + df['n_bsize4']*w[3] + df['n_bsize5']*w[4])
        ask_depth = (df['n_asize1']*w[0] + df['n_asize2']*w[1] + df['n_asize3']*w[2] + df['n_asize4']*w[3] + df['n_asize5']*w[4])
        feats['weighted_depth_imbalance'] = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-10)
        
        ofi_1_val = feats['ofi_1'] if 'ofi_1' in feats else (df['ofi_1'] if 'ofi_1' in df.columns else pd.Series(0, index=df.index))
        mid_diff_val = df['midprice_delta'] if 'midprice_delta' in df.columns else grouped['n_midprice'].transform(lambda x: x.diff()).fillna(0)
        feats['ofi_momentum_sync'] = ofi_1_val * mid_diff_val
        
        if 'volatility_10' in df.columns:
            feats['vov_10'] = grouped['volatility_10'].transform(lambda x: x.rolling(window=5, min_periods=1).std()).fillna(0)
        else:
            temp_mid = 2 + df['n_ask1'] + df['n_bid1']
            temp_mid_df = pd.DataFrame({'sym': df['sym'], 'date': df['date'], 'temp_mid': temp_mid}, index=df.index)
            vol10 = temp_mid_df.groupby(['sym', 'date'])['temp_mid'].transform(lambda x: (x / (x.shift(10) + 1e-10) - 1).fillna(0))
            vol10_df = pd.DataFrame({'sym': df['sym'], 'date': df['date'], 'vol10': vol10}, index=df.index)
            feats['vov_10'] = vol10_df.groupby(['sym', 'date'])['vol10'].transform(lambda x: x.rolling(window=5, min_periods=1).std()).fillna(0)
        
        feats['bid_convexity'] = (df['n_bsize1'] + df['n_bsize5'] - 2 * df['n_bsize3']) / (df['n_bsize3'] + 1e-10)
        feats['ask_convexity'] = (df['n_asize1'] + df['n_asize5'] - 2 * df['n_asize3']) / (df['n_asize3'] + 1e-10)
        
        feats['buy_intensity'] = df['amount_delta'] / (df['n_asize1'] + 1e-10)
        feats['sell_intensity'] = df['amount_delta'] / (df['n_bsize1'] + 1e-10)
        
        ofi_1_for_ema = feats['ofi_1'] if 'ofi_1' in feats else (df['ofi_1'] if 'ofi_1' in df.columns else pd.Series(0, index=df.index))
        ofi_ema_df = pd.DataFrame({'sym': df['sym'], 'date': df['date'], 'ofi_1': ofi_1_for_ema}, index=df.index)
        ofi_ema_grouped = ofi_ema_df.groupby(['sym', 'date'])
        feats['ofi_ema_5'] = ofi_ema_grouped['ofi_1'].transform(lambda x: x.ewm(span=5, adjust=False).mean()).fillna(0)
        feats['ofi_ema_10'] = ofi_ema_grouped['ofi_1'].transform(lambda x: x.ewm(span=10, adjust=False).mean()).fillna(0)
        
        amount_abs = df['amount_delta'].abs()
        amount_df = pd.DataFrame({'sym': df['sym'], 'date': df['date'], 'amount_abs': amount_abs}, index=df.index)
        amount_grouped = amount_df.groupby(['sym', 'date'])
        for period in [5, 20]:
            amount_vol = amount_grouped['amount_abs'].transform(lambda x: x.rolling(window=period, min_periods=1).std()).fillna(1e-10)
            amount_mean = amount_grouped['amount_abs'].transform(lambda x: x.rolling(window=period, min_periods=1).mean()).fillna(1e-10)
            feats[f'vpin_{period}'] = amount_vol / (amount_mean + 1e-10)
        
        feats['book_curvature'] = (feats['bid_convexity'] + feats['ask_convexity']) / 2
        feats['depth_pressure'] = (bid_depth + ask_depth) * feats['weighted_depth_imbalance']
        
        new_df = pd.concat([df, pd.DataFrame(feats, index=df.index)], axis=1)
        
        for col in self.base_feature_names:
            if col in new_df.columns:
                new_df[col] = new_df[col].replace([np.inf, -np.inf], 0).fillna(0)
        
        return new_df.copy()
    
    def get_train_test_split(self, label_name='label_5', test_size=0.2, random_state=42, window=100, cache_dir='./cache', batch_size=5000):
        if label_name not in self.label_columns:
            raise ValueError(f"Label {label_name} not found. Available labels: {self.label_columns}")
        
        os.makedirs(cache_dir, exist_ok=True)
        train_meta = os.path.join(cache_dir, f'train_{label_name}.meta')
        val_meta = os.path.join(cache_dir, f'val_{label_name}.meta')
        
        train_files, val_files = self._split_files(test_size)
        
        sample_file = train_files[0] if train_files else val_files[0]
        sample_df = pd.read_csv(sample_file, nrows=100)
        sample_df = self._add_derived_features(sample_df)
        available_features = [col for col in self.feature_columns if col in sample_df.columns]
        
        print(f"\n=== Feature Availability Check ===")
        print(f"Base features defined: {len(self.feature_columns)}")
        print(f"Final features (with time lags): {len(self.final_feature_names)}")
        print(f"Available base features after derivation: {len(available_features)}")
        if len(available_features) < len(self.feature_columns):
            missing_features = set(self.feature_columns) - set(available_features)
            print(f"Missing features ({len(missing_features)}): {list(missing_features)[:10]}")
        print("====================================\n")
        print("Processing training data and saving to binary DMatrix blocks...")
        train_label_counts = self._process_and_save_binary_blocks(train_files, label_name, window, available_features, train_meta, cache_dir, batch_size)
        
        print("Processing validation data and saving to binary DMatrix blocks...")
        self._process_and_save_binary_blocks(val_files, label_name, window, available_features, val_meta, cache_dir, batch_size)
        
        return train_meta, val_meta, train_label_counts
    
    def _split_files(self, test_size):
        pattern = os.path.join(self.data_dir, 'snapshot_sym*_date*_*.csv')
        csv_files = sorted(glob.glob(pattern))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.data_dir}")
        
        def _parse(file_path):
            name = os.path.basename(file_path)
            # name: snapshot_symX_dateY_am/pm.csv
            try:
                sym_part = name.split('_')[1]
                date_part = name.split('_')[2] 
                session_part = name.split('_')[3].split('.')[0]
                sym_idx = int(sym_part.replace('sym', '').replace('sym', ''))
                date_idx = int(date_part.replace('date', ''))
                session_idx = 0 if 'am' in session_part else 1
                return (date_idx, session_idx, sym_idx)
            except Exception:
                return (0, 0, 0)
        
        csv_files = sorted(csv_files, key=_parse)
        
        split_file_idx = int(len(csv_files) * (1 - test_size))
        train_files = csv_files[:split_file_idx]
        val_files = csv_files[split_file_idx:]
        return train_files, val_files

    def _is_limit_up_down(self, df):
        limit_up = (df['n_ask1'] == 0) | (df['n_close'] >= 0.095)
        limit_down = (df['n_bid1'] == 0) | (df['n_close'] <= -0.095)
        return limit_up | limit_down
    
    def _process_and_save_binary_blocks(self, csv_files, label_name, window, available_features, meta_path, cache_dir, batch_size=5000):
        batch_features = []
        batch_labels = []
        batch_price_diffs = []
        label_counts = {0: 0, 1: 0, 2: 0}
        buffer_files = []
        buffer_idx = 0
        total_samples = 0
        removed_limit_samples = 0
        
        n_shift = int(label_name.split('_')[1])
        base_name = os.path.splitext(os.path.basename(meta_path))[0]
        
        def save_buffer():
            nonlocal buffer_idx, batch_features, batch_labels, batch_price_diffs, total_samples
            if not batch_features:
                return 0
            X_batch = np.array(batch_features, dtype=np.float32)
            y_batch = np.array(batch_labels, dtype=np.int32)
            p_batch = np.array(batch_price_diffs, dtype=np.float32)
            
            buffer_file = os.path.join(cache_dir, f'{base_name}_{buffer_idx}.buffer')
            price_file = buffer_file.replace('.buffer', '.price.npy')
            
            dtrain_batch = xgb.DMatrix(X_batch, label=y_batch)
            dtrain_batch.save_binary(buffer_file)
            np.save(price_file, p_batch)
            
            num_samples = len(batch_labels)
            buffer_files.append(buffer_file)
            batch_features = []
            batch_labels = []
            batch_price_diffs = []
            buffer_idx += 1
            return num_samples
        
        import gc
        
        for file in tqdm(csv_files, desc="Processing files", ncols=80):
            try:
                df = pd.read_csv(file)
                df = df.sort_values(['sym', 'date', 'time']).reset_index(drop=True)
                
                limit_mask = self._is_limit_up_down(df)
                removed_limit_samples += limit_mask.sum()
                
                df['price_diff_raw'] = df.groupby(['sym', 'date'])['n_midprice'].shift(-n_shift) - df['n_midprice']
                
                df = self._add_derived_features(df)
                
                feature_df = df[available_features].copy()
                for col in feature_df.columns:
                    feature_df[col] = feature_df[col].replace([np.inf, -np.inf], 0).fillna(0).astype(np.float32)
                
                mid_idx = available_features.index('n_midprice') if 'n_midprice' in available_features else 0
                imb_idx = available_features.index('total_imbalance') if 'total_imbalance' in available_features else 0
                
                grouped = df.groupby(['sym', 'date'])
                for (sym, date), group in grouped:
                    group_features = feature_df.loc[group.index].values
                    group_labels = group[label_name].fillna(1).astype(int).values
                    group_diffs = group['price_diff_raw'].fillna(0).values
                    group_limit_mask = limit_mask.loc[group.index].values
                    
                    for i in range(len(group)):
                        if group_limit_mask[i]:
                            continue
                        
                        if i < 4:
                            continue
                        
                        pyramid_vec = self._assemble_pyramid_vector(group_features, i, mid_idx, imb_idx)
                        batch_features.append(pyramid_vec)
                        label = int(group_labels[i])
                        batch_labels.append(label)
                        batch_price_diffs.append(group_diffs[i])
                        label_counts[label] = label_counts.get(label, 0) + 1
                        
                        if len(batch_features) >= batch_size:
                            total_samples += save_buffer()
                
                del df, feature_df
                gc.collect()
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue
        
        total_samples += save_buffer()
        
        if not buffer_files:
            raise ValueError("No data processed")
        
        with open(meta_path, 'w', encoding='utf-8') as f:
            for buffer_file in buffer_files:
                f.write(f'{buffer_file}\n')
        
        print(f"Saved {len(buffer_files)} binary DMatrix blocks to {meta_path} ({total_samples} total samples)")
        print(f"Removed {removed_limit_samples} limit up/down samples")
        
        return label_counts
    