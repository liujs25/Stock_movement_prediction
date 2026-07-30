"""Feature engineering for previous-report reproduction experiments.

Two builders are intentionally kept here:

* ``pdf_report`` implements only the feature families explicitly described in
  ``Final_Report.pdf``.
* ``previous_code`` keeps the richer 922-dimensional feature vector from the
  old project code and the current submission pipeline.

The PDF does not fully specify the final XGBoost vectorization, so keeping both
paths makes the missing information explicit instead of silently mixing them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
import pandas as pd


LABEL_COLUMNS = ["label_5", "label_10", "label_20", "label_40", "label_60"]
PDF_LEVEL_MODES = {
    "1-5": (1, 2, 3, 4, 5),
    "1-3": (1, 2, 3),
}


@dataclass
class FeatureBuilder:
    """Build legacy XGBoost features and final pyramid vectors."""

    feature_set: str = "previous_code"
    vector_mode: str = "pyramid_recent5_summary6"
    min_history: int = 4

    base_feature_names: List[str] = field(default_factory=lambda: [
        "n_close", "amount_delta", "n_midprice",
        "n_bid1", "n_bsize1", "n_bid2", "n_bsize2", "n_bid3", "n_bsize3",
        "n_bid4", "n_bsize4", "n_bid5", "n_bsize5",
        "n_ask1", "n_asize1", "n_ask2", "n_asize2", "n_ask3", "n_asize3",
        "n_ask4", "n_asize4", "n_ask5", "n_asize5",
        "spread_1", "spread_3", "spread_5",
        "mid_price_1", "mid_price_3", "mid_price_5",
        "relative_bid_density_1", "relative_ask_density_1",
        "relative_bid_density_3", "relative_ask_density_3",
        "weighted_ab_1", "weighted_ab_3",
        "vol1_rel_diff", "vol3_rel_diff", "vol5_rel_diff",
        "amount_normalized",
        "log_bsize1", "log_asize1", "log_bsize3", "log_asize3", "log_bsize5", "log_asize5",
        "close_delta", "bid1_delta", "ask1_delta", "midprice_delta",
        "close_mean", "close_std", "close_vs_mean",
        "bid1_mean", "bid1_std", "bid1_vs_mean",
        "bid3_mean", "bid3_std", "bid3_vs_mean",
        "bid5_mean", "bid5_std", "bid5_vs_mean",
        "ask1_mean", "ask1_std", "ask1_vs_mean",
        "ask3_mean", "ask3_std", "ask3_vs_mean",
        "ask5_mean", "ask5_std", "ask5_vs_mean",
        "bsize1_mean", "bsize1_std", "bsize1_vs_mean",
        "bsize3_mean", "bsize3_std", "bsize3_vs_mean",
        "bsize5_mean", "bsize5_std", "bsize5_vs_mean",
        "asize1_mean", "asize1_std", "asize1_vs_mean",
        "asize3_mean", "asize3_std", "asize3_vs_mean",
        "asize5_mean", "asize5_std", "asize5_vs_mean",
        "midprice_mean", "midprice_std",
        "mid_price_1_mean", "mid_price_1_std",
        "mid_price_3_mean", "mid_price_3_std",
        "mid_price_5_mean", "mid_price_5_std",
        "time_seconds", "time_interval",
        "bid1_plus1", "bid3_plus1", "bid5_plus1",
        "ask1_plus1", "ask3_plus1", "ask5_plus1",
        "cross_weighted_1", "cross_weighted_2",
        "midprice_ma5",
        "volatility_5", "volatility_10", "volatility_20", "volatility_40", "volatility_60",
        "macd_dif", "macd_dea", "macd_bar",
        "kdj_k", "kdj_d", "kdj_j",
        "roc_1", "roc_5", "roc_10", "roc_30", "roc_60", "roc_100",
        "vol1_rel_diff_mean_5", "vol1_rel_diff_mean_20",
        "price_zscore_20", "price_zscore_100", "price_zscore_300",
        "price_slope_20", "price_slope_100", "price_slope_300",
        "price_percentile_100",
        "amount_zscore_20", "amount_zscore_100", "amount_zscore_300",
        "amount_slope_20", "amount_slope_100", "amount_slope_300",
        "lag_mid_1", "lag_mid_5", "lag_mid_20",
        "lag_bid1_1", "lag_bid1_5", "lag_ask1_1", "lag_ask1_5",
        "lag_bsize1_1", "lag_bsize1_5", "lag_asize1_1", "lag_asize1_5",
        "volume_flow_5", "volume_flow_20", "volume_flow_60",
        "total_imbalance", "total_imbalance_weighted",
        "bid_slope", "ask_slope",
        "price_elasticity_10", "orderbook_pressure",
        "ofi_1", "ofi_avg_3", "ofi_1_rolling_5", "ofi_spread_ratio",
        "midprice_accel", "imb_velocity", "imb_accel", "energy_burst",
        "micro_price", "micro_price_diff", "weighted_depth_imbalance",
        "ofi_momentum_sync", "vov_10", "bid_convexity", "ask_convexity",
        "buy_intensity", "sell_intensity", "ofi_ema_5", "ofi_ema_10",
        "vpin_5", "vpin_20", "book_curvature", "depth_pressure",
    ])
    recent_lags: Sequence[int] = (0, 1, 2, 3, 4)
    summary_lags: Sequence[int] = (5, 10, 20, 40, 80, 100)

    @property
    def final_feature_names(self) -> List[str]:
        names: List[str] = []
        for lag in self.recent_lags:
            names.extend(f"{col}_t{lag}" for col in self.base_feature_names)
        for lag in self.summary_lags:
            names.extend([f"mid_lag{lag}", f"imb_lag{lag}"])
        return names

    def available_features(self, df: pd.DataFrame) -> List[str]:
        featured = self.add_derived_features(df)
        return [col for col in self.base_feature_names if col in featured.columns]

    def prepare_input_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        if "date" not in result.columns:
            result["date"] = 0
        if "sym" not in result.columns:
            result["sym"] = 0
        if "time" in result.columns:
            result = result.sort_values("time", kind="mergesort").reset_index(drop=True)

        numeric_candidates = [
            "date", "sym", "n_close", "amount_delta", "n_midprice",
            *[f"n_bid{i}" for i in range(1, 6)],
            *[f"n_bsize{i}" for i in range(1, 6)],
            *[f"n_ask{i}" for i in range(1, 6)],
            *[f"n_asize{i}" for i in range(1, 6)],
        ]
        for col in numeric_candidates:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce")
        return result

    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.prepare_input_frame(df)
        feats = {}

        t_seconds = df["time"].map(_time_to_seconds) if "time" in df.columns else pd.Series(0, index=df.index)
        feats["time_seconds"] = t_seconds
        feats["time_interval"] = t_seconds.map(lambda x: min(int((x - 34200) / 1800), 7) if x >= 34200 else 0)

        for i in [1, 3, 5]:
            feats[f"spread_{i}"] = df[f"n_ask{i}"] - df[f"n_bid{i}"]
            feats[f"mid_price_{i}"] = (df[f"n_ask{i}"] + df[f"n_bid{i}"]) / 2
            total_size = df[f"n_bsize{i}"] + df[f"n_asize{i}"]
            feats[f"relative_bid_density_{i}"] = df[f"n_bsize{i}"] / (total_size + 1e-10)
            feats[f"relative_ask_density_{i}"] = df[f"n_asize{i}"] / (total_size + 1e-10)

        for i in [1, 3]:
            feats[f"weighted_ab_{i}"] = (
                df[f"n_bid{i}"] * df[f"n_asize{i}"] + df[f"n_ask{i}"] * df[f"n_bsize{i}"]
            ) / (df[f"n_bsize{i}"] + df[f"n_asize{i}"] + 1e-10)

        feats["vol1_rel_diff"] = (df["n_bsize1"] - df["n_asize1"]) / (df["n_bsize1"] + df["n_asize1"] + 1e-10)
        feats["vol3_rel_diff"] = (
            df["n_bsize1"] + df["n_bsize2"] + df["n_bsize3"] - df["n_asize1"] - df["n_asize2"] - df["n_asize3"]
        ) / (df["n_bsize1"] + df["n_bsize2"] + df["n_bsize3"] + df["n_asize1"] + df["n_asize2"] + df["n_asize3"] + 1e-10)
        feats["vol5_rel_diff"] = (
            df["n_bsize1"] + df["n_bsize2"] + df["n_bsize3"] + df["n_bsize4"] + df["n_bsize5"]
            - df["n_asize1"] - df["n_asize2"] - df["n_asize3"] - df["n_asize4"] - df["n_asize5"]
        ) / (
            df["n_bsize1"] + df["n_bsize2"] + df["n_bsize3"] + df["n_bsize4"] + df["n_bsize5"]
            + df["n_asize1"] + df["n_asize2"] + df["n_asize3"] + df["n_asize4"] + df["n_asize5"] + 1e-10
        )

        feats["amount_normalized"] = np.log1p(df["amount_delta"] / (1 + df["n_midprice"]))

        for i in [1, 3, 5]:
            feats[f"log_bsize{i}"] = np.log1p(df[f"n_bsize{i}"])
            feats[f"log_asize{i}"] = np.log1p(df[f"n_asize{i}"])

        grouped = df.groupby(["sym", "date"], sort=False)
        feats["close_delta"] = grouped["n_close"].diff()
        feats["bid1_delta"] = grouped["n_bid1"].diff()
        feats["ask1_delta"] = grouped["n_ask1"].diff()
        feats["midprice_delta"] = grouped["n_midprice"].diff()

        feats["close_mean"] = grouped["n_close"].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
        feats["close_std"] = grouped["n_close"].transform(lambda x: x.rolling(window=10, min_periods=1).std())
        feats["close_vs_mean"] = df["n_close"] / (feats["close_mean"] + 1e-10)

        for i in [1, 3, 5]:
            feats[f"bid{i}_mean"] = grouped[f"n_bid{i}"].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
            feats[f"bid{i}_std"] = grouped[f"n_bid{i}"].transform(lambda x: x.rolling(window=10, min_periods=1).std())
            feats[f"bid{i}_vs_mean"] = df[f"n_bid{i}"] / (feats[f"bid{i}_mean"] + 1e-10)

            feats[f"ask{i}_mean"] = grouped[f"n_ask{i}"].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
            feats[f"ask{i}_std"] = grouped[f"n_ask{i}"].transform(lambda x: x.rolling(window=10, min_periods=1).std())
            feats[f"ask{i}_vs_mean"] = df[f"n_ask{i}"] / (feats[f"ask{i}_mean"] + 1e-10)

            feats[f"bsize{i}_mean"] = grouped[f"n_bsize{i}"].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
            feats[f"bsize{i}_std"] = grouped[f"n_bsize{i}"].transform(lambda x: x.rolling(window=10, min_periods=1).std())
            feats[f"bsize{i}_vs_mean"] = df[f"n_bsize{i}"] / (feats[f"bsize{i}_mean"] + 1e-10)

            feats[f"asize{i}_mean"] = grouped[f"n_asize{i}"].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
            feats[f"asize{i}_std"] = grouped[f"n_asize{i}"].transform(lambda x: x.rolling(window=10, min_periods=1).std())
            feats[f"asize{i}_vs_mean"] = df[f"n_asize{i}"] / (feats[f"asize{i}_mean"] + 1e-10)

            temp_mid_price_df = pd.DataFrame({"sym": df["sym"], "date": df["date"], "mid_price": feats[f"mid_price_{i}"]}, index=df.index)
            temp_grouped = temp_mid_price_df.groupby(["sym", "date"], sort=False)
            feats[f"mid_price_{i}_mean"] = temp_grouped["mid_price"].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
            feats[f"mid_price_{i}_std"] = temp_grouped["mid_price"].transform(lambda x: x.rolling(window=10, min_periods=1).std())

        feats["midprice_mean"] = grouped["n_midprice"].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
        feats["midprice_std"] = grouped["n_midprice"].transform(lambda x: x.rolling(window=10, min_periods=1).std())

        for i in [1, 3, 5]:
            feats[f"bid{i}_plus1"] = df[f"n_bid{i}"] + 1
            feats[f"ask{i}_plus1"] = df[f"n_ask{i}"] + 1

        feats["cross_weighted_1"] = (df["n_ask1"] * df["n_bsize2"] + df["n_ask2"] * df["n_bsize1"]) / (df["n_bsize1"] + df["n_bsize2"] + 1e-10)
        feats["cross_weighted_2"] = (df["n_bid1"] * df["n_asize2"] + df["n_bid2"] * df["n_asize1"]) / (df["n_asize1"] + df["n_asize2"] + 1e-10)
        feats["midprice_ma5"] = grouped["n_midprice"].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

        temp_mid = 2 + df["n_ask1"] + df["n_bid1"]
        temp_mid_df = pd.DataFrame({"sym": df["sym"], "date": df["date"], "temp_mid": temp_mid}, index=df.index)
        temp_mid_grouped = temp_mid_df.groupby(["sym", "date"], sort=False)
        for period in [5, 10, 20, 40, 60]:
            feats[f"volatility_{period}"] = temp_mid_grouped["temp_mid"].transform(
                lambda x: (x / (x.shift(period) + 1e-10) - 1).fillna(0)
            )

        ema12 = grouped["n_midprice"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
        ema26 = grouped["n_midprice"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
        feats["macd_dif"] = ema12 - ema26
        temp_df = pd.DataFrame({"sym": df["sym"], "date": df["date"], "dif": feats["macd_dif"]}, index=df.index)
        feats["macd_dea"] = temp_df.groupby(["sym", "date"], sort=False)["dif"].transform(lambda x: x.ewm(span=9, adjust=False).mean())
        feats["macd_bar"] = feats["macd_dif"] - feats["macd_dea"]

        low_9 = grouped["n_bid1"].transform(lambda x: x.rolling(window=9, min_periods=1).min())
        high_9 = grouped["n_ask1"].transform(lambda x: x.rolling(window=9, min_periods=1).max())
        rsv = 100 * (df["n_midprice"] - low_9) / (high_9 - low_9 + 1e-10)
        rsv_df = pd.DataFrame({"sym": df["sym"], "date": df["date"], "rsv": rsv}, index=df.index)
        feats["kdj_k"] = rsv_df.groupby(["sym", "date"], sort=False)["rsv"].transform(lambda x: x.ewm(alpha=1 / 3, adjust=False).mean())
        k_df = pd.DataFrame({"sym": df["sym"], "date": df["date"], "k": feats["kdj_k"]}, index=df.index)
        feats["kdj_d"] = k_df.groupby(["sym", "date"], sort=False)["k"].transform(lambda x: x.ewm(alpha=1 / 3, adjust=False).mean())
        feats["kdj_j"] = 3 * feats["kdj_k"] - 2 * feats["kdj_d"]

        new_df = pd.concat([df, pd.DataFrame(feats, index=df.index)], axis=1)
        new_df = self._add_advanced_features(new_df)
        return new_df.copy()

    def _add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        grouped = df.groupby(["sym", "date"], sort=False)
        feats = {}
        mid = df["n_midprice"]

        for w in [1, 5, 10, 30, 60, 100]:
            feats[f"roc_{w}"] = grouped["n_midprice"].transform(lambda x: x / x.shift(w) - 1).fillna(0)

        vol1_rel_diff = df["vol1_rel_diff"] if "vol1_rel_diff" in df.columns else (df["n_bsize1"] - df["n_asize1"]) / (df["n_bsize1"] + df["n_asize1"] + 1e-10)
        vol1_df = pd.DataFrame({"sym": df["sym"], "date": df["date"], "vol1": vol1_rel_diff}, index=df.index)
        feats["vol1_rel_diff_mean_5"] = vol1_df.groupby(["sym", "date"], sort=False)["vol1"].transform(lambda x: x.rolling(window=5, min_periods=1).mean()).fillna(0)
        feats["vol1_rel_diff_mean_20"] = vol1_df.groupby(["sym", "date"], sort=False)["vol1"].transform(lambda x: x.rolling(window=20, min_periods=1).mean()).fillna(0)

        for window in [20, 100, 300]:
            roll_mid = grouped["n_midprice"].rolling(window=window, min_periods=1)
            mid_mean = roll_mid.mean().reset_index(level=[0, 1], drop=True)
            mid_std = roll_mid.std().reset_index(level=[0, 1], drop=True)
            feats[f"price_zscore_{window}"] = (mid - mid_mean) / (mid_std + 1e-10)

            mid_recent_mean = grouped["n_midprice"].transform(lambda x: x.rolling(window=window // 3, min_periods=1).mean())
            mid_early_mean = grouped["n_midprice"].transform(lambda x: x.shift(window * 2 // 3).rolling(window=window // 3, min_periods=1).mean())
            feats[f"price_slope_{window}"] = (mid_recent_mean - mid_early_mean) / (window * 2 // 3 + 1e-10)

            roll_amount = grouped["amount_delta"].rolling(window=window, min_periods=1)
            amount_mean = roll_amount.mean().reset_index(level=[0, 1], drop=True)
            amount_std = roll_amount.std().reset_index(level=[0, 1], drop=True)
            feats[f"amount_zscore_{window}"] = (df["amount_delta"] - amount_mean) / (amount_std + 1e-10)

            amount_recent_mean = grouped["amount_delta"].transform(lambda x: x.rolling(window=window // 3, min_periods=1).mean())
            amount_early_mean = grouped["amount_delta"].transform(lambda x: x.shift(window * 2 // 3).rolling(window=window // 3, min_periods=1).mean())
            feats[f"amount_slope_{window}"] = (amount_recent_mean - amount_early_mean) / (window * 2 // 3 + 1e-10)

        roll100_max = grouped["n_midprice"].rolling(window=100, min_periods=1).max().reset_index(level=[0, 1], drop=True)
        roll100_min = grouped["n_midprice"].rolling(window=100, min_periods=1).min().reset_index(level=[0, 1], drop=True)
        feats["price_percentile_100"] = (mid - roll100_min) / (roll100_max - roll100_min + 1e-10)

        total_bid_size = df["n_bsize1"] + df["n_bsize2"] + df["n_bsize3"] + df["n_bsize4"] + df["n_bsize5"]
        total_ask_size = df["n_asize1"] + df["n_asize2"] + df["n_asize3"] + df["n_asize4"] + df["n_asize5"]
        feats["total_imbalance"] = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size + 1e-10)

        weighted_bid = (
            df["n_bid1"] * df["n_bsize1"] + df["n_bid2"] * df["n_bsize2"] + df["n_bid3"] * df["n_bsize3"]
            + df["n_bid4"] * df["n_bsize4"] + df["n_bid5"] * df["n_bsize5"]
        ) / (total_bid_size + 1e-10)
        weighted_ask = (
            df["n_ask1"] * df["n_asize1"] + df["n_ask2"] * df["n_asize2"] + df["n_ask3"] * df["n_asize3"]
            + df["n_ask4"] * df["n_asize4"] + df["n_ask5"] * df["n_asize5"]
        ) / (total_ask_size + 1e-10)
        feats["total_imbalance_weighted"] = (weighted_bid - weighted_ask) / (weighted_bid + weighted_ask + 1e-10)

        feats["bid_slope"] = (df["n_bid1"] - df["n_bid5"]) / (df["n_bid1"] + 1e-10)
        feats["ask_slope"] = (df["n_ask5"] - df["n_ask1"]) / (df["n_ask1"] + 1e-10)
        price_change_10 = grouped["n_midprice"].transform(lambda x: x.diff(10).fillna(0))
        volume_sum_10 = grouped["amount_delta"].transform(lambda x: x.rolling(window=10, min_periods=1).sum())
        feats["price_elasticity_10"] = price_change_10 / (volume_sum_10 + 1e-10)
        feats["orderbook_pressure"] = (total_bid_size - total_ask_size) * (df["n_midprice"] - df["n_bid1"]) / (df["n_ask1"] - df["n_bid1"] + 1e-10)

        def calc_ofi_series(bid_p: pd.Series, bid_v: pd.Series, ask_p: pd.Series, ask_v: pd.Series) -> pd.Series:
            prev_bid_p = bid_p.shift(1)
            prev_bid_v = bid_v.shift(1)
            prev_ask_p = ask_p.shift(1)
            prev_ask_v = ask_v.shift(1)
            ofi_bid = np.where(bid_p > prev_bid_p, bid_v, np.where(bid_p == prev_bid_p, bid_v - prev_bid_v, -prev_bid_v))
            ofi_ask = np.where(ask_p < prev_ask_p, ask_v, np.where(ask_p == prev_ask_p, ask_v - prev_ask_v, -prev_ask_v))
            return pd.Series(ofi_bid - ofi_ask, index=bid_p.index)

        ofi1_list = []
        ofi2_list = []
        ofi3_list = []
        for _, group in grouped:
            ofi1_list.append(calc_ofi_series(group["n_bid1"], group["n_bsize1"], group["n_ask1"], group["n_asize1"]).fillna(0))
            ofi2_list.append(calc_ofi_series(group["n_bid2"], group["n_bsize2"], group["n_ask2"], group["n_asize2"]).fillna(0))
            ofi3_list.append(calc_ofi_series(group["n_bid3"], group["n_bsize3"], group["n_ask3"], group["n_asize3"]).fillna(0))

        feats["ofi_1"] = pd.concat(ofi1_list).reindex(df.index).fillna(0) if ofi1_list else pd.Series(0, index=df.index)
        ofi2_series = pd.concat(ofi2_list).reindex(df.index).fillna(0) if ofi2_list else pd.Series(0, index=df.index)
        ofi3_series = pd.concat(ofi3_list).reindex(df.index).fillna(0) if ofi3_list else pd.Series(0, index=df.index)
        feats["ofi_avg_3"] = (feats["ofi_1"] + ofi2_series + ofi3_series) / 3
        temp_ofi_df = pd.DataFrame({"sym": df["sym"], "date": df["date"], "ofi_1": feats["ofi_1"]}, index=df.index)
        feats["ofi_1_rolling_5"] = temp_ofi_df.groupby(["sym", "date"], sort=False)["ofi_1"].transform(lambda x: x.rolling(window=5, min_periods=1).sum()).fillna(0)
        spread_1 = df["spread_1"] if "spread_1" in df.columns else (df["n_ask1"] - df["n_bid1"])
        feats["ofi_spread_ratio"] = feats["ofi_1"] / (spread_1 + 1e-10)

        if "midprice_delta" in df.columns:
            feats["midprice_accel"] = grouped["midprice_delta"].diff().fillna(0)
        else:
            feats["midprice_accel"] = grouped["n_midprice"].transform(lambda x: x.diff().diff()).fillna(0)
        feats["energy_burst"] = df["amount_delta"] * feats["midprice_accel"]

        imb_vel_list = []
        imb_accel_list = []
        for _, group in grouped:
            group_imb = feats["total_imbalance"].loc[group.index]
            group_vel = group_imb.diff().fillna(0)
            imb_vel_list.append(group_vel)
            imb_accel_list.append(group_vel.diff().fillna(0))
        feats["imb_velocity"] = pd.concat(imb_vel_list).reindex(df.index).fillna(0) if imb_vel_list else pd.Series(0, index=df.index)
        feats["imb_accel"] = pd.concat(imb_accel_list).reindex(df.index).fillna(0) if imb_accel_list else pd.Series(0, index=df.index)

        for lag in [1, 5, 20]:
            feats[f"lag_mid_{lag}"] = grouped["n_midprice"].shift(lag).fillna(0)
        for lag in [1, 5]:
            feats[f"lag_bid1_{lag}"] = grouped["n_bid1"].shift(lag).fillna(0)
            feats[f"lag_ask1_{lag}"] = grouped["n_ask1"].shift(lag).fillna(0)
            feats[f"lag_bsize1_{lag}"] = grouped["n_bsize1"].shift(lag).fillna(0)
            feats[f"lag_asize1_{lag}"] = grouped["n_asize1"].shift(lag).fillna(0)

        amount_mean_100 = grouped["amount_delta"].transform(lambda x: x.rolling(window=100, min_periods=1).mean()).fillna(1e-10)
        for period in [5, 20, 60]:
            amount_sum = grouped["amount_delta"].transform(lambda x: x.rolling(window=period, min_periods=1).sum()).fillna(0)
            feats[f"volume_flow_{period}"] = amount_sum / (amount_mean_100 + 1e-10)

        feats["micro_price"] = (df["n_bid1"] * df["n_asize1"] + df["n_ask1"] * df["n_bsize1"]) / (df["n_bsize1"] + df["n_asize1"] + 1e-10)
        micro_price_df = pd.DataFrame({"sym": df["sym"], "date": df["date"], "micro_price": feats["micro_price"]}, index=df.index)
        feats["micro_price_diff"] = micro_price_df.groupby(["sym", "date"], sort=False)["micro_price"].transform(lambda x: x.diff()).fillna(0)

        weights = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        bid_depth = sum(df[f"n_bsize{i}"] * weights[i - 1] for i in range(1, 6))
        ask_depth = sum(df[f"n_asize{i}"] * weights[i - 1] for i in range(1, 6))
        feats["weighted_depth_imbalance"] = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-10)
        mid_diff_val = df["midprice_delta"] if "midprice_delta" in df.columns else grouped["n_midprice"].transform(lambda x: x.diff()).fillna(0)
        feats["ofi_momentum_sync"] = feats["ofi_1"] * mid_diff_val

        temp_mid = 2 + df["n_ask1"] + df["n_bid1"]
        temp_mid_df = pd.DataFrame({"sym": df["sym"], "date": df["date"], "temp_mid": temp_mid}, index=df.index)
        vol10 = temp_mid_df.groupby(["sym", "date"], sort=False)["temp_mid"].transform(lambda x: (x / (x.shift(10) + 1e-10) - 1).fillna(0))
        vol10_df = pd.DataFrame({"sym": df["sym"], "date": df["date"], "vol10": vol10}, index=df.index)
        feats["vov_10"] = vol10_df.groupby(["sym", "date"], sort=False)["vol10"].transform(lambda x: x.rolling(window=5, min_periods=1).std()).fillna(0)

        feats["bid_convexity"] = (df["n_bsize1"] + df["n_bsize5"] - 2 * df["n_bsize3"]) / (df["n_bsize3"] + 1e-10)
        feats["ask_convexity"] = (df["n_asize1"] + df["n_asize5"] - 2 * df["n_asize3"]) / (df["n_asize3"] + 1e-10)
        feats["buy_intensity"] = df["amount_delta"] / (df["n_asize1"] + 1e-10)
        feats["sell_intensity"] = df["amount_delta"] / (df["n_bsize1"] + 1e-10)

        ofi_ema_df = pd.DataFrame({"sym": df["sym"], "date": df["date"], "ofi_1": feats["ofi_1"]}, index=df.index)
        ofi_ema_grouped = ofi_ema_df.groupby(["sym", "date"], sort=False)
        feats["ofi_ema_5"] = ofi_ema_grouped["ofi_1"].transform(lambda x: x.ewm(span=5, adjust=False).mean()).fillna(0)
        feats["ofi_ema_10"] = ofi_ema_grouped["ofi_1"].transform(lambda x: x.ewm(span=10, adjust=False).mean()).fillna(0)

        amount_abs = df["amount_delta"].abs()
        amount_df = pd.DataFrame({"sym": df["sym"], "date": df["date"], "amount_abs": amount_abs}, index=df.index)
        amount_grouped = amount_df.groupby(["sym", "date"], sort=False)
        for period in [5, 20]:
            amount_vol = amount_grouped["amount_abs"].transform(lambda x: x.rolling(window=period, min_periods=1).std()).fillna(1e-10)
            amount_mean = amount_grouped["amount_abs"].transform(lambda x: x.rolling(window=period, min_periods=1).mean()).fillna(1e-10)
            feats[f"vpin_{period}"] = amount_vol / (amount_mean + 1e-10)

        feats["book_curvature"] = (feats["bid_convexity"] + feats["ask_convexity"]) / 2
        feats["depth_pressure"] = (bid_depth + ask_depth) * feats["weighted_depth_imbalance"]

        new_df = pd.concat([df, pd.DataFrame(feats, index=df.index)], axis=1)
        for col in self.base_feature_names:
            if col in new_df.columns:
                new_df[col] = _clean_numeric_series(new_df[col])
        return new_df.copy()

    def build_feature_matrix_fast_single_group(
        self,
        df: pd.DataFrame,
        available_features: Sequence[str],
    ) -> pd.DataFrame | None:
        df = self.prepare_input_frame(df)
        if len(df) == 0:
            return pd.DataFrame(index=df.index)
        if df[["sym", "date"]].drop_duplicates().shape[0] != 1:
            return None

        feats = {col: df[col] for col in self.base_feature_names if col in df.columns}
        mid = df["n_midprice"]
        amount = df["amount_delta"]

        t_seconds = df["time"].map(_time_to_seconds) if "time" in df.columns else pd.Series(0, index=df.index)
        feats["time_seconds"] = t_seconds
        feats["time_interval"] = t_seconds.map(lambda x: min(int((x - 34200) / 1800), 7) if x >= 34200 else 0)

        for i in [1, 3, 5]:
            bid = df[f"n_bid{i}"]
            ask = df[f"n_ask{i}"]
            bsize = df[f"n_bsize{i}"]
            asize = df[f"n_asize{i}"]
            total_size = bsize + asize
            feats[f"spread_{i}"] = ask - bid
            feats[f"mid_price_{i}"] = (ask + bid) / 2
            feats[f"relative_bid_density_{i}"] = bsize / (total_size + 1e-10)
            feats[f"relative_ask_density_{i}"] = asize / (total_size + 1e-10)
            feats[f"log_bsize{i}"] = np.log1p(bsize)
            feats[f"log_asize{i}"] = np.log1p(asize)
            feats[f"bid{i}_plus1"] = bid + 1
            feats[f"ask{i}_plus1"] = ask + 1

        for i in [1, 3]:
            feats[f"weighted_ab_{i}"] = (
                df[f"n_bid{i}"] * df[f"n_asize{i}"] + df[f"n_ask{i}"] * df[f"n_bsize{i}"]
            ) / (df[f"n_bsize{i}"] + df[f"n_asize{i}"] + 1e-10)

        feats["vol1_rel_diff"] = (df["n_bsize1"] - df["n_asize1"]) / (df["n_bsize1"] + df["n_asize1"] + 1e-10)
        feats["vol3_rel_diff"] = (
            df["n_bsize1"] + df["n_bsize2"] + df["n_bsize3"] - df["n_asize1"] - df["n_asize2"] - df["n_asize3"]
        ) / (df["n_bsize1"] + df["n_bsize2"] + df["n_bsize3"] + df["n_asize1"] + df["n_asize2"] + df["n_asize3"] + 1e-10)
        feats["vol5_rel_diff"] = (
            df["n_bsize1"] + df["n_bsize2"] + df["n_bsize3"] + df["n_bsize4"] + df["n_bsize5"]
            - df["n_asize1"] - df["n_asize2"] - df["n_asize3"] - df["n_asize4"] - df["n_asize5"]
        ) / (
            df["n_bsize1"] + df["n_bsize2"] + df["n_bsize3"] + df["n_bsize4"] + df["n_bsize5"]
            + df["n_asize1"] + df["n_asize2"] + df["n_asize3"] + df["n_asize4"] + df["n_asize5"] + 1e-10
        )
        feats["amount_normalized"] = np.log1p(amount / (1 + mid))

        feats["close_delta"] = df["n_close"].diff()
        feats["bid1_delta"] = df["n_bid1"].diff()
        feats["ask1_delta"] = df["n_ask1"].diff()
        feats["midprice_delta"] = mid.diff()

        feats["close_mean"] = df["n_close"].rolling(window=10, min_periods=1).mean()
        feats["close_std"] = df["n_close"].rolling(window=10, min_periods=1).std()
        feats["close_vs_mean"] = df["n_close"] / (feats["close_mean"] + 1e-10)

        for i in [1, 3, 5]:
            for side, prefix in [("n_bid", "bid"), ("n_ask", "ask"), ("n_bsize", "bsize"), ("n_asize", "asize")]:
                series = df[f"{side}{i}"]
                mean = series.rolling(window=10, min_periods=1).mean()
                feats[f"{prefix}{i}_mean"] = mean
                feats[f"{prefix}{i}_std"] = series.rolling(window=10, min_periods=1).std()
                feats[f"{prefix}{i}_vs_mean"] = series / (mean + 1e-10)
            mid_price = feats[f"mid_price_{i}"]
            feats[f"mid_price_{i}_mean"] = mid_price.rolling(window=10, min_periods=1).mean()
            feats[f"mid_price_{i}_std"] = mid_price.rolling(window=10, min_periods=1).std()

        feats["midprice_mean"] = mid.rolling(window=10, min_periods=1).mean()
        feats["midprice_std"] = mid.rolling(window=10, min_periods=1).std()
        feats["cross_weighted_1"] = (df["n_ask1"] * df["n_bsize2"] + df["n_ask2"] * df["n_bsize1"]) / (df["n_bsize1"] + df["n_bsize2"] + 1e-10)
        feats["cross_weighted_2"] = (df["n_bid1"] * df["n_asize2"] + df["n_bid2"] * df["n_asize1"]) / (df["n_asize1"] + df["n_asize2"] + 1e-10)
        feats["midprice_ma5"] = mid.rolling(window=5, min_periods=1).mean()

        temp_mid = 2 + df["n_ask1"] + df["n_bid1"]
        for period in [5, 10, 20, 40, 60]:
            feats[f"volatility_{period}"] = (temp_mid / (temp_mid.shift(period) + 1e-10) - 1).fillna(0)

        ema12 = mid.ewm(span=12, adjust=False).mean()
        ema26 = mid.ewm(span=26, adjust=False).mean()
        feats["macd_dif"] = ema12 - ema26
        feats["macd_dea"] = feats["macd_dif"].ewm(span=9, adjust=False).mean()
        feats["macd_bar"] = feats["macd_dif"] - feats["macd_dea"]

        low_9 = df["n_bid1"].rolling(window=9, min_periods=1).min()
        high_9 = df["n_ask1"].rolling(window=9, min_periods=1).max()
        rsv = 100 * (mid - low_9) / (high_9 - low_9 + 1e-10)
        feats["kdj_k"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
        feats["kdj_d"] = feats["kdj_k"].ewm(alpha=1 / 3, adjust=False).mean()
        feats["kdj_j"] = 3 * feats["kdj_k"] - 2 * feats["kdj_d"]

        for w in [1, 5, 10, 30, 60, 100]:
            feats[f"roc_{w}"] = (mid / mid.shift(w) - 1).fillna(0)

        feats["vol1_rel_diff_mean_5"] = feats["vol1_rel_diff"].rolling(window=5, min_periods=1).mean().fillna(0)
        feats["vol1_rel_diff_mean_20"] = feats["vol1_rel_diff"].rolling(window=20, min_periods=1).mean().fillna(0)

        for window in [20, 100, 300]:
            mid_mean = mid.rolling(window=window, min_periods=1).mean()
            mid_std = mid.rolling(window=window, min_periods=1).std()
            feats[f"price_zscore_{window}"] = (mid - mid_mean) / (mid_std + 1e-10)
            mid_recent_mean = mid.rolling(window=window // 3, min_periods=1).mean()
            mid_early_mean = mid.shift(window * 2 // 3).rolling(window=window // 3, min_periods=1).mean()
            feats[f"price_slope_{window}"] = (mid_recent_mean - mid_early_mean) / (window * 2 // 3 + 1e-10)

            amount_mean = amount.rolling(window=window, min_periods=1).mean()
            amount_std = amount.rolling(window=window, min_periods=1).std()
            feats[f"amount_zscore_{window}"] = (amount - amount_mean) / (amount_std + 1e-10)
            amount_recent_mean = amount.rolling(window=window // 3, min_periods=1).mean()
            amount_early_mean = amount.shift(window * 2 // 3).rolling(window=window // 3, min_periods=1).mean()
            feats[f"amount_slope_{window}"] = (amount_recent_mean - amount_early_mean) / (window * 2 // 3 + 1e-10)

        roll100_max = mid.rolling(window=100, min_periods=1).max()
        roll100_min = mid.rolling(window=100, min_periods=1).min()
        feats["price_percentile_100"] = (mid - roll100_min) / (roll100_max - roll100_min + 1e-10)

        total_bid_size = sum(df[f"n_bsize{i}"] for i in range(1, 6))
        total_ask_size = sum(df[f"n_asize{i}"] for i in range(1, 6))
        feats["total_imbalance"] = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size + 1e-10)
        weighted_bid = sum(df[f"n_bid{i}"] * df[f"n_bsize{i}"] for i in range(1, 6)) / (total_bid_size + 1e-10)
        weighted_ask = sum(df[f"n_ask{i}"] * df[f"n_asize{i}"] for i in range(1, 6)) / (total_ask_size + 1e-10)
        feats["total_imbalance_weighted"] = (weighted_bid - weighted_ask) / (weighted_bid + weighted_ask + 1e-10)
        feats["bid_slope"] = (df["n_bid1"] - df["n_bid5"]) / (df["n_bid1"] + 1e-10)
        feats["ask_slope"] = (df["n_ask5"] - df["n_ask1"]) / (df["n_ask1"] + 1e-10)
        price_change_10 = mid.diff(10).fillna(0)
        volume_sum_10 = amount.rolling(window=10, min_periods=1).sum()
        feats["price_elasticity_10"] = price_change_10 / (volume_sum_10 + 1e-10)
        feats["orderbook_pressure"] = (total_bid_size - total_ask_size) * (mid - df["n_bid1"]) / (df["n_ask1"] - df["n_bid1"] + 1e-10)

        def calc_ofi_series(bid_p: pd.Series, bid_v: pd.Series, ask_p: pd.Series, ask_v: pd.Series) -> pd.Series:
            prev_bid_p = bid_p.shift(1)
            prev_bid_v = bid_v.shift(1)
            prev_ask_p = ask_p.shift(1)
            prev_ask_v = ask_v.shift(1)
            ofi_bid = np.where(bid_p > prev_bid_p, bid_v, np.where(bid_p == prev_bid_p, bid_v - prev_bid_v, -prev_bid_v))
            ofi_ask = np.where(ask_p < prev_ask_p, ask_v, np.where(ask_p == prev_ask_p, ask_v - prev_ask_v, -prev_ask_v))
            return pd.Series(ofi_bid - ofi_ask, index=bid_p.index).fillna(0)

        feats["ofi_1"] = calc_ofi_series(df["n_bid1"], df["n_bsize1"], df["n_ask1"], df["n_asize1"])
        ofi_2 = calc_ofi_series(df["n_bid2"], df["n_bsize2"], df["n_ask2"], df["n_asize2"])
        ofi_3 = calc_ofi_series(df["n_bid3"], df["n_bsize3"], df["n_ask3"], df["n_asize3"])
        feats["ofi_avg_3"] = (feats["ofi_1"] + ofi_2 + ofi_3) / 3
        feats["ofi_1_rolling_5"] = feats["ofi_1"].rolling(window=5, min_periods=1).sum().fillna(0)
        feats["ofi_spread_ratio"] = feats["ofi_1"] / (feats["spread_1"] + 1e-10)
        feats["midprice_accel"] = feats["midprice_delta"].diff().fillna(0)
        feats["energy_burst"] = amount * feats["midprice_accel"]
        feats["imb_velocity"] = feats["total_imbalance"].diff().fillna(0)
        feats["imb_accel"] = feats["imb_velocity"].diff().fillna(0)

        for lag in [1, 5, 20]:
            feats[f"lag_mid_{lag}"] = mid.shift(lag).fillna(0)
        for lag in [1, 5]:
            feats[f"lag_bid1_{lag}"] = df["n_bid1"].shift(lag).fillna(0)
            feats[f"lag_ask1_{lag}"] = df["n_ask1"].shift(lag).fillna(0)
            feats[f"lag_bsize1_{lag}"] = df["n_bsize1"].shift(lag).fillna(0)
            feats[f"lag_asize1_{lag}"] = df["n_asize1"].shift(lag).fillna(0)

        amount_mean_100 = amount.rolling(window=100, min_periods=1).mean().fillna(1e-10)
        for period in [5, 20, 60]:
            amount_sum = amount.rolling(window=period, min_periods=1).sum().fillna(0)
            feats[f"volume_flow_{period}"] = amount_sum / (amount_mean_100 + 1e-10)

        feats["micro_price"] = (df["n_bid1"] * df["n_asize1"] + df["n_ask1"] * df["n_bsize1"]) / (df["n_bsize1"] + df["n_asize1"] + 1e-10)
        feats["micro_price_diff"] = feats["micro_price"].diff().fillna(0)
        weights = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        bid_depth = sum(df[f"n_bsize{i}"] * weights[i - 1] for i in range(1, 6))
        ask_depth = sum(df[f"n_asize{i}"] * weights[i - 1] for i in range(1, 6))
        feats["weighted_depth_imbalance"] = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-10)
        feats["ofi_momentum_sync"] = feats["ofi_1"] * feats["midprice_delta"]
        vol10 = (temp_mid / (temp_mid.shift(10) + 1e-10) - 1).fillna(0)
        feats["vov_10"] = vol10.rolling(window=5, min_periods=1).std().fillna(0)
        feats["bid_convexity"] = (df["n_bsize1"] + df["n_bsize5"] - 2 * df["n_bsize3"]) / (df["n_bsize3"] + 1e-10)
        feats["ask_convexity"] = (df["n_asize1"] + df["n_asize5"] - 2 * df["n_asize3"]) / (df["n_asize3"] + 1e-10)
        feats["buy_intensity"] = amount / (df["n_asize1"] + 1e-10)
        feats["sell_intensity"] = amount / (df["n_bsize1"] + 1e-10)
        feats["ofi_ema_5"] = feats["ofi_1"].ewm(span=5, adjust=False).mean().fillna(0)
        feats["ofi_ema_10"] = feats["ofi_1"].ewm(span=10, adjust=False).mean().fillna(0)

        amount_abs = amount.abs()
        for period in [5, 20]:
            amount_vol = amount_abs.rolling(window=period, min_periods=1).std().fillna(1e-10)
            amount_mean = amount_abs.rolling(window=period, min_periods=1).mean().fillna(1e-10)
            feats[f"vpin_{period}"] = amount_vol / (amount_mean + 1e-10)

        feats["book_curvature"] = (feats["bid_convexity"] + feats["ask_convexity"]) / 2
        feats["depth_pressure"] = (bid_depth + ask_depth) * feats["weighted_depth_imbalance"]

        matrix = pd.DataFrame({col: feats[col] for col in available_features}, index=df.index)
        for col in matrix.columns:
            matrix[col] = _clean_numeric_series(matrix[col]).astype(np.float32)
        return matrix

    def transform_window_numpy_single_group(
        self,
        df: pd.DataFrame,
        available_features: Sequence[str],
    ) -> np.ndarray | None:
        if len(df) == 0:
            raise ValueError("Cannot transform an empty input window")
        frame = self.prepare_input_frame(df)
        if frame[["sym", "date"]].drop_duplicates().shape[0] != 1:
            return None

        idx_last = len(frame) - 1
        target_indices = [max(0, idx_last - lag) for lag in self.recent_lags]
        summary_indices = [max(0, idx_last - lag) for lag in self.summary_lags]
        cols = _frame_arrays(frame)
        feats = _build_numpy_feature_arrays(frame, cols)

        values = []
        for row_idx in target_indices:
            values.extend(float(_clean_numpy_array(feats[col])[row_idx]) for col in available_features)
        mid = _clean_numpy_array(feats["n_midprice"])
        imb = _clean_numpy_array(feats["total_imbalance"])
        for row_idx in summary_indices:
            values.extend([float(mid[row_idx]), float(imb[row_idx])])
        return np.asarray(values, dtype=np.float32)

    def build_feature_matrix(self, df: pd.DataFrame, available_features: Sequence[str]) -> pd.DataFrame:
        featured = self.add_derived_features(df)
        matrix = featured.loc[:, list(available_features)].copy()
        for col in matrix.columns:
            matrix[col] = _clean_numeric_series(matrix[col]).astype(np.float32)
        return matrix

    def assemble_pyramid_vector(
        self,
        feature_matrix: np.ndarray,
        row_idx: int,
        available_features: Sequence[str],
    ) -> np.ndarray:
        mid_idx = available_features.index("n_midprice") if "n_midprice" in available_features else 0
        imb_idx = available_features.index("total_imbalance") if "total_imbalance" in available_features else 0

        frames = []
        for lag in self.recent_lags:
            idx = max(0, row_idx - lag)
            frames.append(feature_matrix[idx])
        for lag in self.summary_lags:
            idx = max(0, row_idx - lag)
            frames.append(np.array([feature_matrix[idx][mid_idx], feature_matrix[idx][imb_idx]], dtype=np.float32))
        return np.concatenate(frames).astype(np.float32)

    def transform_window(self, df: pd.DataFrame, available_features: Sequence[str]) -> np.ndarray:
        fast_matrix = self.build_feature_matrix_fast_single_group(df, available_features)
        matrix = (fast_matrix if fast_matrix is not None else self.build_feature_matrix(df, available_features)).values
        if len(matrix) == 0:
            raise ValueError("Cannot transform an empty input window")
        return self.assemble_pyramid_vector(matrix, len(matrix) - 1, available_features)


def _normalize_pdf_level_mode(pdf_level_mode: str) -> str:
    if pdf_level_mode not in PDF_LEVEL_MODES:
        raise ValueError(f"Unsupported PDF level mode: {pdf_level_mode}. Expected one of {sorted(PDF_LEVEL_MODES)}")
    return pdf_level_mode


def _pdf_report_base_feature_names(levels: Sequence[int]) -> List[str]:
    names = [
        "time_seconds", "time_interval",
        "n_close", "amount_delta", "n_midprice",
        "n_bid1", "n_bsize1", "n_bid2", "n_bsize2", "n_bid3", "n_bsize3",
        "n_bid4", "n_bsize4", "n_bid5", "n_bsize5",
        "n_ask1", "n_asize1", "n_ask2", "n_asize2", "n_ask3", "n_asize3",
        "n_ask4", "n_asize4", "n_ask5", "n_asize5",
    ]
    names += [f"spread_{i}" for i in levels]
    names += [f"mid_price_{i}" for i in levels]
    for i in levels:
        names += [f"relative_bid_density_{i}", f"relative_ask_density_{i}"]
    names += [f"weighted_ab_{i}" for i in levels]
    names += [
        "vol1_rel_diff", "vol3_rel_diff", "vol5_rel_diff",
        "amount_log",
        "close_delta",
        "close_mean", "close_std", "close_vs_mean",
    ]
    for i in levels:
        names += [f"log_bsize{i}", f"log_asize{i}"]
    for i in levels:
        names += [
            f"bid{i}_delta", f"ask{i}_delta", f"mid_price_{i}_delta",
            f"bsize{i}_delta", f"asize{i}_delta",
        ]
    for i in levels:
        names += [
            f"bid{i}_mean", f"bid{i}_std", f"bid{i}_vs_mean",
            f"ask{i}_mean", f"ask{i}_std", f"ask{i}_vs_mean",
            f"bsize{i}_mean", f"bsize{i}_std", f"bsize{i}_vs_mean",
            f"asize{i}_mean", f"asize{i}_std", f"asize{i}_vs_mean",
            f"mid_price_{i}_mean", f"mid_price_{i}_std",
        ]
    return names


@dataclass
class PDFReportFeatureBuilder(FeatureBuilder):
    """Feature builder limited to the feature groups stated in the PDF report."""

    feature_set: str = "pdf_report"
    vector_mode: str = "window_flatten"
    window_size: int = 100
    min_history: int = 99
    pdf_level_mode: str = "1-5"
    pdf_levels: Sequence[int] = field(default_factory=lambda: PDF_LEVEL_MODES["1-5"])
    recent_lags: Sequence[int] = field(default_factory=tuple)
    summary_lags: Sequence[int] = field(default_factory=tuple)
    base_feature_names: List[str] = field(init=False)

    def __post_init__(self) -> None:
        self.pdf_level_mode = _normalize_pdf_level_mode(self.pdf_level_mode)
        self.pdf_levels = PDF_LEVEL_MODES[self.pdf_level_mode]
        self.base_feature_names = _pdf_report_base_feature_names(self.pdf_levels)

    @property
    def final_feature_names(self) -> List[str]:
        names: List[str] = []
        for offset in range(self.window_size - 1, -1, -1):
            names.extend(f"{col}_t-{offset}" for col in self.base_feature_names)
        return names

    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.prepare_input_frame(df)
        feats = {}

        t_seconds = df["time"].map(_time_to_seconds) if "time" in df.columns else pd.Series(0, index=df.index)
        feats["time_seconds"] = t_seconds
        feats["time_interval"] = t_seconds.map(lambda x: 0 if x < 37800 else min(int((x - 37800) / 1800) + 1, 7))

        for i in self.pdf_levels:
            bid = df[f"n_bid{i}"]
            ask = df[f"n_ask{i}"]
            bsize = df[f"n_bsize{i}"]
            asize = df[f"n_asize{i}"]
            total_size = bsize + asize
            feats[f"spread_{i}"] = ask - bid
            feats[f"mid_price_{i}"] = (ask + bid) / 2
            feats[f"relative_bid_density_{i}"] = bsize / (total_size + 1e-10)
            feats[f"relative_ask_density_{i}"] = asize / (total_size + 1e-10)
            feats[f"log_bsize{i}"] = np.log1p(bsize)
            feats[f"log_asize{i}"] = np.log1p(asize)

        for i in self.pdf_levels:
            feats[f"weighted_ab_{i}"] = (
                df[f"n_bid{i}"] * df[f"n_asize{i}"] + df[f"n_ask{i}"] * df[f"n_bsize{i}"]
            ) / (df[f"n_bsize{i}"] + df[f"n_asize{i}"] + 1e-10)

        feats["vol1_rel_diff"] = (df["n_bsize1"] - df["n_asize1"]) / (df["n_bsize1"] + df["n_asize1"] + 1e-10)
        feats["vol3_rel_diff"] = (
            df["n_bsize1"] + df["n_bsize2"] + df["n_bsize3"] - df["n_asize1"] - df["n_asize2"] - df["n_asize3"]
        ) / (df["n_bsize1"] + df["n_bsize2"] + df["n_bsize3"] + df["n_asize1"] + df["n_asize2"] + df["n_asize3"] + 1e-10)
        feats["vol5_rel_diff"] = (
            df["n_bsize1"] + df["n_bsize2"] + df["n_bsize3"] + df["n_bsize4"] + df["n_bsize5"]
            - df["n_asize1"] - df["n_asize2"] - df["n_asize3"] - df["n_asize4"] - df["n_asize5"]
        ) / (
            df["n_bsize1"] + df["n_bsize2"] + df["n_bsize3"] + df["n_bsize4"] + df["n_bsize5"]
            + df["n_asize1"] + df["n_asize2"] + df["n_asize3"] + df["n_asize4"] + df["n_asize5"] + 1e-10
        )
        feats["amount_log"] = np.log1p(df["amount_delta"])

        grouped = df.groupby(["sym", "date"], sort=False)
        feats["close_delta"] = grouped["n_close"].diff()
        for i in self.pdf_levels:
            feats[f"bid{i}_delta"] = grouped[f"n_bid{i}"].diff()
            feats[f"ask{i}_delta"] = grouped[f"n_ask{i}"].diff()
            mid_col = f"mid_price_{i}"
            mid_df = pd.DataFrame({"sym": df["sym"], "date": df["date"], mid_col: feats[mid_col]}, index=df.index)
            feats[f"{mid_col}_delta"] = mid_df.groupby(["sym", "date"], sort=False)[mid_col].diff()
            feats[f"bsize{i}_delta"] = grouped[f"n_bsize{i}"].diff()
            feats[f"asize{i}_delta"] = grouped[f"n_asize{i}"].diff()

        feats["close_mean"] = grouped["n_close"].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
        feats["close_std"] = grouped["n_close"].transform(lambda x: x.rolling(window=10, min_periods=1).std())
        feats["close_vs_mean"] = df["n_close"] / (feats["close_mean"] + 1e-10)

        for i in self.pdf_levels:
            for raw_prefix, out_prefix in [("n_bid", "bid"), ("n_ask", "ask"), ("n_bsize", "bsize"), ("n_asize", "asize")]:
                series = df[f"{raw_prefix}{i}"]
                mean = grouped[f"{raw_prefix}{i}"].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
                feats[f"{out_prefix}{i}_mean"] = mean
                feats[f"{out_prefix}{i}_std"] = grouped[f"{raw_prefix}{i}"].transform(lambda x: x.rolling(window=10, min_periods=1).std())
                feats[f"{out_prefix}{i}_vs_mean"] = series / (mean + 1e-10)

            mid_col = f"mid_price_{i}"
            mid_df = pd.DataFrame({"sym": df["sym"], "date": df["date"], mid_col: feats[mid_col]}, index=df.index)
            mid_group = mid_df.groupby(["sym", "date"], sort=False)[mid_col]
            feats[f"{mid_col}_mean"] = mid_group.transform(lambda x: x.rolling(window=10, min_periods=1).mean())
            feats[f"{mid_col}_std"] = mid_group.transform(lambda x: x.rolling(window=10, min_periods=1).std())

        new_df = pd.concat([df, pd.DataFrame(feats, index=df.index)], axis=1)
        for col in self.base_feature_names:
            if col in new_df.columns:
                new_df[col] = _clean_numeric_series(new_df[col])
        return new_df.copy()

    def assemble_pyramid_vector(
        self,
        feature_matrix: np.ndarray,
        row_idx: int,
        available_features: Sequence[str],
    ) -> np.ndarray:
        del available_features
        start = row_idx - self.window_size + 1
        if start >= 0:
            window = feature_matrix[start:row_idx + 1]
        else:
            pad_count = -start
            first = feature_matrix[[0]]
            window = np.vstack([np.repeat(first, pad_count, axis=0), feature_matrix[:row_idx + 1]])
        return window.astype(np.float32).reshape(-1)

    def build_feature_matrix_fast_single_group(
        self,
        df: pd.DataFrame,
        available_features: Sequence[str],
    ) -> pd.DataFrame | None:
        return None


def build_feature_builder(feature_set: str, pdf_level_mode: str = "1-5") -> FeatureBuilder:
    if feature_set == "previous_code":
        return FeatureBuilder()
    if feature_set == "pdf_report":
        return PDFReportFeatureBuilder(pdf_level_mode=pdf_level_mode)
    raise ValueError(f"Unsupported feature set: {feature_set}")


def _time_to_seconds(value) -> int:
    try:
        parts = str(value).split(":")
        if len(parts) != 3:
            return 0
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(float(parts[2]))
    except Exception:
        return 0


def _clean_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], 0).fillna(0)


def _frame_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    arrays = {}
    numeric_cols = [
        "n_close", "amount_delta", "n_midprice",
        *[f"n_bid{i}" for i in range(1, 6)],
        *[f"n_bsize{i}" for i in range(1, 6)],
        *[f"n_ask{i}" for i in range(1, 6)],
        *[f"n_asize{i}" for i in range(1, 6)],
    ]
    for col in numeric_cols:
        arrays[col] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
    return arrays


def _build_numpy_feature_arrays(df: pd.DataFrame, c: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    f = {name: arr for name, arr in c.items()}
    n = len(df)
    mid = c["n_midprice"]
    amount = c["amount_delta"]

    if "time" in df.columns:
        time_seconds = np.asarray([_time_to_seconds(value) for value in df["time"].to_numpy()], dtype=np.float64)
    else:
        time_seconds = np.zeros(n, dtype=np.float64)
    f["time_seconds"] = time_seconds
    f["time_interval"] = np.asarray([min(int((x - 34200) / 1800), 7) if x >= 34200 else 0 for x in time_seconds], dtype=np.float64)

    for i in [1, 3, 5]:
        bid = c[f"n_bid{i}"]
        ask = c[f"n_ask{i}"]
        bsize = c[f"n_bsize{i}"]
        asize = c[f"n_asize{i}"]
        total_size = bsize + asize
        f[f"spread_{i}"] = ask - bid
        f[f"mid_price_{i}"] = (ask + bid) / 2
        f[f"relative_bid_density_{i}"] = bsize / (total_size + 1e-10)
        f[f"relative_ask_density_{i}"] = asize / (total_size + 1e-10)
        f[f"log_bsize{i}"] = np.log1p(bsize)
        f[f"log_asize{i}"] = np.log1p(asize)
        f[f"bid{i}_plus1"] = bid + 1
        f[f"ask{i}_plus1"] = ask + 1

    for i in [1, 3]:
        f[f"weighted_ab_{i}"] = (
            c[f"n_bid{i}"] * c[f"n_asize{i}"] + c[f"n_ask{i}"] * c[f"n_bsize{i}"]
        ) / (c[f"n_bsize{i}"] + c[f"n_asize{i}"] + 1e-10)

    f["vol1_rel_diff"] = (c["n_bsize1"] - c["n_asize1"]) / (c["n_bsize1"] + c["n_asize1"] + 1e-10)
    f["vol3_rel_diff"] = (
        c["n_bsize1"] + c["n_bsize2"] + c["n_bsize3"] - c["n_asize1"] - c["n_asize2"] - c["n_asize3"]
    ) / (c["n_bsize1"] + c["n_bsize2"] + c["n_bsize3"] + c["n_asize1"] + c["n_asize2"] + c["n_asize3"] + 1e-10)
    f["vol5_rel_diff"] = (
        c["n_bsize1"] + c["n_bsize2"] + c["n_bsize3"] + c["n_bsize4"] + c["n_bsize5"]
        - c["n_asize1"] - c["n_asize2"] - c["n_asize3"] - c["n_asize4"] - c["n_asize5"]
    ) / (
        c["n_bsize1"] + c["n_bsize2"] + c["n_bsize3"] + c["n_bsize4"] + c["n_bsize5"]
        + c["n_asize1"] + c["n_asize2"] + c["n_asize3"] + c["n_asize4"] + c["n_asize5"] + 1e-10
    )
    f["amount_normalized"] = np.log1p(amount / (1 + mid))

    f["close_delta"] = _diff(c["n_close"])
    f["bid1_delta"] = _diff(c["n_bid1"])
    f["ask1_delta"] = _diff(c["n_ask1"])
    f["midprice_delta"] = _diff(mid)

    f["close_mean"] = _rolling_mean(c["n_close"], 10)
    f["close_std"] = _rolling_std(c["n_close"], 10)
    f["close_vs_mean"] = c["n_close"] / (f["close_mean"] + 1e-10)

    for i in [1, 3, 5]:
        for side, prefix in [("n_bid", "bid"), ("n_ask", "ask"), ("n_bsize", "bsize"), ("n_asize", "asize")]:
            arr = c[f"{side}{i}"]
            mean = _rolling_mean(arr, 10)
            f[f"{prefix}{i}_mean"] = mean
            f[f"{prefix}{i}_std"] = _rolling_std(arr, 10)
            f[f"{prefix}{i}_vs_mean"] = arr / (mean + 1e-10)
        mid_price = f[f"mid_price_{i}"]
        f[f"mid_price_{i}_mean"] = _rolling_mean(mid_price, 10)
        f[f"mid_price_{i}_std"] = _rolling_std(mid_price, 10)

    f["midprice_mean"] = _rolling_mean(mid, 10)
    f["midprice_std"] = _rolling_std(mid, 10)
    f["cross_weighted_1"] = (c["n_ask1"] * c["n_bsize2"] + c["n_ask2"] * c["n_bsize1"]) / (c["n_bsize1"] + c["n_bsize2"] + 1e-10)
    f["cross_weighted_2"] = (c["n_bid1"] * c["n_asize2"] + c["n_bid2"] * c["n_asize1"]) / (c["n_asize1"] + c["n_asize2"] + 1e-10)
    f["midprice_ma5"] = _rolling_mean(mid, 5)

    temp_mid = 2 + c["n_ask1"] + c["n_bid1"]
    for period in [5, 10, 20, 40, 60]:
        f[f"volatility_{period}"] = _fill_nan(temp_mid / (_shift(temp_mid, period) + 1e-10) - 1, 0)

    ema12 = _ewm(mid, span=12)
    ema26 = _ewm(mid, span=26)
    f["macd_dif"] = ema12 - ema26
    f["macd_dea"] = _ewm(f["macd_dif"], span=9)
    f["macd_bar"] = f["macd_dif"] - f["macd_dea"]

    low_9 = _rolling_min(c["n_bid1"], 9)
    high_9 = _rolling_max(c["n_ask1"], 9)
    rsv = 100 * (mid - low_9) / (high_9 - low_9 + 1e-10)
    f["kdj_k"] = _ewm(rsv, alpha=1 / 3)
    f["kdj_d"] = _ewm(f["kdj_k"], alpha=1 / 3)
    f["kdj_j"] = 3 * f["kdj_k"] - 2 * f["kdj_d"]

    for w in [1, 5, 10, 30, 60, 100]:
        with np.errstate(divide="ignore", invalid="ignore"):
            f[f"roc_{w}"] = _fill_nan(mid / _shift(mid, w) - 1, 0)

    f["vol1_rel_diff_mean_5"] = _fill_nan(_rolling_mean(f["vol1_rel_diff"], 5), 0)
    f["vol1_rel_diff_mean_20"] = _fill_nan(_rolling_mean(f["vol1_rel_diff"], 20), 0)

    for window in [20, 100, 300]:
        mid_mean = _rolling_mean(mid, window)
        mid_std = _rolling_std(mid, window)
        f[f"price_zscore_{window}"] = (mid - mid_mean) / (mid_std + 1e-10)
        mid_recent_mean = _rolling_mean(mid, window // 3)
        mid_early_mean = _rolling_mean(_shift(mid, window * 2 // 3), window // 3)
        f[f"price_slope_{window}"] = (mid_recent_mean - mid_early_mean) / (window * 2 // 3 + 1e-10)

        amount_mean = _rolling_mean(amount, window)
        amount_std = _rolling_std(amount, window)
        f[f"amount_zscore_{window}"] = (amount - amount_mean) / (amount_std + 1e-10)
        amount_recent_mean = _rolling_mean(amount, window // 3)
        amount_early_mean = _rolling_mean(_shift(amount, window * 2 // 3), window // 3)
        f[f"amount_slope_{window}"] = (amount_recent_mean - amount_early_mean) / (window * 2 // 3 + 1e-10)

    roll100_max = _rolling_max(mid, 100)
    roll100_min = _rolling_min(mid, 100)
    f["price_percentile_100"] = (mid - roll100_min) / (roll100_max - roll100_min + 1e-10)

    total_bid_size = sum(c[f"n_bsize{i}"] for i in range(1, 6))
    total_ask_size = sum(c[f"n_asize{i}"] for i in range(1, 6))
    f["total_imbalance"] = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size + 1e-10)
    weighted_bid = sum(c[f"n_bid{i}"] * c[f"n_bsize{i}"] for i in range(1, 6)) / (total_bid_size + 1e-10)
    weighted_ask = sum(c[f"n_ask{i}"] * c[f"n_asize{i}"] for i in range(1, 6)) / (total_ask_size + 1e-10)
    f["total_imbalance_weighted"] = (weighted_bid - weighted_ask) / (weighted_bid + weighted_ask + 1e-10)
    f["bid_slope"] = (c["n_bid1"] - c["n_bid5"]) / (c["n_bid1"] + 1e-10)
    f["ask_slope"] = (c["n_ask5"] - c["n_ask1"]) / (c["n_ask1"] + 1e-10)
    f["price_elasticity_10"] = _fill_nan(_diff(mid, 10), 0) / (_rolling_sum(amount, 10) + 1e-10)
    f["orderbook_pressure"] = (total_bid_size - total_ask_size) * (mid - c["n_bid1"]) / (c["n_ask1"] - c["n_bid1"] + 1e-10)

    f["ofi_1"] = _calc_ofi(c["n_bid1"], c["n_bsize1"], c["n_ask1"], c["n_asize1"])
    ofi_2 = _calc_ofi(c["n_bid2"], c["n_bsize2"], c["n_ask2"], c["n_asize2"])
    ofi_3 = _calc_ofi(c["n_bid3"], c["n_bsize3"], c["n_ask3"], c["n_asize3"])
    f["ofi_avg_3"] = (f["ofi_1"] + ofi_2 + ofi_3) / 3
    f["ofi_1_rolling_5"] = _fill_nan(_rolling_sum(f["ofi_1"], 5), 0)
    f["ofi_spread_ratio"] = f["ofi_1"] / (f["spread_1"] + 1e-10)
    f["midprice_accel"] = _fill_nan(_diff(f["midprice_delta"]), 0)
    f["energy_burst"] = amount * f["midprice_accel"]
    f["imb_velocity"] = _fill_nan(_diff(f["total_imbalance"]), 0)
    f["imb_accel"] = _fill_nan(_diff(f["imb_velocity"]), 0)

    for lag in [1, 5, 20]:
        f[f"lag_mid_{lag}"] = _fill_nan(_shift(mid, lag), 0)
    for lag in [1, 5]:
        f[f"lag_bid1_{lag}"] = _fill_nan(_shift(c["n_bid1"], lag), 0)
        f[f"lag_ask1_{lag}"] = _fill_nan(_shift(c["n_ask1"], lag), 0)
        f[f"lag_bsize1_{lag}"] = _fill_nan(_shift(c["n_bsize1"], lag), 0)
        f[f"lag_asize1_{lag}"] = _fill_nan(_shift(c["n_asize1"], lag), 0)

    amount_mean_100 = _fill_nan(_rolling_mean(amount, 100), 1e-10)
    for period in [5, 20, 60]:
        f[f"volume_flow_{period}"] = _fill_nan(_rolling_sum(amount, period), 0) / (amount_mean_100 + 1e-10)

    f["micro_price"] = (c["n_bid1"] * c["n_asize1"] + c["n_ask1"] * c["n_bsize1"]) / (c["n_bsize1"] + c["n_asize1"] + 1e-10)
    f["micro_price_diff"] = _fill_nan(_diff(f["micro_price"]), 0)
    weights = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    bid_depth = sum(c[f"n_bsize{i}"] * weights[i - 1] for i in range(1, 6))
    ask_depth = sum(c[f"n_asize{i}"] * weights[i - 1] for i in range(1, 6))
    f["weighted_depth_imbalance"] = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-10)
    f["ofi_momentum_sync"] = f["ofi_1"] * f["midprice_delta"]
    vol10 = _fill_nan(temp_mid / (_shift(temp_mid, 10) + 1e-10) - 1, 0)
    f["vov_10"] = _fill_nan(_rolling_std(vol10, 5), 0)
    f["bid_convexity"] = (c["n_bsize1"] + c["n_bsize5"] - 2 * c["n_bsize3"]) / (c["n_bsize3"] + 1e-10)
    f["ask_convexity"] = (c["n_asize1"] + c["n_asize5"] - 2 * c["n_asize3"]) / (c["n_asize3"] + 1e-10)
    f["buy_intensity"] = amount / (c["n_asize1"] + 1e-10)
    f["sell_intensity"] = amount / (c["n_bsize1"] + 1e-10)
    f["ofi_ema_5"] = _fill_nan(_ewm(f["ofi_1"], span=5), 0)
    f["ofi_ema_10"] = _fill_nan(_ewm(f["ofi_1"], span=10), 0)

    amount_abs = np.abs(amount)
    for period in [5, 20]:
        amount_vol = _fill_nan(_rolling_std(amount_abs, period), 1e-10)
        amount_mean = _fill_nan(_rolling_mean(amount_abs, period), 1e-10)
        f[f"vpin_{period}"] = amount_vol / (amount_mean + 1e-10)

    f["book_curvature"] = (f["bid_convexity"] + f["ask_convexity"]) / 2
    f["depth_pressure"] = (bid_depth + ask_depth) * f["weighted_depth_imbalance"]
    return f


def _clean_numpy_array(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _fill_nan(values: np.ndarray, fill_value: float) -> np.ndarray:
    return np.where(np.isfinite(values), values, fill_value)


def _shift(values: np.ndarray, periods: int) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    if periods < len(values):
        result[periods:] = values[:-periods]
    return result


def _diff(values: np.ndarray, periods: int = 1) -> np.ndarray:
    return values - _shift(values, periods)


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    cleaned = np.where(valid, values, 0.0)
    sums = _rolling_cumsum(cleaned, window)
    counts = _rolling_cumsum(valid.astype(np.float64), window)
    return np.where(counts > 0, sums, np.nan)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    cleaned = np.where(valid, values, 0.0)
    sums = _rolling_cumsum(cleaned, window)
    counts = _rolling_cumsum(valid.astype(np.float64), window)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    cleaned = np.where(valid, values, 0.0)
    sums = _rolling_cumsum(cleaned, window)
    sums_sq = _rolling_cumsum(cleaned * cleaned, window)
    counts = _rolling_cumsum(valid.astype(np.float64), window)
    with np.errstate(divide="ignore", invalid="ignore"):
        variance = (sums_sq - (sums * sums) / counts) / (counts - 1)
    variance = np.where(variance < 0, 0, variance)
    return np.where(counts > 1, np.sqrt(variance), np.nan)


def _rolling_cumsum(values: np.ndarray, window: int) -> np.ndarray:
    cumsum = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - window)
    return cumsum[end] - cumsum[start]


def _rolling_min(values: np.ndarray, window: int) -> np.ndarray:
    result = np.empty(len(values), dtype=np.float64)
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        window_values = values[start:idx + 1]
        valid = window_values[np.isfinite(window_values)]
        result[idx] = np.min(valid) if valid.size else np.nan
    return result


def _rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    result = np.empty(len(values), dtype=np.float64)
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        window_values = values[start:idx + 1]
        valid = window_values[np.isfinite(window_values)]
        result[idx] = np.max(valid) if valid.size else np.nan
    return result


def _ewm(values: np.ndarray, span: int | None = None, alpha: float | None = None) -> np.ndarray:
    if alpha is None:
        alpha = 2.0 / (span + 1.0)
    result = np.empty(len(values), dtype=np.float64)
    if len(values) == 0:
        return result
    result[0] = values[0]
    for idx in range(1, len(values)):
        result[idx] = alpha * values[idx] + (1 - alpha) * result[idx - 1]
    return result


def _calc_ofi(bid_p: np.ndarray, bid_v: np.ndarray, ask_p: np.ndarray, ask_v: np.ndarray) -> np.ndarray:
    prev_bid_p = _shift(bid_p, 1)
    prev_bid_v = _shift(bid_v, 1)
    prev_ask_p = _shift(ask_p, 1)
    prev_ask_v = _shift(ask_v, 1)
    ofi_bid = np.where(bid_p > prev_bid_p, bid_v, np.where(bid_p == prev_bid_p, bid_v - prev_bid_v, -prev_bid_v))
    ofi_ask = np.where(ask_p < prev_ask_p, ask_v, np.where(ask_p == prev_ask_p, ask_v - prev_ask_v, -prev_ask_v))
    return _fill_nan(ofi_bid - ofi_ask, 0)
