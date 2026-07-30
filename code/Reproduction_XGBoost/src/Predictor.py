"""Platform submission Predictor for the XGBoost stock movement model."""

from __future__ import annotations

import json
import os
from typing import List

import numpy as np
import pandas as pd
import xgboost as xgb

try:
    from .feature_builder import build_feature_builder
except ImportError:
    from feature_builder import build_feature_builder


class Predictor:
    def __init__(self):
        base_dir = os.path.dirname(__file__)
        self.labels = ["label_5", "label_10", "label_20", "label_40", "label_60"]

        spec_path = os.path.join(base_dir, "feature_spec.json")
        thresholds_path = os.path.join(base_dir, "thresholds.json")
        with open(spec_path, "r", encoding="utf-8") as f:
            self.feature_spec = json.load(f)
        with open(thresholds_path, "r", encoding="utf-8") as f:
            self.thresholds = json.load(f)

        self.builder = build_feature_builder(
            self.feature_spec.get("feature_set", "previous_code"),
            self.feature_spec.get("pdf_level_mode", "1-5"),
        )
        self.available_features = self.feature_spec["available_features"]
        self.models = {}
        for label in self.labels:
            model_path = os.path.join(base_dir, f"model_{label}.json")
            booster = xgb.Booster()
            booster.load_model(model_path)
            self.models[label] = booster

    def predict(self, x: List[pd.DataFrame]) -> List[List[int]]:
        if not x:
            return []

        vectors = []
        for df in x:
            vector = self.builder.transform_window(df.copy(), self.available_features)
            vectors.append(vector)
        x_matrix = np.vstack(vectors).astype(np.float32)
        dmatrix = xgb.DMatrix(x_matrix)

        outputs: List[List[int]] = []
        label_predictions = []
        for label in self.labels:
            proba = self.models[label].predict(dmatrix)
            threshold = float(self.thresholds.get(label, 0.88))
            label_predictions.append(_apply_threshold(proba, threshold))

        stacked = np.vstack(label_predictions).T
        for row in stacked:
            outputs.append([int(value) for value in row])
        return outputs


def _apply_threshold(proba: np.ndarray, threshold: float) -> np.ndarray:
    if threshold <= 0:
        return proba.argmax(axis=1).astype(np.int64)

    pred = np.ones(proba.shape[0], dtype=np.int64)
    down_mask = proba[:, 0] > threshold
    up_mask = proba[:, 2] > threshold
    choose_up = up_mask & (~down_mask | (proba[:, 2] >= proba[:, 0]))
    choose_down = down_mask & ~choose_up
    pred[choose_down] = 0
    pred[choose_up] = 2
    return pred
