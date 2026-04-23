"""
quant/ml/baseline.py — scikit-learn GradientBoosting baseline.

Serves two roles:
    1. Honest benchmark for the Transformer.
    2. Tree-based model that SHAP (TreeExplainer) can explain quickly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class BaselineForecaster:
    feature_cols: List[str]
    target_col: str = "target_fwd_ret"
    n_estimators: int = 300
    max_depth: int = 3
    learning_rate: float = 0.05
    random_state: int = 42
    pipeline: Pipeline = None

    def fit(self, train_df: pd.DataFrame) -> "BaselineForecaster":
        X = train_df[self.feature_cols].astype(float).values
        y = train_df[self.target_col].astype(float).values

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("gbr", GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                subsample=0.8,
            )),
        ])
        self.pipeline.fit(X, y)
        logger.info("Baseline GBR fitted on %d samples, %d features", len(X), X.shape[1])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].astype(float).values
        return self.pipeline.predict(X)

    @property
    def model(self) -> GradientBoostingRegressor:
        return self.pipeline.named_steps["gbr"]

    @property
    def scaler(self) -> StandardScaler:
        return self.pipeline.named_steps["scaler"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from pipeline.ingest import fetch_prices
    from quant.ml.features import build_feature_panel, FEATURE_COLS
    prices = fetch_prices(start_date="2021-01-01", end_date="2024-01-01")
    panel = build_feature_panel(prices)
    cutoff = panel["date"].quantile(0.8)
    train = panel[panel["date"] < cutoff]
    test = panel[panel["date"] >= cutoff]
    fc = BaselineForecaster(feature_cols=FEATURE_COLS).fit(train)
    preds = fc.predict(test)
    print("test R²-ish corr:", np.corrcoef(preds, test["target_fwd_ret"].values)[0, 1])
