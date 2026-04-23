"""
quant/ml/conformal.py — Split-conformal prediction intervals.

Why conformal: our models output a point estimate of the forward return. In
real asset-allocation we need a *calibrated* uncertainty band — one that
actually contains the realised return 1-alpha fraction of the time, without
assuming Gaussian residuals.

This module implements Split Conformal Prediction (Vovk et al.; Lei et al.
2018): fit the base model on a proper training fold, compute residuals on a
held-out calibration fold, and take the (1-alpha) quantile of |residuals| as
the interval half-width.

Works with any object that exposes .fit(df) / .predict(df)-returning-array or
.predict(df)-returning-DataFrame with a 'prediction' column.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _as_array(pred, n_expected: int) -> np.ndarray:
    """Normalise various predict() return shapes to a 1-D array."""
    if isinstance(pred, pd.DataFrame):
        if "prediction" not in pred.columns:
            raise ValueError("DataFrame prediction must have a 'prediction' column")
        return pred["prediction"].to_numpy().astype(float)
    return np.asarray(pred, dtype=float)


@dataclass
class SplitConformalWrapper:
    """
    Wraps a base forecaster with split-conformal calibration.

    Attributes
    ----------
    base_model        : the underlying forecaster (already-fit on train-proper)
    calibration_width : the absolute-residual quantile (half-width of interval)
    alpha             : 1 - target coverage (e.g. 0.1 for a 90% interval)
    context_df        : optional historical context passed through to sequence
                        models' predict() so they can build full lookback
                        windows on calibration/test rows.
    """
    base_model: Any
    alpha: float = 0.1
    calibration_width: float = float("nan")
    calibration_residuals: np.ndarray = None  # type: ignore[assignment]
    context_df: Any = None

    def _predict(self, df: pd.DataFrame):
        if self.context_df is not None:
            try:
                return self.base_model.predict(df, context_df=self.context_df)
            except TypeError:
                pass
        return self.base_model.predict(df)

    def calibrate(
        self,
        calibration_df: pd.DataFrame,
        target_col: str = "target_fwd_ret",
    ) -> "SplitConformalWrapper":
        """
        Compute |residuals| on calibration_df and store the (1-alpha) quantile.
        calibration_df must NOT overlap with the base model's training set.
        """
        pred_raw = self._predict(calibration_df)

        if isinstance(pred_raw, pd.DataFrame):
            # Align preds with calibration rows by (ticker, date)
            merged = calibration_df.merge(
                pred_raw, on=["ticker", "date"], how="inner"
            )
            y_true = merged[target_col].to_numpy(dtype=float)
            y_pred = merged["prediction"].to_numpy(dtype=float)
        else:
            y_true = calibration_df[target_col].to_numpy(dtype=float)
            y_pred = _as_array(pred_raw, len(y_true))

        if len(y_true) == 0:
            raise ValueError("No calibration samples — base model emitted 0 predictions.")

        residuals = np.abs(y_true - y_pred)
        # Finite-sample correction: q_level = ceil((n+1)(1-alpha)) / n
        n = len(residuals)
        q_level = min(1.0, (np.ceil((n + 1) * (1 - self.alpha))) / n)
        self.calibration_width = float(np.quantile(residuals, q_level))
        self.calibration_residuals = residuals

        logger.info(
            "Conformal calibrated: n=%d, alpha=%.2f, half-width=%.5f",
            n, self.alpha, self.calibration_width,
        )
        return self

    def predict_interval(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame with columns: ticker, date, prediction, lower, upper.
        """
        if np.isnan(self.calibration_width):
            raise RuntimeError("Call .calibrate() first")

        pred_raw = self._predict(df)

        if isinstance(pred_raw, pd.DataFrame):
            out = pred_raw.copy()
        else:
            out = df[["ticker", "date"]].copy().reset_index(drop=True)
            out["prediction"] = _as_array(pred_raw, len(out))

        out["lower"] = out["prediction"] - self.calibration_width
        out["upper"] = out["prediction"] + self.calibration_width
        return out

    def empirical_coverage(
        self,
        df: pd.DataFrame,
        target_col: str = "target_fwd_ret",
    ) -> float:
        """Fraction of rows where true target falls inside the conformal interval."""
        intervals = self.predict_interval(df)
        merged = df.merge(intervals, on=["ticker", "date"], how="inner")
        y = merged[target_col].to_numpy(dtype=float)
        lo = merged["lower"].to_numpy(dtype=float)
        hi = merged["upper"].to_numpy(dtype=float)
        inside = ((y >= lo) & (y <= hi)).mean()
        return float(inside)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from pipeline.ingest import fetch_prices
    from quant.ml.features import build_feature_panel, FEATURE_COLS
    from quant.ml.baseline import BaselineForecaster

    prices = fetch_prices(start_date="2021-01-01", end_date="2024-01-01")
    panel = build_feature_panel(prices)
    panel = panel.sort_values("date").reset_index(drop=True)
    q = panel["date"].quantile([0.6, 0.8]).tolist()
    train = panel[panel["date"] < q[0]]
    calib = panel[(panel["date"] >= q[0]) & (panel["date"] < q[1])]
    test = panel[panel["date"] >= q[1]]

    base = BaselineForecaster(feature_cols=FEATURE_COLS).fit(train)
    conf = SplitConformalWrapper(base_model=base, alpha=0.1).calibrate(calib)
    cov = conf.empirical_coverage(test)
    print(f"Test empirical coverage (target 0.90): {cov:.3f}")
