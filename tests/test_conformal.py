"""
Tests for split-conformal prediction intervals.

Core invariants:
  1. Empirical coverage on a fresh test set converges to the nominal 1-α.
  2. Calibration raises on empty data (won't silently succeed).
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import pytest

from quant.ml.conformal import SplitConformalWrapper


@dataclass
class _LinearToyModel:
    """Trivial model: y = w · x + noise, exposes .predict(df) → array."""
    w: np.ndarray = None
    feature_cols: List[str] = None

    def fit(self, df: pd.DataFrame):
        X = df[self.feature_cols].values
        y = df["target_fwd_ret"].values
        # Closed-form OLS
        self.w = np.linalg.lstsq(X, y, rcond=None)[0]
        return self

    def predict(self, df: pd.DataFrame):
        return df[self.feature_cols].values @ self.w


def _make_toy_data(n: int, rng: np.random.Generator) -> pd.DataFrame:
    X = rng.standard_normal((n, 3)).astype(np.float32)
    true_w = np.array([0.3, -0.2, 0.1], dtype=np.float32)
    noise = rng.standard_normal(n).astype(np.float32) * 0.05
    y = X @ true_w + noise
    return pd.DataFrame({
        "ticker": ["T0"] * n,
        "date": pd.date_range("2020-01-01", periods=n, freq="B"),
        "f0": X[:, 0], "f1": X[:, 1], "f2": X[:, 2],
        "target_fwd_ret": y,
    })


def test_conformal_coverage_converges():
    """With n_cal = 5000, empirical coverage should be within 3pp of nominal."""
    rng = np.random.default_rng(7)
    train = _make_toy_data(2000, rng)
    calib = _make_toy_data(5000, rng)
    test = _make_toy_data(5000, rng)

    model = _LinearToyModel(feature_cols=["f0", "f1", "f2"]).fit(train)
    conf = SplitConformalWrapper(base_model=model, alpha=0.1).calibrate(calib)
    cov = conf.empirical_coverage(test)

    assert 0.87 <= cov <= 0.93, f"coverage={cov} far from nominal 0.90"


def test_conformal_width_scales_with_noise():
    """Tighter-fitting model → narrower intervals."""
    rng = np.random.default_rng(0)
    train = _make_toy_data(2000, rng)
    calib_low_noise = _make_toy_data(2000, rng)

    # Create a calibration set with 10× noise
    calib_high_noise = calib_low_noise.copy()
    calib_high_noise["target_fwd_ret"] += rng.standard_normal(len(calib_high_noise)) * 0.5

    model = _LinearToyModel(feature_cols=["f0", "f1", "f2"]).fit(train)
    conf_tight = SplitConformalWrapper(base_model=model, alpha=0.1).calibrate(calib_low_noise)
    conf_wide = SplitConformalWrapper(base_model=model, alpha=0.1).calibrate(calib_high_noise)

    assert conf_wide.calibration_width > conf_tight.calibration_width


def test_predict_interval_contains_prediction():
    rng = np.random.default_rng(5)
    train = _make_toy_data(500, rng)
    calib = _make_toy_data(500, rng)
    test = _make_toy_data(100, rng)

    model = _LinearToyModel(feature_cols=["f0", "f1", "f2"]).fit(train)
    conf = SplitConformalWrapper(base_model=model, alpha=0.1).calibrate(calib)
    iv = conf.predict_interval(test)

    assert (iv["lower"] <= iv["prediction"]).all()
    assert (iv["upper"] >= iv["prediction"]).all()


def test_predict_without_calibration_raises():
    rng = np.random.default_rng(0)
    train = _make_toy_data(200, rng)
    model = _LinearToyModel(feature_cols=["f0", "f1", "f2"]).fit(train)
    conf = SplitConformalWrapper(base_model=model, alpha=0.1)
    with pytest.raises(RuntimeError):
        conf.predict_interval(train)
