"""
Tests for the uncertainty-aware allocator.
"""

import numpy as np
import pandas as pd

from quant.ml.allocator import (
    run_allocator,
    vanilla_ls_weights,
    width_scaled_weights,
)


def _make_preds(n_dates: int, n_tickers: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_dates, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    rows = []
    for d in dates:
        for t in tickers:
            pred = rng.standard_normal() * 0.02
            width = abs(rng.standard_normal()) * 0.03 + 0.05
            rows.append({
                "ticker": t, "date": d,
                "prediction": pred,
                "lower": pred - width,
                "upper": pred + width,
                "target_fwd_ret_21d": rng.standard_normal() * 0.03,
            })
    return pd.DataFrame(rows)


def test_vanilla_long_short_weights_sum_correctly():
    df = _make_preds(1, 10)
    day = df[df["date"] == df["date"].iloc[0]]
    w = vanilla_ls_weights(day, top_k=3)
    longs = w[w["weight"] > 0]
    shorts = w[w["weight"] < 0]
    assert abs(longs["weight"].sum() - 1.0) < 1e-6
    assert abs(shorts["weight"].sum() + 1.0) < 1e-6


def test_width_scaled_weights_gross_is_one():
    df = _make_preds(1, 10)
    day = df[df["date"] == df["date"].iloc[0]]
    w = width_scaled_weights(day, top_k=3)
    assert abs(w["weight"].abs().sum() - 1.0) < 1e-6


def test_width_scaled_downweights_wide_intervals():
    """A ticker with a huge conformal interval should get less weight than a
    tight-interval ticker with the same prediction rank."""
    day = pd.DataFrame({
        "ticker": ["A", "B", "C", "D", "E", "F"],
        "date": [pd.Timestamp("2022-01-01")] * 6,
        "prediction": [0.05, 0.04, 0.03, -0.03, -0.04, -0.05],
        "lower": [0.0, 0.0, 0.0, -0.06, -0.07, -0.08],
        "upper": [0.1, 2.0, 0.06, 0.0, -0.01, -0.02],   # B has huge upper
    })
    w = width_scaled_weights(day, top_k=3)
    a = float(w.loc[w["ticker"] == "A", "weight"].iloc[0])
    b = float(w.loc[w["ticker"] == "B", "weight"].iloc[0])
    assert a > b, f"A (tight) should outweigh B (wide): {a} vs {b}"


def test_run_allocator_returns_valid_summary():
    df = _make_preds(200, 8)
    out = run_allocator(df, policy="width_scaled", horizon_days=21)
    assert "pnl_df" in out
    assert "sharpe" in out
    assert out["n_periods"] >= 1
    assert out["gross_leverage_mean"] > 0
