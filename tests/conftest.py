"""Shared fixtures."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def synthetic_prices():
    """Deterministic synthetic OHLCV across 5 tickers × 500 business days."""
    rng = np.random.default_rng(42)
    tickers = ["AAA.NS", "BBB.NS", "CCC.NS", "DDD.NS", "EEE.NS"]
    dates = pd.bdate_range("2020-01-01", periods=500)
    frames = []
    for t in tickers:
        # GBM with ticker-specific drift and vol
        drift = rng.uniform(0.05, 0.15)
        vol = rng.uniform(0.15, 0.35)
        daily = np.exp((drift - 0.5 * vol**2) / 252 + vol / np.sqrt(252) * rng.standard_normal(len(dates)))
        close = 100.0 * np.cumprod(daily)
        frames.append(pd.DataFrame({
            "ticker": t,
            "date": dates,
            "open": close * (1 + rng.standard_normal(len(dates)) * 0.005),
            "high": close * (1 + np.abs(rng.standard_normal(len(dates))) * 0.01),
            "low": close * (1 - np.abs(rng.standard_normal(len(dates))) * 0.01),
            "close": close,
            "volume": rng.integers(100000, 5000000, size=len(dates)),
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="session")
def synthetic_panel(synthetic_prices):
    from quant.ml.features import build_feature_panel
    return build_feature_panel(synthetic_prices)
