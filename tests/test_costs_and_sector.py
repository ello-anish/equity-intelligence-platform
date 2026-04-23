"""
Tests for:
  - sector_neutral_weights — per-sector net exposure sums to ~0
  - monthly_rebalance_backtest — tc drag scales with tc_bps; turnover >= 0
  - run_allocator with tc_bps — net Sharpe <= gross Sharpe
"""

import numpy as np
import pandas as pd

from quant.ml.allocator import (
    DEFAULT_SECTOR_MAP,
    run_allocator,
    sector_neutral_weights,
)
from quant.ml.evaluation import monthly_rebalance_backtest


def _make_preds(n_dates: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_dates, freq="B")
    tickers = list(DEFAULT_SECTOR_MAP.keys())   # 10 NSE tickers with sectors
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


def test_sector_neutral_net_zero_per_sector():
    df = _make_preds(1)
    day = df[df["date"] == df["date"].iloc[0]]
    w = sector_neutral_weights(day, top_k=3)
    w["sector"] = w["ticker"].map(DEFAULT_SECTOR_MAP)
    per_sector = w.groupby("sector")["weight"].sum()
    # Sector-neutral: each sector's net exposure should be ~0
    # (up to numerical noise). Some sectors may drop out if all their
    # names were in the same tail — we allow small deviations.
    assert (per_sector.abs() < 0.1).all(), f"non-zero sector: {per_sector.to_dict()}"


def test_monthly_rebalance_tc_reduces_net_return():
    preds = _make_preds(200)
    r0 = monthly_rebalance_backtest(preds, top_k=3, tc_bps=0)
    r_costly = monthly_rebalance_backtest(preds, top_k=3, tc_bps=50)
    assert r_costly["total_return"] <= r0["total_return"]
    assert r_costly["total_tc_drag"] > 0


def test_monthly_rebalance_turnover_nonnegative():
    preds = _make_preds(200)
    r = monthly_rebalance_backtest(preds, top_k=3, tc_bps=20)
    assert (r["pnl_df"]["turnover"] >= 0).all()
    assert r["avg_turnover"] >= 0


def test_allocator_net_sharpe_leq_gross_sharpe():
    preds = _make_preds(250)
    r = run_allocator(preds, policy="width_scaled", tc_bps=30, horizon_days=21)
    # Net Sharpe should be <= gross Sharpe when tc > 0 and turnover > 0.
    # Allow tiny tolerance for numerical noise.
    assert r["sharpe"] <= r["sharpe_gross"] + 1e-9
    assert r["total_tc_drag"] >= 0


def test_allocator_sector_neutral_policy_runs():
    preds = _make_preds(120)
    r = run_allocator(preds, policy="sector_neutral", tc_bps=10, horizon_days=21)
    assert "error" not in r
    assert r["policy"] == "sector_neutral"
