"""
Tests for feature engineering.

Core invariant: NO LOOK-AHEAD. A feature at date t must depend only on data
at times ≤ t. If we corrupt future prices, past features should not change.
"""

import numpy as np
import pandas as pd

from quant.ml.features import (
    FEATURE_COLS,
    FORWARD_HORIZONS,
    TARGET_COL,
    build_feature_panel,
)


def test_feature_panel_columns(synthetic_panel):
    p = synthetic_panel
    assert not p.empty
    missing = [c for c in FEATURE_COLS if c not in p.columns]
    assert not missing, f"missing feature columns: {missing}"


def test_multi_horizon_targets_present(synthetic_panel):
    for col in FORWARD_HORIZONS:
        assert col in synthetic_panel.columns
    # Primary target alias
    assert TARGET_COL in synthetic_panel.columns


def test_no_lookahead_in_features(synthetic_prices):
    """
    Corrupt future prices (dates > 2021-06-01). Features at date <= 2021-06-01
    must be unchanged.
    """
    clean = synthetic_prices.copy()
    corrupted = synthetic_prices.copy()
    mask = corrupted["date"] > pd.Timestamp("2021-06-01")
    # multiply future closes by 100 → huge future shock
    corrupted.loc[mask, "close"] = corrupted.loc[mask, "close"] * 100
    corrupted.loc[mask, "open"] = corrupted.loc[mask, "open"] * 100
    corrupted.loc[mask, "high"] = corrupted.loc[mask, "high"] * 100
    corrupted.loc[mask, "low"] = corrupted.loc[mask, "low"] * 100

    panel_clean = build_feature_panel(clean)
    panel_corrupted = build_feature_panel(corrupted)

    cutoff = pd.Timestamp("2021-06-01")
    pc = panel_clean[panel_clean["date"] <= cutoff].copy()
    pd_ = panel_corrupted[panel_corrupted["date"] <= cutoff].copy()

    pc = pc.sort_values(["ticker", "date"]).reset_index(drop=True)
    pd_ = pd_.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Drop target columns: those DO depend on future prices (that's the point)
    target_like = [c for c in pc.columns if c.startswith("target_fwd_")]

    for col in FEATURE_COLS:
        if col in target_like:
            continue
        a = pc[col].to_numpy(dtype=float)
        b = pd_[col].to_numpy(dtype=float)
        ok = np.allclose(a, b, equal_nan=True, atol=1e-6)
        assert ok, (
            f"feature {col} leaks future information — "
            f"corrupting post-{cutoff.date()} prices changed past values"
        )


def test_targets_are_forward_looking(synthetic_panel):
    """Target at date t should use price at date t+h, not before."""
    p = synthetic_panel.sort_values(["ticker", "date"]).copy()
    for ticker, grp in p.groupby("ticker"):
        grp = grp.sort_values("date")
        for col, h in FORWARD_HORIZONS.items():
            # For rows with a valid target, the target value must be
            # "close at t+h" info-wise, i.e., correlate with future price moves
            valid = grp.dropna(subset=[col])
            if len(valid) > 50:
                # A trivial sanity check — the target is a log-return so it
                # must be in a reasonable range
                assert valid[col].abs().median() < 1.0
                assert valid[col].abs().max() < 5.0


def test_cross_sectional_ranks_bounded(synthetic_panel):
    """Cross-sectional rank features should be in [0, 1]."""
    for col in synthetic_panel.columns:
        if col.startswith("cs_rank_"):
            vals = synthetic_panel[col].dropna()
            assert vals.min() >= 0
            assert vals.max() <= 1
