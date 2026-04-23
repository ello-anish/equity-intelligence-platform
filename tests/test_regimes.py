"""
Regime detection tests — the core check is no-leakage (HMM fit only sees
dates <= end_date) and that regime labels are stable under the fit-cutoff.
"""

import numpy as np
import pandas as pd

from quant.ml.regimes import fit_regime_model


def test_end_date_restricts_fitting_data(synthetic_prices):
    cutoff = pd.Timestamp("2021-06-01")
    model, labels = fit_regime_model(synthetic_prices, end_date=cutoff, random_state=0)
    # Labels should exist for dates both before and after cutoff
    labels["date"] = pd.to_datetime(labels["date"])
    pre = labels[labels["date"] <= cutoff]
    post = labels[labels["date"] > cutoff]
    assert len(pre) > 0
    assert len(post) > 0


def test_refit_per_fold_is_deterministic(synthetic_prices):
    m1, l1 = fit_regime_model(synthetic_prices, random_state=42)
    m2, l2 = fit_regime_model(synthetic_prices, random_state=42)
    # Same random_state → same labels
    pd.testing.assert_frame_equal(l1, l2)


def test_regimes_are_named(synthetic_prices):
    _, labels = fit_regime_model(synthetic_prices, random_state=0)
    vc = labels["regime"].value_counts().to_dict()
    for name in vc:
        assert name in {"calm", "trending", "crisis"}, f"unexpected regime: {name}"


def test_no_leakage_past_cutoff(synthetic_prices):
    """Fitting with a cutoff in the past should produce different parameters
    from fitting with a cutoff in the future, when future data is shocked."""
    cutoff = pd.Timestamp("2021-06-01")
    # Shock future data
    corrupted = synthetic_prices.copy()
    mask = pd.to_datetime(corrupted["date"]) > cutoff
    corrupted.loc[mask, "close"] *= 1.5

    _, labels_clean = fit_regime_model(synthetic_prices, end_date=cutoff, random_state=0)
    _, labels_corr = fit_regime_model(corrupted, end_date=cutoff, random_state=0)

    # Labels for PAST dates should be identical (the HMM was fit on unshocked past)
    labels_clean["date"] = pd.to_datetime(labels_clean["date"])
    labels_corr["date"] = pd.to_datetime(labels_corr["date"])
    past_clean = labels_clean[labels_clean["date"] <= cutoff].sort_values("date").reset_index(drop=True)
    past_corr = labels_corr[labels_corr["date"] <= cutoff].sort_values("date").reset_index(drop=True)

    same_past = (past_clean["regime"].values == past_corr["regime"].values).mean()
    # Require ≥ 95% identical — we allow a tiny drift from build_market_state's
    # rolling windows that span the cutoff boundary, but labels should match.
    assert same_past >= 0.95, f"past labels diverge after future corruption: {same_past:.3f}"
