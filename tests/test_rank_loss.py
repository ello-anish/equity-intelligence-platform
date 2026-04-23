"""
Tests for the pairwise rank-loss configuration of the Transformer.
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from quant.ml.features import FEATURE_COLS
from quant.ml.transformer_model import TransformerConfig, TransformerForecaster


def test_pairwise_loss_trains_and_predicts(synthetic_panel):
    """The pairwise loss path should train without NaN and produce predictions
    with sensible IC (not necessarily better than MSE on this tiny synthetic
    data — we're checking the code path is wired correctly)."""
    panel = synthetic_panel.dropna(subset=["target_fwd_ret_21d"]).reset_index(drop=True)
    feature_cols = [c for c in FEATURE_COLS if c in panel.columns]

    cutoff = panel["date"].quantile(0.8)
    train = panel[panel["date"] < cutoff].reset_index(drop=True)
    test = panel[panel["date"] >= cutoff].reset_index(drop=True)

    cfg = TransformerConfig(
        epochs=2, seq_len=30, d_model=32, n_heads=2, n_layers=1,
        loss_fn="pairwise",
        target_cols=("target_fwd_ret_21d",),
        primary_target="target_fwd_ret_21d",
    )
    fc = TransformerForecaster(feature_cols=feature_cols, config=cfg).fit(train)
    preds = fc.predict(test, context_df=train)
    assert len(preds) > 0
    # No NaNs in predictions
    assert not preds["prediction"].isna().any()
    # Finite
    assert np.isfinite(preds["prediction"]).all()
