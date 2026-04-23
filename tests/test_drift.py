"""
Tests for quant.ml.drift — KS-based feature distribution drift detection.
"""

import numpy as np
import pandas as pd

from quant.ml.drift import detect_feature_drift


def test_detects_clear_drift():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=400, freq="B")
    df = pd.DataFrame({"date": dates})
    df["feat_ok"] = rng.standard_normal(len(df))
    df["feat_bad"] = np.concatenate([
        rng.standard_normal(200),
        rng.standard_normal(200) + 3.0,   # big mean shift
    ])
    rep = detect_feature_drift(
        df, ["feat_ok", "feat_bad"],
        reference_start="2020-01-01", reference_end="2020-10-01",
        production_start="2020-10-02", production_end="2021-08-01",
    )
    by_feat = {r["feature"]: r for _, r in rep.per_feature.iterrows()}
    assert bool(by_feat["feat_bad"]["drift"])
    # feat_ok may or may not flag — we only assert the bad one is flagged


def test_no_drift_when_same_distribution():
    rng = np.random.default_rng(5)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    df = pd.DataFrame({"date": dates})
    df["f"] = rng.standard_normal(len(df))
    rep = detect_feature_drift(
        df, ["f"],
        reference_start="2020-01-01", reference_end="2020-10-01",
        production_start="2020-10-02", production_end="2021-11-01",
    )
    assert not bool(rep.per_feature.iloc[0]["drift"])
    # p-value should be sensible — not ultra-tiny
    assert rep.per_feature.iloc[0]["p_value"] > 0.01


def test_drift_summary_counts():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    df = pd.DataFrame({"date": dates})
    df["a"] = rng.standard_normal(len(df))
    df["b"] = np.concatenate([rng.standard_normal(150), rng.standard_normal(150) + 4])
    rep = detect_feature_drift(
        df, ["a", "b"],
        reference_start="2020-01-01", reference_end="2020-07-15",
        production_start="2020-07-16", production_end="2021-03-01",
    )
    assert rep.n_features == 2
    assert rep.n_drifted >= 1
