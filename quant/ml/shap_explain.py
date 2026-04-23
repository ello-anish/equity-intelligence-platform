"""
quant/ml/shap_explain.py — SHAP feature attribution for the baseline model.

Produces:
  - Global feature importance (mean |SHAP|)
  - Per-regime mean SHAP (which features matter in crisis vs. calm?)
  - Optional force-plot data for a single prediction

Uses TreeExplainer on the fitted GradientBoostingRegressor inside
BaselineForecaster.pipeline. Falls back to a simple permutation importance
if the `shap` library is not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ShapReport:
    global_importance: pd.DataFrame                  # feature, mean_abs_shap
    per_regime: Optional[pd.DataFrame] = None        # feature × regime matrix
    method: str = "shap_tree"                        # or "permutation_fallback"


def _permutation_importance(model, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """Fallback if shap isn't available."""
    from sklearn.metrics import mean_squared_error
    base_pred = model.predict(X)
    base_err = mean_squared_error(y, base_pred)
    rng = np.random.default_rng(0)
    rows = []
    for i, name in enumerate(feature_names):
        Xp = X.copy()
        rng.shuffle(Xp[:, i])
        err = mean_squared_error(y, model.predict(Xp))
        rows.append({"feature": name, "mean_abs_shap": err - base_err})
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)


def explain_baseline(
    baseline,
    df: pd.DataFrame,
    regimes_df: Optional[pd.DataFrame] = None,
    max_samples: int = 1000,
) -> ShapReport:
    """
    Parameters
    ----------
    baseline   : BaselineForecaster (already fit)
    df         : feature panel to explain (rows sampled if too large)
    regimes_df : optional DataFrame with columns: date, regime
    """
    feature_cols = baseline.feature_cols

    # Sample for speed
    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=0).reset_index(drop=True)

    X_raw = df[feature_cols].astype(float).values
    X_scaled = baseline.scaler.transform(X_raw)
    model = baseline.model

    method = "shap_tree"
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
    except Exception as e:
        logger.warning("shap unavailable (%s) — falling back to permutation importance", e)
        method = "permutation_fallback"
        y = df[baseline.target_col].astype(float).values
        global_imp = _permutation_importance(model, X_scaled, y, feature_cols)
        return ShapReport(global_importance=global_imp, method=method)

    mean_abs = np.abs(shap_values).mean(axis=0)
    global_imp = (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    per_regime = None
    if regimes_df is not None:
        tagged = df[["date"]].copy()
        tagged["date"] = pd.to_datetime(tagged["date"])
        reg = regimes_df.copy()
        reg["date"] = pd.to_datetime(reg["date"])
        tagged = tagged.merge(reg[["date", "regime"]], on="date", how="left")

        per_regime_rows: List[Dict[str, float]] = []
        for regime_name in sorted(tagged["regime"].dropna().unique()):
            mask = (tagged["regime"] == regime_name).values
            if mask.sum() < 10:
                continue
            mean_abs_r = np.abs(shap_values[mask]).mean(axis=0)
            row = {"regime": regime_name}
            row.update({f: float(v) for f, v in zip(feature_cols, mean_abs_r)})
            per_regime_rows.append(row)
        per_regime = pd.DataFrame(per_regime_rows)

    return ShapReport(global_importance=global_imp, per_regime=per_regime, method=method)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from pipeline.ingest import fetch_prices
    from quant.ml.features import build_feature_panel, FEATURE_COLS
    from quant.ml.baseline import BaselineForecaster

    prices = fetch_prices(start_date="2021-01-01", end_date="2024-01-01")
    panel = build_feature_panel(prices)
    cutoff = panel["date"].quantile(0.8)
    train = panel[panel["date"] < cutoff]
    test = panel[panel["date"] >= cutoff]
    base = BaselineForecaster(feature_cols=FEATURE_COLS).fit(train)
    report = explain_baseline(base, test)
    print(report.global_importance.head(10))
