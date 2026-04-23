"""
quant/ml/drift.py — Feature distribution drift detection.

Runs pairwise Kolmogorov-Smirnov tests comparing a reference window (the
training distribution) against a production window (recent OOS data) for
every feature. Flags features whose KS statistic exceeds a threshold.

Why this matters in production: the model was trained on one distribution;
if the live distribution shifts materially, predictions become unreliable —
independent of whether any individual forecast is "wrong" on a given day.

Intended integration: run on a weekly/monthly cadence against the feature
panel, alert if any feature crosses the threshold. A trigger for retraining
or for the allocator to reduce exposure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    per_feature: pd.DataFrame         # feature, ks_stat, p_value, drift
    n_drifted: int
    n_features: int
    threshold_p: float
    threshold_stat: float
    reference_period: tuple
    production_period: tuple


def detect_feature_drift(
    panel: pd.DataFrame,
    feature_cols: List[str],
    reference_start: str,
    reference_end: str,
    production_start: str,
    production_end: str,
    p_threshold: float = 0.01,
    stat_threshold: float = 0.2,
) -> DriftReport:
    """
    KS test per feature between two date windows. Drift flagged if either:
        - p-value < p_threshold   (statistically significant distribution shift)
        - KS statistic > stat_threshold   (effect-size gate, independent of n)

    Parameters
    ----------
    panel : feature DataFrame with columns: date, <feature_cols...>
    reference_start/_end : training-distribution window
    production_start/_end : live/OOS window to check against reference
    """
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    ref = df[(df["date"] >= pd.Timestamp(reference_start)) & (df["date"] <= pd.Timestamp(reference_end))]
    prod = df[(df["date"] >= pd.Timestamp(production_start)) & (df["date"] <= pd.Timestamp(production_end))]

    if len(ref) < 50 or len(prod) < 20:
        logger.warning(
            "Drift check degenerate: ref=%d prod=%d rows — need more data", len(ref), len(prod)
        )

    rows = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        a = ref[col].dropna().to_numpy(dtype=float)
        b = prod[col].dropna().to_numpy(dtype=float)
        if len(a) < 20 or len(b) < 10:
            rows.append({
                "feature": col, "ks_stat": np.nan, "p_value": np.nan, "drift": False,
                "n_ref": len(a), "n_prod": len(b), "reason": "insufficient_data",
            })
            continue
        try:
            ks_stat, p = stats.ks_2samp(a, b, alternative="two-sided", method="asymp")
        except Exception as e:
            rows.append({
                "feature": col, "ks_stat": np.nan, "p_value": np.nan, "drift": False,
                "n_ref": len(a), "n_prod": len(b), "reason": f"error: {e}",
            })
            continue
        drifted = bool(p < p_threshold and ks_stat > stat_threshold)
        rows.append({
            "feature": col,
            "ks_stat": float(ks_stat),
            "p_value": float(p),
            "drift": drifted,
            "n_ref": len(a),
            "n_prod": len(b),
            "reason": "drift" if drifted else "ok",
        })

    per_feat = pd.DataFrame(rows).sort_values("ks_stat", ascending=False).reset_index(drop=True)
    n_drifted = int(per_feat["drift"].sum())

    return DriftReport(
        per_feature=per_feat,
        n_drifted=n_drifted,
        n_features=len(per_feat),
        threshold_p=p_threshold,
        threshold_stat=stat_threshold,
        reference_period=(reference_start, reference_end),
        production_period=(production_start, production_end),
    )


def drift_summary_line(report: DriftReport) -> str:
    return (
        f"drift: {report.n_drifted}/{report.n_features} features flagged  "
        f"(ref={report.reference_period[0]}→{report.reference_period[1]}, "
        f"prod={report.production_period[0]}→{report.production_period[1]})"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=400, freq="B")
    df = pd.DataFrame({"date": dates})
    df["stable"] = rng.standard_normal(len(df))
    df["drifting"] = np.concatenate([
        rng.standard_normal(200),
        rng.standard_normal(200) + 2.0,    # huge shift in second half
    ])
    report = detect_feature_drift(
        df, ["stable", "drifting"],
        reference_start="2020-01-01", reference_end="2020-10-01",
        production_start="2020-10-01", production_end="2021-08-01",
    )
    print(drift_summary_line(report))
    print(report.per_feature)
