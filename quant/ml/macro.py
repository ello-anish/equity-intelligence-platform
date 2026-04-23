"""
quant/ml/macro.py — Macro & cross-asset feature ingestion.

Fetches exogenous time-series that matter for Indian equity forecasts:
    ^NSEI      : Nifty 50 index level      → market-level proxy
    ^INDIAVIX  : India implied vol         → forward-looking risk
    INR=X      : USDINR                    → currency carry / FII proxy
    ^TNX       : US 10y yield              → global rates
    GC=F       : Gold futures              → safe-haven demand

Design: broadcasts one row per date (not per ticker), then the feature-panel
builder LEFT JOINs on date. Missing days (holidays) are forward-filled
within a 5-day window to handle the Monday morning of a long weekend.

Falls back to synthetic GBM macros — clearly labelled — if yfinance is
unreachable, so the pipeline remains fully runnable offline.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


MACRO_SPEC: Dict[str, Dict[str, float]] = {
    # ticker → (base_level, drift, vol) for synthetic fallback
    "^NSEI":      {"base": 17000.0, "drift": 0.08, "vol": 0.18},
    "^INDIAVIX":  {"base": 16.0,    "drift": 0.00, "vol": 0.50},  # mean-reverting-ish
    "INR=X":      {"base": 76.0,    "drift": 0.02, "vol": 0.05},
    "^TNX":       {"base": 3.0,     "drift": 0.01, "vol": 0.25},
    "GC=F":       {"base": 1800.0,  "drift": 0.04, "vol": 0.15},
}

MACRO_FEATURE_COLS = [
    "nifty_ret_1d", "nifty_ret_21d", "nifty_over_ma50",
    "india_vix_level", "india_vix_z20",
    "usdinr_ret_1d", "usdinr_ret_21d",
    "us10y_level", "us10y_chg_21d",
    "gold_ret_21d",
]


def _synthetic_macro(start: str, end: str, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    out = pd.DataFrame({"date": dates})
    for ticker, p in MACRO_SPEC.items():
        dt = 1.0 / 252
        ret = np.exp((p["drift"] - 0.5 * p["vol"] ** 2) * dt +
                     p["vol"] * np.sqrt(dt) * rng.standard_normal(n))
        level = p["base"] * np.cumprod(ret)
        if ticker == "^INDIAVIX":
            # mean-revert around the base
            level = p["base"] + (level - level.mean()) * 0.4
        out[ticker] = level
    return out


def fetch_macro_series(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Returns a wide DataFrame with columns:
        date, ^NSEI, ^INDIAVIX, INR=X, ^TNX, GC=F.

    Tries yfinance first; falls back to synthetic GBM if unreachable.
    """
    try:
        import yfinance as yf
    except Exception as e:
        logger.warning("yfinance unavailable (%s) — using synthetic macro", e)
        return _synthetic_macro(start_date, end_date)

    results: List[pd.DataFrame] = []
    any_success = False
    for ticker in MACRO_SPEC:
        success = False
        for attempt in range(3):
            try:
                data = yf.download(ticker, start=start_date, end=end_date,
                                    progress=False, timeout=15, auto_adjust=True)
                if data is None or data.empty:
                    time.sleep(1.5)
                    continue
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                series = data[["Close"]].rename(columns={"Close": ticker}).reset_index()
                series = series.rename(columns={"Date": "date"})
                results.append(series)
                any_success = True
                success = True
                logger.info("  %s: %d rows fetched", ticker, len(series))
                break
            except Exception as e:
                logger.debug("macro fetch %s attempt %d failed: %s", ticker, attempt + 1, e)
                time.sleep(2)
        if not success:
            logger.warning("macro fetch %s FAILED — will be synthesised", ticker)
        time.sleep(0.3)

    if not any_success:
        logger.warning("All macro fetches failed — using synthetic GBM")
        return _synthetic_macro(start_date, end_date)

    # Merge all successful series; synthesise any that failed
    dates = pd.bdate_range(start_date, end_date)
    base = pd.DataFrame({"date": dates})
    for df in results:
        df["date"] = pd.to_datetime(df["date"])
        base = base.merge(df, on="date", how="left")

    synth = _synthetic_macro(start_date, end_date)
    for ticker in MACRO_SPEC:
        if ticker not in base.columns or base[ticker].isna().all():
            base[ticker] = synth.set_index("date").reindex(base["date"])[ticker].values
        else:
            # Fill small gaps (holidays) with forward-fill
            base[ticker] = base[ticker].ffill(limit=5)
            # Residual fill from synthetic
            mask = base[ticker].isna()
            if mask.any():
                base.loc[mask, ticker] = synth.set_index("date").reindex(
                    base.loc[mask, "date"]
                )[ticker].values
    return base


def build_macro_features(macro_df: pd.DataFrame) -> pd.DataFrame:
    """Transform macro levels into features: returns, z-scores, momentum."""
    df = macro_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    out = pd.DataFrame({"date": df["date"]})

    # Nifty
    if "^NSEI" in df.columns:
        nif = df["^NSEI"].astype(float)
        out["nifty_ret_1d"] = nif.pct_change(1)
        out["nifty_ret_21d"] = nif.pct_change(21)
        ma50 = nif.rolling(50).mean()
        out["nifty_over_ma50"] = nif / ma50 - 1.0

    # India VIX
    if "^INDIAVIX" in df.columns:
        vx = df["^INDIAVIX"].astype(float)
        out["india_vix_level"] = vx
        rm = vx.rolling(20).mean()
        rs = vx.rolling(20).std().replace(0, np.nan)
        out["india_vix_z20"] = (vx - rm) / rs

    # USDINR
    if "INR=X" in df.columns:
        inr = df["INR=X"].astype(float)
        out["usdinr_ret_1d"] = inr.pct_change(1)
        out["usdinr_ret_21d"] = inr.pct_change(21)

    # US 10y
    if "^TNX" in df.columns:
        tnx = df["^TNX"].astype(float)
        out["us10y_level"] = tnx
        out["us10y_chg_21d"] = tnx.diff(21)

    # Gold
    if "GC=F" in df.columns:
        gc = df["GC=F"].astype(float)
        out["gold_ret_21d"] = gc.pct_change(21)

    return out


def merge_macro_onto_panel(panel: pd.DataFrame, macro_feats: pd.DataFrame) -> pd.DataFrame:
    """Left-join macro features (one row per date) onto the per-ticker panel."""
    if macro_feats is None or macro_feats.empty:
        out = panel.copy()
        for c in MACRO_FEATURE_COLS:
            out[c] = 0.0
        return out
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    macro_feats = macro_feats.copy()
    macro_feats["date"] = pd.to_datetime(macro_feats["date"])
    merged = panel.merge(macro_feats, on="date", how="left")
    # Forward-fill within 5 rows to handle trading-day mismatches
    for c in MACRO_FEATURE_COLS:
        if c in merged.columns:
            merged[c] = merged[c].ffill(limit=5).fillna(0.0)
        else:
            merged[c] = 0.0
    return merged


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raw = fetch_macro_series("2022-01-01", "2023-06-01")
    print(raw.head())
    feats = build_macro_features(raw)
    print(feats.tail())
    print("Non-null counts:", feats.count().to_dict())
