"""
quant/ml/features.py — Feature engineering for the ML return-forecasting layer.

Produces a panel DataFrame keyed by (ticker, date) with ~20 engineered features
plus a forward-return target. Features are designed so the Transformer can
attend over both time-series (per-ticker) and cross-sectional (per-date)
structure.
"""

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Forward target horizons (trading days). Multi-horizon forecasting:
# 5 ≈ 1 week, 21 ≈ 1 month (primary), 63 ≈ 3 months.
FORWARD_HORIZON = 21
FORWARD_HORIZONS = {
    "target_fwd_ret_5d": 5,
    "target_fwd_ret_21d": 21,
    "target_fwd_ret_63d": 63,
}


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / window, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / window, adjust=False).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": hist,
    })


def _per_ticker_features(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values("date").copy()
    close = g["close"].astype(float)
    volume = g["volume"].astype(float)

    # Returns
    g["ret_1d"] = close.pct_change(1)
    g["ret_5d"] = close.pct_change(5)
    g["ret_21d"] = close.pct_change(21)

    # Momentum (log) over multiple horizons
    log_close = np.log(close.replace(0.0, np.nan))
    g["mom_1m"] = log_close - log_close.shift(21)
    g["mom_3m"] = log_close - log_close.shift(63)
    g["mom_6m"] = log_close - log_close.shift(126)
    g["mom_12m"] = log_close - log_close.shift(252)

    # Volatility (annualised)
    daily_ret = close.pct_change()
    g["vol_20d"] = daily_ret.rolling(20).std() * np.sqrt(252)
    g["vol_60d"] = daily_ret.rolling(60).std() * np.sqrt(252)

    # Volume features
    log_vol = np.log(volume.replace(0.0, np.nan))
    g["vol_z_20d"] = (log_vol - log_vol.rolling(20).mean()) / log_vol.rolling(20).std()

    # Price relative to moving averages
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    g["px_over_ma20"] = close / ma20 - 1.0
    g["px_over_ma50"] = close / ma50 - 1.0
    g["px_over_ma200"] = close / ma200 - 1.0

    # Technicals
    g["rsi_14"] = _rsi(close, 14)
    macd_df = _macd(close)
    g["macd"] = macd_df["macd"]
    g["macd_hist"] = macd_df["macd_hist"]

    # Skew / kurt of recent returns
    g["ret_skew_60d"] = daily_ret.rolling(60).skew()
    g["ret_kurt_60d"] = daily_ret.rolling(60).kurt()

    # Forward targets at multiple horizons (log-returns)
    for col, h in FORWARD_HORIZONS.items():
        g[col] = log_close.shift(-h) - log_close
    # Back-compat alias (primary horizon)
    g["target_fwd_ret"] = g["target_fwd_ret_21d"]

    # Vol-scaled (risk-adjusted) targets: forward return / trailing realised vol.
    # Rationale: a 5% forward return means something very different for a
    # low-vol defensive stock vs a high-vol mid-cap. Predicting a risk-adjusted
    # target is a standard systematic-strategy move and typically improves IC.
    trailing_vol_63d = daily_ret.rolling(63).std() * np.sqrt(252)
    eps = 1e-6
    g["target_fwd_ret_21d_vs"] = g["target_fwd_ret_21d"] / (trailing_vol_63d + eps)

    return g


def _cross_sectional_ranks(panel: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Add per-date cross-sectional rank (0-1) of each column."""
    out = panel.copy()
    for col in cols:
        if col in out.columns:
            out[f"cs_rank_{col}"] = (
                out.groupby("date")[col]
                .rank(pct=True, method="average")
            )
    return out


def _market_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Market-level features broadcast to every (ticker, date) row."""
    daily = panel.groupby("date").agg(
        mkt_ret=("ret_1d", "mean"),
        mkt_disp=("ret_1d", "std"),
    ).reset_index()
    daily["mkt_vol_20d"] = daily["mkt_ret"].rolling(20).std() * np.sqrt(252)
    daily["mkt_mom_21d"] = daily["mkt_ret"].rolling(21).sum()
    return daily


def build_feature_panel(
    prices_df: pd.DataFrame,
    include_macro: bool = True,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    prices_df : pd.DataFrame
        Columns: ticker, date, open, high, low, close, volume
    include_macro : if True, fetch and merge real macro features (Nifty,
        India VIX, USDINR, 10y, gold). Falls back to synthetic GBM if
        yfinance unreachable.

    Returns
    -------
    pd.DataFrame
        Panel with engineered features + 'target_fwd_ret' target.
        Rows with insufficient history or missing target are dropped.
    """
    logger.info("Building feature panel on %d rows", len(prices_df))

    df = prices_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Per-ticker features
    frames = []
    for ticker, grp in df.groupby("ticker"):
        frames.append(_per_ticker_features(grp))
    panel = pd.concat(frames, ignore_index=True)

    # Cross-sectional ranks
    rank_cols = ["mom_1m", "mom_3m", "mom_6m", "vol_20d", "vol_60d", "rsi_14"]
    panel = _cross_sectional_ranks(panel, rank_cols)

    # Market-level features
    mkt = _market_features(panel)
    panel = panel.merge(mkt, on="date", how="left")

    # Macro features (Nifty, India VIX, USDINR, 10y, gold)
    if include_macro:
        try:
            from quant.ml.macro import fetch_macro_series, build_macro_features, merge_macro_onto_panel
            start = panel["date"].min().strftime("%Y-%m-%d")
            end = (panel["date"].max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
            macro_raw = fetch_macro_series(start, end)
            macro_feats = build_macro_features(macro_raw)
            panel = merge_macro_onto_panel(panel, macro_feats)
            logger.info("Macro features merged: panel=%s", panel.shape)
        except Exception as e:
            logger.warning("Macro features unavailable (%s) — continuing without them", e)

    # Drop rows missing the primary (21d) target or with <1yr of history.
    # Longer-horizon targets (63d) may still be missing for the last 63 days;
    # keep those rows — downstream consumers filter on the target they use.
    panel = panel.dropna(subset=["target_fwd_ret_21d", "mom_12m"]).reset_index(drop=True)

    logger.info("Feature panel ready: %d rows, %d columns", len(panel), panel.shape[1])
    return panel


FEATURE_COLS: List[str] = [
    # Returns / momentum
    "ret_1d", "ret_5d", "ret_21d",
    "mom_1m", "mom_3m", "mom_6m", "mom_12m",
    # Vol / distribution
    "vol_20d", "vol_60d",
    "ret_skew_60d", "ret_kurt_60d",
    # Volume
    "vol_z_20d",
    # Trend
    "px_over_ma20", "px_over_ma50", "px_over_ma200",
    # Technicals
    "rsi_14", "macd", "macd_hist",
    # Cross-sectional ranks
    "cs_rank_mom_1m", "cs_rank_mom_3m", "cs_rank_mom_6m",
    "cs_rank_vol_20d", "cs_rank_vol_60d", "cs_rank_rsi_14",
    # Market state
    "mkt_ret", "mkt_disp", "mkt_vol_20d", "mkt_mom_21d",
]

# Real macro features (from yfinance: Nifty, India VIX, USDINR, US10Y, Gold).
# Added to the panel by build_feature_panel(include_macro=True).
MACRO_FEATURE_COLS: List[str] = [
    "nifty_ret_1d", "nifty_ret_21d", "nifty_over_ma50",
    "india_vix_level", "india_vix_z20",
    "usdinr_ret_1d", "usdinr_ret_21d",
    "us10y_level", "us10y_chg_21d",
    "gold_ret_21d",
]

# Default full-feature list when macro is merged in
ALL_FEATURE_COLS: List[str] = FEATURE_COLS + MACRO_FEATURE_COLS

TARGET_COL = "target_fwd_ret"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from pipeline.ingest import fetch_prices
    prices = fetch_prices(start_date="2021-01-01", end_date="2024-01-01")
    panel = build_feature_panel(prices)
    print(panel[["ticker", "date"] + FEATURE_COLS[:6] + [TARGET_COL]].tail(10))
    print(f"\nShape: {panel.shape}")
    print(f"Features: {len(FEATURE_COLS)}")
