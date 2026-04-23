"""
quant/ml/allocator.py — Uncertainty-aware portfolio allocator.

Consumes a DataFrame of (ticker, date, prediction, lower, upper) from the
conformal-wrapped forecaster and produces per-date portfolio weights that
*discount positions by the width of their conformal prediction interval*.

The intuition: when the model is uncertain (wide conformal band), shrink the
position. In crisis regimes where intervals widen empirically, the allocator
automatically scales back — the intended "risk-aware degradation detection"
behaviour.

Two allocator policies:
  1. "vanilla" long-short top-k (no uncertainty) — baseline for comparison.
  2. "width-scaled" — each position is weighted by prediction / (interval
     width + ε), then L1-normalised so the gross book sums to 1.0.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Static sector map for the 10 NSE tickers. Hard-coded because the universe
# is fixed; if you expand the universe, replace this with a Bloomberg /
# Refinitiv sector lookup.
DEFAULT_SECTOR_MAP: Dict[str, str] = {
    "RELIANCE.NS":   "Energy",
    "TCS.NS":        "IT",
    "INFY.NS":       "IT",
    "WIPRO.NS":      "IT",
    "HDFCBANK.NS":   "Financials",
    "ICICIBANK.NS":  "Financials",
    "AXISBANK.NS":   "Financials",
    "SBIN.NS":       "Financials",
    "BAJFINANCE.NS": "Financials",
    "LT.NS":         "Industrials",
}


def sector_neutral_weights(
    day: pd.DataFrame,
    top_k: int = 3,
    sector_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Equal-weight long the top-k predictions and short the bottom-k, BUT
    rebalance so each sector has net-zero exposure. Prevents the allocator
    from making a disguised sector bet (e.g. long banks, short IT) instead
    of a cross-sectional stock-selection bet.
    """
    smap = sector_map or DEFAULT_SECTOR_MAP
    df = day.copy()
    df["sector"] = df["ticker"].map(smap).fillna("Unknown")

    # Start from vanilla long-short
    sorted_day = df.sort_values("prediction", ascending=False)
    long = sorted_day.head(top_k).assign(side=1)
    short = sorted_day.tail(top_k).assign(side=-1)
    book = pd.concat([long, short], ignore_index=True)

    # Per-sector, within-sector neutralisation:
    # - If a sector has both longs and shorts, assign weights so the sector's
    #   net exposure is exactly zero.
    # - If a sector has only longs or only shorts (e.g., only one name
    #   picked), DROP those positions — we can't neutralise with one side.
    rows = []
    for sector, grp in book.groupby("sector"):
        sides = grp["side"].values
        n_long = int((sides > 0).sum())
        n_short = int((sides < 0).sum())
        if n_long == 0 or n_short == 0:
            # Sector is one-sided; drop to preserve neutrality
            continue
        w = np.where(sides > 0, 1.0 / n_long, -1.0 / n_short)
        sub = grp.copy()
        sub["weight"] = w
        rows.append(sub)

    if not rows:
        # Degenerate book — fall back to vanilla to avoid returning nothing
        return vanilla_ls_weights(day, top_k=top_k)

    stack = pd.concat(rows, ignore_index=True)
    # Re-scale so gross = 2 (matches vanilla long-short convention)
    gross = float(stack["weight"].abs().sum())
    if gross > 0:
        stack["weight"] = stack["weight"] * (2.0 / gross)
    return stack[["ticker", "date", "weight", "prediction"]]


def compute_turnover(weights_frames: List[pd.DataFrame]) -> pd.DataFrame:
    """
    For each rebalance, compute L1 distance between new and previous weight
    vector — this is portfolio turnover. Ticker universes can shift between
    rebalances, so we union the tickers and treat missing as 0.
    """
    rows = []
    prev = None
    for w in weights_frames:
        cur = dict(zip(w["ticker"], w["weight"]))
        d = pd.Timestamp(w["date"].iloc[0])
        if prev is None:
            rows.append({"date": d, "turnover": float(sum(abs(v) for v in cur.values()))})
        else:
            all_tickers = set(prev) | set(cur)
            turn = sum(abs(cur.get(t, 0.0) - prev.get(t, 0.0)) for t in all_tickers)
            rows.append({"date": d, "turnover": float(turn)})
        prev = cur
    return pd.DataFrame(rows)


def vanilla_ls_weights(day: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """Equal-weight long top-k, short bottom-k."""
    sorted_day = day.sort_values("prediction", ascending=False)
    long = sorted_day.head(top_k).assign(weight=1.0 / top_k)
    short = sorted_day.tail(top_k).assign(weight=-1.0 / top_k)
    return pd.concat([long, short], ignore_index=True)[["ticker", "date", "weight", "prediction"]]


def width_scaled_weights(day: pd.DataFrame, top_k: int = 3, eps: float = 1e-4) -> pd.DataFrame:
    """
    Start from vanilla top-k long-short, then scale each position by
    1 / (conformal_width + eps) so wider-interval positions shrink.
    Re-normalise to gross = 1.0 (sum of |weights|).
    """
    width = (day["upper"] - day["lower"]).clip(lower=eps)
    confidence = 1.0 / (width + eps)

    sorted_day = day.assign(_conf=confidence).sort_values("prediction", ascending=False)
    longs = sorted_day.head(top_k).copy()
    shorts = sorted_day.tail(top_k).copy()

    longs["raw"] = longs["_conf"].values
    shorts["raw"] = -shorts["_conf"].values

    stack = pd.concat([longs, shorts], ignore_index=True)
    gross = np.abs(stack["raw"]).sum()
    if gross == 0:
        stack["weight"] = 0.0
    else:
        stack["weight"] = stack["raw"] / gross
    return stack[["ticker", "date", "weight", "prediction"]]


def run_allocator(
    preds_df: pd.DataFrame,
    policy: str = "width_scaled",
    top_k: int = 3,
    target_col: str = "target_fwd_ret_21d",
    horizon_days: int = 21,
    regimes_df: pd.DataFrame | None = None,
    tc_bps: float = 0.0,
    sector_map: Optional[Dict[str, str]] = None,
) -> dict:
    """
    Apply allocator at monthly rebalance dates, compute realised PnL using the
    `target_col` forward return, return summary stats.

    Parameters
    ----------
    policy : "vanilla" | "width_scaled" | "sector_neutral"
    tc_bps : one-way transaction cost in basis points applied to the L1
        turnover at each rebalance (typical Indian equity round-trip is
        10-30bps; 20 is a reasonable default for a real backtest).
    sector_map : {ticker: sector} override. If None, uses DEFAULT_SECTOR_MAP.

    Returns
    -------
    dict with
      - weights_df : concatenated per-rebal weights
      - pnl_df     : per-rebal realised portfolio return (gross) and net
      - turnover_df : per-rebal L1 turnover
      - sharpe (net), sharpe_gross, total_return, max_drawdown
      - gross_leverage, avg_turnover
      - per_regime (optional) sharpe breakdown
    """
    if preds_df.empty:
        return {"error": "no predictions"}

    df = preds_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "target_fwd_ret" not in df.columns and "target_fwd_ret_21d" in df.columns:
        df["target_fwd_ret"] = df["target_fwd_ret_21d"]
    tgt = target_col if target_col in df.columns else "target_fwd_ret"

    unique_dates = sorted(df["date"].unique())
    rebal = []
    last = None
    for d in unique_dates:
        if last is None or (d - last).days >= horizon_days:
            rebal.append(d)
            last = d

    if policy == "sector_neutral":
        def policy_fn(day, top_k):
            return sector_neutral_weights(day, top_k=top_k, sector_map=sector_map)
    elif policy == "width_scaled":
        policy_fn = width_scaled_weights
    else:
        policy_fn = vanilla_ls_weights

    weights_frames: List[pd.DataFrame] = []
    pnl_rows = []
    prev_weights: Dict[str, float] = {}

    for d in rebal:
        day = df[df["date"] == d]
        if len(day) < 2 * top_k:
            continue
        w = policy_fn(day, top_k=top_k)
        realised = day.merge(w[["ticker", "weight"]], on="ticker")
        gross_ret = float((realised["weight"] * realised[tgt]).sum())

        cur_w = dict(zip(w["ticker"], w["weight"]))
        all_tickers = set(cur_w) | set(prev_weights)
        turnover = sum(abs(cur_w.get(t, 0.0) - prev_weights.get(t, 0.0)) for t in all_tickers)
        tc = turnover * (tc_bps / 10000.0)
        net_ret = gross_ret - tc

        pnl_rows.append({
            "date": d,
            "gross_ret": gross_ret,
            "turnover": float(turnover),
            "tc": float(tc),
            "ret": net_ret,
            "gross": float(w["weight"].abs().sum()),
        })
        weights_frames.append(w)
        prev_weights = cur_w

    if not pnl_rows:
        return {"error": "empty pnl"}

    pnl = pd.DataFrame(pnl_rows).sort_values("date").reset_index(drop=True)
    pnl["cum"] = (1 + pnl["ret"]).cumprod()
    pnl["cum_gross"] = (1 + pnl["gross_ret"]).cumprod()

    periods_per_year = 252 / horizon_days
    std = pnl["ret"].std()
    sharpe_net = float(pnl["ret"].mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0
    std_g = pnl["gross_ret"].std()
    sharpe_g = float(pnl["gross_ret"].mean() / std_g * np.sqrt(periods_per_year)) if std_g > 0 else 0.0
    total = float(pnl["cum"].iloc[-1] - 1.0)
    cummax = pnl["cum"].cummax()
    dd = float(((pnl["cum"] - cummax) / cummax).min())

    out = {
        "policy": policy,
        "tc_bps": float(tc_bps),
        "weights_df": pd.concat(weights_frames, ignore_index=True),
        "pnl_df": pnl,
        "sharpe": sharpe_net,
        "sharpe_gross": sharpe_g,
        "total_return": total,
        "max_drawdown": dd,
        "gross_leverage_mean": float(pnl["gross"].mean()),
        "avg_turnover": float(pnl["turnover"].mean()),
        "total_tc_drag": float(pnl["tc"].sum()),
        "n_periods": int(len(pnl)),
        "horizon_days": horizon_days,
        "top_k": top_k,
    }

    if regimes_df is not None and not regimes_df.empty:
        rg = regimes_df.copy()
        rg["date"] = pd.to_datetime(rg["date"])
        pnl_r = pnl.merge(rg[["date", "regime"]], on="date", how="left")
        per_regime = []
        for regime, grp in pnl_r.groupby("regime"):
            if len(grp) < 3:
                continue
            s = grp["ret"].std()
            per_regime.append({
                "regime": str(regime),
                "n": int(len(grp)),
                "mean": float(grp["ret"].mean()),
                "std": float(s),
                "sharpe": float(grp["ret"].mean() / s * np.sqrt(periods_per_year)) if s > 0 else 0.0,
            })
        out["per_regime"] = pd.DataFrame(per_regime)

    return out
