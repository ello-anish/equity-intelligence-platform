"""
pipeline/transform.py — Factor computation module for momentum and low-volatility factors.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_momentum(df, window=126):
    """
    Compute 6-month price momentum for each ticker.
    
    momentum = (close_today / close_126_days_ago) - 1

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: ticker, date, close
    window : int
        Lookback window in trading days (default 126 ≈ 6 months).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ticker, date, momentum_score
    """
    logger.info(f"Computing momentum with window={window}")

    results = []

    for ticker, group in df.groupby("ticker"):
        group = group.sort_values("date").copy()
        group["momentum_score"] = group["close"] / group["close"].shift(window) - 1
        results.append(group[["ticker", "date", "momentum_score"]])

    result = pd.concat(results, ignore_index=True).dropna(subset=["momentum_score"])
    logger.info(f"Momentum computed: {len(result)} rows")
    return result


def compute_low_volatility(df, window=63):
    """
    Compute 3-month rolling annualised volatility.
    
    low_vol_score = -1 * rolling_std(daily_returns, 63) * sqrt(252)
    (Lower volatility = higher score)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: ticker, date, close
    window : int
        Lookback window in trading days (default 63 ≈ 3 months).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ticker, date, low_vol_score
    """
    logger.info(f"Computing low volatility with window={window}")

    results = []

    for ticker, group in df.groupby("ticker"):
        group = group.sort_values("date").copy()
        daily_returns = group["close"].pct_change()
        rolling_vol = daily_returns.rolling(window=window).std() * np.sqrt(252)
        group["low_vol_score"] = -rolling_vol  # Negative: lower vol → higher score
        results.append(group[["ticker", "date", "low_vol_score"]])

    result = pd.concat(results, ignore_index=True).dropna(subset=["low_vol_score"])
    logger.info(f"Low-vol computed: {len(result)} rows")
    return result


def compute_composite_score(momentum_df, low_vol_df):
    """
    Compute composite factor score by cross-sectionally ranking each factor.
    
    Composite = 0.5 * momentum_rank + 0.5 * low_vol_rank
    (Rank 1 = best)

    Parameters
    ----------
    momentum_df : pd.DataFrame
        DataFrame with columns: ticker, date, momentum_score
    low_vol_df : pd.DataFrame
        DataFrame with columns: ticker, date, low_vol_score

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ticker, date, momentum_score, low_vol_score,
        composite_score, rank
    """
    logger.info("Computing composite scores")

    merged = pd.merge(
        momentum_df, low_vol_df,
        on=["ticker", "date"],
        how="inner"
    )

    def rank_group(group):
        # Rank descending: highest score = rank 1
        group["momentum_rank"] = group["momentum_score"].rank(ascending=False, method="min")
        group["low_vol_rank"] = group["low_vol_score"].rank(ascending=False, method="min")
        group["composite_score"] = 0.5 * group["momentum_rank"] + 0.5 * group["low_vol_rank"]
        # Overall rank: lowest composite_score = best = rank 1
        group["rank"] = group["composite_score"].rank(ascending=True, method="min").astype(int)
        return group

    result = merged.groupby("date", group_keys=False).apply(rank_group)
    result = result[["ticker", "date", "momentum_score", "low_vol_score", "composite_score", "rank"]]

    logger.info(f"Composite scores computed: {len(result)} rows")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Test with sample data
    from pipeline.ingest import fetch_prices
    prices = fetch_prices(start_date="2022-01-01", end_date="2024-01-01")
    mom = compute_momentum(prices)
    lvol = compute_low_volatility(prices)
    scores = compute_composite_score(mom, lvol)
    print(scores.tail(20))
