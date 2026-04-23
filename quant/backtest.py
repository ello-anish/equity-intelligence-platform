"""
quant/backtest.py — Backtesting engine for the equity intelligence platform.
"""

import sys
import os
import numpy as np
import pandas as pd
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.ingest import fetch_prices, DEFAULT_TICKERS
from pipeline.quality import run_quality_checks
from pipeline.transform import compute_momentum, compute_low_volatility, compute_composite_score
from quant.optimizer import mean_variance_optimize

logger = logging.getLogger(__name__)


def run_backtest(prices_df=None, weights_df=None,
                 start_date="2021-01-01", end_date="2024-01-01"):
    """
    Run a full backtest with monthly rebalancing.

    Parameters
    ----------
    prices_df : pd.DataFrame, optional
        Price DataFrame. If None, will fetch from yfinance.
    weights_df : pd.DataFrame, optional
        Portfolio weights. If None, will compute from factor model + optimizer.
    start_date : str
        Backtest start date.
    end_date : str
        Backtest end date.

    Returns
    -------
    tuple of (pd.DataFrame, dict)
        - backtest_results DataFrame with columns: date, portfolio_return,
          benchmark_return, cumulative_portfolio, cumulative_benchmark
        - summary_metrics dict with: total_return, annualised_return,
          sharpe_ratio, max_drawdown, alpha_vs_benchmark, win_rate
    """
    logger.info(f"Running backtest from {start_date} to {end_date}")

    # Step 1: Get prices if not provided
    if prices_df is None:
        prices_df = fetch_prices(start_date=start_date, end_date=end_date)

    prices_df = prices_df.copy()
    prices_df["date"] = pd.to_datetime(prices_df["date"])
    prices_df = prices_df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Step 2: Compute daily returns
    daily_returns_list = []
    for ticker, group in prices_df.groupby("ticker"):
        group = group.sort_values("date").copy()
        group["daily_return"] = group["close"].pct_change()
        daily_returns_list.append(group[["ticker", "date", "daily_return"]])

    daily_returns_df = pd.concat(daily_returns_list, ignore_index=True).dropna()

    # Step 3: Compute factor scores and optimize if weights not provided
    if weights_df is None:
        momentum_df = compute_momentum(prices_df, window=126)
        low_vol_df = compute_low_volatility(prices_df, window=63)
        scores_df = compute_composite_score(momentum_df, low_vol_df)
        weights_df = mean_variance_optimize(scores_df, daily_returns_df, top_n=5)

    if weights_df.empty:
        logger.error("No portfolio weights generated")
        return pd.DataFrame(), {}

    weights_df = weights_df.copy()
    weights_df["rebalance_date"] = pd.to_datetime(weights_df["rebalance_date"])

    # Step 4: Build daily portfolio and benchmark returns
    # Pivot returns to wide format
    returns_wide = daily_returns_df.pivot_table(
        index="date", columns="ticker", values="daily_return"
    ).fillna(0)

    # Get all unique dates
    all_dates = sorted(returns_wide.index.tolist())
    rebalance_dates = sorted(weights_df["rebalance_date"].unique())

    portfolio_daily_returns = []
    benchmark_daily_returns = []

    # Benchmark: equal-weight across all 10 tickers
    all_tickers = [t for t in DEFAULT_TICKERS if t in returns_wide.columns]
    n_benchmark = len(all_tickers)
    benchmark_weight = 1.0 / n_benchmark if n_benchmark > 0 else 0

    current_weights = {}

    for date in all_dates:
        # Check if we need to rebalance
        if len(rebalance_dates) > 0:
            applicable_rebalances = [d for d in rebalance_dates if d <= date]
            if applicable_rebalances:
                latest_rebalance = max(applicable_rebalances)
                reb_weights = weights_df[weights_df["rebalance_date"] == latest_rebalance]
                current_weights = dict(zip(reb_weights["ticker"], reb_weights["weight"]))

        # Portfolio return
        if current_weights:
            port_ret = sum(
                w * returns_wide.loc[date].get(t, 0)
                for t, w in current_weights.items()
            )
        else:
            port_ret = 0

        # Benchmark return
        bench_ret = sum(
            benchmark_weight * returns_wide.loc[date].get(t, 0)
            for t in all_tickers
        )

        portfolio_daily_returns.append({"date": date, "portfolio_return": port_ret})
        benchmark_daily_returns.append({"date": date, "benchmark_return": bench_ret})

    port_df = pd.DataFrame(portfolio_daily_returns)
    bench_df = pd.DataFrame(benchmark_daily_returns)

    results_df = pd.merge(port_df, bench_df, on="date")
    results_df = results_df.sort_values("date").reset_index(drop=True)

    # Cumulative returns
    results_df["cumulative_portfolio"] = (1 + results_df["portfolio_return"]).cumprod()
    results_df["cumulative_benchmark"] = (1 + results_df["benchmark_return"]).cumprod()

    # Convert dates to date objects for storage
    results_df["date"] = results_df["date"].dt.date

    # Step 5: Compute summary metrics
    total_return = results_df["cumulative_portfolio"].iloc[-1] - 1
    n_years = len(results_df) / 252
    annualised_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    port_returns = results_df["portfolio_return"]
    sharpe_ratio = (port_returns.mean() / port_returns.std() * np.sqrt(252)) if port_returns.std() > 0 else 0

    # Max drawdown
    cum_max = results_df["cumulative_portfolio"].cummax()
    drawdowns = (results_df["cumulative_portfolio"] - cum_max) / cum_max
    max_drawdown = drawdowns.min()

    # Alpha vs benchmark
    benchmark_total_return = results_df["cumulative_benchmark"].iloc[-1] - 1
    alpha_vs_benchmark = total_return - benchmark_total_return

    # Win rate
    win_rate = (port_returns > 0).sum() / len(port_returns) if len(port_returns) > 0 else 0

    summary_metrics = {
        "total_return": float(total_return),
        "annualised_return": float(annualised_return),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown),
        "alpha_vs_benchmark": float(alpha_vs_benchmark),
        "win_rate": float(win_rate)
    }

    logger.info(f"Backtest complete. Summary: {summary_metrics}")
    return results_df, summary_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results, metrics = run_backtest()
    print(f"\nResults shape: {results.shape}")
    print(f"\nSummary Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
