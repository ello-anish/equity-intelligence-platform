"""
quant/optimizer.py — Mean-variance portfolio optimization using scipy.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


def mean_variance_optimize(scores_df, returns_df, top_n=5):
    """
    Select top_n stocks by composite_score on each rebalance date and
    optimize weights to maximise the Sharpe ratio.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Factor scores with columns: ticker, date, composite_score, rank
    returns_df : pd.DataFrame
        Daily returns with columns: ticker, date, daily_return
    top_n : int
        Number of top stocks to include in portfolio.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: rebalance_date, ticker, weight
    """
    logger.info(f"Running mean-variance optimization with top_n={top_n}")

    # Get unique rebalance dates (first trading day of each month)
    scores_df = scores_df.copy()
    scores_df["date"] = pd.to_datetime(scores_df["date"])
    returns_df = returns_df.copy()
    returns_df["date"] = pd.to_datetime(returns_df["date"])

    # Identify first trading day of each month from scores
    scores_df["year_month"] = scores_df["date"].dt.to_period("M")
    rebalance_dates = scores_df.groupby("year_month")["date"].min().values

    all_weights = []

    for reb_date in rebalance_dates:
        reb_date = pd.Timestamp(reb_date)

        # Get scores on this date
        date_scores = scores_df[scores_df["date"] == reb_date].copy()
        if len(date_scores) == 0:
            continue

        # Select top N by rank (rank 1 = best)
        top_stocks = date_scores.nsmallest(top_n, "rank")["ticker"].tolist()

        if len(top_stocks) < 2:
            continue

        # Get historical returns for these stocks (lookback 63 days)
        lookback_start = reb_date - pd.Timedelta(days=120)
        hist_returns = returns_df[
            (returns_df["date"] >= lookback_start) &
            (returns_df["date"] <= reb_date) &
            (returns_df["ticker"].isin(top_stocks))
        ]

        # Pivot to get returns matrix
        returns_matrix = hist_returns.pivot_table(
            index="date", columns="ticker", values="daily_return"
        ).dropna()

        if len(returns_matrix) < 20 or len(returns_matrix.columns) < 2:
            # Not enough data, use equal weights
            n = len(top_stocks)
            for t in top_stocks:
                all_weights.append({
                    "rebalance_date": reb_date.date(),
                    "ticker": t,
                    "weight": 1.0 / n
                })
            continue

        # Available tickers (some may have been dropped due to NaN)
        available_tickers = returns_matrix.columns.tolist()
        n_assets = len(available_tickers)

        mean_returns = returns_matrix.mean().values
        cov_matrix = returns_matrix.cov().values

        # Optimize for max Sharpe ratio
        def neg_sharpe(weights):
            port_return = np.dot(weights, mean_returns) * 252
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
            if port_vol == 0:
                return 0
            return -(port_return / port_vol)

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        ]
        bounds = [(0.05, 0.40)] * n_assets
        x0 = np.array([1.0 / n_assets] * n_assets)

        try:
            result = minimize(
                neg_sharpe, x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000}
            )

            if result.success:
                opt_weights = result.x
            else:
                opt_weights = x0  # Fallback to equal weights
        except Exception as e:
            logger.warning(f"Optimization failed for {reb_date}: {e}")
            opt_weights = x0

        for i, ticker in enumerate(available_tickers):
            all_weights.append({
                "rebalance_date": reb_date.date(),
                "ticker": ticker,
                "weight": float(opt_weights[i])
            })

    weights_df = pd.DataFrame(all_weights)
    logger.info(f"Optimization complete: {len(weights_df)} weight entries across {len(weights_df['rebalance_date'].unique()) if len(weights_df) > 0 else 0} rebalance dates")
    return weights_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Optimizer module loaded successfully")
