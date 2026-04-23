"""
quant/factors.py — Wrapper module that calls transform.py functions to produce
scored factor DataFrames.
"""

import sys
import os
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.transform import compute_momentum, compute_low_volatility, compute_composite_score

logger = logging.getLogger(__name__)


def compute_factor_scores(prices_df):
    """
    Compute all factor scores from a price DataFrame.

    Parameters
    ----------
    prices_df : pd.DataFrame
        DataFrame with columns: ticker, date, open, high, low, close, volume

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ticker, date, momentum_score, low_vol_score,
        composite_score, rank
    """
    logger.info("Computing factor scores...")

    # Compute individual factors
    momentum_df = compute_momentum(prices_df, window=126)
    low_vol_df = compute_low_volatility(prices_df, window=63)

    # Compute composite scores
    scores_df = compute_composite_score(momentum_df, low_vol_df)

    logger.info(f"Factor scores computed: {len(scores_df)} rows")
    return scores_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from pipeline.ingest import fetch_prices
    prices = fetch_prices(start_date="2022-01-01", end_date="2024-01-01")
    scores = compute_factor_scores(prices)
    print(scores.tail(20))
