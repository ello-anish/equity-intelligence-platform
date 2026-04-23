"""
pipeline/quality.py — Data quality checks and cleansing module.
"""

import pandas as pd
import logging
from datetime import date

logger = logging.getLogger(__name__)


def run_quality_checks(df):
    """
    Run data quality checks on a price DataFrame.

    Checks:
        1. No null values in 'close' or 'volume' columns
        2. No negative prices (open, high, low, close)
        3. No future dates (date <= today)
        4. Volume > 0

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: ticker, date, open, high, low, close, volume

    Returns
    -------
    tuple of (pd.DataFrame, dict)
        Cleaned DataFrame and quality report dictionary.
    """
    logger.info(f"Running quality checks on {len(df)} rows")
    
    quality_report = {
        "total_rows_input": len(df),
        "checks": {},
        "rows_dropped": 0,
        "total_rows_output": 0
    }

    rows_to_drop = set()

    # Check 1: No null values in close or volume
    null_mask = df["close"].isna() | df["volume"].isna()
    null_rows = df[null_mask]
    if len(null_rows) > 0:
        logger.warning(f"Check 1 FAILED: {len(null_rows)} rows with null close/volume")
        logger.warning(f"  Failing rows:\n{null_rows.to_string()}")
        rows_to_drop.update(null_rows.index.tolist())
    quality_report["checks"]["null_close_volume"] = {
        "passed": len(null_rows) == 0,
        "failing_rows": len(null_rows)
    }

    # Check 2: No negative prices
    price_cols = ["open", "high", "low", "close"]
    existing_price_cols = [c for c in price_cols if c in df.columns]
    neg_mask = (df[existing_price_cols] < 0).any(axis=1)
    neg_rows = df[neg_mask]
    if len(neg_rows) > 0:
        logger.warning(f"Check 2 FAILED: {len(neg_rows)} rows with negative prices")
        logger.warning(f"  Failing rows:\n{neg_rows.to_string()}")
        rows_to_drop.update(neg_rows.index.tolist())
    quality_report["checks"]["negative_prices"] = {
        "passed": len(neg_rows) == 0,
        "failing_rows": len(neg_rows)
    }

    # Check 3: No future dates
    today = date.today()
    df["date_parsed"] = pd.to_datetime(df["date"]).dt.date
    future_mask = df["date_parsed"] > today
    future_rows = df[future_mask]
    if len(future_rows) > 0:
        logger.warning(f"Check 3 FAILED: {len(future_rows)} rows with future dates")
        logger.warning(f"  Failing rows:\n{future_rows.to_string()}")
        rows_to_drop.update(future_rows.index.tolist())
    quality_report["checks"]["future_dates"] = {
        "passed": len(future_rows) == 0,
        "failing_rows": len(future_rows)
    }
    df = df.drop(columns=["date_parsed"])

    # Check 4: Volume > 0
    vol_mask = df["volume"] <= 0
    vol_rows = df[vol_mask]
    if len(vol_rows) > 0:
        logger.warning(f"Check 4 FAILED: {len(vol_rows)} rows with volume <= 0")
        logger.warning(f"  Failing rows:\n{vol_rows.head(10).to_string()}")
        rows_to_drop.update(vol_rows.index.tolist())
    quality_report["checks"]["volume_positive"] = {
        "passed": len(vol_rows) == 0,
        "failing_rows": len(vol_rows)
    }

    # Drop all failing rows
    cleaned_df = df.drop(index=list(rows_to_drop)).reset_index(drop=True)

    quality_report["rows_dropped"] = len(rows_to_drop)
    quality_report["total_rows_output"] = len(cleaned_df)

    logger.info(
        f"Quality checks complete: {len(rows_to_drop)} rows dropped, "
        f"{len(cleaned_df)} rows remaining"
    )

    return cleaned_df, quality_report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from pipeline.ingest import fetch_prices
    prices = fetch_prices(start_date="2023-01-01", end_date="2024-01-01")
    cleaned, report = run_quality_checks(prices)
    print(f"\nQuality Report: {report}")
    print(f"Cleaned shape: {cleaned.shape}")
