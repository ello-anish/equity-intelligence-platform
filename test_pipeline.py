"""
test_pipeline.py — End-to-end pipeline test.
"""

import logging
import sys
import os

logging.basicConfig(level=logging.INFO)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("EQUITY INTELLIGENCE PLATFORM — End-to-End Pipeline Test")
print("=" * 60)

# Step 1: Fetch Prices
print("\n--- Step 1: Fetch Prices ---")
from pipeline.ingest import fetch_prices
prices = fetch_prices(start_date="2021-01-01", end_date="2024-01-01")
print(f"  Rows fetched: {len(prices)}")
print(f"  Tickers: {prices['ticker'].unique().tolist()}")

# Step 2: Quality Checks
print("\n--- Step 2: Quality Checks ---")
from pipeline.quality import run_quality_checks
cleaned, report = run_quality_checks(prices)
print(f"  Input rows: {report['total_rows_input']}")
print(f"  Rows dropped: {report['rows_dropped']}")
print(f"  Output rows: {report['total_rows_output']}")

# Step 3: Factor Scores
print("\n--- Step 3: Factor Scores ---")
from quant.factors import compute_factor_scores
import pandas as pd
cleaned["date"] = pd.to_datetime(cleaned["date"])
scores = compute_factor_scores(cleaned)
print(f"  Score rows: {len(scores)}")

# Latest scores
latest_date = scores["date"].max()
latest = scores[scores["date"] == latest_date].sort_values("rank")
print("\n  Latest Factor Scores:")
print(latest[["ticker", "momentum_score", "low_vol_score", "composite_score", "rank"]].to_string(index=False))

# Step 4: Backtest
print("\n--- Step 4: Backtest ---")
from quant.backtest import run_backtest
results, metrics = run_backtest(prices_df=cleaned)
print(f"  Backtest days: {len(results)}")
print("\n  Summary Metrics:")
for k, v in metrics.items():
    print(f"    {k}: {v:.4f}")

print("\n" + "=" * 60)
print("PIPELINE TEST COMPLETE — All steps passed!")
print("=" * 60)
