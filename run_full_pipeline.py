"""
run_full_pipeline.py - Full pipeline: Ingest -> Quality -> Transform -> Optimize -> Backtest -> Load to PostgreSQL.
"""

import logging
import sys
import os
import io
import pandas as pd

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("EQUITY INTELLIGENCE PLATFORM -- Full Pipeline with PostgreSQL")
print("=" * 70)

# Step 1: Fetch Prices from Yahoo Finance
print("\n--- Step 1: Fetching Real Prices from Yahoo Finance ---")
from pipeline.ingest import fetch_prices
prices = fetch_prices(start_date="2021-01-01", end_date="2024-01-01")
print(f"  [OK] Rows fetched: {len(prices)}")
print(f"  [OK] Tickers: {prices['ticker'].unique().tolist()}")

# Step 2: Quality Checks
print("\n--- Step 2: Running Quality Checks ---")
from pipeline.quality import run_quality_checks
cleaned, report = run_quality_checks(prices)
print(f"  [OK] Input rows: {report['total_rows_input']}")
print(f"  [OK] Rows dropped: {report['rows_dropped']}")
print(f"  [OK] Output rows: {report['total_rows_output']}")

# Step 3: Factor Scores
print("\n--- Step 3: Computing Factor Scores ---")
from quant.factors import compute_factor_scores
cleaned["date"] = pd.to_datetime(cleaned["date"])
scores = compute_factor_scores(cleaned)
print(f"  [OK] Factor score rows: {len(scores)}")

latest_date = scores["date"].max()
latest = scores[scores["date"] == latest_date].sort_values("rank")
print("\n  Latest Factor Rankings:")
print(latest[["ticker", "momentum_score", "low_vol_score", "composite_score", "rank"]].to_string(index=False))

# Step 4: Backtest
print("\n--- Step 4: Running Backtest ---")
from quant.backtest import run_backtest
backtest_results, metrics = run_backtest(prices_df=cleaned)
print(f"  [OK] Backtest days: {len(backtest_results)}")
print("\n  Summary Metrics:")
for k, v in metrics.items():
    print(f"    {k}: {v:.4f}")

# Step 5: Compute portfolio weights for loading
print("\n--- Step 5: Computing Portfolio Weights ---")
from quant.optimizer import mean_variance_optimize

daily_returns = []
for ticker, group in cleaned.groupby("ticker"):
    group = group.sort_values("date").copy()
    group["daily_return"] = group["close"].pct_change()
    daily_returns.append(group[["ticker", "date", "daily_return"]])
daily_returns_df = pd.concat(daily_returns, ignore_index=True).dropna()

weights_df = mean_variance_optimize(scores, daily_returns_df, top_n=5)
print(f"  [OK] Portfolio weight entries: {len(weights_df)}")

# Step 6: Load to PostgreSQL
print("\n--- Step 6: Loading Data into PostgreSQL ---")
from pipeline.load import get_connection, upsert_prices, upsert_factor_scores, upsert_portfolio_weights, upsert_backtest_results

conn = get_connection()
print("  [OK] Connected to PostgreSQL")

# Load prices
upsert_prices(cleaned, conn)
print(f"  [OK] Loaded {len(cleaned)} price rows")

# Load factor scores
upsert_factor_scores(scores, conn)
print(f"  [OK] Loaded {len(scores)} factor score rows")

# Load portfolio weights
upsert_portfolio_weights(weights_df, conn)
print(f"  [OK] Loaded {len(weights_df)} portfolio weight rows")

# Load backtest results
upsert_backtest_results(backtest_results, conn)
print(f"  [OK] Loaded {len(backtest_results)} backtest result rows")

conn.close()

# Verify
print("\n--- Step 7: Verifying Database Contents ---")
conn = get_connection()
cur = conn.cursor()

tables = ["raw_prices", "factor_scores", "portfolio_weights", "backtest_results"]
for table in tables:
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    count = cur.fetchone()[0]
    print(f"  [OK] {table}: {count} rows")

cur.close()
conn.close()

print("\n" + "=" * 70)
print("FULL PIPELINE COMPLETE -- All data loaded to PostgreSQL!")
print("=" * 70)
print("\nNext steps:")
print("  * Dashboard: streamlit run dashboard/app.py")
print("  * Airflow:   docker compose up -d (for DAG scheduling)")
