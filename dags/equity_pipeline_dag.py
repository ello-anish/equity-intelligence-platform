"""
dags/equity_pipeline_dag.py — Airflow DAG for the Equity Intelligence Pipeline.

Schedule: @daily
Tasks: fetch → quality check → compute factors → load to postgres → backtest
"""

import json
import sys
import os
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Add project root to path
sys.path.insert(0, "/opt/airflow")

logger = logging.getLogger(__name__)

# Default args
default_args = {
    "owner": "equity_intelligence",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Tickers
TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "WIPRO.NS", "BAJFINANCE.NS", "AXISBANK.NS", "LT.NS", "SBIN.NS"
]

DB_HOST = "postgres"  # Docker service name


def task_fetch_raw_data(**context):
    """Fetch raw price data from yfinance."""
    from pipeline.ingest import fetch_prices

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")

    df = fetch_prices(tickers=TICKERS, start_date=start_date, end_date=end_date)

    # Serialise for XCom
    df["date"] = df["date"].astype(str)
    context["ti"].xcom_push(key="raw_prices", value=df.to_json(orient="records"))
    logger.info(f"Fetched {len(df)} rows")


def task_run_quality_checks(**context):
    """Run quality checks on raw price data."""
    import pandas as pd
    from pipeline.quality import run_quality_checks

    raw_json = context["ti"].xcom_pull(task_ids="fetch_raw_data", key="raw_prices")
    df = pd.read_json(raw_json, orient="records")

    cleaned_df, quality_report = run_quality_checks(df)

    cleaned_df["date"] = cleaned_df["date"].astype(str)
    context["ti"].xcom_push(key="cleaned_prices", value=cleaned_df.to_json(orient="records"))
    context["ti"].xcom_push(key="quality_report", value=json.dumps(quality_report))
    logger.info(f"Quality checks complete. Report: {quality_report}")


def task_compute_factors(**context):
    """Compute factor scores from cleaned price data."""
    import pandas as pd
    from quant.factors import compute_factor_scores

    cleaned_json = context["ti"].xcom_pull(task_ids="run_quality_checks", key="cleaned_prices")
    df = pd.read_json(cleaned_json, orient="records")

    scores_df = compute_factor_scores(df)

    scores_df["date"] = scores_df["date"].astype(str)
    context["ti"].xcom_push(key="factor_scores", value=scores_df.to_json(orient="records"))
    logger.info(f"Computed factor scores: {len(scores_df)} rows")


def task_load_to_postgres(**context):
    """Load all data into PostgreSQL."""
    import pandas as pd
    import psycopg2
    from pipeline.load import upsert_prices, upsert_factor_scores

    conn = psycopg2.connect(
        host=DB_HOST, port=5432,
        dbname="equity_db", user="equity_user", password="equity_pass"
    )

    # Load prices
    cleaned_json = context["ti"].xcom_pull(task_ids="run_quality_checks", key="cleaned_prices")
    prices_df = pd.read_json(cleaned_json, orient="records")
    upsert_prices(prices_df, conn)

    # Load factor scores
    scores_json = context["ti"].xcom_pull(task_ids="compute_factors", key="factor_scores")
    scores_df = pd.read_json(scores_json, orient="records")
    upsert_factor_scores(scores_df, conn)

    conn.close()
    logger.info("Data loaded to PostgreSQL")


def task_run_backtest(**context):
    """Run backtest and store results."""
    import pandas as pd
    import psycopg2
    from quant.backtest import run_backtest
    from pipeline.load import upsert_portfolio_weights, upsert_backtest_results

    # Run full backtest
    results_df, summary_metrics = run_backtest()

    if results_df.empty:
        logger.warning("Backtest produced no results")
        return

    # Connect and store results
    conn = psycopg2.connect(
        host=DB_HOST, port=5432,
        dbname="equity_db", user="equity_user", password="equity_pass"
    )

    upsert_backtest_results(results_df, conn)

    conn.close()

    context["ti"].xcom_push(key="backtest_metrics", value=json.dumps(summary_metrics))
    logger.info(f"Backtest complete. Metrics: {summary_metrics}")


# DAG definition
with DAG(
    dag_id="equity_intelligence_pipeline",
    default_args=default_args,
    description="End-to-end equity intelligence pipeline: ingest, quality, factors, load, backtest",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["equity", "quant", "nse"],
) as dag:

    fetch_raw_data = PythonOperator(
        task_id="fetch_raw_data",
        python_callable=task_fetch_raw_data,
    )

    run_quality_checks = PythonOperator(
        task_id="run_quality_checks",
        python_callable=task_run_quality_checks,
    )

    compute_factors = PythonOperator(
        task_id="compute_factors",
        python_callable=task_compute_factors,
    )

    load_to_postgres = PythonOperator(
        task_id="load_to_postgres",
        python_callable=task_load_to_postgres,
    )

    run_backtest = PythonOperator(
        task_id="run_backtest",
        python_callable=task_run_backtest,
    )

    # Task dependencies
    fetch_raw_data >> run_quality_checks >> compute_factors >> load_to_postgres >> run_backtest
