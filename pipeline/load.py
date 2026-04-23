"""
pipeline/load.py — Database loading module using psycopg2 upsert operations.
"""

import psycopg2
import psycopg2.extras
import pandas as pd
import logging

logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "equity_db",
    "user": "equity_user",
    "password": "equity_pass"
}


def get_connection(host=None):
    """Get a psycopg2 connection to the equity database."""
    config = DB_CONFIG.copy()
    if host:
        config["host"] = host
    return psycopg2.connect(**config)


def upsert_prices(df, conn):
    """
    Upsert price data into raw_prices table.
    INSERT ... ON CONFLICT (ticker, date) DO NOTHING.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: ticker, date, open, high, low, close, volume
    conn : psycopg2 connection
    """
    logger.info(f"Upserting {len(df)} rows into raw_prices")

    sql = """
        INSERT INTO raw_prices (ticker, date, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (ticker, date) DO NOTHING
    """

    cursor = conn.cursor()
    rows = [
        (
            row["ticker"], row["date"],
            float(row["open"]), float(row["high"]),
            float(row["low"]), float(row["close"]),
            int(row["volume"])
        )
        for _, row in df.iterrows()
    ]

    psycopg2.extras.execute_batch(cursor, sql, rows, page_size=500)
    conn.commit()
    cursor.close()
    logger.info(f"Upserted prices successfully")


def upsert_factor_scores(df, conn):
    """
    Upsert factor scores into factor_scores table.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: ticker, date, momentum_score, low_vol_score,
        composite_score, rank
    conn : psycopg2 connection
    """
    logger.info(f"Upserting {len(df)} rows into factor_scores")

    sql = """
        INSERT INTO factor_scores (ticker, date, momentum_score, low_vol_score, composite_score, rank)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (ticker, date) DO NOTHING
    """

    cursor = conn.cursor()
    rows = [
        (
            row["ticker"], row["date"],
            float(row["momentum_score"]), float(row["low_vol_score"]),
            float(row["composite_score"]), int(row["rank"])
        )
        for _, row in df.iterrows()
    ]

    psycopg2.extras.execute_batch(cursor, sql, rows, page_size=500)
    conn.commit()
    cursor.close()
    logger.info(f"Upserted factor scores successfully")


def upsert_portfolio_weights(df, conn):
    """
    Upsert portfolio weights into portfolio_weights table.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: rebalance_date, ticker, weight
    conn : psycopg2 connection
    """
    logger.info(f"Upserting {len(df)} rows into portfolio_weights")

    sql = """
        INSERT INTO portfolio_weights (rebalance_date, ticker, weight)
        VALUES (%s, %s, %s)
        ON CONFLICT (rebalance_date, ticker) DO NOTHING
    """

    cursor = conn.cursor()
    rows = [
        (row["rebalance_date"], row["ticker"], float(row["weight"]))
        for _, row in df.iterrows()
    ]

    psycopg2.extras.execute_batch(cursor, sql, rows, page_size=500)
    conn.commit()
    cursor.close()
    logger.info(f"Upserted portfolio weights successfully")


def upsert_backtest_results(df, conn):
    """
    Upsert backtest results into backtest_results table.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: date, portfolio_return, benchmark_return,
        cumulative_portfolio, cumulative_benchmark
    conn : psycopg2 connection
    """
    logger.info(f"Upserting {len(df)} rows into backtest_results")

    # Truncate and reinsert since backtest_results doesn't have a unique constraint on date
    cursor = conn.cursor()
    cursor.execute("DELETE FROM backtest_results")

    sql = """
        INSERT INTO backtest_results (date, portfolio_return, benchmark_return, 
                                       cumulative_portfolio, cumulative_benchmark)
        VALUES (%s, %s, %s, %s, %s)
    """

    rows = [
        (
            row["date"],
            float(row["portfolio_return"]),
            float(row["benchmark_return"]),
            float(row["cumulative_portfolio"]),
            float(row["cumulative_benchmark"])
        )
        for _, row in df.iterrows()
    ]

    psycopg2.extras.execute_batch(cursor, sql, rows, page_size=500)
    conn.commit()
    cursor.close()
    logger.info(f"Upserted backtest results successfully")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    conn = get_connection()
    print("Connection successful")
    conn.close()
