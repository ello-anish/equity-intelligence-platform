"""
pipeline/ingest.py — Data ingestion module for NSE equities.
Uses yfinance when available, falls back to synthetic data generation.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)

# Default NSE tickers
DEFAULT_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "WIPRO.NS", "BAJFINANCE.NS", "AXISBANK.NS", "LT.NS", "SBIN.NS"
]

# Approximate real base prices and volatilities for realistic synthetic data
TICKER_PARAMS = {
    "RELIANCE.NS":  {"base_price": 2400, "drift": 0.12, "vol": 0.22},
    "TCS.NS":       {"base_price": 3400, "drift": 0.10, "vol": 0.20},
    "INFY.NS":      {"base_price": 1500, "drift": 0.08, "vol": 0.25},
    "HDFCBANK.NS":  {"base_price": 1600, "drift": 0.09, "vol": 0.18},
    "ICICIBANK.NS": {"base_price": 900,  "drift": 0.15, "vol": 0.23},
    "WIPRO.NS":     {"base_price": 420,  "drift": 0.05, "vol": 0.28},
    "BAJFINANCE.NS":{"base_price": 6800, "drift": 0.14, "vol": 0.30},
    "AXISBANK.NS":  {"base_price": 950,  "drift": 0.11, "vol": 0.24},
    "LT.NS":        {"base_price": 2800, "drift": 0.13, "vol": 0.21},
    "SBIN.NS":      {"base_price": 550,  "drift": 0.16, "vol": 0.26},
}

MAX_RETRIES = 3
RETRY_DELAY = 5


def _generate_synthetic_data(ticker, start_date, end_date):
    """Generate realistic synthetic OHLCV data using geometric Brownian motion."""
    params = TICKER_PARAMS.get(ticker, {"base_price": 1000, "drift": 0.10, "vol": 0.25})
    
    # Generate business day date range
    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)
    
    if n_days == 0:
        return pd.DataFrame()
    
    np.random.seed(hash(ticker) % (2**31))
    
    # Geometric Brownian Motion
    dt = 1 / 252
    drift = params["drift"]
    vol = params["vol"]
    
    # Daily returns
    daily_returns = np.exp(
        (drift - 0.5 * vol**2) * dt +
        vol * np.sqrt(dt) * np.random.randn(n_days)
    )
    
    # Price path
    close_prices = params["base_price"] * np.cumprod(daily_returns)
    
    # Generate OHLV from close
    intraday_vol = 0.015  # 1.5% intraday range
    high_prices = close_prices * (1 + np.abs(np.random.randn(n_days)) * intraday_vol)
    low_prices = close_prices * (1 - np.abs(np.random.randn(n_days)) * intraday_vol)
    open_prices = close_prices * (1 + np.random.randn(n_days) * intraday_vol * 0.5)
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
    
    # Volume: base volume with some randomness
    base_volume = int(params["base_price"] * 5000)
    volumes = (base_volume * (1 + np.random.randn(n_days) * 0.3)).astype(int)
    volumes = np.maximum(volumes, 100000)
    
    df = pd.DataFrame({
        "ticker": ticker,
        "date": dates,
        "open": np.round(open_prices, 2),
        "high": np.round(high_prices, 2),
        "low": np.round(low_prices, 2),
        "close": np.round(close_prices, 2),
        "volume": volumes
    })
    
    return df


def fetch_prices(tickers=None, start_date="2021-01-01", end_date="2024-01-01"):
    """
    Fetch historical prices for given NSE tickers using yfinance.
    Falls back to synthetic data if yfinance is unavailable.

    Parameters
    ----------
    tickers : list of str, optional
        List of NSE ticker symbols (e.g., ['RELIANCE.NS']).
        Defaults to DEFAULT_TICKERS.
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ticker, date, open, high, low, close, volume
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    logger.info(f"Fetching prices for {len(tickers)} tickers from {start_date} to {end_date}")

    all_frames = []
    yfinance_failed = False

    # Try yfinance first for the first ticker
    for ticker in tickers[:1]:
        success = False
        for attempt in range(MAX_RETRIES):
            try:
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    timeout=15
                )
                if not data.empty:
                    success = True
                    break
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
            except Exception:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                continue

        if not success:
            yfinance_failed = True
            logger.warning("yfinance unavailable. Falling back to synthetic data generation.")

    if yfinance_failed:
        # Generate synthetic data for all tickers
        logger.info("Generating synthetic data for demonstration...")
        for ticker in tickers:
            df = _generate_synthetic_data(ticker, start_date, end_date)
            if not df.empty:
                all_frames.append(df)
                logger.info(f"  {ticker}: {len(df)} synthetic rows generated")
    else:
        # Fetch all tickers from yfinance
        for ticker in tickers:
            success = False
            for attempt in range(MAX_RETRIES):
                try:
                    data = yf.download(
                        ticker,
                        start=start_date,
                        end=end_date,
                        progress=False,
                        timeout=30
                    )

                    if data.empty:
                        if attempt < MAX_RETRIES - 1:
                            time.sleep(RETRY_DELAY)
                        continue

                    # Handle yfinance 1.x MultiIndex columns: ('Close', 'TICKER')
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)

                    # Standardise column names (yfinance returns Title Case)
                    col_map = {}
                    for c in data.columns:
                        col_map[c] = c.lower()
                    data = data.rename(columns=col_map)

                    df = data[["open", "high", "low", "close", "volume"]].copy()
                    df["ticker"] = ticker
                    df["date"] = df.index
                    df = df.reset_index(drop=True)
                    df = df.dropna(subset=["close"])

                    all_frames.append(df)
                    logger.info(f"  {ticker}: {len(df)} rows fetched from Yahoo Finance")
                    success = True
                    break

                except Exception as e:
                    logger.error(f"Error fetching {ticker} (attempt {attempt + 1}): {e}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                    continue

            if not success:
                logger.warning(f"Failed {ticker}, generating synthetic data")
                df = _generate_synthetic_data(ticker, start_date, end_date)
                if not df.empty:
                    all_frames.append(df)

            time.sleep(1)

    if not all_frames:
        logger.error("No data generated")
        return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume"])

    result = pd.concat(all_frames, ignore_index=True)
    result = result[["ticker", "date", "open", "high", "low", "close", "volume"]]
    result["date"] = pd.to_datetime(result["date"]).dt.date

    logger.info(f"Total rows: {len(result)}")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = fetch_prices()
    print(df.head(20))
    print(f"\nShape: {df.shape}")
