-- Equity Intelligence Platform - Database Schema Initialization

CREATE TABLE IF NOT EXISTS raw_prices (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20),
    date DATE,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume BIGINT,
    ingested_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(ticker, date)
);

CREATE TABLE IF NOT EXISTS factor_scores (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20),
    date DATE,
    momentum_score FLOAT,
    low_vol_score FLOAT,
    composite_score FLOAT,
    rank INTEGER,
    UNIQUE(ticker, date)
);

CREATE TABLE IF NOT EXISTS portfolio_weights (
    id SERIAL PRIMARY KEY,
    rebalance_date DATE,
    ticker VARCHAR(20),
    weight FLOAT,
    UNIQUE(rebalance_date, ticker)
);

CREATE TABLE IF NOT EXISTS backtest_results (
    id SERIAL PRIMARY KEY,
    date DATE,
    portfolio_return FLOAT,
    benchmark_return FLOAT,
    cumulative_portfolio FLOAT,
    cumulative_benchmark FLOAT
);
