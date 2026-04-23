"""
dashboard/app.py — Streamlit dashboard for the Equity Intelligence Platform.

Sections:
1. Live Factor Scores
2. Current Portfolio Weights
3. Backtest Performance
4. Individual Stock Deep-Dive

Sidebar: Last updated timestamp + Refresh button
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Equity Intelligence Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Database Connection ──────────────────────────────────────────────────────
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "equity_db",
    "user": "equity_user",
    "password": "equity_pass"
}

TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "WIPRO.NS", "BAJFINANCE.NS", "AXISBANK.NS", "LT.NS", "SBIN.NS"
]


@st.cache_resource
def get_connection():
    """Get a database connection."""
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        st.warning(f"Database connection failed: {e}. Running in demo mode with live data.")
        return None


def query_db(sql, conn):
    """Execute a SQL query and return a DataFrame."""
    try:
        return pd.read_sql(sql, conn)
    except Exception:
        return pd.DataFrame()


def get_live_data_fallback():
    """Fetch live data using yfinance when DB is not available."""
    from pipeline.ingest import fetch_prices
    from pipeline.quality import run_quality_checks
    from pipeline.transform import compute_momentum, compute_low_volatility, compute_composite_score
    from quant.optimizer import mean_variance_optimize
    from quant.backtest import run_backtest

    # Fetch prices
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    prices_df = fetch_prices(start_date=start_date, end_date=end_date)

    if prices_df.empty:
        return None, None, None, None, None

    # Quality checks
    prices_df, _ = run_quality_checks(prices_df)
    prices_df["date"] = pd.to_datetime(prices_df["date"])

    # Factor scores
    mom = compute_momentum(prices_df, window=126)
    lvol = compute_low_volatility(prices_df, window=63)
    scores = compute_composite_score(mom, lvol)

    # Latest scores
    latest_date = scores["date"].max()
    latest_scores = scores[scores["date"] == latest_date].copy()

    # Daily returns for optimizer
    daily_returns = []
    for ticker, group in prices_df.groupby("ticker"):
        group = group.sort_values("date").copy()
        group["daily_return"] = group["close"].pct_change()
        daily_returns.append(group[["ticker", "date", "daily_return"]])
    daily_returns_df = pd.concat(daily_returns, ignore_index=True).dropna()

    # Weights
    weights_df = mean_variance_optimize(scores, daily_returns_df, top_n=5)

    # Backtest
    backtest_df, metrics = run_backtest(prices_df=prices_df)

    # Latest weights
    if not weights_df.empty:
        latest_reb = weights_df["rebalance_date"].max()
        latest_weights = weights_df[weights_df["rebalance_date"] == latest_reb].copy()
    else:
        latest_weights = pd.DataFrame()

    return prices_df, latest_scores, latest_weights, backtest_df, metrics


# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(120deg, #1a73e8, #00c853);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1a73e8;
        border-bottom: 3px solid #1a73e8;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border-left: 4px solid #1a73e8;
    }
    .stDataFrame {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📈 Equity Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">NSE Factor Model • Mean-Variance Optimization • Real-Time Dashboard</div>',
    unsafe_allow_html=True
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/stock-market.png", width=80)
    st.markdown("### ⚙️ Controls")

    conn = get_connection()

    if conn:
        last_updated_df = query_db(
            "SELECT MAX(ingested_at) as last_updated FROM raw_prices", conn
        )
        if not last_updated_df.empty and last_updated_df["last_updated"].iloc[0]:
            st.info(f"🕐 Last Updated: {last_updated_df['last_updated'].iloc[0]}")
        else:
            st.info("🕐 No data loaded yet")
    else:
        st.info("🕐 Running in live data mode")

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Universe")
    st.markdown("**10 NSE Large-Caps:**")
    for t in TICKERS:
        st.markdown(f"• {t.replace('.NS', '')}")

# ── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_all_data():
    """Load all data from database or fallback to live computation."""
    conn = get_connection()

    if conn:
        try:
            factor_scores = query_db(
                "SELECT * FROM factor_scores ORDER BY date DESC, rank ASC", conn
            )
            portfolio_weights = query_db(
                "SELECT * FROM portfolio_weights ORDER BY rebalance_date DESC", conn
            )
            backtest_results = query_db(
                "SELECT * FROM backtest_results ORDER BY date ASC", conn
            )
            raw_prices = query_db(
                "SELECT * FROM raw_prices ORDER BY ticker, date", conn
            )

            if not factor_scores.empty:
                latest_date = factor_scores["date"].max()
                latest_scores = factor_scores[factor_scores["date"] == latest_date]

                if not portfolio_weights.empty:
                    latest_reb = portfolio_weights["rebalance_date"].max()
                    latest_weights = portfolio_weights[
                        portfolio_weights["rebalance_date"] == latest_reb
                    ]
                else:
                    latest_weights = pd.DataFrame()

                # Compute metrics from backtest results
                metrics = {}
                if not backtest_results.empty:
                    total_ret = backtest_results["cumulative_portfolio"].iloc[-1] - 1
                    bench_ret = backtest_results["cumulative_benchmark"].iloc[-1] - 1
                    port_rets = backtest_results["portfolio_return"]
                    sharpe = port_rets.mean() / port_rets.std() * np.sqrt(252) if port_rets.std() > 0 else 0
                    cum_max = backtest_results["cumulative_portfolio"].cummax()
                    drawdowns = (backtest_results["cumulative_portfolio"] - cum_max) / cum_max
                    metrics = {
                        "total_return": total_ret,
                        "sharpe_ratio": sharpe,
                        "max_drawdown": drawdowns.min(),
                        "alpha_vs_benchmark": total_ret - bench_ret,
                        "annualised_return": (1 + total_ret) ** (252 / len(backtest_results)) - 1 if len(backtest_results) > 0 else 0,
                        "win_rate": (port_rets > 0).sum() / len(port_rets) if len(port_rets) > 0 else 0
                    }

                return raw_prices, latest_scores, latest_weights, backtest_results, metrics
        except Exception as e:
            st.warning(f"Error loading from DB: {e}")

    # Fallback to live computation
    return get_live_data_fallback()


with st.spinner("Loading data... This may take a moment on first load."):
    data = load_all_data()

if data is None or data[0] is None:
    st.error("Unable to load data. Please ensure the pipeline has run at least once.")
    st.stop()

prices_df, latest_scores, latest_weights, backtest_df, metrics = data

# ── Section 1: Live Factor Scores ────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Section 1 — Live Factor Scores</div>', unsafe_allow_html=True)

if latest_scores is not None and not latest_scores.empty:
    display_scores = latest_scores[["ticker", "momentum_score", "low_vol_score", "composite_score", "rank"]].copy()
    display_scores = display_scores.sort_values("rank")
    display_scores.columns = ["Ticker", "Momentum Score", "Low-Vol Score", "Composite Score", "Rank"]
    display_scores = display_scores.reset_index(drop=True)

    # Style function for rank coloring
    def color_rank(val):
        if val <= 5:
            return "background-color: #c8e6c9; color: #1b5e20; font-weight: bold"
        else:
            return "background-color: #ffcdd2; color: #b71c1c; font-weight: bold"

    styled_df = display_scores.style.map(
        color_rank, subset=["Rank"]
    ).format({
        "Momentum Score": "{:.4f}",
        "Low-Vol Score": "{:.4f}",
        "Composite Score": "{:.1f}",
        "Rank": "{:.0f}"
    })

    st.dataframe(styled_df, use_container_width=True, hide_index=True)
else:
    st.info("No factor scores available. Run the pipeline first.")


# ── Section 2: Current Portfolio Weights ─────────────────────────────────────
st.markdown('<div class="section-header">💰 Section 2 — Current Portfolio Weights</div>', unsafe_allow_html=True)

if latest_weights is not None and not latest_weights.empty:
    weights_display = latest_weights[["ticker", "weight"]].copy()
    weights_display["weight_pct"] = weights_display["weight"] * 100
    weights_display = weights_display.sort_values("weight_pct", ascending=True)
    weights_display["ticker_clean"] = weights_display["ticker"].str.replace(".NS", "", regex=False)

    fig_weights = px.bar(
        weights_display,
        x="weight_pct",
        y="ticker_clean",
        orientation="h",
        text=weights_display["weight_pct"].apply(lambda x: f"{x:.1f}%"),
        labels={"weight_pct": "Weight (%)", "ticker_clean": "Stock"},
        template="plotly_white",
        color="weight_pct",
        color_continuous_scale=["#bbdefb", "#1a73e8"],
    )
    fig_weights.update_traces(textposition="outside")
    fig_weights.update_layout(
        height=350,
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=20, r=80, t=30, b=20),
        font=dict(size=14),
        yaxis=dict(title=""),
        xaxis=dict(title="Weight (%)"),
    )

    st.plotly_chart(fig_weights, use_container_width=True)
else:
    st.info("No portfolio weights available. Run the pipeline first.")


# ── Section 3: Backtest Performance ──────────────────────────────────────────
st.markdown('<div class="section-header">📈 Section 3 — Backtest Performance</div>', unsafe_allow_html=True)

if backtest_df is not None and not backtest_df.empty:
    # Metrics row
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Sharpe Ratio",
                f"{metrics.get('sharpe_ratio', 0):.2f}",
                delta=None
            )
        with col2:
            st.metric(
                "Max Drawdown",
                f"{metrics.get('max_drawdown', 0):.2%}",
                delta=None
            )
        with col3:
            st.metric(
                "Total Return",
                f"{metrics.get('total_return', 0):.2%}",
                delta=None
            )
        with col4:
            st.metric(
                "Alpha vs Benchmark",
                f"{metrics.get('alpha_vs_benchmark', 0):.2%}",
                delta=None
            )

    st.markdown("")

    # Line chart
    bt_plot = backtest_df.copy()
    bt_plot["date"] = pd.to_datetime(bt_plot["date"])

    # Date range selector
    col_start, col_end = st.columns(2)
    with col_start:
        date_start = st.date_input(
            "Start Date",
            value=bt_plot["date"].min().date(),
            min_value=bt_plot["date"].min().date(),
            max_value=bt_plot["date"].max().date()
        )
    with col_end:
        date_end = st.date_input(
            "End Date",
            value=bt_plot["date"].max().date(),
            min_value=bt_plot["date"].min().date(),
            max_value=bt_plot["date"].max().date()
        )

    bt_filtered = bt_plot[
        (bt_plot["date"].dt.date >= date_start) &
        (bt_plot["date"].dt.date <= date_end)
    ]

    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(
        x=bt_filtered["date"],
        y=bt_filtered["cumulative_portfolio"],
        name="Portfolio",
        line=dict(color="#1a73e8", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(26, 115, 232, 0.08)"
    ))
    fig_bt.add_trace(go.Scatter(
        x=bt_filtered["date"],
        y=bt_filtered["cumulative_benchmark"],
        name="Benchmark (Equal-Weight)",
        line=dict(color="#ea4335", width=2, dash="dash"),
    ))
    fig_bt.update_layout(
        template="plotly_white",
        height=450,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        font=dict(size=13),
    )

    st.plotly_chart(fig_bt, use_container_width=True)
else:
    st.info("No backtest results available. Run the pipeline first.")


# ── Section 4: Individual Stock Deep-Dive ────────────────────────────────────
st.markdown('<div class="section-header">🔍 Section 4 — Individual Stock Deep-Dive</div>', unsafe_allow_html=True)

selected_ticker = st.selectbox(
    "Select a ticker",
    options=TICKERS,
    format_func=lambda x: x.replace(".NS", "") + f" ({x})"
)

if selected_ticker:
    with st.spinner(f"Loading {selected_ticker} data..."):
        # Try to load from DB first, fallback to prices_df
        stock_data = None
        conn = get_connection()
        if conn:
            stock_data = query_db(
                f"SELECT date, close FROM raw_prices WHERE ticker = '{selected_ticker}' "
                f"AND date >= CURRENT_DATE - INTERVAL '1 year' ORDER BY date",
                conn
            )

        if (stock_data is None or stock_data.empty) and prices_df is not None and not prices_df.empty:
            # Use already-loaded prices data
            ticker_data = prices_df[prices_df["ticker"] == selected_ticker].copy()
            if not ticker_data.empty:
                ticker_data["date"] = pd.to_datetime(ticker_data["date"])
                # Get last 1 year of data
                max_date = ticker_data["date"].max()
                one_year_ago = max_date - pd.Timedelta(days=365)
                stock_data = ticker_data[ticker_data["date"] >= one_year_ago][["date", "close"]].copy()
            else:
                stock_data = pd.DataFrame()

        if not stock_data.empty:
            stock_data["date"] = pd.to_datetime(stock_data["date"])
            stock_data = stock_data.sort_values("date")
            stock_data["close"] = stock_data["close"].astype(float)
            stock_data["MA20"] = stock_data["close"].rolling(window=20).mean()
            stock_data["MA50"] = stock_data["close"].rolling(window=50).mean()

            fig_stock = go.Figure()
            fig_stock.add_trace(go.Scatter(
                x=stock_data["date"],
                y=stock_data["close"],
                name="Close Price",
                line=dict(color="#1a73e8", width=2),
            ))
            fig_stock.add_trace(go.Scatter(
                x=stock_data["date"],
                y=stock_data["MA20"],
                name="20-Day MA",
                line=dict(color="#fbbc05", width=1.5, dash="dot"),
            ))
            fig_stock.add_trace(go.Scatter(
                x=stock_data["date"],
                y=stock_data["MA50"],
                name="50-Day MA",
                line=dict(color="#ea4335", width=1.5, dash="dash"),
            ))

            ticker_clean = selected_ticker.replace(".NS", "")
            fig_stock.update_layout(
                title=f"{ticker_clean} — 1 Year Price Chart with Moving Averages",
                template="plotly_white",
                height=450,
                margin=dict(l=20, r=20, t=60, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                font=dict(size=13),
            )

            st.plotly_chart(fig_stock, use_container_width=True)

            # Summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                current_price = stock_data["close"].iloc[-1]
                st.metric("Current Price", f"₹{current_price:,.2f}")
            with col2:
                price_change = (stock_data["close"].iloc[-1] / stock_data["close"].iloc[0] - 1)
                st.metric("1Y Change", f"{price_change:.2%}")
            with col3:
                daily_rets = stock_data["close"].pct_change().dropna()
                vol = daily_rets.std() * np.sqrt(252)
                st.metric("Annualised Vol", f"{vol:.2%}")
        else:
            st.warning(f"No data available for {selected_ticker}")

# ── Section 5: ML Model Performance ──────────────────────────────────────────
st.markdown(
    '<div class="section-header">🤖 Section 5 — ML Model Performance (Transformer vs. Baseline)</div>',
    unsafe_allow_html=True,
)

ART_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")

@st.cache_data(ttl=300)
def _load_ml_artifacts():
    def _try(path, loader):
        full = os.path.join(ART_DIR, path)
        if os.path.exists(full):
            try:
                return loader(full)
            except Exception:
                return None
        return None

    return {
        "baseline": _try("oos_preds_baseline.parquet", pd.read_parquet),
        "transformer": _try("oos_preds_transformer.parquet", pd.read_parquet),
        "regimes": _try("regime_labels.parquet", pd.read_parquet),
        "shap_global": _try("shap_global.csv", pd.read_csv),
        "shap_per_regime": _try("shap_per_regime.csv", pd.read_csv),
        "evaluation_summary": _try(
            "evaluation_summary.json",
            lambda p: __import__("json").load(open(p, "r", encoding="utf-8")),
        ),
    }


ml = _load_ml_artifacts()
if ml["baseline"] is None and ml["transformer"] is None:
    st.info(
        "ML artifacts not found. Run `python run_ml_pipeline.py` first to generate "
        "out-of-sample predictions, conformal intervals, regime labels, and SHAP values."
    )
else:
    # ── KPIs from evaluation summary ─────────────────────────────────────
    summary = ml.get("evaluation_summary") or {}
    colA, colB, colC, colD = st.columns(4)
    if "transformer" in summary and summary["transformer"]["overall"]:
        ov = summary["transformer"]["overall"][0]
        colA.metric("Transformer OOS IC", f"{ov.get('ic', 0):.4f}")
        colB.metric("Transformer Dir. Acc.", f"{ov.get('dir_acc', 0):.2%}")
        colC.metric("TFT Long-Short Sharpe-like", f"{ov.get('ls_sharpe_like', 0):.2f}")
    if "transformer" in summary and summary["transformer"]["calibration"]:
        cal = [r for r in summary["transformer"]["calibration"] if r["regime"] == "overall"]
        if cal:
            colD.metric(
                "Conformal Coverage (nominal 0.90)",
                f"{cal[0]['empirical_coverage']:.2%}",
            )

    # ── Out-of-sample long-short PnL chart ───────────────────────────────
    st.markdown("#### Out-of-sample long-short portfolio — Transformer vs. Baseline")

    def _cum_ls(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        rows = []
        for d, g in df.groupby("date"):
            if len(g) < 6:
                continue
            g_sorted = g.sort_values("prediction", ascending=False)
            lret = g_sorted.head(3)["target_fwd_ret"].mean()
            sret = g_sorted.tail(3)["target_fwd_ret"].mean()
            rows.append({"date": pd.Timestamp(d), "ls": float(lret - sret)})
        out = pd.DataFrame(rows).sort_values("date")
        out["cum"] = (1 + out["ls"]).cumprod()
        return out

    cum_base = _cum_ls(ml["baseline"])
    cum_tft = _cum_ls(ml["transformer"])

    fig_pnl = go.Figure()
    if not cum_tft.empty:
        fig_pnl.add_trace(go.Scatter(
            x=cum_tft["date"], y=cum_tft["cum"],
            name="Transformer", line=dict(color="#1a73e8", width=2.5),
        ))
    if not cum_base.empty:
        fig_pnl.add_trace(go.Scatter(
            x=cum_base["date"], y=cum_base["cum"],
            name="Baseline GBR", line=dict(color="#ea4335", width=2, dash="dash"),
        ))
    fig_pnl.update_layout(
        template="plotly_white",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Date",
        yaxis_title="Cumulative long-short return",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_pnl, use_container_width=True)

    # ── Per-regime metrics table ─────────────────────────────────────────
    st.markdown("#### Per-regime out-of-sample performance")
    rows = []
    for model_name in ("baseline_gbr", "transformer"):
        if model_name in summary:
            for r in summary[model_name]["per_regime"]:
                rows.append({
                    "model": model_name,
                    "regime": r["regime"],
                    "n": r["n_samples"],
                    "RMSE": f"{r['rmse']:.4f}",
                    "Dir. Acc.": f"{r['dir_acc']:.2%}",
                    "IC": f"{r['ic']:.4f}",
                    "LS mean": f"{r.get('ls_mean', 0):.4f}",
                    "LS Sharpe~": f"{r.get('ls_sharpe_like', 0):.2f}",
                })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Conformal calibration ────────────────────────────────────────────
    st.markdown("#### Conformal prediction calibration (empirical vs. 90% nominal)")
    cal_rows = []
    for model_name in ("baseline_gbr", "transformer"):
        if model_name in summary and summary[model_name]["calibration"]:
            for r in summary[model_name]["calibration"]:
                cal_rows.append({
                    "model": model_name,
                    "regime": r["regime"],
                    "n": r["n_samples"],
                    "empirical_cov": r["empirical_coverage"],
                    "nominal_cov": r["nominal_coverage"],
                    "mean_width": r["mean_interval_width"],
                })
    if cal_rows:
        cdf = pd.DataFrame(cal_rows)
        fig_cal = px.bar(
            cdf, x="regime", y="empirical_cov", color="model", barmode="group",
            labels={"empirical_cov": "Empirical coverage"},
            template="plotly_white",
            color_discrete_map={"baseline_gbr": "#ea4335", "transformer": "#1a73e8"},
        )
        fig_cal.add_hline(y=0.90, line_dash="dash", line_color="black",
                          annotation_text="nominal 0.90", annotation_position="top right")
        fig_cal.update_layout(height=330, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_cal, use_container_width=True)

    # ── SHAP feature importance ──────────────────────────────────────────
    if ml["shap_global"] is not None and not ml["shap_global"].empty:
        st.markdown("#### Global SHAP feature importance (baseline GBR)")
        shap_df = ml["shap_global"].head(15).sort_values("mean_abs_shap")
        fig_shap = px.bar(
            shap_df, x="mean_abs_shap", y="feature",
            orientation="h", template="plotly_white",
            labels={"mean_abs_shap": "Mean |SHAP value|", "feature": ""},
        )
        fig_shap.update_layout(height=450, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_shap, use_container_width=True)

    # ── Regime timeline ──────────────────────────────────────────────────
    if ml["regimes"] is not None and not ml["regimes"].empty:
        st.markdown("#### HMM-detected market regimes")
        reg = ml["regimes"].copy()
        reg["date"] = pd.to_datetime(reg["date"])
        regime_color = {"calm": "#34a853", "trending": "#1a73e8", "crisis": "#ea4335"}
        fig_reg = px.scatter(
            reg, x="date", y="regime", color="regime",
            color_discrete_map=regime_color,
            template="plotly_white",
        )
        fig_reg.update_traces(marker=dict(size=5))
        fig_reg.update_layout(
            height=230, margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False, yaxis=dict(title=""),
        )
        st.plotly_chart(fig_reg, use_container_width=True)


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #999; font-size: 0.85rem;">'
    '🏗️ Equity Intelligence Platform • Built with Streamlit, Airflow, PostgreSQL, PyTorch, scikit-learn & yfinance'
    '</div>',
    unsafe_allow_html=True
)
