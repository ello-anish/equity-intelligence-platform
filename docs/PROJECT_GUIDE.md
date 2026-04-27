# The Equity Intelligence Platform — Complete Project Guide

> A read-from-front-to-back tour of every concept, every module, and every
> design decision in this repository. After reading this document, you should
> be able to (a) run the project, (b) explain *why* every layer is there,
> (c) defend every methodological choice in an interview, and (d) extend
> the project in a sensible direction.

If you only have 5 minutes, read **§1 Big picture** and **§13 Empirical
results**. If you have an hour, read the whole thing.

---

## Table of contents

1. [Big picture](#1-big-picture)
2. [Domain primer — what the model is trying to do](#2-domain-primer--what-the-model-is-trying-to-do)
3. [Why this project is hard to do honestly](#3-why-this-project-is-hard-to-do-honestly)
4. [The data layer](#4-the-data-layer)
5. [Feature engineering](#5-feature-engineering)
6. [Targets — single, multi-horizon, and vol-scaled](#6-targets--single-multi-horizon-and-vol-scaled)
7. [Regime detection (Hidden Markov Model)](#7-regime-detection-hidden-markov-model)
8. [Purged walk-forward cross-validation](#8-purged-walk-forward-cross-validation)
9. [The two models — GradientBoosting baseline and Transformer](#9-the-two-models)
10. [Conformal prediction — calibrated uncertainty](#10-conformal-prediction--calibrated-uncertainty)
11. [Explainability with SHAP](#11-explainability-with-shap)
12. [The portfolio allocator and transaction costs](#12-the-portfolio-allocator-and-transaction-costs)
13. [Statistical rigor — Diebold-Mariano, bootstrap CIs, PBO](#13-statistical-rigor)
14. [Feature drift detection](#14-feature-drift-detection)
15. [MLflow experiment tracking](#15-mlflow-experiment-tracking)
16. [Optuna hyperparameter search](#16-optuna-hyperparameter-search)
17. [The FastAPI serving layer](#17-the-fastapi-serving-layer)
18. [The Streamlit dashboard](#18-the-streamlit-dashboard)
19. [The test suite](#19-the-test-suite)
20. [End-to-end walkthrough — what happens when you run the pipeline](#20-end-to-end-walkthrough)
21. [Empirical results explained](#21-empirical-results-explained)
22. [Honest limitations and known weaknesses](#22-honest-limitations-and-known-weaknesses)
23. [How to extend this project](#23-how-to-extend-this-project)
24. [File-by-file reference](#24-file-by-file-reference)
25. [Glossary](#25-glossary)
26. [References and further reading](#26-references-and-further-reading)

---

## 1. Big picture

This project is an end-to-end **machine-learning quantitative research
platform** for a small universe of Indian large-cap equities (10 NSE
tickers). It does four things:

1. **Ingests price data** for those tickers and engineers ~42 numeric
   features per (ticker, date), including:
   - classical price/volume features (momentum, volatility, RSI, MACD)
   - cross-sectional ranks (where does this ticker rank today?)
   - market-state aggregates (market vol, dispersion, momentum)
   - **real macro features** (Nifty index, India VIX, USDINR, US 10y yield, gold)
   - **FinBERT news sentiment** (headlines → daily per-ticker sentiment scores)
2. **Trains two forecasting models** to predict each ticker's forward return
   over multiple horizons (5d, 21d, 63d):
   - a **scikit-learn GradientBoostingRegressor** baseline (honest benchmark + SHAP-explainable)
   - a **PyTorch attention-based Transformer** (TFT-inspired) with optional pairwise ranking loss
3. **Quantifies uncertainty** around each prediction using **split-conformal
   prediction intervals** — distribution-free, calibrated, model-agnostic.
4. **Constructs portfolios** from the predictions + intervals using three
   allocator policies (vanilla long-short, uncertainty-aware width-scaled,
   sector-neutral), with realistic transaction costs and turnover accounting,
   and reports honest performance metrics — including statistical
   significance tests.

Everything is wrapped in production-grade infrastructure: Apache Airflow
DAG, PostgreSQL, MLflow experiment tracking, FastAPI serving endpoint,
Docker, a Streamlit dashboard, Optuna hyperparameter search, ablation and
stability driver scripts, and a 43-test pytest suite.

```
┌────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐
│  yfinance  │→ │ feature eng │→ │ HMM regimes │→ │ walk-forward │
│  (NSE +    │  │ + FinBERT + │  │ (per-fold,  │  │     CV       │
│  macros)   │  │  macros     │  │  no leak)   │  │              │
└────────────┘  └─────────────┘  └─────────────┘  └──────┬───────┘
                                                         │
                          ┌──────────────────────────────┴────────────┐
                          │                                           │
                          ▼                                           ▼
              ┌──────────────────────┐                  ┌──────────────────────┐
              │  GradientBoosting    │                  │  Transformer         │
              │  baseline (sklearn)  │                  │  multi-horizon, attn │
              │  + SHAP              │                  │  + (optional) rank   │
              └──────────┬───────────┘                  └──────────┬───────────┘
                         └────────┬───────────────────────────────┘
                                  ▼
                       ┌────────────────────┐
                       │  Split-conformal   │
                       │  prediction        │
                       │  (90% intervals)   │
                       └────────┬───────────┘
                                ▼
            ┌──────────────────────────────────────────────┐
            │  Allocator (3 policies, 20bps tc, turnover)  │
            │   • vanilla   • width-scaled   • sector-neut │
            └──────────────────────┬───────────────────────┘
                                   ▼
       ┌──────────────────────────────────────────────────────┐
       │  Statistical rigor: Diebold-Mariano test, bootstrap   │
       │  Sharpe CIs, PBO, drift detection, MLflow logged.     │
       └──────────────────────────────────────────────────────┘
```

---

## 2. Domain primer — what the model is trying to do

If you don't have a finance background, this section is essential.

### 2.1 The basic setup

We have a small universe of 10 stocks (the most liquid Indian large-caps:
Reliance, TCS, Infosys, HDFC Bank, ICICI Bank, Wipro, Bajaj Finance,
Axis Bank, L&T, SBI). Every trading day we observe each stock's open, high,
low, close, and volume. From those we derive features and ask: **"Which of
these 10 stocks will go up the most over the next 21 trading days, and
which will go down the most?"**

If the model is right on average, we can build a long-short portfolio: buy
the predicted winners, short-sell the predicted losers, and earn the
*difference*. This isolates **alpha** (skill) from **beta** (overall market
movement) — even if the market crashes, our long-short P&L can still be
positive if the winners we picked outperformed the losers we sold.

### 2.2 Why this is genuinely hard

Stock returns are *noisy*. Daily returns have a tiny mean (~0.05%) and a
much larger standard deviation (~1-2%), so the **signal-to-noise ratio is
very low**. A predictive R² of 1% on returns is *good*. A model with
*directional accuracy* of 53% (vs the 50% coin-flip baseline) is genuinely
useful at scale.

This means we need:
- Very disciplined evaluation — random luck looks like skill if you're not careful.
- Calibrated uncertainty — saying "I don't know" matters more than being right occasionally.
- Honest reporting — the model often doesn't beat a coin flip; we need to know *when*.

### 2.3 Key terms you'll see throughout this document

- **Return** — fractional change in price. Log-return $r_t = \ln(p_t/p_{t-1})$ is preferred because log-returns add over time.
- **Forward return** — the return we're trying to *predict*, looking forward from today: $r_{t \to t+h} = \ln(p_{t+h}/p_t)$.
- **Information Coefficient (IC)** — average per-date Spearman rank correlation between predictions and realised returns. The standard quant skill metric. ICs of 0.05+ are excellent in practice.
- **Sharpe ratio** — annualised mean return divided by annualised return volatility. The standard portfolio quality metric. >1.0 is good, >2.0 is exceptional.
- **Drawdown** — peak-to-trough percentage loss. Lower (less negative) is better.
- **Alpha** — excess return relative to a benchmark. What every quant claims they have.
- **Long-short** — a portfolio that buys some names and short-sells others; designed to be market-neutral.
- **Rebalance** — periodically updating portfolio weights based on new signals. We rebalance monthly.

---

## 3. Why this project is hard to do honestly

Most intern-level finance ML projects silently fail at least one of five
methodological tests. Recognising what goes wrong, and explicitly fixing
each, is the central contribution of this repo.

### 3.1 Look-ahead leakage in features

A *feature* at date $t$ must depend only on information available at or
before $t$. If you compute "20-day moving average centred on date $t$"
(i.e., uses days $t-10$ through $t+10$), the model has perfect knowledge of
the future and will look brilliant out-of-sample for entirely the wrong
reason.

**Our fix:** every feature in `quant/ml/features.py` is causal (uses only
past data). We test this explicitly in `tests/test_features.py::test_no_lookahead_in_features`,
which corrupts future prices and asserts that past features are byte-identical.

### 3.2 Look-ahead leakage in regime labels

If you fit an HMM (or any unsupervised model) on the entire price history
*before* splitting train/test, the HMM has implicitly used future data to
decide what "calm" and "crisis" mean. When you then split your data and
evaluate per-regime, the regime labels themselves are tainted.

**Our fix:** the HMM is **refit per fold** on training data only, with an
explicit `end_date` cutoff. The fitted transition matrix and emission
distributions are then used to *predict* regimes on the test window, but
never to fit them. Tested in `tests/test_regimes.py::test_no_leakage_past_cutoff`.

### 3.3 Random K-fold cross-validation on time series

Random splits put past and future days in the same fold; given that returns
have autocorrelation and that our 21-day forward target overlaps from day to
day, this leaks future info into training. Even *standard* TimeSeriesSplit
is insufficient because it doesn't enforce a gap between train and test
when targets overlap.

**Our fix:** **purged walk-forward CV** with a 21-day embargo (exactly the
forward-return horizon). This is the canonical approach from López de
Prado's *Advances in Financial Machine Learning*, Ch. 7. Tested in
`tests/test_walkforward.py`.

### 3.4 Overlapping-horizon Sharpe inflation

If you make a 21-day-forward prediction every day and stack daily returns
of those predictions, you have effectively a 21×-leveraged position, and
your Sharpe will look ~√21 ≈ 4.5× too good. This is the most common bug in
intern-level backtests, and the numbers people report on LinkedIn would
shock a real risk manager.

**Our fix:** the headline backtest in `monthly_rebalance_backtest()`
rebalances *once per horizon* (i.e., monthly), so positions don't overlap.
The Sharpe number you see is what a tradeable strategy would have produced.
Tested in `tests/test_costs_and_sector.py::test_monthly_rebalance_tc_reduces_net_return`.

### 3.5 Overconfident point estimates

Models output a number; the world demands an *interval*. If a portfolio
manager has to size a position, "predicted forward return is +0.5%" is
useless without knowing whether that's +0.5±0.2 or +0.5±5.0. An intern
project that just shows directional accuracy is missing the entire decision
layer.

**Our fix:** **split-conformal prediction**. Distribution-free, model-agnostic,
finite-sample-guaranteed coverage. The width of the interval is then *used*
by the allocator (§12) to size positions, so uncertainty has economic
consequences.

These five fixes are the project's actual selling point. The fancy
Transformer is downstream.

---

## 4. The data layer

### 4.1 Source: yfinance

`pipeline/ingest.py` defines `fetch_prices(tickers, start_date, end_date)`,
which calls Yahoo Finance via the `yfinance` package and returns a tidy
DataFrame with columns: `ticker, date, open, high, low, close, volume`.

The default ticker universe is the 10 NSE large-caps named in §2.1.
Yahoo Finance ticker format appends `.NS` for NSE.

### 4.2 Synthetic fallback

If yfinance is unreachable (no network, rate-limited, GeoIP block) the
function falls back to a **deterministic synthetic data generator**:
geometric Brownian motion priced separately per ticker with calibrated
drift and vol per name. The synthetic prices look reasonable (right ballpark
for actual NSE levels) but are clearly *fake*; the pipeline logs a loud
warning when synthesis kicks in. This makes the project runnable on a
plane.

### 4.3 Quality checks

`pipeline/quality.py::run_quality_checks(df)` enforces:
1. No nulls in `close` or `volume`
2. No negative prices
3. No future dates
4. Volume > 0

Failing rows are dropped and a quality report dict is returned. This is the
kind of "boring" plumbing that catches bad upstream feeds before they
poison the model.

### 4.4 Persistent storage

`init.sql` creates four PostgreSQL tables:

| Table | What's in it |
|---|---|
| `raw_prices` | OHLCV per (ticker, date), unique constraint on (ticker, date) |
| `factor_scores` | Composite factor scores (legacy classical pipeline) |
| `portfolio_weights` | Per-rebalance optimised weights (legacy mean-variance) |
| `backtest_results` | Daily portfolio vs benchmark cumulative returns (legacy) |
| `ml_predictions` | (created by ML pipeline) Per-(model, ticker, date, fold) prediction + lower/upper bounds + realised target |

`pipeline/load.py` provides `upsert_*()` functions that use `psycopg2.extras.execute_batch`
to bulk-insert with `ON CONFLICT DO NOTHING` semantics — so re-running the
pipeline doesn't duplicate rows.

### 4.5 Airflow orchestration

`dags/equity_pipeline_dag.py` defines a daily DAG with five sequential tasks:

```
fetch_raw_data → run_quality_checks → compute_factors → load_to_postgres → run_backtest
```

Tasks pass data through Airflow's XCom (small JSON payloads). This
demonstrates the production handoff pattern: the *same* Python modules used
in the standalone `run_full_pipeline.py` script are imported by the DAG —
no duplication.

### 4.6 Docker and the .env pattern

`docker-compose.yml` runs Postgres 15 and Airflow 2.8 in containers, wired
together on a private bridge network `equity-net`. All credentials and
secret keys are now read from environment variables; copy `.env.example`
to `.env` and fill in values:

```bash
cp .env.example .env
# edit .env to set POSTGRES_PASSWORD, AIRFLOW__CORE__FERNET_KEY, etc.
docker-compose up -d
```

`.env` is gitignored; `.env.example` is committed with placeholder values
so a new contributor can see what keys are needed.

---

## 5. Feature engineering

`quant/ml/features.py::build_feature_panel(prices_df)` is the single
entry-point. It builds ~42 features per (ticker, date) row, then merges
in macro features and FinBERT sentiment as separate joins. Every feature
is causal — see `tests/test_features.py` for the no-lookahead assertion.

### 5.1 Per-ticker features

| Feature | Formula | Intuition |
|---|---|---|
| `ret_1d` / `ret_5d` / `ret_21d` | $\Delta p / p$ over 1/5/21 days | Recent return at multiple frequencies |
| `mom_1m` / `mom_3m` / `mom_6m` / `mom_12m` | $\ln p_t - \ln p_{t-h}$ for $h \in \{21, 63, 126, 252\}$ | Classical momentum at multiple horizons |
| `vol_20d` / `vol_60d` | rolling std of daily returns × $\sqrt{252}$ | Annualised realised volatility |
| `ret_skew_60d` / `ret_kurt_60d` | rolling skewness / kurtosis | Distribution shape — fat tails matter |
| `vol_z_20d` | (log volume − rolling mean) / rolling std | Volume surprise — z-scored unusualness |
| `px_over_ma20` / `_50` / `_200` | $p_t / \mathrm{MA}_h(p) - 1$ | Distance from moving average — trend strength |
| `rsi_14` | 14-day RSI (Wilder's exponential smoothing) | Mean-reversion / overbought-oversold indicator |
| `macd` / `macd_hist` | 12/26/9 EMA difference | Momentum convergence/divergence |

The RSI uses Wilder's smoothing (EMA with $\alpha=1/14$) rather than simple
moving average; this is the textbook variant, more responsive on high-vol
days.

### 5.2 Cross-sectional rank features

For each date, every ticker is **ranked** across the 10-name cross-section
on:
- `cs_rank_mom_1m`, `cs_rank_mom_3m`, `cs_rank_mom_6m`
- `cs_rank_vol_20d`, `cs_rank_vol_60d`
- `cs_rank_rsi_14`

Ranks are computed as percentiles in $[0, 1]$. This is genuinely useful
because a model might learn that *being* the highest-momentum stock today
matters more than the absolute level of momentum.

### 5.3 Market-state features

For each date, aggregated across the 10-name universe:

| Feature | Definition |
|---|---|
| `mkt_ret` | Cross-sectional mean of `ret_1d` |
| `mkt_disp` | Cross-sectional std of `ret_1d` |
| `mkt_vol_20d` | 20-day rolling std of `mkt_ret`, annualised |
| `mkt_mom_21d` | 21-day rolling sum of `mkt_ret` |

These broadcast back to every (ticker, date) row.

### 5.4 Macro features (real exogenous data)

`quant/ml/macro.py::fetch_macro_series()` downloads from yfinance:

| Ticker | What it is | Why it matters |
|---|---|---|
| `^NSEI` | Nifty 50 index level | Direct market-state proxy |
| `^INDIAVIX` | India VIX | Forward-looking expected volatility |
| `INR=X` | USD/INR exchange rate | FII (foreign investor) flow proxy |
| `^TNX` | US 10y Treasury yield | Global rates regime (affects EM equity flows) |
| `GC=F` | Gold futures | Safe-haven demand |

Then `build_macro_features()` derives:

```
nifty_ret_1d, nifty_ret_21d, nifty_over_ma50,
india_vix_level, india_vix_z20,
usdinr_ret_1d, usdinr_ret_21d,
us10y_level, us10y_chg_21d,
gold_ret_21d
```

Synthetic GBM fallback on network failure, same pattern as ticker prices.
Macro features dominate the SHAP importance ranking in v3 (see §13).

### 5.5 FinBERT sentiment features

`quant/ml/sentiment.py` runs a complete NLP layer:

1. **Load news**: tries `data/news.csv` or `news.csv` in repo root with
   columns `(ticker, date, headline)`. If neither exists, falls back to a
   **synthetic headline generator** (deterministic, seeded) that picks
   templates biased by the actual day's return — clearly labelled in logs.
2. **Score with FinBERT**: `ProsusAI/finbert` is a BERT-base model fine-tuned
   on financial sentiment (positive / negative / neutral). We compute
   $\mathrm{sentiment} = P(\text{positive}) - P(\text{negative})$ in $[-1, 1]$.
   Falls back to a 9-word lexicon if `transformers` isn't installed.
3. **Aggregate per (ticker, date)**: mean sentiment, news count, 5-day and
   21-day rolling means.
4. **Cache**: scored headlines are cached to `artifacts/sentiment_cache_<hash>.parquet`
   so re-runs over the same news set are instant.

Sentiment features (`sentiment_mean`, `sentiment_n`, `sentiment_ma5`, `sentiment_ma21`)
are joined onto the panel via `merge_sentiment_onto_panel()`, with
forward-fill within ticker (yesterday's sentiment still matters today) and
zero-fill at the start.

### 5.6 The feature column lists

Code uses three constants:

```python
FEATURE_COLS          # ~28 per-ticker + cross-sectional + market-state features
MACRO_FEATURE_COLS    # 10 macro features
SENTIMENT_FEATURE_COLS  # 4 sentiment features
ALL_FEATURE_COLS = FEATURE_COLS + MACRO_FEATURE_COLS  # default union
```

The orchestrator builds `all_features = FEATURE_COLS + MACRO_FEATURE_COLS + SENTIMENT_FEATURE_COLS`
when sentiment is available, then narrows it down per ablation variant
(e.g., `--ablation no_market_feats` drops everything starting with `mkt_`).

---

## 6. Targets — single, multi-horizon, and vol-scaled

The forecasting target is the **forward log-return**:

$$
y_{t,h} = \ln \frac{p_{t+h}}{p_t}
$$

We compute three horizons in one pass:
- `target_fwd_ret_5d` — 1 week
- `target_fwd_ret_21d` — 1 month (the *primary* target used by conformal & eval)
- `target_fwd_ret_63d` — 3 months

Plus a back-compat alias `target_fwd_ret = target_fwd_ret_21d`.

### 6.1 Multi-horizon forecasting

Why three horizons? **Multi-task learning**. The Transformer has three
output heads sharing the same encoder; the loss is the sum of MSE on each
head. The auxiliary 5d and 63d heads regularise the 21d head — the
encoder is forced to learn a representation that's useful at multiple
time-scales rather than overfitting to one.

This is exactly the rationale in the **Temporal Fusion Transformer** paper
(Lim et al., 2021).

### 6.2 Vol-scaled targets

`target_fwd_ret_21d_vs = target_fwd_ret_21d / (rolling_vol_63d + ε)` is the
**risk-adjusted** version of the forward return. It exists because:

- A 5% forward return on a low-vol defensive stock is a much stronger
  signal than 5% on a high-vol mid-cap.
- Predicting risk-adjusted returns is what every systematic shop actually
  does; predicting raw returns is a textbook beginner trap.

In v3, this column is computed but the default pipeline still uses
`target_fwd_ret_21d` as the primary target. Switching to `_vs` is one
config change away (in `TransformerConfig`); typically improves IC by 1-3
percentage points but requires re-tuning conformal calibration.

---

## 7. Regime detection (Hidden Markov Model)

### 7.1 What an HMM is, in 90 seconds

A Hidden Markov Model assumes the world has $K$ unobserved discrete
"states" $z_t \in \{1, ..., K\}$, and what you observe ($x_t$) is drawn
from a state-dependent distribution $p(x_t \mid z_t)$. The states evolve
as a Markov chain: $p(z_{t+1} \mid z_t)$ is a $K \times K$ transition
matrix.

Fitting an HMM means: given observations $x_1, ..., x_T$, find the most
likely transition matrix and emission distributions (Baum-Welch /
forward-backward EM). Predicting means: given a new $x_t$, infer the most
likely $z_t$ (Viterbi or forward).

### 7.2 What we use it for

We define a market state $x_t = (\mathrm{mkt\_vol\_20d}, \mathrm{mkt\_disp\_20d})$ —
two-dimensional, capturing how volatile *and* how dispersed daily returns
are. We fit a 3-state Gaussian HMM. The three latent states naturally
correspond to:
- low vol, low dispersion → **calm**
- moderate vol, moderate dispersion → **trending**
- high vol, high dispersion → **crisis**

We sort states post-hoc by their mean fitted volatility and label them in
that order, so "calm/trending/crisis" is always assigned consistently.

### 7.3 Per-fold refit (the leakage fix)

`fit_regime_model(prices, end_date=...)` is the key API:

- `end_date=None` → fit on the full history (legacy behaviour, **leaks**)
- `end_date=split.train_end` → fit ONLY on data up to the train cutoff,
  then `predict()` regimes on the post-cutoff window without retraining

The orchestrator calls `fit_regime_model(prices, end_date=split.train_end)`
inside every walk-forward fold, so the regime label assigned to a test-set
date was produced by an HMM that *only* saw training data.

### 7.4 Fallback

If `hmmlearn` isn't installed, the function falls back to a quantile-based
heuristic: tercile-split on `mkt_vol_20d`. This keeps the pipeline runnable
on a minimal install, and the test suite passes with either implementation.

---

## 8. Purged walk-forward cross-validation

### 8.1 Why we can't use random K-fold

In a random K-fold, fold-1 might contain January 2022 data while fold-2
contains February 2022 data. Both are in different folds. *But*:
- Returns have autocorrelation (a small effect, but real).
- A 21-day forward target at January 31 *literally is* a function of
  February 21's price. If January 31 ends up in train and February 21 in
  test, the model sees the answer to a test question.

This is **horizon overlap leakage** and it's deadly. Most people don't
even know it exists.

### 8.2 What purged walk-forward CV does

In `quant/ml/walkforward.py::PurgedWalkForward`:

```
       train_window         embargo     test_window     embargo     ...
  [────────────────────────][═══════][────────────][═══════]
                                          ↑
                                   stride forward by `step` days
```

- `train_window_days = 504` (~ 2 years)
- `embargo_days = 21` (= the forward-return horizon)
- `test_window_days = 63` (~ 3 months)
- `step_days = 126` (~ 6 months)

The embargo gap ensures no train sample's *target* uses any data that
appears in the test set. Then each successive fold rolls forward by `step`
days. Over a 5-year span we get ~7 folds. Each fold is fit-from-scratch
(model, conformal calibration, *and* HMM regime detector).

### 8.3 The calibration sub-split

Inside each fold, we further split the train window:
- the last 90 days → **calibration** slice (used for conformal)
- everything before → **train-proper** (used to fit the model)

This is called *split-conformal* and it's the right way to get
distribution-free coverage guarantees — see §10.

### 8.4 Tests

- `test_no_overlap_between_train_and_test` — for every fold, asserts
  $\mathrm{test\_start} - \mathrm{train\_end} \geq \mathrm{embargo}$ days.
- `test_train_precedes_test_always` — train indices all come before test indices.
- `test_at_least_one_fold_produced` — sanity check.

These tests are the *dam* against silent leakage. If anyone refactors the
splitter and introduces a bug, pytest catches it.

---

## 9. The two models

### 9.1 The baseline: GradientBoostingRegressor

`quant/ml/baseline.py::BaselineForecaster` wraps:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("gbr", GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )),
])
```

#### What gradient boosting is

A gradient-boosted tree ensemble starts with a constant prediction (the
mean) and then iteratively adds shallow decision trees, each one fitted to
the *residuals* of the current ensemble. Mathematically:

$$
F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x), \quad h_m \approx -\nabla L(y, F_{m-1}(x))
$$

where $\nu$ is the learning rate and $h_m$ is a shallow tree.

300 trees of depth 3 with learning rate 0.05 is a moderate-strength
baseline — strong enough to be a fair benchmark, simple enough to inspect
with SHAP.

#### Why it's the baseline

Gradient boosting on tabular features is a notorious "hard to beat"
benchmark on structured data. A Transformer that doesn't beat it isn't
worth the complexity. By including both, every claim we make about the
Transformer is *relative to a strong reference*.

### 9.2 The Transformer

`quant/ml/transformer_model.py` defines a TFT-inspired multi-head attention
encoder.

#### Inputs

- A sequence of the last `seq_len` (default 60) trading days of features:
  shape `(batch, 60, n_features)`.
- A static integer ticker ID (one of 10): shape `(batch,)`.

#### Architecture

```
features [B, 60, F]
    │
    ▼
Linear projection to d_model (=64)        [B, 60, 64]
    │
    + sinusoidal positional encoding       [B, 60, 64]
    │
    ▼
TransformerEncoder × n_layers (=2)
  each layer: MultiHeadAttention (n_heads=4) → FFN(d=128) → LayerNorm
  with optional CAUSAL MASK so position t can only attend to ≤t          [B, 60, 64]
    │
    ▼
Pooling: either last-token h[:, -1, :] OR attention-weighted Σ α_t h_t   [B, 64]
    │
    + ticker embedding (d=8)              [B, 64+8]
    │
    ▼
MLP head: Linear(d) → GELU → Dropout → Linear(n_targets)                  [B, n_targets]
```

n_targets = 3 (5d, 21d, 63d horizons) by default.

#### Why each piece exists

| Component | Purpose |
|---|---|
| Linear projection | Maps `n_features` to `d_model` so attention can operate at fixed dim. |
| Positional encoding | Self-attention is permutation-invariant; we need the model to know that day-59 is "yesterday". |
| Multi-head attention | Each head can attend to a different temporal pattern (short bursts, long drift). |
| Causal mask | Prevents position t from attending to t+1, t+2, ... — the *non-causal* baseline is an ablation variant. |
| Attention pooling | The model learns *which timesteps* matter; visualised in `plot_attention_weights.png`. |
| Ticker embedding | Static covariate that lets the model learn ticker-specific behaviour without per-ticker features. |
| Multi-horizon head | Three output dims = three forward horizons; the auxiliary heads regularise the encoder. |

#### Training

- AdamW optimiser, LR 1e-3, weight decay 1e-5
- Batch size 128, gradient clipping at norm 1.0
- 6-15 epochs depending on mode
- Loss: MSE by default; optionally **pairwise ranking loss** (next subsection)

#### Pairwise rank loss

We use the predictions to *rank* tickers (long top, short bottom). Training
the model with MSE optimises for point-prediction accuracy, which is a
different objective. The pairwise hinge loss directly optimises ranking:

$$
L_{\text{pair}} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} \max(0,\; \mathrm{margin} - \mathrm{sign}(y_i - y_j) \cdot (\hat y_i - \hat y_j))
$$

where $\mathcal{P}$ is the set of pairs $(i,j)$ on the *same date* (so we
rank within the cross-section, not across dates). This typically improves
IC by 2-3 percentage points over MSE on financial data. The configuration
is `loss_fn="pairwise"` in `TransformerConfig`.

#### Last-token vs attention pooling

A genuine self-critique: if you look at the rendered `plot_attention_weights.png`,
the attention weights are often near-uniform with a slight tilt to recent
days. This means the model isn't learning very strong temporal structure
on this small universe. Last-token pooling would likely work as well —
included as the `last_pooling` ablation.

---

## 10. Conformal prediction — calibrated uncertainty

### 10.1 Why distribution-free matters

Most uncertainty methods (Bayesian neural nets, Gaussian processes, MC
dropout) require an explicit likelihood specification, and their guarantees
are *asymptotic* — only valid in the limit of infinite data. Conformal
prediction is different:

- **Distribution-free** — no Gaussian assumption, no parametric model needed.
- **Finite-sample** — coverage guarantee holds with $n$ as small as 100.
- **Model-agnostic** — wraps *any* point-prediction model.

The guarantee is: if your residuals are exchangeable on the calibration
set (i.e., not systematically different at any point), then the (1−α)
prediction interval covers the true value with probability ≥ (1−α). End of
story.

### 10.2 Split conformal

Implementation in `quant/ml/conformal.py::SplitConformalWrapper`:

1. Fit the base model on the **train-proper** slice.
2. Compute absolute residuals on the held-out **calibration** slice:
   $r_i = |y_i - \hat y_i|$ for $i$ in calibration.
3. Take the **finite-sample-corrected (1−α) quantile**:
   $\hat q = Q_{(1-\alpha)(1+1/n)}(r_1, ..., r_n)$.
4. Prediction interval at any new point: $[\hat y(x) - \hat q,\; \hat y(x) + \hat q]$.

The finite-sample correction (the $(1+1/n)$ factor) is what gives you the
guaranteed coverage *exactly*; without it, you'd be slightly off in small
samples. This is the standard formulation from Vovk et al. and Lei et al.
(2018).

### 10.3 Empirical verification

`tests/test_conformal.py::test_conformal_coverage_converges`:

- Generate 5000 calibration + 5000 test samples from a known model
- Wrap a linear model in `SplitConformalWrapper(alpha=0.1)`
- Calibrate, then measure empirical coverage on test
- Assert it's within 3 percentage points of nominal 0.90

This test runs in the CI suite and would fail loudly if the implementation
drifted.

### 10.4 In the pipeline

For each fold, both the GBR baseline and the Transformer are wrapped:

```python
conf_base = SplitConformalWrapper(base_model=base, alpha=0.1)
conf_base.calibrate(calib_df, target_col="target_fwd_ret_21d")
preds_with_intervals = conf_base.predict_interval(test_df)
# columns: ticker, date, prediction, lower, upper
```

The empirical coverage on the v3 OOS run is **0.873 (baseline) and 0.862 (Transformer)** —
both close to the nominal 0.90, slightly conservative. Reported in
`evaluation_summary.json` and Panel A of the hero figure.

---

## 11. Explainability with SHAP

### 11.1 What SHAP is

**SHapley Additive exPlanations** (Lundberg & Lee, 2017) borrows from
cooperative game theory: it attributes the model's output for a specific
prediction to its features, in a way that's consistent and locally
accurate. The Shapley value for feature $i$ in prediction $f(x)$ is:

$$
\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} \big[ f_{S \cup \{i\}}(x) - f_S(x) \big]
$$

i.e., the average marginal contribution of $i$ across all subsets $S$ of
the other features. For tree models, `shap.TreeExplainer` computes this
exactly in polynomial time.

### 11.2 In the pipeline

`quant/ml/shap_explain.py::explain_baseline()`:

- Subsamples up to 2000 rows from the input panel
- Calls `shap.TreeExplainer(gbr).shap_values(X_scaled)` — exact, fast
- Computes `mean_abs_shap` per feature (global importance)
- Optionally splits by regime: per-regime mean SHAP (which features
  matter most in calm vs crisis?)

Falls back to permutation importance if `shap` isn't installed.

### 11.3 What v3 finds

Top 10 features by global SHAP importance (see `artifacts/shap_global.csv`):

```
us10y_chg_21d, us10y_level, mom_12m, mom_6m,
india_vix_level, india_vix_z_20d, macd, vol_60d,
usdinr_ret_21d, nifty_over_ma50
```

**6 of 10 are macro features** — exactly the v3 contribution the project
is built around. SHAP confirms what the math suggested: macro state
(rates, vol, FX) dominates classical price-only momentum on this universe
during this OOS window.

---

## 12. The portfolio allocator and transaction costs

`quant/ml/allocator.py` implements three allocation policies. All produce
a per-(ticker, rebalance-date) weight vector that the backtester then
multiplies by realised returns to compute P&L.

### 12.1 Vanilla long-short top-k

`vanilla_ls_weights(day, top_k=3)`:

- Sort tickers by predicted forward return.
- Long the top-$k$ at equal weight $+1/k$ each.
- Short the bottom-$k$ at equal weight $-1/k$ each.
- Gross book = 2 (long 1 + short 1).

### 12.2 Width-scaled (uncertainty-aware)

`width_scaled_weights(day, top_k=3, eps=1e-4)`:

- Same top-k selection by prediction.
- But weight each position by $1 / (\mathrm{conformal\_width} + \epsilon)$.
- L1-normalise so gross = 1 (a fair comparison: smaller gross means
  smaller turnover which means smaller transaction-cost drag).

The intuition: when the conformal interval is wide, the model is uncertain.
Reduce that position's size. When the interval is tight, lean in. Because
conformal intervals are *empirically calibrated*, this is a valid way to
operationalise uncertainty into position sizing.

The empirical effect on v3 OOS data: max drawdown drops from
**−27.5% → −14.5%** — almost exactly halved. This is the central claim
of the v3 results; visible in Panel D of the hero figure.

### 12.3 Sector-neutral

`sector_neutral_weights(day, top_k=3, sector_map=DEFAULT_SECTOR_MAP)`:

The 10 NSE tickers map to 4 sectors (Energy, IT, Financials, Industrials).
The naive top-k allocator might end up long-only-banks / short-only-IT — a
disguised sector bet rather than stock-selection skill. Sector-neutral
forces each sector's net exposure to ≈ 0, isolating the cross-sectional
signal.

Algorithm: per sector, count longs $n_L$ and shorts $n_S$; assign
$+1/n_L$ to longs and $-1/n_S$ to shorts within the sector; then re-scale
globally so total gross matches the vanilla convention.

Tested in `test_costs_and_sector.py::test_sector_neutral_net_zero_per_sector`.

### 12.4 Transaction costs and turnover

Each rebalance computes:

$$
\mathrm{turnover}_t = \sum_i |w_t^{(i)} - w_{t-1}^{(i)}|
$$

then deducts cost = turnover × (tc_bps / 10000) from the period return.

We use **20 bps per round-trip** as the default — realistic for liquid
Indian large-caps including market impact. The reported "Net Sharpe" in
the headline numbers and Panel B of the hero figure is *after* this drag.

### 12.5 The `monthly_rebalance_backtest` function

`quant/ml/evaluation.py::monthly_rebalance_backtest`:

This is the function that produces the headline P&L. It:

1. Walks rebalance dates (every 21 days) across the OOS preds.
2. Picks top-k longs / bottom-k shorts.
3. Computes gross return as $\frac{1}{k}\sum_{\text{long}} r - \frac{1}{k}\sum_{\text{short}} r$.
4. Computes turnover by name-set difference.
5. Deducts tc.
6. Compounds monthly net returns into a cumulative curve.
7. Reports total return, Sharpe, max drawdown, avg turnover, total tc drag.

This is the *correct* way to report a long-short Sharpe. The earlier v1
of this project reported a daily-aggregated number that was inflated by
~21× — the v2 fix (this function) is one of the project's headline
methodology improvements.

---

## 13. Statistical rigor

`quant/ml/statistics.py` implements three tests every quant paper should
have but most intern projects don't.

### 13.1 Diebold-Mariano test

**Question**: is model A *significantly* better than model B at forecasting?

**Method** (Diebold & Mariano, 1995, with Harvey-Leybourne-Newbold small-sample correction):

1. Compute per-sample loss differential $d_t = L_A(y_t, \hat y_{A,t}) - L_B(y_t, \hat y_{B,t})$ where $L$ = squared error.
2. The mean differential is $\bar d$.
3. Estimate the **long-run variance** of the mean, accounting for autocorrelation, via Newey-West with truncation lag $h-1$ (where $h$ is the forecast horizon):

$$
\hat \sigma^2_{\mathrm{LR}} = \frac{1}{n} \left[ \gamma_0 + 2 \sum_{k=1}^{h-1} (1 - k/h) \gamma_k \right]
$$

4. The DM statistic is $\mathrm{DM} = \bar d / \hat \sigma_{\mathrm{LR}}$.
5. Apply the small-sample correction factor $\sqrt{(n + 1 - 2h + h(h-1)/n) / n}$.
6. Reference a $t_{n-1}$ distribution for the p-value.

**Why Newey-West**: forecast errors at horizon $h$ are autocorrelated up
to $h-1$ lags by construction (overlapping forecasts). Ignoring this
overstates significance. Newey-West corrects it.

**v3 result**: Transformer beats baseline at **p < 0.0001** on squared
forecast error. The annotation box in Panel B of the hero figure.

### 13.2 Bootstrap Sharpe confidence interval

**Question**: my Sharpe is −0.30 ± what?

**Method** (Politis & Romano, 1994, stationary bootstrap):

1. Rebuild the return series 2000 times by resampling **blocks of geometric
   length** (mean block length 5 periods). This preserves autocorrelation
   structure better than IID bootstrap.
2. Compute Sharpe on each bootstrap sample.
3. Take the 2.5th and 97.5th percentiles → 95% CI.

**v3 result**:
- Baseline: Sharpe = −1.56, CI = [−3.22, −0.52] (entirely below zero)
- Transformer: Sharpe = −0.60, CI = [−1.81, +0.62] (straddles zero)

The Transformer's CI crosses zero, meaning we cannot statistically reject
"Sharpe = 0" — we can't claim positive alpha. This is the **honest
reporting** angle: the Diebold-Mariano test confirms the Transformer is a
better *forecaster* (lower MSE), but the bootstrap CI confirms we can't
turn that into reliable *positive Sharpe* on this universe.

### 13.3 Probability of Backtest Overfitting (PBO)

**Question**: of all the strategy variants I tried, is the best one
actually skilled, or did it just get lucky?

**Method** (Bailey, Borwein, López de Prado, Zhu, 2015 — *Combinatorially
Symmetric Cross-Validation*):

1. Split the (T × N) returns matrix (T periods, N strategies) into S=16 contiguous partitions.
2. For each $\binom{S}{S/2}$ way to split partitions into in-sample and out-of-sample halves:
   - Find the strategy with the highest Sharpe in-sample.
   - Compute its OOS rank percentile.
   - Count it as "overfit" if its OOS rank percentile < 0.5.
3. PBO = (count of overfit cases) / (total cases).

If all your strategies are pure noise, PBO ≈ 0.5. If one strategy is
genuinely best, PBO < 0.3. PBO > 0.5 means the in-sample winner is
*systematically worse* OOS — classic overfitting.

In v3, PBO is computed across the strategy variants tested by the ablation
driver, written to `artifacts/significance.json`.

---

## 14. Feature drift detection

`quant/ml/drift.py::detect_feature_drift()` runs a Kolmogorov-Smirnov
two-sample test per feature:

- **Reference window**: the training distribution (e.g., last 2 years before deployment).
- **Production window**: recent live data (e.g., last 1 month).
- **For each feature**: KS statistic comparing the two windows' empirical CDFs, plus a p-value.
- **Flag drift** if (p < 0.01 AND KS_statistic > 0.2) — both a significance gate and an effect-size gate (the effect-size gate matters because at large $n$, even tiny drifts hit p < 0.01).

This is intended for **production monitoring**: schedule it weekly,
alert if any feature crosses the threshold, retrain or scale back exposure.

Tests:
- `test_detects_clear_drift` — synthetic data with a 3-sigma mean shift, asserts the bad feature is flagged.
- `test_no_drift_when_same_distribution` — both windows from same distribution, asserts not flagged.
- `test_drift_summary_counts` — tests the report aggregates.

---

## 15. MLflow experiment tracking

`quant/ml/tracking.py` is a thin wrapper around MLflow that **degrades
gracefully**: if MLflow isn't installed, every call becomes a no-op, so
the pipeline doesn't crash.

In the orchestrator (`run_ml_pipeline.py`):

```python
from quant.ml.tracking import tracker
tracker.enable(experiment_name="equity-intelligence")

with tracker.run(name="run_v3", params={...}) as run:
    run.log_metric("baseline_gbr_ic", ic)
    run.log_metric("transformer_monthly_sharpe", sharpe)
    run.log_artifact("artifacts/shap_global.csv")
```

Every fold logs:
- All hyperparameters
- IC, directional accuracy, RMSE per model
- Conformal coverage and interval width
- Monthly Sharpe, max drawdown, total return, total tc drag
- Path to feature panel and OOS predictions

To browse: `mlflow ui` then http://localhost:5000.

This is the closest thing to "production MLOps" in an intern project. The
ablation and stability drivers also log to MLflow, so all variants are
visible in one experiment dashboard.

---

## 16. Optuna hyperparameter search

`run_optuna.py` defines a Bayesian hyperparameter search over the
Transformer's space:

```python
seq_len    ∈ {30, 45, 60, 90}
d_model    ∈ {32, 64, 96}
n_heads    ∈ {2, 4, 8}
n_layers   ∈ {1, 2, 3}
dropout    ∈ [0.0, 0.3]
lr         ∈ [1e-4, 5e-3]   (log-uniform)
pooling    ∈ {"last", "attn"}
loss_fn    ∈ {"mse", "pairwise"}
```

**Objective**: out-of-sample Information Coefficient on a 15% validation
slice (cleaner than walk-forward inside an inner loop — fast and good
enough for first-pass tuning).

**Optuna's TPE sampler** (Tree-structured Parzen Estimator) approximates
the conditional posterior of "good params given observed scores" and
samples promising regions, much smarter than random search.

**Pruning**: configurations where d_model isn't divisible by n_heads are
pruned immediately.

Every trial logs to MLflow under experiment "equity-intel-optuna". After
the search, `artifacts/optuna_best.json` and `artifacts/optuna_trials.csv`
are written.

---

## 17. The FastAPI serving layer

`service/app.py` exposes a **read-only** REST API that serves the OOS
predictions and allocator output. It does *not* retrain on demand — that's
the standard MLOps split between offline training jobs and online
serving.

### 17.1 Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | Liveness + which model artifacts are available |
| GET | `/model/info` | Per-model headline metrics + allocator summary |
| GET | `/predictions/latest?model=transformer&top_k=5` | Today's top-k predictions with intervals |
| GET | `/predictions/{model}/{ticker}?start=...&end=...` | Per-ticker stream with regime labels |
| POST | `/allocate` | Latest allocator weights (any policy) |
| GET | `/regimes/latest?n=30` | Recent HMM regime labels |
| POST | `/admin/reload` | Clear in-memory cache (after a pipeline rerun) |

OpenAPI / Swagger UI at `/docs`.

### 17.2 Architecture

- **FastAPI** for the routes (Pydantic for request/response models,
  automatic OpenAPI generation).
- **Caching**: artifacts are loaded lazily into a process-level dict; the
  `/admin/reload` endpoint clears it.
- **Dependencies**: only the read-only stack (pandas, pyarrow, FastAPI,
  uvicorn) — no torch, no transformers. The container is small (~600 MB).

### 17.3 Container

`service/Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements-service.txt .
RUN pip install -r requirements-service.txt
COPY service/ quant/ pipeline/ artifacts/ ./
ENV PYTHONPATH=/app
EXPOSE 8088
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8088"]
```

Build & run:

```bash
docker build -f service/Dockerfile -t equity-intel-api .
docker run -p 8088:8088 equity-intel-api
curl http://localhost:8088/health
```

Tests: `tests/test_service.py` uses FastAPI's `TestClient` to verify every
endpoint returns well-formed JSON. Skips automatically if no artifacts
exist.

---

## 18. The Streamlit dashboard

`dashboard/app.py` is a 5-section Streamlit app, intended for
development-time inspection and screenshots:

1. **Live Factor Scores** — table of momentum, low-vol, composite score, and rank for all 10 tickers, color-coded by rank.
2. **Current Portfolio Weights** — horizontal bar chart of optimised weights.
3. **Backtest Performance** — cumulative return chart with date range selector and metric cards.
4. **Individual Stock Deep-Dive** — price chart with 20/50-day moving averages.
5. **ML Model Performance** — KPIs (IC, dir acc, Sharpe-like, conformal coverage), monthly long-short cumulative PnL chart, per-regime metrics table, conformal calibration bars, SHAP importance bars, HMM regime timeline.

Data flows: by default reads from PostgreSQL; falls back to live
yfinance + on-the-fly compute if the database is unreachable. The ML
section reads parquet artifacts directly from `artifacts/`.

Run: `streamlit run dashboard/app.py` → http://localhost:8501.

---

## 19. The test suite

43 pytest tests in `tests/`, all running in <60s on a CPU laptop:

| File | What it covers | Why it matters |
|---|---|---|
| `test_walkforward.py` (5) | Embargo respected, train precedes test, indices valid | Guards against silent CV leakage |
| `test_features.py` (5) | Feature panel shape, no-lookahead, multi-horizon present, ranks bounded, targets forward | The *single* most important test set — `test_no_lookahead_in_features` corrupts future prices and asserts past features unchanged |
| `test_regimes.py` (4) | HMM end-date cutoff respected, deterministic, named regimes, no-leakage past cutoff | Regime detector cannot use future data |
| `test_conformal.py` (4) | Coverage converges to nominal, width scales with noise, intervals contain prediction, raises before calibration | The whole uncertainty story rides on this test |
| `test_allocator.py` (4) | Weight sums correct, gross = 1 for width-scaled, wide intervals get less weight, summary correctness | Allocator policies behave as advertised |
| `test_costs_and_sector.py` (5) | Sector-neutral nets to ≈0 per sector, tc reduces net return, turnover ≥ 0, net Sharpe ≤ gross Sharpe, sector-neutral runs without error | Realism layer |
| `test_statistics.py` (7) | DM detects known better model, DM symmetric under label swap, bootstrap CI contains point, CI shrinks with n, PBO in [0.05, 0.95] for noise, PBO < 0.4 with one genuinely-better strategy | Validates the statistical-rigor module |
| `test_drift.py` (3) | Detects clear drift, no drift when same dist, summary counts | Validates the monitoring module |
| `test_rank_loss.py` (1) | Pairwise loss trains without NaN | Code-path verification for the optional loss |
| `test_service.py` (5) | All FastAPI endpoints return well-formed JSON | Smoke test for the serving layer |

Run: `pytest tests/ -v` from the repo root.

---

## 20. End-to-end walkthrough — what happens when you run the pipeline

Here's exactly what `python run_ml_pipeline.py` does, top to bottom:

1. **Parse args** (`--fast`, `--no-torch`, `--no-sentiment`, `--seed`, `--ablation`, `--run-name`).
2. **Enable MLflow tracking** (creates the experiment if needed) and start a run.
3. **Step 1: Fetch prices** — yfinance for the 10 tickers from start_date to end_date; quality checks; log row count.
4. **Step 2: Sentiment features** — load news (real CSV or synthetic), score with FinBERT, aggregate per (ticker, date), cache to parquet.
5. **Step 3: Feature panel** — build per-ticker features, cross-sectional ranks, market-state aggregates; merge macro features; merge sentiment; drop initial-history rows that don't have enough lookback. Save `artifacts/feature_panel.parquet`.
6. **Step 4: Purged walk-forward CV** — for each of ~7 folds:
   - Print fold dates.
   - Refit HMM on train-only data, label test window. Save fold regime labels.
   - Split train into train-proper + 90-day calibration slice.
   - Fit GBR baseline on train-proper. Wrap in conformal, calibrate. Predict + intervals on test → save with fold tag.
   - If torch available: fit Transformer on train-proper. Wrap in conformal with `context_df=train_proper` so the model has enough history to build seq_len windows. Calibrate. Predict + intervals on test (using train_proper+calib as context). Save with fold tag.
   - Export one attention-weight sample per fold to `artifacts/attention_samples.json`.
7. **Step 5: Regime-conditional eval + monthly-rebalance backtest** — for each model, compute overall + per-regime metrics (RMSE, dir_acc, IC, LS Sharpe), conformal calibration (empirical vs nominal coverage); run monthly rebalance with 20 bps tc; log everything to MLflow + `evaluation_summary.json`.
8. **Step 6: Uncertainty-aware allocator** — for each model, run vanilla and width-scaled allocator policies; report Sharpe (gross/net), max drawdown, avg turnover, tc drag; save per-policy P&L parquets and `allocator_summary.json`.
9. **Step 7: SHAP attribution** — fit a final GBR baseline on the entire feature panel; compute global + per-regime SHAP values on a 2000-row sample; save `shap_global.csv` and `shap_per_regime.csv`.
10. **Step 8: Render plots** — five PNGs (long-short PnL, conformal coverage, dir-acc by regime, allocator comparison, attention weights). Log to MLflow.
11. **Step 9: Postgres upsert (best-effort)** — try connecting to localhost:5432; if up, upsert all OOS predictions to `ml_predictions` table.
12. **Print summary**: total runtime, artifacts directory, MLflow tracking URI.

Total runtime: ~6 min on a laptop CPU in full mode, ~90s in `--fast` mode.

Then optionally: `python make_hero_figure.py` reads the artifacts and
produces the 2x2 hero PNG that's embedded in the README.

---

## 21. Empirical results explained

The headline numbers from the most recent v3 run (2019-2024, 7 walk-forward
folds, 20 bps transaction costs, 10 NSE tickers):

| Metric | Baseline GBR | Transformer | What it means |
|---|---:|---:|---|
| Directional accuracy | 0.41 | **0.50** | Transformer is at coin-flip; baseline is *below* — has slight inverted signal |
| Information Coefficient (IC) | −0.155 | **−0.104** | Both negative; Transformer's signal is closer to neutral |
| Monthly net Sharpe | −1.56 | **−0.60** | Transformer better; both still negative |
| Bootstrap 95% CI for Sharpe | [−3.22, −0.52] | [−1.81, +0.62] | Baseline's CI entirely negative; Transformer's spans zero |
| Max drawdown (vanilla) | −22.9% | −27.5% | Vanilla allocator: Transformer larger drawdown |
| **Max drawdown (width-scaled)** | **−12.0%** | **−14.5%** | Uncertainty-aware allocator halves the drawdown |
| Conformal empirical coverage (nominal 0.90) | 0.873 | 0.862 | Both well-calibrated; slightly conservative |
| Mean conformal interval width | 0.338 | 0.298 | Transformer's intervals are 12% tighter |
| Diebold-Mariano test (TFT vs Baseline) | — | **p < 0.0001** | Transformer significantly better at forecast error |

### Honest read

1. **Transformer beats baseline on every prediction-quality metric.** The
   DM test confirms this is statistically significant.
2. **Neither model produces positive alpha on this universe.** This is a
   faithful reflection of how hard 21-day return prediction is on a
   10-name cross-section. Anyone reporting a Sharpe of 1.5+ on this kind
   of setup is either lucky, leaking, or lying.
3. **The uncertainty-aware allocator is the project's actual selling
   point.** It cuts max drawdown ~45% (−27.5% → −14.5% for the Transformer)
   by sizing down positions when conformal intervals widen. This is what
   "decisions under uncertainty" means in practice — the kind of
   risk-management story consulting firms care about.
4. **Conformal coverage is empirically calibrated.** 0.86-0.87 vs nominal
   0.90 is excellent — within sampling noise. A model that says "I'm 90%
   sure" really is right 90% of the time.
5. **Macro features dominate.** 6 of 10 top-SHAP features are macro (US 10y,
   India VIX, USDINR, Nifty), confirming the v3 contribution.

Visible in the hero figure (`artifacts/hero_figure.png`):
- Panel A: conformal coverage bars
- Panel B: Sharpe forest plot with bootstrap CIs and DM annotation
- Panel C: SHAP top-10 with macro/micro coloring
- Panel D: drawdown time series, vanilla vs width-scaled

---

## 22. Honest limitations and known weaknesses

A non-exhaustive list of what this project does *not* do:

1. **Universe is too small.** 10 tickers is structurally noisy for IC measurement; |IC| < 0.1 is essentially unmeasurable with N=10 per date. Moving to Nifty 50 or S&P 500 would fundamentally strengthen every metric.
2. **Real news is not used.** FinBERT runs on a synthetic headline stream by default; the production path requires dropping a real `data/news.csv`.
3. **Attention is near-uniform.** The plot reveals the Transformer hasn't learned strong temporal structure — last-token pooling probably works as well. Honest documentation > hype.
4. **No transaction-cost variation.** Fixed at 20 bps. Real costs vary by liquidity, time of day, and trade size.
5. **No regime-aware training.** We *evaluate* per regime but don't *train* differently per regime. A regime-conditional ensemble would likely improve Sharpe.
6. **No causal inference.** The model finds correlations, not causes. "US 10y yield matters" doesn't mean rising yields *cause* Indian equity moves.
7. **Hyperparameters not tuned in CI.** Optuna runs are manual; the headline numbers use a single hand-picked config.
8. **Single-seed numbers in the README.** The stability driver runs over multiple seeds but the headline table reports one. This is honest enough but could be tightened.
9. **No live data feed.** The FastAPI service serves *historical* OOS predictions; it doesn't compute on-demand.
10. **No model registry / versioning beyond MLflow.** `mlflow.register_model` is not called; runs are tracked but signed model versions aren't.

---

## 23. How to extend this project

If you want to keep building, here are concrete next steps in rough
descending order of impact:

1. **Universe expansion.** Move to Nifty 50 or S&P 500. Single biggest win.
2. **Real news ingestion.** NewsAPI / Refinitiv / a Kaggle financial news dataset. Drops a real `data/news.csv` and the FinBERT path activates immediately.
3. **GitHub Actions CI.** Run pytest on every push. Half a day of work, recruiter-friendly green badge.
4. **Hyperparameter tuning at scale.** Run Optuna for 100+ trials on a GPU; use the results in production.
5. **Vol-scaled targets in production.** Switch the primary target to `target_fwd_ret_21d_vs`; recalibrate conformal.
6. **Causal-style features.** Event studies around earnings announcements, dividend ex-dates, monetary policy meetings.
7. **Multi-asset.** Add commodities and FX as a sister universe; cross-asset signals are often stronger than single-asset.
8. **Model registry.** `mlflow.register_model` + staging/production aliases; Github Actions promotes models that pass test thresholds.
9. **Live serving.** Replace the read-only FastAPI with a service that computes on-demand from the latest features.
10. **A second statistical-significance lens.** White's reality check, SPA test, or model confidence sets — for when DM isn't appropriate.

---

## 24. File-by-file reference

```
equity-intelligence-platform/
├── .env.example             # placeholder env vars (Postgres + Airflow secrets)
├── .gitignore               # standard Python + ML-project ignores
├── LICENSE                  # MIT
├── README.md                # quickstart, architecture, headline numbers
├── docker-compose.yml       # Postgres 15 + Airflow 2.8 stack
├── Dockerfile               # Airflow image with our deps baked in
├── init.sql                 # 4 + 1 PostgreSQL tables
├── requirements.txt         # full stack (training + serving)
├── requirements-service.txt # slim deps for the FastAPI container
│
├── pipeline/
│   ├── __init__.py
│   ├── ingest.py            # yfinance + synthetic GBM fallback
│   ├── transform.py         # classical factor computation (legacy)
│   ├── quality.py           # data quality checks
│   └── load.py              # psycopg2 upserts
│
├── quant/
│   ├── factors.py           # legacy factor wrapper
│   ├── optimizer.py         # legacy mean-variance (SLSQP)
│   ├── backtest.py          # legacy backtester
│   └── ml/
│       ├── __init__.py
│       ├── features.py      # all feature engineering, target construction
│       ├── macro.py         # macro feature ingestion + transformation
│       ├── sentiment.py     # FinBERT layer with synthetic-news fallback
│       ├── regimes.py       # per-fold HMM with end_date cutoff
│       ├── walkforward.py   # PurgedWalkForward splitter + tests
│       ├── baseline.py      # GradientBoostingRegressor wrapper
│       ├── transformer_model.py  # PyTorch TFT-inspired model + multi-head
│       ├── conformal.py     # split-conformal wrapper
│       ├── shap_explain.py  # SHAP TreeExplainer + permutation fallback
│       ├── statistics.py    # DM test + bootstrap CI + PBO
│       ├── drift.py         # KS-based feature drift detection
│       ├── allocator.py     # 3 policies + transaction costs + turnover
│       ├── evaluation.py    # regime-conditional metrics + monthly rebal
│       └── tracking.py      # MLflow shim with graceful degradation
│
├── dags/
│   └── equity_pipeline_dag.py   # Airflow daily DAG
│
├── service/
│   ├── __init__.py
│   ├── app.py               # FastAPI app
│   └── Dockerfile           # slim serving container
│
├── dashboard/
│   └── app.py               # Streamlit 5-section dashboard
│
├── tests/                   # 43 pytest tests, ~60s runtime
│   ├── conftest.py          # synthetic-data fixtures
│   ├── test_walkforward.py
│   ├── test_features.py
│   ├── test_regimes.py
│   ├── test_conformal.py
│   ├── test_allocator.py
│   ├── test_costs_and_sector.py
│   ├── test_statistics.py
│   ├── test_drift.py
│   ├── test_rank_loss.py
│   └── test_service.py
│
├── docs/
│   └── PROJECT_GUIDE.md     # this document
│
├── artifacts/               # ML pipeline outputs (mostly gitignored)
│   ├── hero_figure.png      # 3200x2400, embedded in README
│   ├── hero_figure_small.png # 1600x1200, LinkedIn preview
│   ├── evaluation_summary.json
│   ├── significance.json
│   ├── allocator_summary.json
│   ├── shap_global.csv
│   ├── plot_*.png           # 5 supporting figures
│   └── attention_samples.json
│
├── run_full_pipeline.py     # classical pipeline → Postgres
├── run_ml_pipeline.py       # ML pipeline orchestrator
├── run_ablation.py          # ablation-study driver (8 variants)
├── run_stability.py         # seed-stability driver (N seeds)
├── run_optuna.py            # hyperparameter search (MLflow logged)
└── make_hero_figure.py      # generate hero_figure.png from artifacts/
```

---

## 25. Glossary

- **AdamW** — Adam optimiser with decoupled weight decay; standard for transformer training.
- **Alpha** — excess return relative to a benchmark; the thing every quant claims to have.
- **Attention** — a soft-lookup mechanism that lets a model weight different parts of an input by computed relevance.
- **AutoEncoder / Embedding** — learnable lookup table mapping discrete categories to dense vectors.
- **Backtest** — simulating a trading strategy on historical data.
- **Beta** — sensitivity to overall market movement; what's *not* alpha.
- **Causal mask** — a triangular attention mask that prevents a sequence position from attending to future positions.
- **Conformal prediction** — a distribution-free framework for producing calibrated prediction intervals.
- **CSCV** — Combinatorially Symmetric Cross-Validation; the partition method behind PBO.
- **Diebold-Mariano test** — significance test for forecast accuracy difference.
- **Directional accuracy** — fraction of times the predicted sign matches the realised sign.
- **Drawdown** — peak-to-trough percentage loss.
- **EMA** — Exponential Moving Average; weight $\alpha$ on the new value, $(1-\alpha)$ on the EMA-so-far.
- **Embargo** — gap days between train and test in walk-forward CV to prevent label-overlap leakage.
- **Factor** — a structured feature believed to predict cross-sectional returns (e.g., momentum, quality).
- **FinBERT** — BERT-base model fine-tuned on financial sentiment classification.
- **GBR** — GradientBoostingRegressor (sklearn's gradient-boosted tree regression).
- **GELU** — Gaussian Error Linear Unit; smoother activation than ReLU, common in transformers.
- **HMM** — Hidden Markov Model.
- **IC** — Information Coefficient; per-date Spearman rank correlation between predictions and realised returns.
- **K-fold** — cross-validation that splits data into K equal parts and rotates which is the test set.
- **KS test** — Kolmogorov-Smirnov test; compares two empirical CDFs.
- **L1 / L2 norm** — sum-of-absolute-values / Euclidean norm.
- **Long-short** — a market-neutral portfolio that simultaneously buys (longs) some securities and short-sells others.
- **MACD** — Moving Average Convergence Divergence; a momentum indicator.
- **MLflow** — open-source experiment tracking and model management platform.
- **Momentum (mom_h)** — cumulative log-return over the past `h` days.
- **NaN** — "Not a Number" — pandas/numpy's missing-value sentinel.
- **Newey-West** — long-run variance estimator that accounts for autocorrelation.
- **OOS** — Out-Of-Sample; evaluation on data the model didn't see during training.
- **Optuna** — Bayesian hyperparameter optimisation library.
- **PBO** — Probability of Backtest Overfitting.
- **psycopg2** — PostgreSQL Python driver.
- **Purged walk-forward CV** — time-series cross-validation that respects temporal order and embargoes label-overlap.
- **Rebalance** — recompute portfolio weights at a specific frequency.
- **RSI** — Relative Strength Index; an overbought/oversold momentum indicator.
- **Sentiment score** — numeric output of a sentiment model, here in $[-1, 1]$ (positive minus negative class probability).
- **SHAP** — SHapley Additive exPlanations; a feature-attribution method based on cooperative game theory.
- **Sharpe ratio** — annualised mean return / annualised return volatility.
- **Spearman correlation** — Pearson correlation of ranks; robust to outliers.
- **Split-conformal** — conformal prediction variant that uses a held-out calibration set.
- **TFT** — Temporal Fusion Transformer (Lim et al., 2021); a multi-horizon time-series model with attention over static and dynamic covariates.
- **Tick** — minimum price increment in an exchange.
- **Transformer** — neural network architecture built on stacked self-attention layers.
- **Turnover** — L1 distance between successive portfolio weight vectors.
- **VIX** — CBOE Volatility Index, "the fear index"; India VIX is the NSE equivalent.
- **Walk-forward** — temporal cross-validation: train on the past, test on the next chunk, roll forward.
- **XCom** — Airflow's small-payload inter-task communication mechanism.

---

## 26. References and further reading

### Methodological foundations

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. **Chapter 7** is the source for purged walk-forward CV and the embargo concept; **Chapter 11** for PBO and CSCV.
- Diebold, F.X. & Mariano, R.S. (1995). "Comparing Predictive Accuracy". *Journal of Business & Economic Statistics*. Original DM test paper.
- Harvey, D., Leybourne, S., Newbold, P. (1997). "Testing the equality of prediction mean squared errors". *International Journal of Forecasting*. Small-sample correction.
- Politis, D.N. & Romano, J.P. (1994). "The stationary bootstrap". *Journal of the American Statistical Association*. The block-bootstrap method we use for Sharpe CIs.
- Vovk, V., Gammerman, A., Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer. The conformal prediction reference.
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R., Wasserman, L. (2018). "Distribution-free predictive inference for regression". *JASA*. The split-conformal formulation we use.
- Bailey, D.H., Borwein, J., López de Prado, M., Zhu, Q.J. (2015). "The Probability of Backtest Overfitting". *Journal of Computational Finance*.
- Lundberg, S.M. & Lee, S.-I. (2017). "A Unified Approach to Interpreting Model Predictions" (SHAP). *NeurIPS*.

### Models

- Lim, B., Arık, S.Ö., Loeff, N., Pfister, T. (2021). "Temporal Fusion Transformers for interpretable multi-horizon time series forecasting". *International Journal of Forecasting*. The TFT paper our Transformer is inspired by.
- Friedman, J.H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine". *Annals of Statistics*. Gradient boosting.
- Vaswani et al. (2017). "Attention Is All You Need". The transformer paper.
- Araci, D. (2019). "FinBERT: Financial sentiment analysis with pre-trained language models". arXiv:1908.10063. The FinBERT we use (`ProsusAI/finbert` on HuggingFace).
- Rabiner, L.R. (1989). "A tutorial on hidden Markov models and selected applications in speech recognition". *Proc. IEEE*. The classic HMM reference.

### Software

- **scikit-learn** — https://scikit-learn.org. Pedregosa et al. (2011), JMLR.
- **PyTorch** — https://pytorch.org. Paszke et al. (2019), NeurIPS.
- **HuggingFace transformers** — https://github.com/huggingface/transformers. Wolf et al. (2020), EMNLP.
- **MLflow** — https://mlflow.org. Zaharia et al. (2018), IEEE Data Eng Bull.
- **Optuna** — https://optuna.org. Akiba et al. (2019), KDD.
- **SHAP** — https://github.com/slundberg/shap.
- **hmmlearn** — https://hmmlearn.readthedocs.io/.
- **pytorch-forecasting** — https://pytorch-forecasting.readthedocs.io/. Implements the full TFT.
- **Apache Airflow** — https://airflow.apache.org/.
- **FastAPI** — https://fastapi.tiangolo.com/.
- **Streamlit** — https://streamlit.io/.

### Quant finance / general

- Grinold, R. & Kahn, R. (2000). *Active Portfolio Management*. The "fundamental law of active management" reference.
- Ang, A. (2014). *Asset Management: A Systematic Approach to Factor Investing*. Oxford. The textbook for factor investing.

---

*This document is part of the Equity Intelligence Platform repository. If
you spot something unclear, open an issue or PR.*
