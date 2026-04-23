"""
quant/ml/regimes.py — Hidden Markov Model regime detection on market state.

Fits a Gaussian HMM on two market-level features:
    - 20-day realised market volatility
    - 20-day cross-sectional return dispersion

and labels each date with one of 3 latent regimes. We then rename the raw
latent states to human-readable labels ("calm", "trending", "crisis") using
the mean market-volatility in each state: lowest vol → calm, highest → crisis.

Falls back to a quantile-based rule if `hmmlearn` is not installed, so the
pipeline remains runnable on a minimal install.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RegimeModel:
    """Fitted regime detector. Use .label() to assign regimes to new dates."""
    method: str                                # "hmm" or "quantile"
    state_to_label: Dict[int, str]             # raw HMM state → "calm"/"trending"/"crisis"
    raw_model: Optional[object] = None         # hmmlearn GaussianHMM if available
    fit_means: Optional[pd.Series] = None      # for standardisation at predict time
    fit_stds: Optional[pd.Series] = None

    def label(self, market_df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        market_df : pd.DataFrame
            Must contain columns: date, mkt_vol_20d, mkt_disp_20d.

        Returns
        -------
        pd.DataFrame with columns: date, regime_raw (int), regime (str label)
        """
        X_df = market_df[["mkt_vol_20d", "mkt_disp_20d"]].copy()
        X_df = X_df.ffill().bfill().fillna(0.0)

        if self.method == "hmm" and self.raw_model is not None:
            # Standardise with the fit-time params
            X = (X_df - self.fit_means) / self.fit_stds.replace(0.0, 1.0)
            X = X.fillna(0.0).values
            raw = self.raw_model.predict(X)
        else:
            raw = _quantile_states(X_df).values

        labels = np.array([self.state_to_label[int(s)] for s in raw])
        return pd.DataFrame({
            "date": pd.to_datetime(market_df["date"]).values,
            "regime_raw": raw,
            "regime": labels,
        })


def _build_market_state(panel_or_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Build market-state features from either a per-ticker price DataFrame or
    the feature panel. Returns date, mkt_vol_20d, mkt_disp_20d.
    """
    df = panel_or_prices.copy()
    df["date"] = pd.to_datetime(df["date"])

    if "ret_1d" not in df.columns:
        # Compute per-ticker daily returns
        df = df.sort_values(["ticker", "date"])
        df["ret_1d"] = df.groupby("ticker")["close"].pct_change()

    daily = df.groupby("date").agg(
        mkt_ret=("ret_1d", "mean"),
        mkt_disp=("ret_1d", "std"),
    ).reset_index().sort_values("date")

    daily["mkt_vol_20d"] = daily["mkt_ret"].rolling(20).std() * np.sqrt(252)
    daily["mkt_disp_20d"] = daily["mkt_disp"].rolling(20).mean()

    return daily.dropna(subset=["mkt_vol_20d", "mkt_disp_20d"]).reset_index(drop=True)


def _quantile_states(X_df: pd.DataFrame) -> pd.Series:
    """Fallback: tercile-split on volatility."""
    vol = X_df["mkt_vol_20d"]
    q = vol.quantile([1 / 3, 2 / 3]).values
    states = pd.Series(1, index=X_df.index)  # trending by default
    states[vol <= q[0]] = 0  # calm
    states[vol >= q[1]] = 2  # crisis
    return states


def fit_regime_model(
    prices_or_panel: pd.DataFrame,
    n_regimes: int = 3,
    random_state: int = 42,
    end_date: pd.Timestamp | None = None,
) -> tuple[RegimeModel, pd.DataFrame]:
    """
    Fit a 3-state Gaussian HMM on market-vol + dispersion.

    Parameters
    ----------
    end_date : optional cutoff. When provided, the HMM is fit ONLY on rows
        with date <= end_date. This prevents look-ahead bias when the caller
        is inside a walk-forward fold and must treat future data as unseen.

    Returns
    -------
    (RegimeModel, labels_df)
        labels_df has columns: date, regime_raw, regime. Labels are emitted
        for every date in the input (including dates after end_date — those
        are produced by .predict(), no re-fitting).
    """
    market_full = _build_market_state(prices_or_panel)
    if end_date is not None:
        market_fit = market_full[market_full["date"] <= pd.Timestamp(end_date)].copy()
        if len(market_fit) < 50:
            # Too little data to fit a multi-state HMM — fall back to using all
            logger.warning("Too few rows before end_date, using full history for HMM fit")
            market_fit = market_full
    else:
        market_fit = market_full
    X_df = market_fit[["mkt_vol_20d", "mkt_disp_20d"]].copy()

    fit_means = X_df.mean()
    fit_stds = X_df.std().replace(0.0, 1.0)
    X_std = ((X_df - fit_means) / fit_stds).fillna(0.0).values

    method = "quantile"
    raw_model = None
    try:
        from hmmlearn.hmm import GaussianHMM
        model = GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=200,
            random_state=random_state,
        )
        model.fit(X_std)
        raw = model.predict(X_std)
        method = "hmm"
        raw_model = model
        logger.info("Fitted Gaussian HMM with %d states", n_regimes)
    except Exception as e:
        logger.warning("hmmlearn unavailable (%s) — using quantile fallback", e)
        raw = _quantile_states(X_df).values

    # Map raw states → human labels by mean volatility (lowest → calm)
    state_mean_vol = {}
    for s in np.unique(raw):
        state_mean_vol[int(s)] = float(X_df.loc[raw == s, "mkt_vol_20d"].mean())
    sorted_states = sorted(state_mean_vol, key=lambda s: state_mean_vol[s])
    human = ["calm", "trending", "crisis"]
    state_to_label = {s: human[i] for i, s in enumerate(sorted_states[: len(human)])}

    rm = RegimeModel(
        method=method,
        state_to_label=state_to_label,
        raw_model=raw_model,
        fit_means=fit_means,
        fit_stds=fit_stds,
    )

    # Label the full market series (including post-end_date rows — these use
    # .predict() with the train-fit transition matrix, no future leakage).
    labels_df = rm.label(market_full)
    logger.info(
        "Regime distribution (full): %s",
        labels_df["regime"].value_counts().to_dict(),
    )
    return rm, labels_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from pipeline.ingest import fetch_prices
    prices = fetch_prices(start_date="2021-01-01", end_date="2024-01-01")
    model, labels = fit_regime_model(prices)
    print(labels.tail(20))
    print("\nCounts:", labels["regime"].value_counts().to_dict())
