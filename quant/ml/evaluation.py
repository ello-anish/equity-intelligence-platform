"""
quant/ml/evaluation.py — Regime-conditional performance + conformal calibration.

Given per-sample predictions (optionally with conformal lower/upper bounds),
realised forward returns, and regime labels, compute:

  1. Per-regime prediction quality:
       - RMSE
       - Directional accuracy (sign match)
       - Information Coefficient (Spearman rank-corr across the cross-section)
  2. Per-regime long-short portfolio performance (top-3 minus bottom-3):
       - Mean forward return
       - Sharpe-style ratio (mean / std)
  3. Per-regime conformal calibration (empirical coverage vs. nominal).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = (y_true != 0)
    if mask.sum() == 0:
        return float("nan")
    return float((np.sign(y_true[mask]) == np.sign(y_pred[mask])).mean())


def _information_coefficient(df: pd.DataFrame) -> float:
    """Average per-date Spearman corr between predictions and realised returns."""
    corrs = []
    for _, grp in df.groupby("date"):
        if len(grp) < 3:
            continue
        # Spearman via rank-Pearson
        r1 = grp["prediction"].rank()
        r2 = grp["target_fwd_ret"].rank()
        if r1.std() == 0 or r2.std() == 0:
            continue
        corrs.append(float(np.corrcoef(r1, r2)[0, 1]))
    if not corrs:
        return float("nan")
    return float(np.mean(corrs))


def _long_short_returns(df: pd.DataFrame, top_k: int = 3, target: str = "target_fwd_ret") -> pd.DataFrame:
    """Per-date long-short return: long top-k, short bottom-k by prediction."""
    rows = []
    for date, grp in df.groupby("date"):
        if len(grp) < 2 * top_k:
            continue
        grp = grp.sort_values("prediction", ascending=False)
        long_ret = grp.head(top_k)[target].mean()
        short_ret = grp.tail(top_k)[target].mean()
        rows.append({"date": date, "ls_return": float(long_ret - short_ret)})
    return pd.DataFrame(rows)


def monthly_rebalance_backtest(
    preds_df: pd.DataFrame,
    top_k: int = 3,
    target_col: str = "target_fwd_ret_21d",
    horizon_days: int = 21,
    tc_bps: float = 0.0,
) -> dict:
    """
    Honest long-short backtest: pick one signal per `horizon_days`, hold the
    portfolio for `horizon_days`, then rebalance. Avoids the overlap inflation
    that plagues naive daily-aggregated long-short PnL.

    Parameters
    ----------
    tc_bps : one-way transaction-cost in bps deducted from the L1 turnover
        between successive rebalances. 0 = frictionless (gross), 20 ≈ realistic
        Indian equity round-trip cost.

    Returns a dict with:
        - pnl_df (date, gross_return, turnover, tc, ls_return, cum, cum_gross)
        - total_return (net)
        - sharpe (net), sharpe_gross
        - max_drawdown (net)
        - avg_turnover, total_tc_drag
        - n_periods, horizon_days, top_k, tc_bps
    """
    if preds_df.empty:
        return {"error": "no predictions"}
    df = preds_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "prediction"])

    unique_dates = sorted(df["date"].unique())
    if not unique_dates:
        return {"error": "no dates"}
    rebal_dates = []
    last = None
    for d in unique_dates:
        if last is None or (d - last).days >= horizon_days:
            rebal_dates.append(d)
            last = d

    target = target_col if target_col in df.columns else "target_fwd_ret"

    rows = []
    prev_longs: set = set()
    prev_shorts: set = set()
    eq_long = 1.0 / top_k
    eq_short = 1.0 / top_k

    for d in rebal_dates:
        day = df[df["date"] == d]
        if len(day) < 2 * top_k:
            continue
        sorted_day = day.sort_values("prediction", ascending=False)
        longs_df = sorted_day.head(top_k)
        shorts_df = sorted_day.tail(top_k)
        longs = longs_df[target].mean()
        shorts = shorts_df[target].mean()
        gross = float(longs - shorts)

        # Turnover = L1 distance in weights between new and old book.
        new_longs = set(longs_df["ticker"])
        new_shorts = set(shorts_df["ticker"])
        cur_w = {**{t: eq_long for t in new_longs}, **{t: -eq_short for t in new_shorts}}
        prev_w = {**{t: eq_long for t in prev_longs}, **{t: -eq_short for t in prev_shorts}}
        all_t = set(cur_w) | set(prev_w)
        turn = sum(abs(cur_w.get(t, 0.0) - prev_w.get(t, 0.0)) for t in all_t)
        tc = turn * (tc_bps / 10000.0)
        net = gross - tc

        rows.append({
            "date": d,
            "gross_return": gross,
            "turnover": float(turn),
            "tc": float(tc),
            "ls_return": net,
        })
        prev_longs, prev_shorts = new_longs, new_shorts

    pnl = pd.DataFrame(rows)
    if pnl.empty:
        return {"error": "empty pnl"}

    pnl = pnl.sort_values("date").reset_index(drop=True)
    pnl["cum"] = (1 + pnl["ls_return"]).cumprod()
    pnl["cum_gross"] = (1 + pnl["gross_return"]).cumprod()

    n = len(pnl)
    total_ret = float(pnl["cum"].iloc[-1] - 1.0)
    periods_per_year = 252 / horizon_days
    ann_ret = (1 + total_ret) ** (periods_per_year / n) - 1 if n > 0 else 0.0
    std = pnl["ls_return"].std()
    sharpe = float(pnl["ls_return"].mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0
    std_g = pnl["gross_return"].std()
    sharpe_g = float(pnl["gross_return"].mean() / std_g * np.sqrt(periods_per_year)) if std_g > 0 else 0.0
    cummax = pnl["cum"].cummax()
    dd = ((pnl["cum"] - cummax) / cummax).min()

    return {
        "pnl_df": pnl,
        "total_return": total_ret,
        "annualised_return": float(ann_ret),
        "sharpe": sharpe,
        "sharpe_gross": sharpe_g,
        "max_drawdown": float(dd),
        "avg_turnover": float(pnl["turnover"].mean()),
        "total_tc_drag": float(pnl["tc"].sum()),
        "n_periods": int(n),
        "horizon_days": horizon_days,
        "top_k": top_k,
        "tc_bps": float(tc_bps),
    }


@dataclass
class EvaluationReport:
    overall: pd.DataFrame
    per_regime: pd.DataFrame
    long_short: pd.DataFrame
    calibration: Optional[pd.DataFrame] = None


def evaluate_predictions(
    preds_df: pd.DataFrame,
    regimes_df: Optional[pd.DataFrame] = None,
    nominal_coverage: float = 0.90,
    top_k: int = 3,
) -> EvaluationReport:
    """
    Parameters
    ----------
    preds_df : columns must include ticker, date, prediction, target_fwd_ret.
               Optionally lower, upper (conformal bounds).
    regimes_df : columns date, regime (optional).
    """
    df = preds_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Back-compat: some callers pass `target_fwd_ret_21d` instead of `target_fwd_ret`
    if "target_fwd_ret" not in df.columns and "target_fwd_ret_21d" in df.columns:
        df["target_fwd_ret"] = df["target_fwd_ret_21d"]

    if regimes_df is not None:
        r = regimes_df.copy()
        r["date"] = pd.to_datetime(r["date"])
        df = df.merge(r[["date", "regime"]], on="date", how="left")
    else:
        df["regime"] = "all"

    # Overall metrics
    overall_row = {
        "regime": "overall",
        "n_samples": int(len(df)),
        "rmse": float(np.sqrt(np.mean((df["target_fwd_ret"] - df["prediction"]) ** 2))),
        "dir_acc": _directional_accuracy(
            df["target_fwd_ret"].to_numpy(dtype=float),
            df["prediction"].to_numpy(dtype=float),
        ),
        "ic": _information_coefficient(df),
    }
    ls = _long_short_returns(df, top_k=top_k)
    if len(ls) > 0:
        mu, sd = float(ls["ls_return"].mean()), float(ls["ls_return"].std())
        overall_row["ls_mean"] = mu
        overall_row["ls_sharpe_like"] = (mu / sd * np.sqrt(252 / 21)) if sd > 0 else 0.0
    overall = pd.DataFrame([overall_row])

    # Per-regime metrics
    per_regime_rows = []
    ls_per_regime_rows = []
    for regime_name, grp in df.groupby("regime"):
        if len(grp) < 10:
            continue
        rmse = float(np.sqrt(np.mean((grp["target_fwd_ret"] - grp["prediction"]) ** 2)))
        dacc = _directional_accuracy(
            grp["target_fwd_ret"].to_numpy(dtype=float),
            grp["prediction"].to_numpy(dtype=float),
        )
        ic = _information_coefficient(grp)
        ls_r = _long_short_returns(grp, top_k=top_k)
        if len(ls_r) > 0:
            mu, sd = float(ls_r["ls_return"].mean()), float(ls_r["ls_return"].std())
            ls_mean = mu
            ls_sharpe = (mu / sd * np.sqrt(252 / 21)) if sd > 0 else 0.0
            ls_per_regime_rows.append(ls_r.assign(regime=regime_name))
        else:
            ls_mean, ls_sharpe = float("nan"), float("nan")
        per_regime_rows.append({
            "regime": regime_name,
            "n_samples": int(len(grp)),
            "rmse": rmse,
            "dir_acc": dacc,
            "ic": ic,
            "ls_mean": ls_mean,
            "ls_sharpe_like": ls_sharpe,
        })
    per_regime = pd.DataFrame(per_regime_rows).sort_values("regime").reset_index(drop=True)
    long_short = (
        pd.concat(ls_per_regime_rows, ignore_index=True)
        if ls_per_regime_rows else pd.DataFrame(columns=["date", "ls_return", "regime"])
    )

    # Conformal calibration (if intervals present)
    calibration = None
    if {"lower", "upper"}.issubset(df.columns):
        rows = []
        for regime_name, grp in df.groupby("regime"):
            inside = ((grp["target_fwd_ret"] >= grp["lower"]) & (grp["target_fwd_ret"] <= grp["upper"])).mean()
            mean_width = float((grp["upper"] - grp["lower"]).mean())
            rows.append({
                "regime": regime_name,
                "n_samples": int(len(grp)),
                "nominal_coverage": nominal_coverage,
                "empirical_coverage": float(inside),
                "mean_interval_width": mean_width,
            })
        # Overall calibration row
        inside_all = ((df["target_fwd_ret"] >= df["lower"]) & (df["target_fwd_ret"] <= df["upper"])).mean()
        rows.append({
            "regime": "overall",
            "n_samples": int(len(df)),
            "nominal_coverage": nominal_coverage,
            "empirical_coverage": float(inside_all),
            "mean_interval_width": float((df["upper"] - df["lower"]).mean()),
        })
        calibration = pd.DataFrame(rows)

    return EvaluationReport(
        overall=overall,
        per_regime=per_regime,
        long_short=long_short,
        calibration=calibration,
    )


def pretty_print(report: EvaluationReport) -> str:
    lines = []
    lines.append("─" * 72)
    lines.append("OVERALL")
    lines.append("─" * 72)
    lines.append(report.overall.to_string(index=False))
    lines.append("")
    lines.append("─" * 72)
    lines.append("PER-REGIME")
    lines.append("─" * 72)
    lines.append(report.per_regime.to_string(index=False))
    if report.calibration is not None:
        lines.append("")
        lines.append("─" * 72)
        lines.append("CONFORMAL CALIBRATION")
        lines.append("─" * 72)
        lines.append(report.calibration.to_string(index=False))
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({
        "ticker": rng.choice(list("ABCDEF"), n),
        "date": pd.date_range("2023-01-01", periods=n // 10).repeat(10)[:n],
        "prediction": rng.normal(0, 0.02, n),
        "target_fwd_ret": rng.normal(0, 0.03, n),
    })
    df["lower"] = df["prediction"] - 0.04
    df["upper"] = df["prediction"] + 0.04
    regimes = pd.DataFrame({"date": df["date"].unique(), "regime": rng.choice(["calm", "trending", "crisis"], df["date"].nunique())})
    rep = evaluate_predictions(df, regimes)
    print(pretty_print(rep))
