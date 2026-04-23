"""
quant/ml/statistics.py — Statistical rigor utilities.

Three things every quant paper should have but most intern projects don't:

    1. diebold_mariano(loss_a, loss_b, h)  →  is model A significantly better
       than model B?  Newey-West corrected t-stat of loss differential.
       (Diebold & Mariano, 1995; Harvey, Leybourne, Newbold small-sample
       correction.)

    2. bootstrap_sharpe_ci(returns, n_boot, alpha)  →  stationary-bootstrap
       confidence interval on the Sharpe ratio. Politis & Romano (1994).
       Addresses "your Sharpe is ±what?"

    3. pbo(cv_matrix)  →  Probability of Backtest Overfitting from the
       combinatorially symmetric cross-validation framework of Bailey,
       Borwein, López de Prado, Zhu (2015).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats


# ── Diebold-Mariano ──────────────────────────────────────────────────────────

@dataclass
class DMResult:
    mean_diff: float          # mean(loss_A - loss_B); negative → A is better
    dm_stat: float            # Newey-West corrected t statistic
    p_value: float            # two-sided
    n: int
    horizon: int
    better_model: str         # "A", "B", or "tie"


def _newey_west_var(d: np.ndarray, lag: int) -> float:
    """Newey-West long-run variance estimator with truncated kernel."""
    n = len(d)
    d_bar = d.mean()
    centered = d - d_bar
    gamma0 = (centered @ centered) / n
    total = gamma0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        gamma_k = (centered[k:] @ centered[:-k]) / n
        total += 2.0 * w * gamma_k
    # Long-run variance of the mean:  (gamma_0 + 2 Σ w_k gamma_k) / n
    return max(total / n, 1e-12)


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    horizon: int = 1,
    loss_fn: Literal["squared", "absolute"] = "squared",
) -> DMResult:
    """
    Compare forecast accuracy of two models.

    Parameters
    ----------
    loss_a, loss_b : arrays of per-sample losses (e.g. squared errors).
        Must be aligned.
    horizon : forecast horizon h. For a 21-day forward prediction this is 21.
        Used to set the Newey-West truncation lag (h - 1).
    loss_fn : 'squared' is assumed (you pass SE directly). Included in the
        result only for reporting.

    Returns
    -------
    DMResult
    """
    a = np.asarray(loss_a, dtype=float).ravel()
    b = np.asarray(loss_b, dtype=float).ravel()
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 10:
        raise ValueError(f"need at least 10 aligned observations, got {n}")

    d = a - b
    d_bar = d.mean()
    lag = max(horizon - 1, 0)
    var_lr = _newey_west_var(d, lag)
    dm = d_bar / math.sqrt(var_lr)

    # Harvey-Leybourne-Newbold small-sample correction
    adj = math.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    dm_adj = dm * adj
    # t-distribution with n-1 df
    p = 2.0 * (1.0 - stats.t.cdf(abs(dm_adj), df=n - 1))

    better = "tie"
    if p < 0.10:
        better = "A" if d_bar < 0 else "B"   # lower loss = better

    return DMResult(
        mean_diff=float(d_bar),
        dm_stat=float(dm_adj),
        p_value=float(p),
        n=int(n),
        horizon=int(horizon),
        better_model=better,
    )


# ── Stationary-bootstrap Sharpe CI ───────────────────────────────────────────

def _stationary_bootstrap_indices(n: int, avg_block_len: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap — geometric block lengths."""
    if avg_block_len <= 1.0:
        return rng.integers(0, n, size=n)
    p = 1.0 / avg_block_len
    idx = np.empty(n, dtype=np.int64)
    i = 0
    while i < n:
        start = rng.integers(0, n)
        # geometric block length
        blen = rng.geometric(p)
        for k in range(blen):
            if i >= n:
                break
            idx[i] = (start + k) % n
            i += 1
    return idx


def bootstrap_sharpe_ci(
    returns: np.ndarray,
    periods_per_year: float = 12.0,
    n_boot: int = 2000,
    alpha: float = 0.05,
    avg_block_len: float = 5.0,
    seed: int = 0,
) -> dict:
    """
    Stationary-bootstrap confidence interval for the Sharpe ratio of a return
    series (already in per-period units — e.g., monthly returns).

    Returns
    -------
    dict with 'point', 'lo', 'hi', 'n_boot', 'alpha'.
    """
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 5:
        raise ValueError(f"need ≥5 observations, got {n}")
    rng = np.random.default_rng(seed)

    def sharpe(x):
        s = x.std(ddof=1)
        return (x.mean() / s * math.sqrt(periods_per_year)) if s > 0 else 0.0

    point = sharpe(r)
    samples = np.empty(n_boot)
    for i in range(n_boot):
        idx = _stationary_bootstrap_indices(n, avg_block_len, rng)
        samples[i] = sharpe(r[idx])

    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return {
        "point": float(point),
        "lo": lo,
        "hi": hi,
        "n_boot": int(n_boot),
        "alpha": float(alpha),
        "n": int(n),
        "mean_of_boot": float(samples.mean()),
        "std_of_boot": float(samples.std(ddof=1)),
    }


# ── Probability of Backtest Overfitting ──────────────────────────────────────

def pbo(cv_matrix: np.ndarray, metric_fn=None) -> dict:
    """
    Probability of Backtest Overfitting (Bailey, Borwein, López de Prado, Zhu
    2015). Given an (n_samples × n_strategies) matrix of per-period realised
    returns under each candidate strategy, compute the probability that the
    *in-sample* best strategy is *not* the best out-of-sample.

    Intuition: if a researcher tries many strategies, the one that looks best
    in-sample is usually just lucky; a well-designed backtest should have
    PBO ≈ 0.5 (completely uninformative) if all strategies are equivalent,
    or < 0.3 if there is real signal.

    Implementation: combinatorially-symmetric cross-validation (CSCV) with
    16 partitions — chosen because it's the figure used in the original paper
    and gives stable estimates without blowing up the combinatorial cost.

    Returns
    -------
    dict with 'pbo', 'n_trials', 'n_strategies'.
    """
    M = np.asarray(cv_matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError("cv_matrix must be 2-D")
    T, N = M.shape
    if N < 2:
        raise ValueError("need at least 2 strategies")
    S = 16
    if T < S:
        # Shrink S to something feasible
        S = max(4, T // 2 * 2)
        if S < 2 or S % 2 != 0:
            raise ValueError(f"cv_matrix rows too few ({T}) for CSCV")

    # Split rows into S contiguous partitions of (near-)equal size
    boundaries = np.linspace(0, T, S + 1, dtype=int)
    partitions = [M[boundaries[i]:boundaries[i + 1]] for i in range(S)]

    if metric_fn is None:
        def metric_fn(block):
            mean = block.mean(axis=0)
            std = block.std(axis=0, ddof=1) + 1e-12
            return mean / std

    # Combinations of S/2 partitions out of S for the in-sample set
    from itertools import combinations
    count_overfit = 0
    total = 0
    for is_idx in combinations(range(S), S // 2):
        is_set = set(is_idx)
        oos_set = set(range(S)) - is_set
        is_block = np.concatenate([partitions[i] for i in is_set], axis=0)
        oos_block = np.concatenate([partitions[i] for i in oos_set], axis=0)

        is_rank = metric_fn(is_block)
        oos_rank = metric_fn(oos_block)

        best_is = int(np.argmax(is_rank))
        # OOS rank of the in-sample champion (0 = worst, N-1 = best)
        oos_order = np.argsort(np.argsort(oos_rank))
        pct = oos_order[best_is] / (N - 1)
        if pct < 0.5:
            count_overfit += 1
        total += 1

    return {
        "pbo": count_overfit / total if total > 0 else float("nan"),
        "n_trials": total,
        "n_strategies": int(N),
        "n_partitions": int(S),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Toy: two forecasters, A slightly better
    n = 400
    la = rng.standard_normal(n) ** 2
    lb = la * 1.1 + rng.standard_normal(n) ** 2 * 0.1
    r = diebold_mariano(la, lb, horizon=21)
    print(f"DM: mean_diff={r.mean_diff:.5f}  stat={r.dm_stat:.3f}  p={r.p_value:.4f}  → {r.better_model}")

    rets = rng.standard_normal(60) * 0.05 + 0.005
    ci = bootstrap_sharpe_ci(rets, periods_per_year=12, n_boot=1000)
    print(f"Sharpe {ci['point']:.2f}  95% CI [{ci['lo']:.2f}, {ci['hi']:.2f}]")

    M = rng.standard_normal((100, 20))
    p = pbo(M)
    print(f"PBO on random strategies: {p['pbo']:.2f}  (expect ≈ 0.5)")
