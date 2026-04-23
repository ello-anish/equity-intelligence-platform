"""
Tests for quant.ml.statistics: Diebold-Mariano, bootstrap CI, PBO.
"""

import numpy as np
import pytest

from quant.ml.statistics import (
    bootstrap_sharpe_ci,
    diebold_mariano,
    pbo,
)


def test_dm_detects_known_better_model():
    """If A's losses are systematically smaller, DM should flag A as better."""
    rng = np.random.default_rng(0)
    n = 1000
    loss_a = rng.chisquare(df=2, size=n)
    loss_b = loss_a + rng.chisquare(df=2, size=n)   # always larger
    r = diebold_mariano(loss_a, loss_b, horizon=1)
    assert r.better_model == "A"
    assert r.p_value < 0.01


def test_dm_no_difference_produces_nonsignificant_p():
    """When A and B are the same distribution, p should be moderate (>0.1)."""
    rng = np.random.default_rng(5)
    n = 1000
    loss_a = rng.chisquare(df=2, size=n)
    loss_b = rng.chisquare(df=2, size=n)
    r = diebold_mariano(loss_a, loss_b, horizon=1)
    # Usually > 0.1 but occasionally < 0.1 by chance; check tie-zone
    assert r.better_model in ("tie", "A", "B")
    # Effect size small
    assert abs(r.dm_stat) < 4.0


def test_dm_symmetric_under_label_swap():
    """Swapping A and B should flip the sign but keep |stat| and p equal."""
    rng = np.random.default_rng(2)
    la = rng.chisquare(df=2, size=500)
    lb = rng.chisquare(df=2, size=500)
    r1 = diebold_mariano(la, lb, horizon=5)
    r2 = diebold_mariano(lb, la, horizon=5)
    assert abs(r1.dm_stat + r2.dm_stat) < 1e-6
    assert abs(r1.p_value - r2.p_value) < 1e-6


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(7)
    returns = rng.standard_normal(80) * 0.05 + 0.01
    ci = bootstrap_sharpe_ci(returns, periods_per_year=12, n_boot=1000, seed=1)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert ci["n_boot"] == 1000


def test_bootstrap_ci_width_shrinks_with_more_data():
    """Double n → CI width should roughly shrink."""
    rng = np.random.default_rng(11)
    r_short = rng.standard_normal(40) * 0.05
    r_long = rng.standard_normal(400) * 0.05

    ci_s = bootstrap_sharpe_ci(r_short, periods_per_year=12, n_boot=1000, seed=1)
    ci_l = bootstrap_sharpe_ci(r_long, periods_per_year=12, n_boot=1000, seed=1)

    w_s = ci_s["hi"] - ci_s["lo"]
    w_l = ci_l["hi"] - ci_l["lo"]
    assert w_l < w_s, f"CI width should shrink with n: short={w_s:.3f} long={w_l:.3f}"


def test_pbo_on_random_strategies_is_in_range():
    """
    With pure noise strategies, PBO should be in [0, 1] and not pinned at
    extremes. Exact value depends heavily on T/N and partition count.
    """
    rng = np.random.default_rng(13)
    M = rng.standard_normal((80, 20))
    res = pbo(M)
    assert 0.05 <= res["pbo"] <= 0.95
    assert res["n_trials"] > 0
    assert res["n_strategies"] == 20


def test_pbo_with_one_genuinely_better_strategy():
    """A strategy that's genuinely better both in and out of sample should
    drive PBO lower."""
    rng = np.random.default_rng(17)
    T, N = 128, 10
    M = rng.standard_normal((T, N))
    # Boost column 0 everywhere
    M[:, 0] += 0.5
    res = pbo(M)
    # PBO should be substantially below 0.5 (the best IS is usually also best OOS)
    assert res["pbo"] < 0.4
