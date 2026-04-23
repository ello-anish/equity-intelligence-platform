"""
Tests for the purged walk-forward splitter.

Core invariant being tested: NO OVERLAP between train and test after embargo.
This is the thing that breaks naive TimeSeriesSplit on time-series with
overlapping-horizon targets and causes silent leakage.
"""

import pandas as pd

from quant.ml.walkforward import PurgedWalkForward


def test_no_overlap_between_train_and_test():
    """Each fold's train-end must strictly precede test-start by ≥ embargo."""
    dates = pd.Series(pd.date_range("2020-01-01", "2024-01-01", freq="B"))
    wf = PurgedWalkForward(
        train_window_days=504, test_window_days=63,
        embargo_days=21, step_days=63,
    )
    for split in wf.split(dates):
        train_end = split.train_end
        test_start = split.test_start
        gap_days = (test_start - train_end).days
        assert gap_days >= 21, (
            f"fold {split.fold}: train_end={train_end}, "
            f"test_start={test_start}, gap={gap_days}d — "
            f"embargo violated"
        )


def test_train_precedes_test_always():
    """Train indices must all come before test indices."""
    dates = pd.Series(pd.date_range("2020-01-01", "2024-01-01", freq="B"))
    wf = PurgedWalkForward(504, 63, 21, 63)
    for split in wf.split(dates):
        if len(split.train_idx) and len(split.test_idx):
            assert split.train_idx.max() < split.test_idx.min()


def test_fold_indices_are_valid():
    dates = pd.Series(pd.date_range("2020-01-01", "2024-01-01", freq="B"))
    wf = PurgedWalkForward(504, 63, 21, 63)
    for split in wf.split(dates):
        # train/test indices must lie within dates
        assert split.train_idx.max() < len(dates)
        assert split.test_idx.max() < len(dates)
        # train/test must be non-empty
        assert len(split.train_idx) > 0
        assert len(split.test_idx) > 0


def test_at_least_one_fold_produced():
    dates = pd.Series(pd.date_range("2020-01-01", "2024-01-01", freq="B"))
    wf = PurgedWalkForward(504, 63, 21, 63)
    assert sum(1 for _ in wf.split(dates)) >= 3


def test_embargo_zero_still_works():
    """With embargo=0, test starts immediately after train ends. Still valid."""
    dates = pd.Series(pd.date_range("2020-01-01", "2023-01-01", freq="B"))
    wf = PurgedWalkForward(365, 63, 0, 63)
    for split in wf.split(dates):
        assert split.test_start >= split.train_end
