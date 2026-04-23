"""
quant/ml/walkforward.py — Purged walk-forward cross-validation for time series.

Implements the purged / embargoed walk-forward scheme from Lopez de Prado's
"Advances in Financial Machine Learning" (Ch. 7). Random K-fold CV leaks
information through overlapping forward-return targets; purged walk-forward
CV is the right way to evaluate a return-forecasting model.

Parameters
----------
train_window_days : int   size of each rolling training window
test_window_days  : int   size of each test window (immediately after train)
embargo_days      : int   gap between train end and test start (at least the
                          target horizon, to purge label overlap)
step_days         : int   stride between successive folds
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardSplit:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_idx: np.ndarray
    test_idx: np.ndarray


class PurgedWalkForward:
    def __init__(
        self,
        train_window_days: int = 504,   # ~2 years
        test_window_days: int = 63,     # ~3 months
        embargo_days: int = 21,         # ≥ forward-return horizon
        step_days: int = 63,            # roll every quarter
    ):
        self.train_window = train_window_days
        self.test_window = test_window_days
        self.embargo = embargo_days
        self.step = step_days

    def split(self, dates: pd.Series) -> Iterator[WalkForwardSplit]:
        """
        Yield purged walk-forward splits over `dates` (a pd.Series of datetimes,
        not necessarily unique — one per (ticker, date) sample).
        """
        dates = pd.to_datetime(dates).reset_index(drop=True)
        unique_days = dates.drop_duplicates().sort_values().reset_index(drop=True)
        start = unique_days.iloc[0]
        end = unique_days.iloc[-1]

        train_end = start + pd.Timedelta(days=self.train_window)
        fold = 0
        while True:
            embargo_end = train_end + pd.Timedelta(days=self.embargo)
            test_end = embargo_end + pd.Timedelta(days=self.test_window)
            if test_end > end:
                break

            train_mask = (dates >= start) & (dates < train_end)
            test_mask = (dates >= embargo_end) & (dates < test_end)

            if train_mask.sum() > 0 and test_mask.sum() > 0:
                yield WalkForwardSplit(
                    fold=fold,
                    train_start=start,
                    train_end=train_end,
                    test_start=embargo_end,
                    test_end=test_end,
                    train_idx=np.where(train_mask.values)[0],
                    test_idx=np.where(test_mask.values)[0],
                )
                fold += 1

            train_end = train_end + pd.Timedelta(days=self.step)

    def describe(self, dates: pd.Series) -> pd.DataFrame:
        rows: List[dict] = []
        for s in self.split(dates):
            rows.append(dict(
                fold=s.fold,
                train_start=s.train_start.date(),
                train_end=s.train_end.date(),
                test_start=s.test_start.date(),
                test_end=s.test_end.date(),
                n_train=len(s.train_idx),
                n_test=len(s.test_idx),
            ))
        return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dates = pd.date_range("2021-01-01", "2024-01-01", freq="B")
    s = pd.Series(np.tile(dates, 10))
    wf = PurgedWalkForward()
    print(wf.describe(s))
