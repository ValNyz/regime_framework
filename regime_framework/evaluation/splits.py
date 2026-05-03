"""Time-aware train/test split with purge gap + walk-forward CV.

Purge: prevent forward-window leakage from train into test.
Walk-forward: K expanding-window folds for robust OOS validation.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


def time_aware_split(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    train_fraction: float = 0.80,
    purge_bars: int = 0,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    """Single time-ordered split with purge gap. Returns X_tr, y_tr, d_tr, X_te, y_te, d_te."""
    n = len(X)
    split_idx = int(n * train_fraction)
    train_end = max(split_idx - purge_bars, 0)
    X_tr = X.iloc[:train_end].copy()
    y_tr = y.iloc[:train_end].copy()
    d_tr = dates.iloc[:train_end].copy()
    X_te = X.iloc[split_idx:].copy()
    y_te = y.iloc[split_idx:].copy()
    d_te = dates.iloc[split_idx:].copy()
    return X_tr, y_tr, d_tr, X_te, y_te, d_te


def walk_forward_splits(
    n: int,
    n_folds: int = 5,
    purge_bars: int = 0,
    min_train_fraction: float = 0.40,
) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
    """Yield (train_idx, test_idx, fold_id) for K expanding-window folds.

    Layout:
      | train_0 ... | purge | test_0 |
      | train_0+test_0 ... | purge | test_1 |
      ...

    Args:
        n: total number of bars (aligned X / y / dates after NaN drop)
        n_folds: number of (train, test) folds to generate
        purge_bars: gap between train end and test start (forward-leakage protection)
        min_train_fraction: train data of fold-0 spans 0..min_train_fraction*n.
            Each subsequent fold absorbs the previous test window into its train.

    The remaining (1 - min_train_fraction)*n bars are split into n_folds equal test windows.
    """
    min_train = int(n * min_train_fraction)
    test_total = max(n - min_train, 0)
    if n_folds <= 0 or test_total <= n_folds:
        return
    test_size = test_total // n_folds
    for k in range(n_folds):
        train_end = min_train + k * test_size
        test_start = train_end + purge_bars
        test_end = min(test_start + test_size, n)
        if test_start >= n or test_end <= test_start:
            break
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        yield train_idx, test_idx, k
