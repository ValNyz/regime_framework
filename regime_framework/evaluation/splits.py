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


def leave_one_out_splits(
    n: int,
    n_folds: int = 5,
    purge_bars: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
    """K chronologically equal-sized folds; each fold becomes test once.

    Train = all other folds, with `purge_bars` removed on BOTH sides of the
    test window to avoid forward-window label leakage from train into test.

    Yields:
      (train_idx, test_idx, fold_id) — train_idx is non-contiguous when k != 0
      and k != n_folds-1 (it's the union of past + future folds minus the
      purge gap around test).

    NOTE: this mode INTENTIONALLY uses future data in train. It answers
    "is the predictor robust ON each calendar period?" rather than "what
    would I have observed live at that time?". Use walk_forward_splits()
    for the latter.
    """
    if n_folds <= 1 or n <= n_folds:
        return
    fold_size = n // n_folds
    for k in range(n_folds):
        test_start = k * fold_size
        test_end = (k + 1) * fold_size if k < n_folds - 1 else n
        if test_end - test_start < 10:
            continue

        # Train: everything outside the test window, minus a purge gap on BOTH sides
        train_left = np.arange(0, max(test_start - purge_bars, 0))
        train_right = np.arange(min(test_end + purge_bars, n), n)
        train_idx = np.concatenate([train_left, train_right])
        test_idx = np.arange(test_start, test_end)

        if len(train_idx) < 100:
            continue
        yield train_idx, test_idx, k


def rolling_walk_forward_splits(
    n: int,
    train_window_bars: int,
    test_window_bars: int,
    purge_bars: int = 0,
    step_bars: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
    """Yield (train_idx, test_idx, fold_id) for sliding fixed-size folds.

    Layout (vs walk_forward_splits which is expanding):
      | train_0 (W bars) | purge | test_0 (T bars) |   step S
                                                    \\____
      | train_1 (W bars, slid by S) | purge | test_1 (T bars) |
      ...

    At fold k:
      train spans bars [k*S, k*S + W)
      test  spans bars [k*S + W + purge, k*S + W + purge + T)
      step S defaults to T (non-overlapping consecutive test windows).

    The fixed train width W means every fold has the same training set size —
    the model is always trained on the most recent W bars before its test.
    Useful for non-stationary regimes (e.g. crypto): models see recent data
    only, no contamination from years-old market structure.

    Args:
        n: total number of bars
        train_window_bars: W — fixed-size training window
        test_window_bars: T — fixed-size test window
        purge_bars: gap between train end and test start
        step_bars: S — slide step between consecutive folds. None = T (consecutive).

    Total folds: max(0, (n - W - purge - T) // S + 1)
    """
    if train_window_bars <= 0 or test_window_bars <= 0:
        return
    step = step_bars if step_bars is not None else test_window_bars
    if step <= 0:
        return
    fold_id = 0
    start = 0
    while start + train_window_bars + purge_bars + test_window_bars <= n:
        train_end = start + train_window_bars
        test_start = train_end + purge_bars
        test_end = test_start + test_window_bars
        train_idx = np.arange(start, train_end)
        test_idx = np.arange(test_start, test_end)
        yield train_idx, test_idx, fold_id
        fold_id += 1
        start += step


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
