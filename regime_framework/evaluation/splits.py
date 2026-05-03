"""Time-aware train/test split with purge gap.

The purge gap prevents forward-looking labels (computed from data after t)
from leaking from train into test. With purge = max(L_range), the last
purge_bars rows of train are dropped to ensure no overlap.
"""
from __future__ import annotations

import pandas as pd


def time_aware_split(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    train_fraction: float = 0.80,
    purge_bars: int = 0,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    """Returns X_tr, y_tr, dates_tr, X_te, y_te, dates_te."""
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
