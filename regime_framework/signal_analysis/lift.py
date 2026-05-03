"""Lift analysis — measures how much a binary signal updates regime probability.

For a binary feature S ∈ {0, 1} and a regime label R ∈ {bull, bear}:
  lift_bull(S) = P(R=bull | S=1) / P(R=bull)
  lift_bear(S) = P(R=bear | S=1) / P(R=bear)

A signal with lift_bull = 1.5 means it boosts the bull-base-rate by 50% when
triggered. A signal with lift = 1 carries no regime information (independence).

Conditional accuracy (alongside lift):
  acc_bull(S) = P(R=bull | S=1)
  acc_bear(S) = P(R=bear | S=1)

Returns a table sorted by max(|lift_bull - 1|, |lift_bear - 1|) — i.e. the
strongest deviation from base rate, regardless of direction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_lift_table(
    X: pd.DataFrame,
    y: pd.Series,
    binary_threshold: float = 0.5,
    min_triggers: int = 30,
) -> pd.DataFrame:
    """Compute lift + conditional accuracy for each column of X.

    Args:
        X: feature matrix (any column treated as binary by thresholding > binary_threshold)
        y: regime labels, values in {"bull", "bear"}
        binary_threshold: cutoff to convert continuous feature → binary trigger
        min_triggers: skip columns with fewer than this many triggers (statistically unreliable)

    Returns:
        DataFrame with columns: signal, n_triggers, p_bull_base, p_bull_cond,
        lift_bull, p_bear_base, p_bear_cond, lift_bear, max_abs_lift, deviation
    """
    y_arr = y.values
    n = len(y_arr)
    p_bull = float((y_arr == "bull").sum()) / max(n, 1)
    p_bear = float((y_arr == "bear").sum()) / max(n, 1)

    rows = []
    for col in X.columns:
        try:
            v = np.asarray(X[col].values, dtype=float)  # coerce to float; NaN preserved
        except (TypeError, ValueError):
            # Non-numeric column — skip
            continue

        if not np.isfinite(v).any():
            continue

        # Binarize. Fast-path for already 0/1 columns:
        finite = np.isfinite(v)
        unique_finite = np.unique(v[finite])
        is_binary_already = (
            len(unique_finite) <= 2
            and float(unique_finite.min()) >= 0.0
            and float(unique_finite.max()) <= 1.0
        )
        if is_binary_already:
            trig = (v >= 0.5) & finite
        else:
            trig = (v > binary_threshold) & finite

        n_trig = int(trig.sum())
        if n_trig < min_triggers:
            continue

        try:
            p_bull_cond = float((y_arr[trig] == "bull").sum()) / max(n_trig, 1)
            p_bear_cond = float((y_arr[trig] == "bear").sum()) / max(n_trig, 1)
        except Exception:
            continue

        lift_bull = p_bull_cond / max(p_bull, 1e-12)
        lift_bear = p_bear_cond / max(p_bear, 1e-12)
        max_abs = max(abs(lift_bull - 1), abs(lift_bear - 1))
        deviation = (lift_bull - lift_bear) / 2

        rows.append({
            "signal": col,
            "n_triggers": n_trig,
            "p_bull_base": p_bull,
            "p_bull_cond": p_bull_cond,
            "lift_bull": lift_bull,
            "p_bear_base": p_bear,
            "p_bear_cond": p_bear_cond,
            "lift_bear": lift_bear,
            "max_abs_lift": max_abs,
            "deviation": deviation,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "signal", "n_triggers", "p_bull_base", "p_bull_cond", "lift_bull",
            "p_bear_base", "p_bear_cond", "lift_bear", "max_abs_lift", "deviation",
        ])
    df = pd.DataFrame(rows).sort_values("max_abs_lift", ascending=False).reset_index(drop=True)
    return df
