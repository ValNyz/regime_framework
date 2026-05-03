"""Mutual information between each feature and the regime label.

Uses sklearn.feature_selection.mutual_info_classif. Higher MI = more
information about the regime. Lift is multiplicative; MI is informational.
The two often agree, but MI generalises to continuous features without thresholding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from ..config import LABEL_ORDER


def compute_mutual_info_table(
    X: pd.DataFrame,
    y: pd.Series,
    discrete_features: bool | list[bool] = False,
    random_state: int = 42,
    n_neighbors: int = 5,
    subsample_freq: int = 24,
) -> pd.DataFrame:
    """Compute MI between each column of X and the regime label.

    Args:
        subsample_freq: take every Nth row before computing MI to break temporal
            autocorrelation. With 1h data and N=24 (1 day), close-in-time bars
            sharing the same label no longer inflate the MI score. Set to 1 to
            disable. Default 24 matches the typical regime persistence horizon.

    Returns:
        DataFrame with signal, mi, mi_full (uncorrected baseline), sorted by mi desc.
    """
    cls_to_idx = {c: i for i, c in enumerate(LABEL_ORDER)}
    mask = y.isin(LABEL_ORDER).values
    Xv_full = X.fillna(0).replace([np.inf, -np.inf], 0).values.astype(float)
    Xv = Xv_full[mask]
    y_int = np.array([cls_to_idx[v] for v in y.values[mask]], dtype=np.int64)

    if len(y_int) < 50 or Xv.shape[1] == 0:
        return pd.DataFrame({
            "signal": list(X.columns), "mi": [0.0] * X.shape[1], "mi_full": [0.0] * X.shape[1],
        })

    def _compute(X_arr: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
        try:
            return np.asarray(mutual_info_classif(
                X_arr, y_arr,
                discrete_features=discrete_features,
                random_state=random_state,
                n_neighbors=n_neighbors,
            ), dtype=float)
        except Exception as e:
            print(f"  WARN: mutual_info_classif failed ({e}); per-column fallback.")
            mi = np.zeros(X_arr.shape[1])
            for j in range(X_arr.shape[1]):
                try:
                    m = mutual_info_classif(
                        X_arr[:, [j]], y_arr, discrete_features=discrete_features,
                        random_state=random_state, n_neighbors=n_neighbors,
                    )
                    mi[j] = float(m[0]) if np.isfinite(m[0]) else 0.0
                except Exception:
                    mi[j] = 0.0
            return mi

    # Baseline MI on every bar (potentially inflated by temporal autocorrelation)
    mi_full = _compute(Xv, y_int)

    # Subsampled MI: every `subsample_freq` bars, breaks regime-persistence inflation
    if subsample_freq > 1 and len(y_int) > subsample_freq * 50:
        Xs = Xv[::subsample_freq]
        ys = y_int[::subsample_freq]
        mi_sub = _compute(Xs, ys)
    else:
        mi_sub = mi_full

    mi_full = np.nan_to_num(mi_full, nan=0.0, posinf=0.0, neginf=0.0)
    mi_sub = np.nan_to_num(mi_sub, nan=0.0, posinf=0.0, neginf=0.0)
    df = pd.DataFrame({
        "signal": X.columns,
        "mi": mi_sub,           # primary metric — autocorrelation-corrected
        "mi_full": mi_full,     # uncorrected baseline for diagnosis
    }).sort_values("mi", ascending=False).reset_index(drop=True)
    return df
