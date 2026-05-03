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
) -> pd.DataFrame:
    """Compute MI between each column of X and the regime label.

    Returns: DataFrame with signal, mi (sorted descending).
    """
    cls_to_idx = {c: i for i, c in enumerate(LABEL_ORDER)}
    # Drop rows whose label is not in {bull, bear} (e.g. "" or NaN)
    mask = y.isin(LABEL_ORDER).values
    Xv_full = X.fillna(0).replace([np.inf, -np.inf], 0).values.astype(float)
    Xv = Xv_full[mask]
    y_int = np.array([cls_to_idx[v] for v in y.values[mask]], dtype=np.int64)

    if len(y_int) < 50 or Xv.shape[1] == 0:
        return pd.DataFrame({"signal": list(X.columns), "mi": [0.0] * X.shape[1]})

    try:
        mi = mutual_info_classif(
            Xv, y_int,
            discrete_features=discrete_features,
            random_state=random_state,
            n_neighbors=n_neighbors,
        )
    except Exception as e:
        print(f"  WARN: mutual_info_classif failed ({e}); falling back to per-column MI.")
        # Fallback: compute MI per column independently, replacing NaN with 0
        mi = np.zeros(Xv.shape[1])
        for j in range(Xv.shape[1]):
            try:
                m = mutual_info_classif(
                    Xv[:, [j]], y_int, discrete_features=discrete_features,
                    random_state=random_state, n_neighbors=n_neighbors,
                )
                mi[j] = float(m[0]) if np.isfinite(m[0]) else 0.0
            except Exception:
                mi[j] = 0.0

    mi = np.nan_to_num(np.asarray(mi, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    df = pd.DataFrame({"signal": X.columns, "mi": mi}).sort_values(
        "mi", ascending=False
    ).reset_index(drop=True)
    return df
