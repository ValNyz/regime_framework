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
    y_int = np.array([cls_to_idx[v] for v in y.values], dtype=np.int64)
    Xv = X.fillna(0).values

    mi = mutual_info_classif(
        Xv, y_int,
        discrete_features=discrete_features,
        random_state=random_state,
        n_neighbors=n_neighbors,
    )
    df = pd.DataFrame({"signal": X.columns, "mi": mi}).sort_values(
        "mi", ascending=False
    ).reset_index(drop=True)
    return df
