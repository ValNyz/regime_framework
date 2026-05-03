"""Metric computation for predictor outputs."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from ..config import LABEL_ORDER
from ..predictors.base import PredictionResult


def evaluate(
    name: str,
    family: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata: dict | None = None,
) -> PredictionResult:
    """Compute the standard regime classification metrics."""
    # Filter unlabelled positions
    mask = (y_true != "") & (y_pred != "")
    y_true_f = y_true[mask]
    y_pred_f = y_pred[mask]

    if len(y_true_f) == 0:
        return PredictionResult(
            name=name, family=family, accuracy=float("nan"),
            kappa=float("nan"), f1_macro=float("nan"),
            confusion=[[0, 0], [0, 0]], n_test=0,
            metadata=metadata or {},
        )

    acc = float(accuracy_score(y_true_f, y_pred_f))
    kappa = float(cohen_kappa_score(y_true_f, y_pred_f))
    f1m = float(f1_score(y_true_f, y_pred_f, labels=LABEL_ORDER, average="macro", zero_division=0))
    cm = confusion_matrix(y_true_f, y_pred_f, labels=LABEL_ORDER).tolist()

    return PredictionResult(
        name=name, family=family,
        accuracy=acc, kappa=kappa, f1_macro=f1m,
        confusion=cm, n_test=int(len(y_true_f)),
        metadata=metadata or {},
    )
