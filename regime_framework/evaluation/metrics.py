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


def synth_equity_curve(
    closes: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute the long-bull / short-bear / flat-else synth-equity curve.

    Strategy: at each bar, take a long position if the label is "bull",
    short if "bear", flat otherwise (range / volatile / unlabeled). Hourly
    compounded log returns, no costs/slippage. The curve is normalized to
    start at closes[0] so it overlays cleanly with the actual price series.

    Returns:
        equity: cumulative equity matching len(closes), starting at closes[0]
        total_gain: final fractional return (e.g. 0.125 = +12.5%)

    Used by both the metric path (evaluation.metrics.evaluate) and the plot
    path (visualization.regime_plots._plot_B and friends) — single source of
    truth for the synth-strategy semantics.
    """
    closes = np.asarray(closes, dtype=np.float64)
    labels_arr = np.asarray(labels)
    if len(closes) == 0:
        return np.zeros(0), float("nan")
    log_ret = np.zeros(len(closes))
    log_ret[1:] = np.log(closes[1:] / closes[:-1])
    sign = np.zeros(len(labels_arr), dtype=np.float64)
    sign[labels_arr == "bull"] = +1.0
    sign[labels_arr == "bear"] = -1.0
    cum_log = np.cumsum(sign * log_ret)
    equity = np.exp(cum_log) * float(closes[0])
    total_gain = float(np.exp(cum_log[-1]) - 1.0)
    return equity, total_gain


def evaluate(
    name: str,
    family: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    closes: np.ndarray | None = None,
    metadata: dict | None = None,
) -> PredictionResult:
    """Compute the standard regime classification metrics + synth gain.

    `closes` (optional) is the close price series aligned to y_pred. When
    given, computes synth_gain = total fractional return of the long-bull /
    short-bear strategy on this test slice (uses raw predictions, no
    smoothing). Without `closes`, synth_gain stays NaN.
    """
    # Filter unlabelled positions
    mask = (y_true != "") & (y_pred != "")
    y_true_f = y_true[mask]
    y_pred_f = y_pred[mask]

    if len(y_true_f) == 0:
        return PredictionResult(
            name=name, family=family, accuracy=float("nan"),
            kappa=float("nan"), f1_macro=float("nan"),
            confusion=[[0, 0], [0, 0]], n_test=0,
            synth_gain=float("nan"),
            metadata=metadata or {},
        )

    acc = float(accuracy_score(y_true_f, y_pred_f))
    kappa = float(cohen_kappa_score(y_true_f, y_pred_f))
    f1m = float(f1_score(y_true_f, y_pred_f, labels=LABEL_ORDER, average="macro", zero_division=0))
    cm = confusion_matrix(y_true_f, y_pred_f, labels=LABEL_ORDER).tolist()

    synth_gain = float("nan")
    if closes is not None and len(closes) == len(y_pred):
        # Use the masked-aligned closes + predictions so unlabeled bars are
        # excluded from PnL too (they'd be flat anyway, but cleaner).
        closes_f = np.asarray(closes)[mask]
        _, synth_gain = synth_equity_curve(closes_f, y_pred_f)

    return PredictionResult(
        name=name, family=family,
        accuracy=acc, kappa=kappa, f1_macro=f1m,
        confusion=cm, n_test=int(len(y_true_f)),
        synth_gain=synth_gain,
        metadata=metadata or {},
    )
