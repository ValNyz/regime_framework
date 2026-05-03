"""Metric computation for predictor outputs."""
from __future__ import annotations

import numpy as np
import pandas as pd
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

    Strategy semantics (no look-ahead):
      - At end of bar t, observe close[t] and the model's label[t].
      - Take position sign(label[t]) (+1 long bull, -1 short bear, 0 else).
      - Hold until end of bar t+1 → earn sign(label[t]) * log(close[t+1]/close[t]).
      - That contribution is the strategy log return at step (t → t+1).

    Hourly compounded log returns, no costs/slippage. The curve is normalized
    to start at closes[0] so it overlays cleanly with the actual price.

    Returns:
        equity: cumulative equity matching len(closes), starting at closes[0]
        total_gain: final fractional return (e.g. 0.125 = +12.5%)

    Single source of truth: used by both evaluate() (the synth_gain metric)
    and regime_plots._plot_B / plot_synth_equity_multi / plot_stitched_oos.
    """
    closes = np.asarray(closes, dtype=np.float64)
    labels_arr = np.asarray(labels)
    if len(closes) < 2:
        zero_eq = np.full(len(closes), float(closes[0]) if len(closes) else 0.0)
        return zero_eq, 0.0
    # Per-step relative log return (length n-1):
    #   log_ret_rel[t] = log(close[t+1] / close[t])  for t in [0, n-2]
    log_ret_rel = np.log(closes[1:] / closes[:-1])
    sign = np.zeros(len(labels_arr), dtype=np.float64)
    sign[labels_arr == "bull"] = +1.0
    sign[labels_arr == "bear"] = -1.0
    # Strategy log return at step (t → t+1):
    #   strategy_log_ret[t] = sign(label[t]) * log_ret_rel[t]
    # Position is decided AFTER observing close[t], applied to the next bar.
    strategy_log_ret = sign[:-1] * log_ret_rel
    cum_log = np.zeros(len(closes))
    cum_log[1:] = np.cumsum(strategy_log_ret)
    equity = np.exp(cum_log) * float(closes[0])
    total_gain = float(np.exp(cum_log[-1]) - 1.0)
    return equity, total_gain


def buy_and_hold_gain(closes: np.ndarray) -> float:
    """Return total fractional return of buy-and-hold over the slice.
    Reference baseline for synth_gain comparisons.
    """
    closes = np.asarray(closes, dtype=np.float64)
    if len(closes) < 2:
        return float("nan")
    return float(closes[-1] / closes[0] - 1.0)


def compound_returns(gains: np.ndarray | list[float]) -> float:
    """Compound a sequence of fractional returns: prod(1 + g) - 1.
    NaN-tolerant: drops NaNs before compounding.
    """
    g = np.asarray(gains, dtype=np.float64)
    g = g[~np.isnan(g)]
    if len(g) == 0:
        return float("nan")
    return float(np.prod(1.0 + g) - 1.0)


def synth_gain_by_month(
    closes: np.ndarray,
    labels: np.ndarray,
    dates: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Per-calendar-month synth_gain. Returns {YYYY-MM: fractional_return}.

    Same no-look-ahead semantics as synth_equity_curve: at bar t, the model
    label triggers a position held from t to t+1. The realized log return
    log(close[t+1]/close[t]) is bucketed by date[t] (the position date).
    """
    closes = np.asarray(closes, dtype=np.float64)
    labels_arr = np.asarray(labels)
    if len(closes) < 2:
        return {}
    log_ret_rel = np.log(closes[1:] / closes[:-1])  # length n-1
    sign = np.zeros(len(labels_arr), dtype=np.float64)
    sign[labels_arr == "bull"] = +1.0
    sign[labels_arr == "bear"] = -1.0
    strategy_log_ret = sign[:-1] * log_ret_rel  # length n-1

    months = pd.to_datetime(dates[:-1]).to_period("M").astype(str)
    df = pd.DataFrame({"month": months, "s": strategy_log_ret})
    monthly_log = df.groupby("month")["s"].sum()
    return {str(m): float(np.exp(v) - 1.0) for m, v in monthly_log.items()}


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
