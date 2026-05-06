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


def _strategy_log_returns(
    closes: np.ndarray,
    labels: np.ndarray,
    transaction_cost: float = 0.0,
) -> np.ndarray:
    """Per-step strategy log returns net of transaction cost.

    Returns array of length n-1: strategy_log_ret[t] = sign(label[t]) *
    log(close[t+1]/close[t]) - cost * |sign[t] - sign[t-1]|. The cost
    deduction is applied at the bar where the position changes (entry,
    exit, or flip). Initial position assumed flat (0); first bar's cost
    is charged for the entry move from 0 to sign[0].

    Used by synth_equity_curve, sharpe_ratio, synth_gain_by_month — single
    source of truth for cost-adjusted strategy returns. Set
    transaction_cost=0 to get the original cost-blind series.
    """
    closes = np.asarray(closes, dtype=np.float64)
    labels_arr = np.asarray(labels)
    log_ret_rel = np.log(closes[1:] / closes[:-1])               # length n-1
    sign = np.zeros(len(labels_arr), dtype=np.float64)
    sign[labels_arr == "bull"] = +1.0
    sign[labels_arr == "bear"] = -1.0
    strategy_log_ret = sign[:-1] * log_ret_rel                   # length n-1
    if transaction_cost > 0:
        position_change = np.abs(np.diff(sign, prepend=0.0))     # length n
        strategy_log_ret = strategy_log_ret - transaction_cost * position_change[:-1]
    return strategy_log_ret


def synth_equity_curve(
    closes: np.ndarray,
    labels: np.ndarray,
    transaction_cost: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Compute the long-bull / short-bear / flat-else synth-equity curve.

    Strategy semantics (no look-ahead):
      - At end of bar t, observe close[t] and the model's label[t].
      - Take position sign(label[t]) (+1 long bull, -1 short bear, 0 else).
      - Hold until end of bar t+1 → earn sign(label[t]) * log(close[t+1]/close[t]).
      - Pay `transaction_cost * |sign[t] - sign[t-1]|` at the entry/flip step.
      - Net contribution to log equity at (t → t+1).

    Hourly compounded log returns. The curve is normalized to start at
    closes[0] so it overlays cleanly with the actual price.

    Returns:
        equity: cumulative equity matching len(closes), starting at closes[0]
        total_gain: final fractional return (e.g. 0.125 = +12.5%)

    Single source of truth: used by both evaluate() (the synth_gain metric)
    and regime_plots._plot_B / plot_synth_equity_multi / plot_stitched_oos.
    """
    closes = np.asarray(closes, dtype=np.float64)
    if len(closes) < 2:
        zero_eq = np.full(len(closes), float(closes[0]) if len(closes) else 0.0)
        return zero_eq, 0.0
    strategy_log_ret = _strategy_log_returns(closes, labels, transaction_cost)
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


# Periods per year per timeframe — used for annualizing Sharpe.
# Crypto markets are 24/7, so 365-day year (not 252 trading days like equities).
_PERIODS_PER_YEAR = {
    "1m": 60 * 24 * 365,
    "5m": 12 * 24 * 365,
    "15m": 4 * 24 * 365,
    "30m": 2 * 24 * 365,
    "1h": 24 * 365,
    "2h": 12 * 365,
    "4h": 6 * 365,
    "8h": 3 * 365,
    "12h": 2 * 365,
    "1d": 365,
}


def periods_per_year(timeframe: str) -> int:
    """Return # bars per year for a given timeframe string (e.g. '1h' → 8760).

    Falls back to 1h (8760) for unknown timeframes — emit no warning since
    the metric is still meaningful, just may be miscalibrated.
    """
    return _PERIODS_PER_YEAR.get(timeframe, _PERIODS_PER_YEAR["1h"])


def sharpe_ratio(
    closes: np.ndarray,
    labels: np.ndarray,
    periods_per_year: int = 24 * 365,
    transaction_cost: float = 0.0,
) -> float:
    """Annualized Sharpe of the synth strategy (long-bull / short-bear / flat).

    Sharpe = mean(strategy_log_ret) / std(strategy_log_ret) * sqrt(ppy).
    Risk-free rate assumed 0. transaction_cost (default 0) is deducted on
    every position change. Same no-look-ahead semantics as synth_equity_curve.
    """
    closes = np.asarray(closes, dtype=np.float64)
    if len(closes) < 3:
        return float("nan")
    strategy_log_ret = _strategy_log_returns(closes, labels, transaction_cost)
    sd = float(np.std(strategy_log_ret, ddof=1))
    if sd <= 1e-12:
        return float("nan")
    return float(np.mean(strategy_log_ret) / sd * np.sqrt(periods_per_year))


def max_drawdown(equity: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of the equity curve, as a fraction.

    Returns a NEGATIVE number (or 0 for monotonic equity). E.g. -0.25 means
    the deepest decline from a running peak was 25%.
    """
    eq = np.asarray(equity, dtype=np.float64)
    if len(eq) < 2 or np.all(eq <= 0):
        return float("nan")
    running_max = np.maximum.accumulate(eq)
    dd = (eq - running_max) / running_max
    return float(np.min(dd))


def profit_factor(
    closes: np.ndarray,
    labels: np.ndarray,
    transaction_cost: float = 0.0,
) -> float:
    """sum(positive strategy returns) / |sum(negative strategy returns)|.

    Classic trading metric. >1 = profitable, >1.5 = decent edge, >2 = strong.
    Robust to outliers compared to win-rate. Returns +inf when there are
    only winners, NaN when no trades.
    """
    log_ret = _strategy_log_returns(np.asarray(closes, dtype=np.float64), labels, transaction_cost)
    if len(log_ret) == 0:
        return float("nan")
    pos = float(log_ret[log_ret > 0].sum())
    neg = float(-log_ret[log_ret < 0].sum())
    if neg <= 1e-12:
        return float("inf") if pos > 0 else float("nan")
    return pos / neg


def avg_excess_ratio(
    closes: np.ndarray,
    labels: np.ndarray,
    transaction_cost: float = 0.0,
) -> float:
    """Time-averaged outperformance of strategy vs buy-and-hold.

    Equation:  mean over t of (eq_strategy[t] / closes[t] - 1)

    Captures path-dependent outperformance — a strategy that ends at +85%
    but only crossed B&H late gives a small value; one consistently +85%
    throughout gives a large value. Endpoint `synth_gain` only sees the
    final bar; this captures the area between the strategy equity and
    the B&H curve, normalized by time.
    """
    closes = np.asarray(closes, dtype=np.float64)
    if len(closes) < 2:
        return float("nan")
    equity, _ = synth_equity_curve(closes, labels, transaction_cost=transaction_cost)
    # closes already starts at closes[0] = synth-equity start, so B&H equity == closes.
    return float((equity / np.maximum(closes, 1e-12) - 1.0).mean())


def time_above_bh(
    closes: np.ndarray,
    labels: np.ndarray,
    transaction_cost: float = 0.0,
) -> float:
    """Fraction of bars where strategy equity is at or above B&H equity.

    0–1 range. >0.5 = strategy spent most of the deployment at or ahead of
    buy-and-hold. Uses >= rather than > so the t=0 bar (where strategy and
    B&H are tied at closes[0]) counts toward "not underperforming". Robust
    to single-bar spikes; complements `avg_excess_ratio` (magnitude) with
    a frequency view.
    """
    closes = np.asarray(closes, dtype=np.float64)
    if len(closes) < 2:
        return float("nan")
    equity, _ = synth_equity_curve(closes, labels, transaction_cost=transaction_cost)
    return float((equity >= closes).mean())


def calmar_ratio(total_gain: float, max_dd: float) -> float:
    """Calmar = total_gain / |max_dd|. Risk-adjusted return per unit of pain.

    >1 = decent, >3 = excellent, <0 = lost money. Return NaN when DD is
    undefined or zero (monotonic equity).
    """
    if not np.isfinite(total_gain) or not np.isfinite(max_dd):
        return float("nan")
    if abs(max_dd) < 1e-12:
        return float("nan")
    return float(total_gain / abs(max_dd))


def consistency_positive_months(
    monthly_gains: dict[str, float],
) -> tuple[int, int]:
    """Count of (positive months, total months) from a {YYYY-MM: gain} dict.

    Useful as a stability check: a strategy with high total gain but only
    20% positive months is far more fragile than one with the same gain
    spread across 70% positive months.
    """
    if not monthly_gains:
        return 0, 0
    n_pos = sum(1 for v in monthly_gains.values() if v > 0)
    return int(n_pos), int(len(monthly_gains))


def compound_returns(gains: np.ndarray | list[float]) -> float:
    """Compound a sequence of fractional returns: prod(1 + g) - 1.
    NaN-tolerant: drops NaNs before compounding.
    """
    g = np.asarray(gains, dtype=np.float64)
    g = g[~np.isnan(g)]
    if len(g) == 0:
        return float("nan")
    return float(np.prod(1.0 + g) - 1.0)


def directional_kappa(
    closes: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Cohen's κ between predictions and next-bar return sign.

    For each bar t (0 <= t < n-1):
        truth[t] = 'bull' if close[t+1] > close[t] else 'bear'
        pred[t]  = y_pred[t] (filtered to bull/bear)
    Returns κ on the resulting agreement.

    By construction this metric tracks synth_gain: the strategy at bar t
    takes position sign(y_pred[t]) and earns sign(y_pred[t]) * log_ret[t+1].
    A bull/bear agreement on each bar is exactly what the strategy needs.

    Returns NaN if there's only one class (truth all one direction or
    pred all one direction) — same convention as evaluate's kappa.
    """
    closes = np.asarray(closes, dtype=np.float64)
    y_pred = np.asarray(y_pred)
    if len(closes) < 2 or len(y_pred) < 2:
        return float("nan")
    # Truth: sign of next-bar return at every bar except the last.
    n = min(len(closes), len(y_pred)) - 1
    truth = np.where(closes[1:n + 1] > closes[:n], "bull", "bear")
    pred = y_pred[:n]
    # Filter to bars where prediction is bull or bear (exclude '' / range).
    mask = (pred == "bull") | (pred == "bear")
    if mask.sum() < 2:
        return float("nan")
    truth_f = truth[mask]
    pred_f = pred[mask]
    observed = np.unique(np.concatenate([truth_f, pred_f]))
    if len(observed) <= 1:
        return float("nan")
    return float(cohen_kappa_score(truth_f, pred_f, labels=LABEL_ORDER))


def synth_gain_by_month(
    closes: np.ndarray,
    labels: np.ndarray,
    dates: np.ndarray | pd.Series,
    transaction_cost: float = 0.0,
) -> dict[str, float]:
    """Per-calendar-month synth_gain. Returns {YYYY-MM: fractional_return}.

    Same no-look-ahead semantics as synth_equity_curve. transaction_cost
    (default 0) is deducted on every position change before bucketing.
    """
    closes = np.asarray(closes, dtype=np.float64)
    if len(closes) < 2:
        return {}
    strategy_log_ret = _strategy_log_returns(closes, labels, transaction_cost)

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
    dates: np.ndarray | pd.Series | None = None,
    timeframe: str = "1h",
    transaction_cost: float = 0.0,
    metadata: dict | None = None,
) -> PredictionResult:
    """Compute the standard regime classification metrics + synth gain.

    `closes` (optional) is the close price series aligned to y_pred. When
    given, computes synth_gain plus the trade-quality companions:
      - sharpe: annualized (uses `timeframe` to compute periods_per_year)
      - max_dd: deepest peak-to-trough drawdown of the equity curve
      - n_positive_months / n_total_months: consistency, requires `dates`

    Without `closes`, all the strategy metrics stay NaN/0.
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
    # Cohen's kappa is undefined when only one class is observed (formula
    # divides 0 by 0 — sklearn warns and returns NaN). Detect that explicitly
    # so the warning doesn't fire and downstream tables show NaN cleanly.
    observed_classes = np.unique(np.concatenate([y_true_f, y_pred_f]))
    if len(observed_classes) <= 1:
        kappa = float("nan")
    else:
        kappa = float(cohen_kappa_score(y_true_f, y_pred_f, labels=LABEL_ORDER))
    f1m = float(f1_score(y_true_f, y_pred_f, labels=LABEL_ORDER, average="macro", zero_division=0))
    cm = confusion_matrix(y_true_f, y_pred_f, labels=LABEL_ORDER).tolist()

    synth_gain = float("nan")
    dir_kappa = float("nan")
    sharpe = float("nan")
    max_dd_v = float("nan")
    calmar = float("nan")
    pf = float("nan")
    n_pos_months = 0
    n_total_months = 0
    if closes is not None and len(closes) == len(y_pred):
        # Use the masked-aligned closes + predictions so unlabeled bars are
        # excluded from PnL too (they'd be flat anyway, but cleaner).
        closes_arr = np.asarray(closes)
        closes_f = closes_arr[mask]
        equity, synth_gain = synth_equity_curve(
            closes_f, y_pred_f, transaction_cost=transaction_cost,
        )
        # Directional kappa: agreement between prediction and next-bar return
        # sign. Computed on raw (unmasked) y_pred so we use all bars where the
        # strategy could actually take a position.
        dir_kappa = directional_kappa(closes_arr, y_pred)
        # Sharpe + max DD on the masked-aligned curve, cost-deducted.
        ppy = periods_per_year(timeframe)
        sharpe = sharpe_ratio(
            closes_f, y_pred_f, periods_per_year=ppy,
            transaction_cost=transaction_cost,
        )
        max_dd_v = max_drawdown(equity)
        calmar = calmar_ratio(synth_gain, max_dd_v)
        pf = profit_factor(closes_f, y_pred_f, transaction_cost=transaction_cost)
        # Consistency: needs dates, masked the same way
        if dates is not None and len(dates) == len(y_pred):
            dates_arr = np.asarray(dates)
            dates_f = dates_arr[mask]
            monthly = synth_gain_by_month(
                closes_f, y_pred_f, dates_f, transaction_cost=transaction_cost,
            )
            n_pos_months, n_total_months = consistency_positive_months(monthly)

    return PredictionResult(
        name=name, family=family,
        accuracy=acc, kappa=kappa, f1_macro=f1m,
        confusion=cm, n_test=int(len(y_true_f)),
        synth_gain=synth_gain,
        dir_kappa=dir_kappa,
        sharpe=sharpe,
        max_dd=max_dd_v,
        calmar=calmar,
        profit_factor=pf,
        n_positive_months=n_pos_months,
        n_total_months=n_total_months,
        metadata=metadata or {},
    )
