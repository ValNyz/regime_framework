"""Base interface for all predictors (classical, rule-based, deep, pretrained).

Every predictor follows the same lifecycle:
  1. fit(X_train, y_train, dates_train, df_train)  -> self
  2. predict(X_test, dates_test, df_test)          -> array of labels (object)
  3. evaluate(y_true, y_pred)                      -> PredictionResult dict

Some predictors (rule-based) ignore X and use df directly.
Some (pretrained, sequence models) need the raw OHLCV df rather than the X feature matrix.
The interface accepts both — implementers use what they need and ignore the rest.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import LABEL_ORDER


class MultiCoinAware:
    """Mixin marking a predictor that benefits from multi-coin training data.

    The runner detects this via `is_multi_coin_aware` and, when extra coins
    are configured, calls `set_multi_coin_data` before `fit()` with per-coin
    (X, y, dates, df) tuples filtered to the fold's date range.

    Without this mixin, a predictor only sees the (possibly stacked) X_train
    that the runner already builds via `_stack_with_target`. RL agents and
    pretrained fine_tuned models need separate per-coin price series instead
    of a stacked X — that's what this mixin provides.

    Subclasses use the cached data inside their `fit()`. If `set_multi_coin_data`
    isn't called (single-coin run), `_target_coin_data` / `_extra_coin_data`
    stay None and the predictor falls back to using just (X_train, df_train).
    """
    is_multi_coin_aware: bool = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # cooperative — works in MRO chains
        self._target_coin_data: dict | None = None
        self._extra_coin_data: dict[str, dict] | None = None

    def set_multi_coin_data(
        self,
        target_data: dict,
        extras_data: dict[str, dict],
    ) -> None:
        """target_data and each extras_data[coin] dict has keys:
            X (DataFrame), y (Series), dates (Series), df (DataFrame with close)
        all clipped to the same date range as the fold.
        """
        self._target_coin_data = target_data
        self._extra_coin_data = dict(extras_data)


@dataclass
class PredictionResult:
    name: str
    family: str             # "classical" | "rule_based" | "deep" | "transformer" | "pretrained"
    accuracy: float
    kappa: float
    f1_macro: float
    confusion: list[list[int]]    # row=true, col=pred, in LABEL_ORDER
    n_test: int
    # Total fractional return of the long-bull / short-bear / flat-else strategy
    # on the test slice (same formulation as plot B). NaN if log returns weren't
    # supplied to evaluate(). Costs/slippage = 0 — idealized but tradeable.
    synth_gain: float = float("nan")
    # Directional kappa: Cohen's κ between the predicted label and the SIGN of
    # the next-bar log return (sign(close[t+1]/close[t])). Independent of the
    # training labels, this metric correlates directly with synth_gain — same
    # underlying agreement as the strategy's bar-level decisions. Useful as a
    # gain-aligned alternative to the standard κ when the training label is
    # itself off-target (e.g. trend-scan trained, but we want trade-aligned eval).
    dir_kappa: float = float("nan")
    metadata: dict = field(default_factory=dict)


class BasePredictor(ABC):
    name: str = "base"
    family: str = "base"
    needs_features: bool = True   # if False, fit/predict are called with df only

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        dates_train: pd.Series,
        df_train: pd.DataFrame,
    ) -> "BasePredictor":
        ...

    @abstractmethod
    def predict(
        self,
        X_test: pd.DataFrame,
        dates_test: pd.Series,
        df_test: pd.DataFrame,
    ) -> np.ndarray:
        ...

    def predict_proba(
        self,
        X_test: pd.DataFrame,    # noqa: ARG002 — kept for API parity
        dates_test: pd.Series,   # noqa: ARG002
        df_test: pd.DataFrame,   # noqa: ARG002
    ) -> Any:
        """Return per-class probabilities of shape (n_test, len(LABEL_ORDER)),
        with columns ordered by LABEL_ORDER.

        Default: None — meaning this predictor does not produce calibrated
        probabilities (rule-based deterministic, pretrained zero-shot, etc).
        Used by EnsemblePredictor to soft-vote across base predictors; if a
        predictor returns None it is automatically excluded from the ensemble.
        """
        return None

    def feature_importances(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_repeats: int = 3,
        random_state: int = 42,
    ) -> pd.Series | None:
        """Return feature importance as a Series indexed by column name.

        Default implementation: permutation importance via sklearn — works for
        any predictor that exposes `predict(X, dates, df)`. Subclasses with
        native importance (`.feature_importances_`, `.coef_`) should override
        for speed.

        Returns None when importance is structurally unavailable (rule-based
        deterministic predictors, pretrained zero-shot).
        """
        try:
            return _permutation_importance(
                self, X_test, y_test, n_repeats=n_repeats, random_state=random_state,
            )
        except Exception:
            return None

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"


def _permutation_importance(
    predictor: "BasePredictor",
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 3,
    random_state: int = 42,
) -> pd.Series:
    """Permutation importance: shuffle each column, measure accuracy drop.

    Uses dummy `dates`/`df` since column-shuffling preserves alignment.
    """
    from sklearn.metrics import accuracy_score
    rng = np.random.default_rng(random_state)
    base_pred = predictor.predict(X, pd.Series(np.arange(len(X))), X.iloc[:0])
    base_score = accuracy_score(y.values, base_pred)
    importances = np.zeros(X.shape[1])
    cols = list(X.columns)
    for j, col in enumerate(cols):
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[col] = rng.permutation(X_perm[col].values)
            perm_pred = predictor.predict(X_perm, pd.Series(np.arange(len(X_perm))), X_perm.iloc[:0])
            drops.append(base_score - accuracy_score(y.values, perm_pred))
        importances[j] = float(np.mean(drops))
    return pd.Series(importances, index=cols, name="importance").sort_values(ascending=False)


# Common helpers (used by sub-classes)
def labels_to_int(y: pd.Series) -> np.ndarray:
    m = {c: i for i, c in enumerate(LABEL_ORDER)}
    return np.array([m[v] for v in y.values], dtype=np.int64)


def int_to_labels(arr: np.ndarray) -> np.ndarray:
    inv = {i: c for i, c in enumerate(LABEL_ORDER)}
    return np.array([inv[int(i)] for i in arr], dtype=object)
