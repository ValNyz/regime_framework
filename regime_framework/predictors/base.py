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

import numpy as np
import pandas as pd

from ..config import LABEL_ORDER


@dataclass
class PredictionResult:
    name: str
    family: str             # "classical" | "rule_based" | "deep" | "transformer" | "pretrained"
    accuracy: float
    kappa: float
    f1_macro: float
    confusion: list[list[int]]    # row=true, col=pred, in LABEL_ORDER
    n_test: int
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

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"


# Common helpers (used by sub-classes)
def labels_to_int(y: pd.Series) -> np.ndarray:
    m = {c: i for i, c in enumerate(LABEL_ORDER)}
    return np.array([m[v] for v in y.values], dtype=np.int64)


def int_to_labels(arr: np.ndarray) -> np.ndarray:
    inv = {i: c for i, c in enumerate(LABEL_ORDER)}
    return np.array([inv[int(i)] for i in arr], dtype=object)
