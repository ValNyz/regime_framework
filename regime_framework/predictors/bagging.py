"""Bagging meta-predictor.

Wraps a base predictor in N parallel instances trained on different bootstrap
samples (classical) or different seeds (RL). Aggregation: mean of bag probas
when available, falling back to majority vote on hard predictions.

Two flavors via `bootstrap`:
  - bootstrap=True (classical) : each bag fits on a row-resampled (with
    replacement) view of the training data. Variance source = sample.
  - bootstrap=False (RL)       : every bag fits on the same data with a
    different seed. Variance source = policy stochasticity.

The wrapper is transparent: it inherits name/family from the first fitted
bag, so downstream Ensemble / metrics see it as a single predictor.

Eligibility: only wrap predictors that override BasePredictor.predict_proba
to return non-None. Rule-based / pretrained zero-shot have no proba and
gain nothing from bag-averaging deterministic outputs.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable

import numpy as np
import pandas as pd

from .base import BasePredictor
from ..config import LABEL_ORDER


class BaggingWrapper(BasePredictor):
    """Train N copies of a base predictor; aggregate via averaged proba.

    Args:
        base_factory: callable (seed: int) -> BasePredictor producing a fresh
            unfit instance. Receives a per-bag seed; for RL this is forwarded
            to the constructor; for classical this should mutate the
            underlying sklearn clf's random_state.
        n_bags:    number of base instances to train (must be >= 1).
        base_seed: seed for the first bag; bag i gets `base_seed + i`.
        bootstrap: if True, fit each bag on a bootstrap (with replacement)
            of the training rows. If False, fit on identical data (relies on
            seed-driven variance only). Use True for classical, False for RL.
    """

    def __init__(
        self,
        base_factory: Callable[[int], BasePredictor],
        n_bags: int,
        base_seed: int = 42,
        bootstrap: bool = True,
    ) -> None:
        if n_bags < 1:
            raise ValueError(f"BaggingWrapper.n_bags must be >=1, got {n_bags}")
        self.base_factory = base_factory
        self.n_bags = int(n_bags)
        self.base_seed = int(base_seed)
        self.bootstrap = bool(bootstrap)
        self.bags: list[BasePredictor] = []
        # Display fields populated from the first fitted bag (transparent name).
        self.name = ""
        self.family = ""
        self.needs_features = True

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        dates_train: pd.Series,
        df_train: pd.DataFrame,
    ) -> "BaggingWrapper":
        rng = np.random.default_rng(self.base_seed)
        self.bags = []
        for i in range(self.n_bags):
            base = self.base_factory(self.base_seed + i)
            if i == 0:
                self.name = base.name
                self.family = base.family
                self.needs_features = getattr(base, "needs_features", True)
            if self.bootstrap and len(X_train) > 0:
                idx = rng.integers(0, len(X_train), size=len(X_train))
                Xb = X_train.iloc[idx]
                yb = y_train.iloc[idx]
                db = dates_train.iloc[idx]
                dfb = df_train.iloc[idx]
                base.fit(Xb, yb, db, dfb)
            else:
                base.fit(X_train, y_train, dates_train, df_train)
            self.bags.append(base)
        return self

    def predict_proba(
        self,
        X_test: pd.DataFrame,
        dates_test: pd.Series,
        df_test: pd.DataFrame,
    ):
        probas = []
        for b in self.bags:
            p = b.predict_proba(X_test, dates_test, df_test)
            if p is not None:
                probas.append(np.asarray(p, dtype=np.float64))
        if not probas:
            return None
        return np.mean(np.stack(probas), axis=0)

    def predict(
        self,
        X_test: pd.DataFrame,
        dates_test: pd.Series,
        df_test: pd.DataFrame,
    ) -> np.ndarray:
        avg = self.predict_proba(X_test, dates_test, df_test)
        if avg is not None:
            idx_to_cls = {i: c for i, c in enumerate(LABEL_ORDER)}
            return np.array([idx_to_cls[int(i)] for i in avg.argmax(axis=1)], dtype=object)
        # No bag exposes proba: majority vote across hard predictions.
        preds = np.stack(
            [b.predict(X_test, dates_test, df_test) for b in self.bags], axis=0
        )
        return np.array(
            [Counter(col).most_common(1)[0][0] for col in preds.T], dtype=object
        )

    def feature_importances(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_repeats: int = 3,
        random_state: int = 42,
    ):
        """Average per-bag importances; None if no bag exposes them."""
        imps: list[pd.Series] = []
        for b in self.bags:
            try:
                imp = b.feature_importances(X_test, y_test, n_repeats, random_state)
            except Exception:
                imp = None
            if imp is not None:
                imps.append(imp)
        if not imps:
            return None
        return sum(imps) / len(imps)


def _has_predict_proba(predictor: BasePredictor) -> bool:
    """True iff the predictor's class overrides BasePredictor.predict_proba.

    Cheap structural check; doesn't run predict_proba (some implementations
    only know they have/lack proba once fitted, but the override-vs-default
    test catches all non-rule-based predictors at build time).
    """
    return type(predictor).predict_proba is not BasePredictor.predict_proba
