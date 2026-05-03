"""Soft-voting ensemble predictor.

Averages `predict_proba` of every base predictor in the current run that
exposes one (rule-based and pretrained zero-shot are skipped automatically).
At each fold the runner:
  1. Fits + predicts every base predictor as usual.
  2. Calls `feed_base_probas` on the ensemble with the per-base predict_proba
     dict for the fold's test set.
  3. Calls the ensemble's `predict()` — which argmaxes the weighted average.

Variants:
  - `Ensemble`     (cold)  : uniform 1/N weights every fold.
  - `Ensemble-FT`  (warm) : weights = softmax(prior fold's per-base kappa).
                            Falls back to uniform on fold 1.
"""
from __future__ import annotations

import numpy as np

from .base import BasePredictor
from ..config import LABEL_ORDER


class EnsemblePredictor(BasePredictor):
    """Soft-vote ensemble. Special: handled outside the normal fit/predict path
    by the runner because it depends on other predictors' outputs.
    """
    family = "ensemble"
    base_name = "Ensemble"
    supports_finetune = True
    is_ensemble = True   # marker — runner processes ensembles after base predictors

    def __init__(self, finetune: bool = False) -> None:
        self.finetune = bool(finetune)
        self.is_finetune = self.finetune
        self.name = self.base_name + ("-FT" if self.finetune else "")
        # State held across folds (FT only): predictor_name -> kappa from last fold
        self._prior_kappas: dict[str, float] = {}
        # State held within a fold: predictor_name -> proba (n_test, n_classes)
        self._fold_base_probas: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Runner-facing hooks
    # ------------------------------------------------------------------
    def feed_base_probas(self, base_probas: dict[str, np.ndarray]) -> None:
        """Runner sets the current fold's base probabilities here before predict().
        Keys are base predictor names; values are (n_test, n_classes) arrays.
        """
        # Keep only base predictors that actually returned a proba matrix
        self._fold_base_probas = {
            k: v for k, v in base_probas.items()
            if v is not None and v.ndim == 2 and v.shape[1] == len(LABEL_ORDER)
        }

    def update_prior_kappas(self, kappas: dict[str, float]) -> None:
        """Runner calls this at end of each fold so the FT variant can re-weight
        next fold's vote. No-op for the cold variant.
        """
        self._prior_kappas = {k: float(v) for k, v in kappas.items() if not np.isnan(v)}

    # ------------------------------------------------------------------
    # Standard predictor interface
    # ------------------------------------------------------------------
    def fit(self, X_train, y_train, dates_train, df_train):
        # No fitting on raw features — ensemble aggregates downstream model outputs.
        return self

    def predict(self, X_test, dates_test, df_test):
        proba = self.predict_proba(X_test, dates_test, df_test)
        if proba is None:
            # No base predictors gave probabilities — ensemble can't produce a
            # vote. Return empty strings so the evaluator marks this as failed.
            return np.full(len(X_test), "", dtype=object)
        idx_to_cls = {i: c for i, c in enumerate(LABEL_ORDER)}
        idx = proba.argmax(axis=1)
        return np.array([idx_to_cls[int(i)] for i in idx], dtype=object)

    def predict_proba(self, X_test, dates_test, df_test):
        if not self._fold_base_probas:
            return None
        names = list(self._fold_base_probas.keys())
        weights = self._compute_weights(names)
        # Weighted sum: stack is (M, N, C); weights is (M,).
        stack = np.stack([self._fold_base_probas[n] for n in names], axis=0)
        avg = np.tensordot(weights, stack, axes=1)  # (N, C)
        # Re-normalize for numerical safety (weighted softmax inputs).
        avg = avg / np.maximum(avg.sum(axis=1, keepdims=True), 1e-12)
        return avg.astype(np.float32)

    # ------------------------------------------------------------------
    # Weighting strategies
    # ------------------------------------------------------------------
    def _compute_weights(self, names: list[str]) -> np.ndarray:
        if not (self.finetune and self._prior_kappas):
            # Cold or fold-1 FT: uniform weights.
            return np.full(len(names), 1.0 / len(names), dtype=np.float64)
        # FT with prior kappas: softmax with a temperature scaled by the kappa
        # spread, so weights actually differentiate base predictors. If all
        # base predictors had similar kappa, weights stay near uniform.
        kappas = np.array([self._prior_kappas.get(n, 0.0) for n in names], dtype=np.float64)
        scale = max(0.05, float(kappas.std()))
        z = kappas / scale
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()
