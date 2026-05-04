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

    def __init__(
        self,
        finetune: bool = False,
        bases_filter: list[str] | None = None,
        name_suffix: str = "",
        proba_normalize: bool = False,
    ) -> None:
        """
        bases_filter: when provided, the ensemble only averages over base
            predictors whose cold name (stripped of -FT suffix) matches one of
            these. Used to build subset ensembles (e.g. trees-only, linear-only).
            None = use all bases of the appropriate variant.
        name_suffix: inserted between base_name and the optional -FT, so a
            tree-only ensemble shows up as 'Ensemble-trees' / 'Ensemble-trees-FT'.
        proba_normalize: when True, each base's proba is quantile-calibrated
            to uniform[0.5, 1.0] in max(proba) before averaging. Equalizes
            vote influence across heterogeneous bases (e.g. LogReg's near
            one-hot vs RL's softer softmax). Without this, a base with
            consistently sharper proba (margin ~0.98) drowns out bases with
            softer proba (margin ~0.2) regardless of nominal weights.
        """
        self.finetune = bool(finetune)
        self.is_finetune = self.finetune
        self.name = self.base_name + name_suffix + ("-FT" if self.finetune else "")
        self._bases_filter = list(bases_filter) if bases_filter is not None else None
        self.proba_normalize = bool(proba_normalize)
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

        Deduplication by variant: the cold ensemble averages only cold bases,
        the FT ensemble averages only FT bases. Without this, having both
        LightGBM and LightGBM-FT in the same run would double-count each
        underlying classifier and amplify disagreement on ambiguous bars.
        When `include_finetune=False` there are no `-FT` names, so the cold
        ensemble's filter degrades to "everything", which is correct.
        """
        def _is_ft(name: str) -> bool:
            return name.endswith("-FT")

        def _strip_ft(name: str) -> str:
            return name[:-3] if _is_ft(name) else name

        if self.finetune:
            filtered = {k: v for k, v in base_probas.items() if _is_ft(k)}
        else:
            filtered = {k: v for k, v in base_probas.items() if not _is_ft(k)}

        # Subset filter: keep only bases whose cold name is in self._bases_filter.
        if self._bases_filter is not None:
            allow = set(self._bases_filter)
            filtered = {k: v for k, v in filtered.items() if _strip_ft(k) in allow}

        self._fold_base_probas = {
            k: v for k, v in filtered.items()
            if v is not None and v.ndim == 2 and v.shape[1] == len(LABEL_ORDER)
        }

    def update_prior_kappas(self, kappas: dict[str, float]) -> None:
        """Runner calls this at end of each fold so the FT variant can re-weight
        next fold's vote. No-op for the cold variant.
        """
        self._prior_kappas = {k: float(v) for k, v in kappas.items() if not np.isnan(v)}

    def get_effective_bases(self) -> list[str]:
        """Names of base predictors actually used in this fold's vote — i.e.
        post-deduplication (cold ensembles → cold bases only; FT ensembles →
        FT bases only). For runner display, never the full unfiltered set.
        """
        return list(self._fold_base_probas.keys())

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
        base_probas = self._maybe_normalize_probas(self._fold_base_probas)
        # Weighted sum: stack is (M, N, C); weights is (M,).
        stack = np.stack([base_probas[n] for n in names], axis=0)
        avg = np.tensordot(weights, stack, axes=1)  # (N, C)
        # Re-normalize for numerical safety (weighted softmax inputs).
        avg = avg / np.maximum(avg.sum(axis=1, keepdims=True), 1e-12)
        return avg.astype(np.float32)

    # ------------------------------------------------------------------
    # Per-base proba calibration (optional)
    # ------------------------------------------------------------------
    def _maybe_normalize_probas(
        self, base_probas: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Apply quantile normalization to each base's proba if enabled.
        No-op when proba_normalize=False (cheap reference passthrough)."""
        if not self.proba_normalize:
            return base_probas
        return {n: self._quantile_normalize_proba(p) for n, p in base_probas.items()}

    @staticmethod
    def _quantile_normalize_proba(proba: np.ndarray) -> np.ndarray:
        """Per-base proba calibration: rank max(proba) across bars and map to
        uniform[0.5, 1.0]. The full proba vector is rescaled at each bar so
        the chosen class gets the new normalized max, with the remaining
        (1 - new_max) redistributed across non-max classes proportional to
        their original ratios.

        Equalizes vote influence across heterogeneous bases — a base whose
        native proba is sharp (e.g. LogReg ~ [0.99, 0.01]) is brought down
        to the same calibration as a base whose native proba is soft
        (e.g. RL ~ [0.6, 0.4]). Each base's relative confidence ordering
        within its own bars is preserved; only the absolute scale is
        normalized to [0.5, 1.0].
        """
        proba = np.asarray(proba, dtype=np.float64)
        n, c = proba.shape
        if n == 0:
            return proba.astype(np.float32)
        argmax = proba.argmax(axis=1)              # (N,)
        max_vals = proba[np.arange(n), argmax]     # (N,)
        # Quantile rank max(proba) → uniform [0.5, 1.0]
        order = np.argsort(max_vals, kind="stable")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n)
        rank_fraction = ranks / max(n - 1, 1)      # [0, 1]
        new_max = 0.5 + 0.5 * rank_fraction        # [0.5, 1.0]
        # Build mask: True for non-argmax classes per row
        non_argmax = np.ones((n, c), dtype=bool)
        non_argmax[np.arange(n), argmax] = False
        other_sums = (proba * non_argmax).sum(axis=1)        # (N,)
        remaining = 1.0 - new_max                            # (N,)
        # Scale non-argmax probas: keep ratios, total = remaining
        safe_other = np.maximum(other_sums, 1e-12)
        scale = remaining / safe_other                       # (N,)
        out = proba * non_argmax * scale[:, None]            # (N, C)
        # Rows where all mass was at argmax: distribute uniformly among others
        all_at_argmax = other_sums <= 1e-12
        if all_at_argmax.any() and c > 1:
            uniform_share = remaining[all_at_argmax] / (c - 1)
            out[all_at_argmax] = non_argmax[all_at_argmax] * uniform_share[:, None]
        # Place the new max at the chosen class
        out[np.arange(n), argmax] = new_max
        return out.astype(np.float32)

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


class ConfidenceEnsemblePredictor(EnsemblePredictor):
    """Confidence-weighted soft-voting ensemble. Sibling of EnsemblePredictor:
    same plumbing (feed_base_probas, update_prior_kappas, FT/cold split, base
    deduplication), but each base's vote is weighted *per bar* by its own
    max(predict_proba) — so a base that's very confident at bar t dominates
    the average there even if its global weight is low.

    Combines multiplicatively with the global weight (uniform for cold,
    softmax-of-prior-kappa for FT). The two trust signals — "this base is
    historically reliable" (global) and "this base is confident on this
    specific bar" (per-bar) — compose cleanly.

    With M predictors, N test bars, C classes:
      stack    : (M, N, C)  — per-predictor probas
      cert     : (M, N)     — per-predictor max-class proba per bar
      combined : (M, N)     = global_w[:, None] * cert, normalized over M per bar
      avg      : (N, C)     = sum_m combined[m, t] * stack[m, t, :]
    """
    base_name = "Ensemble-Conf"

    def predict_proba(self, X_test, dates_test, df_test):
        if not self._fold_base_probas:
            return None
        names = list(self._fold_base_probas.keys())
        global_w = self._compute_weights(names)
        base_probas = self._maybe_normalize_probas(self._fold_base_probas)
        stack = np.stack([base_probas[n] for n in names], axis=0)  # (M, N, C)
        cert = stack.max(axis=2)                                              # (M, N)
        combined = global_w[:, None] * cert                                   # (M, N)
        combined = combined / np.maximum(combined.sum(axis=0, keepdims=True), 1e-12)
        avg = (combined[:, :, None] * stack).sum(axis=0)                      # (N, C)
        avg = avg / np.maximum(avg.sum(axis=1, keepdims=True), 1e-12)
        return avg.astype(np.float32)
