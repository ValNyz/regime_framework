"""Huber regression Fitted Q Iteration (FQI).

Linear regressor with M-estimation that caps the influence of outliers.
Useful when Q-targets have heavy tails (rare reward spikes from large
prices moves) — Ridge weights such targets quadratically and gets pulled
around by them; Huber clips at the epsilon threshold and stays robust.

Convergence note: Huber's lbfgs solver routinely hits max_iter on FQI's
noisy Q-targets, even after scaling features and y. The IRLS reweighting
shifts samples each iteration, and bootstrap targets compound the noise.
The partial fit is a valid estimator (just not loss-optimal) — for FQI
that's fine, the targets themselves are noisy bootstrap estimates. We
suppress the ConvergenceWarning at module load since it's structural
rather than actionable.
"""
from __future__ import annotations

import warnings
from typing import Any

from sklearn.exceptions import ConvergenceWarning

from ._fqi import _FQIRLBase
from .env import (
    ACTION_DISCRETE_2,
    ACTION_DISCRETE_3,
)


# See module docstring — Huber's lbfgs convergence on noisy FQI Q-targets
# is structural; partial fits are still valid estimators.
warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning,
    module="sklearn.linear_model._huber",
)


class _HuberRLBase(_FQIRLBase):
    """Huber-FQI approximator. Subclasses set action_space_type."""
    approximator_kind = "huber"

    def __init__(
        self,
        alpha: float = 0.0001,    # L2 regularization
        epsilon: float = 1.35,    # Huber threshold (1.35 = sklearn default)
        max_iter: int = 200,
        tol: float = 1e-3,        # looser than sklearn 1e-5; targets are noisy
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def _make_regressor(self) -> Any:
        # Scale BOTH features and targets. FQI's Q-targets are tiny (~1e-4
        # log-returns); without target scaling the gradient is too small to
        # ever cross lbfgs's tolerance, so the solver always hits max_iter
        # and emits a ConvergenceWarning. TransformedTargetRegressor wraps
        # the y in a StandardScaler at fit and inverse-transforms at predict.
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.linear_model import HuberRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        feature_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("huber", HuberRegressor(
                alpha=self.alpha,
                epsilon=self.epsilon,
                max_iter=self.max_iter,
                tol=self.tol,
            )),
        ])
        return TransformedTargetRegressor(
            regressor=feature_pipeline,
            transformer=StandardScaler(),
        )


class HuberQ2Predictor(_HuberRLBase):
    """Huber-FQI with discrete-2 action space."""
    base_name = "Huber-FQI-2"
    action_space_type = ACTION_DISCRETE_2


class HuberQ3Predictor(_HuberRLBase):
    """Huber-FQI with discrete-3 action space."""
    base_name = "Huber-FQI-3"
    action_space_type = ACTION_DISCRETE_3
