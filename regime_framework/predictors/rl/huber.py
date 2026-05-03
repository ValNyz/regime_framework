"""Huber regression Fitted Q Iteration (FQI).

Linear regressor with M-estimation that caps the influence of outliers.
Useful when Q-targets have heavy tails (rare reward spikes from large
prices moves) — Ridge weights such targets quadratically and gets pulled
around by them; Huber clips at the epsilon threshold and stays robust.

Like Ridge-FQI, this needs StandardScaler in front: features arrive on
heterogeneous scales (prices, z-scores, ratios) and the IRLS solver
underlying Huber is sensitive to conditioning.
"""
from __future__ import annotations

from typing import Any

from ._fqi import _FQIRLBase
from .env import (
    ACTION_DISCRETE_2,
    ACTION_DISCRETE_3,
)


class _HuberRLBase(_FQIRLBase):
    """Huber-FQI approximator. Subclasses set action_space_type."""
    approximator_kind = "huber"

    def __init__(
        self,
        alpha: float = 0.0001,    # L2 regularization
        epsilon: float = 1.35,    # Huber threshold (1.35 = sklearn default)
        max_iter: int = 200,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.max_iter = int(max_iter)

    def _make_regressor(self) -> Any:
        from sklearn.linear_model import HuberRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return Pipeline([
            ("scaler", StandardScaler()),
            ("huber", HuberRegressor(
                alpha=self.alpha,
                epsilon=self.epsilon,
                max_iter=self.max_iter,
            )),
        ])


class HuberQ2Predictor(_HuberRLBase):
    """Huber-FQI with discrete-2 action space."""
    base_name = "Huber-FQI-2"
    action_space_type = ACTION_DISCRETE_2


class HuberQ3Predictor(_HuberRLBase):
    """Huber-FQI with discrete-3 action space."""
    base_name = "Huber-FQI-3"
    action_space_type = ACTION_DISCRETE_3
