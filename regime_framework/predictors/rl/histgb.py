"""sklearn HistGradientBoosting Fitted Q Iteration (FQI).

Same family as LightGBM: histogram-based gradient boosting that bins
features into a fixed number of buckets before splitting. The benchmark
showed HistGB is 50–100× faster than LGBM on noisy Q-targets because
sklearn's implementation has tighter `min_gain_to_split` defaults that
prune away noise-only splits early. Often loses 1–3 points of accuracy
to LGBM on real signal but wins on wall-clock — useful as a fast bench
predictor or when LightGBM isn't installed.
"""
from __future__ import annotations

from typing import Any

from ._fqi import _FQIRLBase
from .env import (
    ACTION_DISCRETE_2,
    ACTION_DISCRETE_3,
)


class _HistGBRLBase(_FQIRLBase):
    """HistGB-FQI approximator. Subclasses set action_space_type."""
    approximator_kind = "histgb"

    def __init__(
        self,
        max_iter: int = 200,           # number of boosting iterations (≈ n_estimators)
        max_depth: int = 6,
        learning_rate: float = 0.05,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.max_iter = int(max_iter)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.min_samples_leaf = int(min_samples_leaf)
        self.l2_regularization = float(l2_regularization)

    def _make_regressor(self) -> Any:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            random_state=self.seed,
        )


class HistGBQ2Predictor(_HistGBRLBase):
    """HistGB-FQI with discrete-2 action space."""
    base_name = "HistGB-FQI-2"
    action_space_type = ACTION_DISCRETE_2


class HistGBQ3Predictor(_HistGBRLBase):
    """HistGB-FQI with discrete-3 action space."""
    base_name = "HistGB-FQI-3"
    action_space_type = ACTION_DISCRETE_3
