"""Random Forest Fitted Q Iteration (FQI).

Mirror of `lgb.py` / `xgb.py` — same FQI loop (in `_fqi.py`), different
regressor. RF has no shrinkage (no learning_rate), so on small per-action
batches it tends to overfit relative to boosted trees, but it's a useful
baseline and very easy to tune (mostly just n_estimators).
"""
from __future__ import annotations

from typing import Any

from ._fqi import _FQIRLBase
from .env import (
    ACTION_DISCRETE_2,
    ACTION_DISCRETE_3,
)


class _RFRLBase(_FQIRLBase):
    """RandomForest-FQI approximator. Subclasses set action_space_type."""
    approximator_kind = "rf"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_estimators = int(n_estimators)
        self.max_depth = None if max_depth is None else int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)

    def _make_regressor(self) -> Any:
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=-1,
            random_state=self.seed,
        )


class RFQ2Predictor(_RFRLBase):
    """RandomForest-FQI with discrete-2 action space."""
    base_name = "RF-FQI-2"
    action_space_type = ACTION_DISCRETE_2


class RFQ3Predictor(_RFRLBase):
    """RandomForest-FQI with discrete-3 action space."""
    base_name = "RF-FQI-3"
    action_space_type = ACTION_DISCRETE_3
