"""XGBoost Fitted Q Iteration (FQI).

Mirror of `lgb.py` — same FQI loop (in `_fqi.py`), different gradient-
boosting backend. XGBoost vs LightGBM: similar accuracy in most tabular
settings, slightly different speed/memory trade-offs. Worth comparing on
a given dataset before committing to one.
"""
from __future__ import annotations

from typing import Any

from ._fqi import _FQIRLBase
from .env import (
    ACTION_DISCRETE_2,
    ACTION_DISCRETE_3,
)


class _XGBRLBase(_FQIRLBase):
    """XGBoost-FQI approximator. Subclasses set action_space_type."""
    approximator_kind = "xgb"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)

    def _make_regressor(self) -> Any:
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise ImportError(
                "xgboost not installed. Install with: pip install xgboost"
            ) from e
        return XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_jobs=-1,
            verbosity=0,
            random_state=self.seed,
            tree_method="hist",  # fast on CPU and matches LightGBM's default
        )


class XGBQ2Predictor(_XGBRLBase):
    """XGBoost-FQI with discrete-2 action space."""
    base_name = "XGB-FQI-2"
    action_space_type = ACTION_DISCRETE_2


class XGBQ3Predictor(_XGBRLBase):
    """XGBoost-FQI with discrete-3 action space."""
    base_name = "XGB-FQI-3"
    action_space_type = ACTION_DISCRETE_3
