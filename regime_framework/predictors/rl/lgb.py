"""LightGBM Fitted Q Iteration (FQI).

The FQI loop lives in `_fqi.py` (regressor-agnostic). This module just
provides the LightGBM regressor factory + the concrete predictor classes.
"""
from __future__ import annotations

from typing import Any

from ._fqi import _FQIRLBase
from .env import (
    ACTION_DISCRETE_2,
    ACTION_DISCRETE_3,
)


class _LGBRLBase(_FQIRLBase):
    """LightGBM-FQI approximator. Subclasses set action_space_type."""
    approximator_kind = "lgb"

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
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_jobs=-1,
            verbosity=-1,
            random_state=self.seed,
        )


class LGBQ2Predictor(_LGBRLBase):
    """LightGBM-FQI with discrete-2 action space."""
    base_name = "LGB-FQI-2"
    action_space_type = ACTION_DISCRETE_2


class LGBQ3Predictor(_LGBRLBase):
    """LightGBM-FQI with discrete-3 action space."""
    base_name = "LGB-FQI-3"
    action_space_type = ACTION_DISCRETE_3
