"""CatBoost Fitted Q Iteration (FQI).

The third major boosting library. Uses oblivious (symmetric) trees and
ordered boosting to mitigate target leakage on noisy data — often wins
on tabular RL targets where Q-values are themselves noisy bootstrap
estimates. Benchmark on synthetic noise: ~2 min/fold vs 26-38 min for
LGB/XGB; on real signal LGB/XGB usually catch up but CatBoost stays
competitive.
"""
from __future__ import annotations

from typing import Any

from ._fqi import _FQIRLBase
from .env import (
    ACTION_DISCRETE_2,
    ACTION_DISCRETE_3,
)


class _CatRLBase(_FQIRLBase):
    """CatBoost-FQI approximator. Subclasses set action_space_type."""
    approximator_kind = "cat"

    def __init__(
        self,
        n_estimators: int = 200,       # CatBoost calls this `iterations`
        max_depth: int = 6,            # CatBoost calls this `depth`
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3.0,      # default L2 regularization
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.l2_leaf_reg = float(l2_leaf_reg)

    def _make_regressor(self) -> Any:
        try:
            from catboost import CatBoostRegressor
        except ImportError as e:
            raise ImportError(
                "catboost not installed. Install with: pip install catboost"
            ) from e
        return CatBoostRegressor(
            iterations=self.n_estimators,
            depth=self.max_depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.seed,
            thread_count=-1,
            verbose=0,
        )


class CatQ2Predictor(_CatRLBase):
    """CatBoost-FQI with discrete-2 action space."""
    base_name = "Cat-FQI-2"
    action_space_type = ACTION_DISCRETE_2


class CatQ3Predictor(_CatRLBase):
    """CatBoost-FQI with discrete-3 action space."""
    base_name = "Cat-FQI-3"
    action_space_type = ACTION_DISCRETE_3
