"""Ridge regression Fitted Q Iteration (FQI).

Closed-form linear regressor for Q(s, a) — same FQI loop as the boosted
variants, but each per-action model is just a regularized linear fit.

Coexists with `LinearQ` (in `linear.py`) which uses online TD(0) updates.
The two differ in *training algorithm*, not hypothesis class:
  - LinearQ:  online stochastic update (TD-error * features)
  - Ridge-FQI: closed-form solve at each FQI iteration on the full batch

LinearQ is faster and matches the standard "Linear Q-learning" recipe.
Ridge-FQI is more stable on noisy rewards (least-squares averages out
noise across the whole batch) but slower per fold for the same budget.
"""
from __future__ import annotations

from typing import Any

from ._fqi import _FQIRLBase
from .env import (
    ACTION_DISCRETE_2,
    ACTION_DISCRETE_3,
)


class _RidgeRLBase(_FQIRLBase):
    """Ridge-FQI approximator. Subclasses set action_space_type."""
    approximator_kind = "ridge"

    def __init__(
        self,
        alpha: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.alpha = float(alpha)

    def _make_regressor(self) -> Any:
        from sklearn.linear_model import Ridge
        return Ridge(alpha=self.alpha, random_state=self.seed)


class RidgeQ2Predictor(_RidgeRLBase):
    """Ridge-FQI with discrete-2 action space."""
    base_name = "Ridge-FQI-2"
    action_space_type = ACTION_DISCRETE_2


class RidgeQ3Predictor(_RidgeRLBase):
    """Ridge-FQI with discrete-3 action space."""
    base_name = "Ridge-FQI-3"
    action_space_type = ACTION_DISCRETE_3
