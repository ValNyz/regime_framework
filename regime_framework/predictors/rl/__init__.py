"""Reinforcement learning predictors.

Three approximators × multiple action spaces, all sharing the same
RegimeTradingEnv and the framework's BasePredictor interface:

  Approximator | Action spaces             | Backend
  -------------|---------------------------|------------------
  NN           | Discrete-2/-3, Continuous | Stable-Baselines3
  Linear-Q     | Discrete-2/-3             | custom (no SB3)
  LightGBM-FQI | Discrete-2/-3             | custom (no SB3)

Continuous-action approximators are NN-only because Linear/LightGBM
value-based RL doesn't extend cleanly to continuous actions (would
require an actor-critic architecture, not just a Q-function).

For continuous predictors, the policy outputs a signed position size
in [-1, +1]. At predict-time, that position is projected back to the
framework's bull/bear/flat label format via `position_to_label`. This
loses the magnitude — synth_gain reflects sign-only behavior. The
underlying training still optimizes on continuous gradient (correct);
just the post-hoc reporting is sign-quantized.

Multi-coin training is supported via the MultiCoinAware mixin (see
regime_framework/predictors/base.py): the runner passes per-coin
(features, closes, dates) tuples to RL predictors before fit(); the
predictors build N envs running in parallel (vec_env for NN, mixed
buffer for Linear/LightGBM-FQI). At test time only the target coin
is used.
"""
from .env import (
    RegimeTradingEnv,
    action_to_position,
    position_to_label,
    ACTION_DISCRETE_2,
    ACTION_DISCRETE_3,
    ACTION_CONTINUOUS,
    VALID_ACTION_SPACES,
)
from .base import RLBasePredictor
from .nn import DQN2Predictor, DQN3Predictor, SACPredictor
from .linear import LinearQ2Predictor, LinearQ3Predictor
from .lgb import LGBQ2Predictor, LGBQ3Predictor


__all__ = [
    "RegimeTradingEnv",
    "action_to_position",
    "position_to_label",
    "ACTION_DISCRETE_2",
    "ACTION_DISCRETE_3",
    "ACTION_CONTINUOUS",
    "VALID_ACTION_SPACES",
    "RLBasePredictor",
    "DQN2Predictor",
    "DQN3Predictor",
    "SACPredictor",
    "LinearQ2Predictor",
    "LinearQ3Predictor",
    "LGBQ2Predictor",
    "LGBQ3Predictor",
]
