"""Base class for RL predictors.

Shared infrastructure across the three approximators (NN, Linear-Q,
LightGBM-FQI):
  - Single class hierarchy via MultiCoinAware mixin.
  - Common fit/predict shape: build env(s) → train approximator → run
    policy at predict time → project actions to bull/bear/flat labels.
  - Subclasses override `_train_approximator` (consumes one or more envs
    and updates internal state) and `_act` (single-step action selection).

The cold/warm dispatcher pattern (used by classical predictors) applies
here too: subclasses can support FT by keeping their model state across
fit() calls instead of re-initializing.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..base import BasePredictor, MultiCoinAware
from ...config import LABEL_ORDER
from .env import (
    RegimeTradingEnv,
    action_to_position,
    position_to_label,
    ACTION_DISCRETE_2,
    ACTION_DISCRETE_3,
    ACTION_CONTINUOUS,
    VALID_ACTION_SPACES,
)


class RLBasePredictor(MultiCoinAware, BasePredictor):
    """Common base for all RL predictors.

    Subclass contract:
      - Set class attributes: base_name, action_space_type, approximator_kind.
      - Override _train_approximator(envs_data: list[tuple]) -> None.
      - Override _act(obs) -> action.
      - Optional: override _has_prior_state() and _warm_train for FT mode.
    """
    family = "rl"
    is_rl = True
    # FT semantics for RL predictors are ambiguous: the current implementation
    # would re-train on the full train window with a reduced budget, which is
    # neither classical "warm-start fine-tune" (only new data) nor a useful
    # comparison to the cold variant. Disabled until the right flavor is
    # decided — flip back to True (and likely add an `ft_only_new_data` knob)
    # if/when we want it.
    supports_finetune = False
    base_name: str = ""
    action_space_type: str = ""
    approximator_kind: str = ""  # "nn" | "linear" | "lgb" | "xgb" | "rf" | "ridge"

    def __init__(
        self,
        finetune: bool = False,
        transaction_cost: float = 0.0,
        flat_threshold: float = 0.05,
        total_timesteps: int = 100000,
        ft_steps_scale: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__()  # MultiCoinAware initializes _target_coin_data etc.
        if self.action_space_type not in VALID_ACTION_SPACES:
            raise ValueError(
                f"{self.__class__.__name__}: action_space_type must be set "
                f"to one of {VALID_ACTION_SPACES}"
            )
        self.finetune = bool(finetune)
        self.is_finetune = self.finetune
        self.name = self.base_name + ("-FT" if self.finetune else "")
        self.transaction_cost = float(transaction_cost)
        self.flat_threshold = float(flat_threshold)
        self.total_timesteps = int(total_timesteps)
        self.ft_steps_scale = float(ft_steps_scale)
        self._extra_kwargs = kwargs  # passed through to subclass-specific configs

    # ------------------------------------------------------------------
    # Helpers (build envs, project actions to labels)
    # ------------------------------------------------------------------
    def _build_env(self, features: np.ndarray, closes: np.ndarray) -> RegimeTradingEnv:
        """Build a single env. Both arrays must be the same length."""
        return RegimeTradingEnv(
            features=features,
            closes=closes,
            action_space_type=self.action_space_type,
            transaction_cost=self.transaction_cost,
            flat_threshold=self.flat_threshold,
        )

    def _envs_data_from_fit_args(
        self,
        X_train: pd.DataFrame,
        df_train: pd.DataFrame,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return list of (features, closes) tuples — one per coin.

        Multi-coin: target + extras when set_multi_coin_data was called.
        Mono-coin: just (X_train.values, df_train.close.values).
        """
        envs_data: list[tuple[np.ndarray, np.ndarray]] = []
        if self._target_coin_data is not None and self._extra_coin_data is not None:
            # Multi-coin path — target + each extra coin
            tdata = self._target_coin_data
            envs_data.append(
                (np.asarray(tdata["X"].values, dtype=np.float32),
                 np.asarray(tdata["df"]["close"].values, dtype=np.float64))
            )
            for _coin, edata in self._extra_coin_data.items():
                envs_data.append(
                    (np.asarray(edata["X"].values, dtype=np.float32),
                     np.asarray(edata["df"]["close"].values, dtype=np.float64))
                )
        else:
            # Mono-coin fallback
            if "close" not in df_train.columns:
                raise ValueError(
                    f"{self.name}: df_train missing 'close' column — required "
                    f"for RL reward. (Either single-coin run with proper df, "
                    f"or set_multi_coin_data must have been called.)"
                )
            envs_data.append(
                (np.asarray(X_train.values, dtype=np.float32),
                 np.asarray(df_train["close"].values, dtype=np.float64))
            )
        return envs_data

    # ------------------------------------------------------------------
    # Standard predictor lifecycle (cold/warm dispatcher inherited via the
    # base flow — RL doesn't strictly need it but we mirror the pattern)
    # ------------------------------------------------------------------
    def fit(self, X_train, y_train, dates_train, df_train):
        envs_data = self._envs_data_from_fit_args(X_train, df_train)
        is_ft = self.finetune and self._has_prior_state()
        timesteps = (
            int(self.total_timesteps * self.ft_steps_scale) if is_ft
            else self.total_timesteps
        )
        self._train_approximator(envs_data, total_timesteps=timesteps, warm=is_ft)
        return self

    def predict(self, X_test, dates_test, df_test):
        features = np.asarray(X_test.values, dtype=np.float32)
        closes = np.asarray(df_test["close"].values, dtype=np.float64) if "close" in df_test.columns else None
        n = len(features)
        idx_to_lbl = {i: c for i, c in enumerate(LABEL_ORDER)}
        out = np.full(n, "", dtype=object)
        # Use closes for env if available; otherwise fake stationary prices
        # (predict-time prices don't affect actions, only the env's returned
        # reward which we ignore).
        if closes is None or len(closes) != n:
            closes = np.ones(n, dtype=np.float64) * 100.0
        env = self._build_env(features, closes)
        obs, _ = env.reset()
        for t in range(n - 1):
            action = self._act(obs)
            position = action_to_position(action, self.action_space_type, self.flat_threshold)
            out[t] = position_to_label(position, self.flat_threshold)
            obs, _, terminated, _, _ = env.step(action)
            if terminated:
                break
        # Last bar: we never decided an action for it (no t+1 return) — copy
        # the previous label for visual continuity.
        if n >= 2:
            out[n - 1] = out[n - 2]
        return out

    def predict_proba(self, X_test, dates_test, df_test):
        # Excluded from Ensemble for now per user spec — return None to skip.
        return None

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------
    def _train_approximator(
        self,
        envs_data: list[tuple[np.ndarray, np.ndarray]],
        total_timesteps: int,
        warm: bool,
    ) -> None:
        """Train the approximator on the given (features, closes) pairs.

        envs_data: one tuple per coin (target first if multi-coin).
        total_timesteps: training budget. Reduced for FT mode.
        warm: True if FT mode AND prior state exists. Subclass may continue
            from existing model weights instead of re-initializing.
        """
        raise NotImplementedError

    def _act(self, obs: np.ndarray):
        """Single-step action selection at predict time. Returns the env-native
        action (int for discrete, float for continuous)."""
        raise NotImplementedError

    def _has_prior_state(self) -> bool:
        """Override to return True when the subclass already has a trained
        approximator from a previous fit() call (used for FT warm-start)."""
        return False
