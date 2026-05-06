"""Gymnasium-compatible environment for regime-detection RL.

A single configurable env class supports all three action spaces:
  - 'discrete-2'  : actions {0=short, 1=long} — always in a position
  - 'discrete-3'  : actions {0=flat,  1=long, 2=short}
  - 'continuous'  : action ∈ [-1, +1] = signed position size

Reward at bar t:
    r_t = a_t * log(close[t+1] / close[t]) - cost * |a_t - a_{t-1}|

where a_t is the signed position in [-1, +1] (-1=full short, 0=flat, +1=full long).
For the discrete spaces we map action ids to signed positions before applying
the formula. The transaction-cost term is disable-able (cost=0).

The episode iterates from bar 0 to bar n-2 (last step has no t+1 return).
Observation = features X[t] (numpy array of shape (n_features,)).
"""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # gymnasium is optional — only required for RL predictors
    gym = None  # type: ignore
    spaces = None  # type: ignore


# Action space identifier strings
ACTION_DISCRETE_2 = "discrete-2"
ACTION_DISCRETE_3 = "discrete-3"
ACTION_CONTINUOUS = "continuous"
VALID_ACTION_SPACES = (ACTION_DISCRETE_2, ACTION_DISCRETE_3, ACTION_CONTINUOUS)


def action_to_position(action: Any, action_space_type: str, flat_threshold: float = 0.05) -> float:
    """Map an env action to a signed position in [-1, +1].

    Single source of truth used both inside the env (reward) and by RL
    predictors at predict() time (when projecting back to bull/bear/flat
    labels for synth_gain integration).
    """
    if action_space_type == ACTION_DISCRETE_2:
        # 0 = short, 1 = long
        return -1.0 if int(action) == 0 else +1.0
    if action_space_type == ACTION_DISCRETE_3:
        # 0 = flat, 1 = long, 2 = short
        a = int(action)
        if a == 0:
            return 0.0
        if a == 1:
            return +1.0
        return -1.0
    if action_space_type == ACTION_CONTINUOUS:
        a = float(np.clip(action, -1.0, 1.0))
        # Threshold near zero → treat as flat (numerical noise / tiny exposures)
        if abs(a) < flat_threshold:
            return 0.0
        return a
    raise ValueError(f"Unknown action_space_type: {action_space_type!r}")


def position_to_label(position: float, flat_threshold: float = 0.05) -> str:
    """Map a signed position back to the framework's label format.

    Loses the magnitude for continuous positions (sign-only). This is an
    accepted trade-off for synth_gain integration — see
    regime_framework/predictors/rl/__init__.py for full rationale.
    """
    if abs(position) < flat_threshold:
        return ""
    return "bull" if position > 0 else "bear"


class RegimeTradingEnv(gym.Env if gym is not None else object):  # type: ignore[misc]
    """One env per coin. Multi-coin training uses N envs in parallel via
    SB3's `make_vec_env` (commit 3) — this class stays single-coin.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        closes: np.ndarray,
        action_space_type: str = ACTION_DISCRETE_3,
        transaction_cost: float = 0.0005,
        flat_threshold: float = 0.05,
    ) -> None:
        if gym is None:
            raise ImportError(
                "gymnasium is required for RL predictors. "
                "Install with: pip install stable-baselines3"
            )
        if action_space_type not in VALID_ACTION_SPACES:
            raise ValueError(
                f"action_space_type must be one of {VALID_ACTION_SPACES} "
                f"(got {action_space_type!r})"
            )
        super().__init__()
        self.features = np.asarray(features, dtype=np.float32)
        self.closes = np.asarray(closes, dtype=np.float64)
        if len(self.features) != len(self.closes):
            raise ValueError(
                f"features ({len(self.features)}) and closes ({len(self.closes)}) length mismatch"
            )
        if len(self.closes) < 2:
            raise ValueError(f"need at least 2 bars (got {len(self.closes)})")
        self.action_space_type = action_space_type
        self.transaction_cost = float(transaction_cost)
        self.flat_threshold = float(flat_threshold)

        # Pre-compute log returns for reward — avoids per-step recomputation.
        self._log_ret = np.log(self.closes[1:] / self.closes[:-1])  # length n-1

        # Build gymnasium action / observation spaces
        n_features = self.features.shape[1] if self.features.ndim == 2 else 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32,
        )
        if action_space_type == ACTION_DISCRETE_2:
            self.action_space = spaces.Discrete(2)
        elif action_space_type == ACTION_DISCRETE_3:
            self.action_space = spaces.Discrete(3)
        else:  # continuous
            self.action_space = spaces.Box(
                low=-1.0, high=+1.0, shape=(1,), dtype=np.float32,
            )

        # Episode state
        self._t = 0
        self._prev_position = 0.0  # for transaction cost

    def _obs(self) -> np.ndarray:
        return self.features[self._t].astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._t = 0
        self._prev_position = 0.0
        return self._obs(), {}

    def step(self, action):
        position = action_to_position(action, self.action_space_type, self.flat_threshold)
        # Reward: position * next-bar log return, minus transaction cost
        if self._t < len(self._log_ret):
            log_ret = float(self._log_ret[self._t])
        else:
            log_ret = 0.0
        reward = position * log_ret
        if self.transaction_cost > 0.0:
            reward -= self.transaction_cost * abs(position - self._prev_position)
        self._prev_position = position

        self._t += 1
        # Last bar (t == n-1) has no t+1 — episode ends.
        terminated = self._t >= len(self.features) - 1
        truncated = False
        # If we're still within bounds, return current observation; otherwise
        # any zero observation works since the next step won't be evaluated.
        if self._t < len(self.features):
            obs = self._obs()
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(reward), terminated, truncated, {}
