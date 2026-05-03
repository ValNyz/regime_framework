"""LightGBM Fitted Q Iteration (FQI) — custom, no SB3 dependency.

Classical batch RL (Ernst, Geurts, Wehenkel 2005). Workflow:

1. Collect transitions {(s, a, r, s', done)} from all envs via a random
   exploration policy (off-policy: any policy works for collection).
2. Iterate K times:
     For each transition (s, a, r, s', done):
       target = r + γ · max_a' Q(s', a')   # γ * 0 if done
     For each action a, fit a LightGBM regressor on
     (s_train_a, target_train_a) — one booster per action.
3. Use the final Q for predictions: argmax_a Q(s, a).

Naturally regularized on tabular features (no overfit knobs to tune
beyond n_estimators / max_depth). Skipped continuous: same reason as
Linear-Q — value-based, not actor-critic.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import RLBasePredictor
from .env import (
    ACTION_DISCRETE_2,
    ACTION_DISCRETE_3,
)


class _FQILearner:
    """Per-action LightGBM regressor of Q(s, a). Lazily-imported lightgbm."""
    def __init__(
        self,
        n_actions: int,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        gamma: float = 0.99,
    ):
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self._params = dict(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate=float(learning_rate),
            n_jobs=-1,
            verbosity=-1,
        )
        self._models: list[Any] = [None] * self.n_actions

    def is_fitted(self) -> bool:
        return all(m is not None for m in self._models)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Predict Q(state, a) for all actions. Returns 0 if not yet fitted."""
        out = np.zeros(self.n_actions, dtype=np.float64)
        if state.ndim == 1:
            state = state.reshape(1, -1)
        for a in range(self.n_actions):
            if self._models[a] is not None:
                out[a] = float(self._models[a].predict(state)[0])
        return out

    def q_values_batch(self, states: np.ndarray) -> np.ndarray:
        """Batch version: returns (n_samples, n_actions) array."""
        n = len(states)
        out = np.zeros((n, self.n_actions), dtype=np.float64)
        for a in range(self.n_actions):
            if self._models[a] is not None:
                out[:, a] = self._models[a].predict(states)
        return out

    def fit_iteration(self, transitions: list[tuple], n_iterations: int = 20) -> None:
        """One full FQI fit: K iterations of (compute targets → fit per-action)."""
        from lightgbm import LGBMRegressor

        if not transitions:
            return
        states = np.array([t[0] for t in transitions], dtype=np.float32)
        actions = np.array([t[1] for t in transitions], dtype=np.int64)
        rewards = np.array([t[2] for t in transitions], dtype=np.float32)
        next_states = np.array([t[3] for t in transitions], dtype=np.float32)
        dones = np.array([t[4] for t in transitions], dtype=bool)

        for k in range(n_iterations):
            # Compute target Q for each transition.
            if k == 0 and not self.is_fitted():
                # Iteration 0 with no model: target = reward (assume Q(s', .) = 0)
                next_q = np.zeros((len(transitions), self.n_actions))
            else:
                next_q = self.q_values_batch(next_states)
            max_next_q = np.max(next_q, axis=1)
            max_next_q[dones] = 0.0
            targets = rewards + self.gamma * max_next_q

            # Fit one regressor per action on its subset of transitions.
            for a in range(self.n_actions):
                mask = actions == a
                if mask.sum() < 5:
                    continue  # too few samples for this action
                X_a = states[mask]
                y_a = targets[mask]
                model = LGBMRegressor(**self._params)
                model.fit(X_a, y_a)
                self._models[a] = model

    def select_action(self, state: np.ndarray) -> int:
        if not self.is_fitted():
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self.q_values(state)))


class _LGBRLBase(RLBasePredictor):
    """LightGBM-FQI approximator. Subclasses set action_space_type."""
    approximator_kind = "lgb"

    def __init__(
        self,
        finetune: bool = False,
        transaction_cost: float = 0.0,
        flat_threshold: float = 0.05,
        total_timesteps: int = 100000,  # used as transition budget
        ft_steps_scale: float = 0.5,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        gamma: float = 0.99,
        iterations: int = 20,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            finetune=finetune,
            transaction_cost=transaction_cost,
            flat_threshold=flat_threshold,
            total_timesteps=total_timesteps,
            ft_steps_scale=ft_steps_scale,
        )
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.iterations = int(iterations)
        self.seed = int(seed)
        self._learner: _FQILearner | None = None

    def _has_prior_state(self) -> bool:
        return self._learner is not None and self._learner.is_fitted()

    def _n_actions(self) -> int:
        if self.action_space_type == ACTION_DISCRETE_2:
            return 2
        if self.action_space_type == ACTION_DISCRETE_3:
            return 3
        raise ValueError(f"LightGBM-FQI doesn't support {self.action_space_type}")

    def _train_approximator(
        self,
        envs_data: list[tuple[np.ndarray, np.ndarray]],
        total_timesteps: int,
        warm: bool,
    ) -> None:
        np.random.seed(self.seed)
        n_actions = self._n_actions()
        if not warm or self._learner is None:
            self._learner = _FQILearner(
                n_actions=n_actions,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
            )

        # 1. Collect transitions from all envs via random exploration
        envs = [self._build_env(f, c) for (f, c) in envs_data]
        transitions: list[tuple] = []
        steps_per_env = max(10, total_timesteps // len(envs))
        for env in envs:
            obs, _ = env.reset()
            for _ in range(steps_per_env):
                # ε-greedy: if model fitted, exploit half the time; else random
                if self._learner.is_fitted() and np.random.rand() > 0.5:
                    action = self._learner.select_action(obs)
                else:
                    action = int(np.random.randint(n_actions))
                next_obs, reward, terminated, _, _ = env.step(action)
                transitions.append((obs.copy(), action, reward, next_obs.copy(), terminated))
                if terminated:
                    obs, _ = env.reset()
                else:
                    obs = next_obs

        # 2. FQI iterations on the collected transitions
        self._learner.fit_iteration(transitions, n_iterations=self.iterations)

    def _act(self, obs: np.ndarray):
        if self._learner is None:
            raise RuntimeError(f"{self.name}: predict() called before fit()")
        return self._learner.select_action(obs)


class LGBQ2Predictor(_LGBRLBase):
    """LightGBM-FQI with discrete-2 action space."""
    base_name = "LGB-FQI-2"
    action_space_type = ACTION_DISCRETE_2


class LGBQ3Predictor(_LGBRLBase):
    """LightGBM-FQI with discrete-3 action space."""
    base_name = "LGB-FQI-3"
    action_space_type = ACTION_DISCRETE_3
