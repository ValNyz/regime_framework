"""Linear Q-learning approximator (custom — no SB3 / no torch).

Q(s, a) = w_a · features(s)  for each discrete action a.

Update rule (TD(0) off-policy Q-learning):
    target = r + γ · max_a' Q(s', a')
    δ      = target - Q(s, a)
    w_a   ← w_a + α · δ · features(s)

ε-greedy exploration with linear decay 1.0 → 0.05 over training.
Multi-coin handled by alternating envs: each env contributes
total_timesteps // n_envs transitions, learner state shared across.

Skipped action_space=continuous: value-based linear Q doesn't extend
naturally to continuous (would need a separate actor — that's SAC,
covered by the NN approximator).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import RLBasePredictor
from .env import (
    action_to_position,
    ACTION_DISCRETE_2,
    ACTION_DISCRETE_3,
)


class _LinearQLearner:
    """Tabular-style Q-learning with linear features. ~5x faster than DQN
    on small feature sets, virtually no overfit risk on 4380 train bars.

    Raw features arrive on wildly different scales (technical indicators,
    z-scores, prices). Without standardization, td_error * features explodes
    after a few hundred updates → matmul overflow → degenerate policy. We
    standardize per-feature using train-set mean/std stored once at fit
    time, applied to every q_values / update / predict call afterwards.
    """
    def __init__(self, n_features: int, n_actions: int, lr: float = 1e-3, gamma: float = 0.99):
        # weights[a] is the (n_features+1,) vector — the +1 is a bias term
        self.W = np.zeros((n_actions, n_features + 1), dtype=np.float64)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.epsilon = 1.0
        self.n_actions = int(n_actions)
        # Standardization stats — set via set_normalization() before training
        self._mu: np.ndarray | None = None
        self._sigma: np.ndarray | None = None

    def set_normalization(self, mu: np.ndarray, sigma: np.ndarray) -> None:
        """Set per-feature mean/std for standardization. sigma is floored to
        1e-8 to avoid division by zero on constant features."""
        self._mu = mu.astype(np.float64)
        self._sigma = np.maximum(sigma.astype(np.float64), 1e-8)

    def _phi(self, state: np.ndarray) -> np.ndarray:
        """Standardize state and append a 1.0 bias term."""
        s = state.astype(np.float64)
        if self._mu is not None and self._sigma is not None:
            s = (s - self._mu) / self._sigma
        return np.concatenate([s, [1.0]])

    def q_values(self, state: np.ndarray) -> np.ndarray:
        return self.W @ self._phi(state)  # (n_actions,)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        if deterministic or np.random.rand() > self.epsilon:
            return int(np.argmax(self.q_values(state)))
        return int(np.random.randint(self.n_actions))

    def update(self, s: np.ndarray, a: int, r: float, s_next: np.ndarray, done: bool) -> None:
        phi_s = self._phi(s)
        q_current = float(self.W[a] @ phi_s)
        q_next = 0.0 if done else float(np.max(self.W @ self._phi(s_next)))
        target = r + self.gamma * q_next
        td_error = target - q_current
        # Clip td_error so a single outlier reward can't blow up W
        td_error = float(np.clip(td_error, -1.0, 1.0))
        self.W[a] += self.lr * td_error * phi_s


class _LinearRLBase(RLBasePredictor):
    """Linear-Q approximator. Subclasses set action_space_type."""
    approximator_kind = "linear"

    def __init__(
        self,
        finetune: bool = False,
        transaction_cost: float = 0.0,
        flat_threshold: float = 0.05,
        total_timesteps: int = 100000,
        ft_steps_scale: float = 0.5,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            finetune=finetune,
            transaction_cost=transaction_cost,
            flat_threshold=flat_threshold,
            total_timesteps=total_timesteps,
            ft_steps_scale=ft_steps_scale,
            **kwargs,  # forward any extra RLBase params (e.g. proba_temperature)
        )
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.seed = int(seed)
        self._learner: _LinearQLearner | None = None

    def _has_prior_state(self) -> bool:
        return self._learner is not None

    def _n_actions(self) -> int:
        if self.action_space_type == ACTION_DISCRETE_2:
            return 2
        if self.action_space_type == ACTION_DISCRETE_3:
            return 3
        raise ValueError(f"Linear approximator doesn't support {self.action_space_type}")

    def _train_approximator(
        self,
        envs_data: list[tuple[np.ndarray, np.ndarray]],
        total_timesteps: int,
        warm: bool,
    ) -> None:
        np.random.seed(self.seed)
        n_features = envs_data[0][0].shape[1]
        n_actions = self._n_actions()
        if not warm or self._learner is None:
            self._learner = _LinearQLearner(
                n_features=n_features, n_actions=n_actions,
                lr=self.learning_rate, gamma=self.gamma,
            )
            # Pool features across all envs to fit standardization stats once.
            # Cold start only — FT keeps the previously fit stats so the
            # warm-started weights stay aligned with the same feature scale.
            pooled = np.concatenate([f for (f, _) in envs_data], axis=0)
            self._learner.set_normalization(pooled.mean(axis=0), pooled.std(axis=0))

        envs = [self._build_env(f, c) for (f, c) in envs_data]
        steps_per_env = max(1, total_timesteps // len(envs))
        # Print 5 progress checkpoints per env, regardless of steps_per_env size
        log_every = max(1, steps_per_env // 5)
        print(
            f"      {self.name} training: {len(envs)} env(s) × {steps_per_env} steps "
            f"= {len(envs)*steps_per_env}"
        )

        for env_idx, env in enumerate(envs):
            obs, _ = env.reset()
            recent_rewards: list[float] = []
            for step in range(steps_per_env):
                # Linear ε decay over this env's budget
                progress = step / max(1, steps_per_env - 1)
                self._learner.epsilon = (
                    self.epsilon_start * (1 - progress) + self.epsilon_end * progress
                )
                action = self._learner.select_action(obs, deterministic=False)
                next_obs, reward, terminated, _, _ = env.step(action)
                self._learner.update(obs, action, reward, next_obs, terminated)
                obs = next_obs
                recent_rewards.append(float(reward))
                if (step + 1) % log_every == 0 or step == steps_per_env - 1:
                    mean_r = float(np.mean(recent_rewards[-log_every:]))
                    print(
                        f"      {self.name} env {env_idx+1}/{len(envs)} "
                        f"step {step+1}/{steps_per_env} "
                        f"eps={self._learner.epsilon:.3f} mean_r={mean_r:+.5f}"
                    )
                if terminated:
                    obs, _ = env.reset()

    def _act(self, obs: np.ndarray):
        if self._learner is None:
            raise RuntimeError(f"{self.name}: predict() called before fit()")
        return self._learner.select_action(obs, deterministic=True)

    def _q_values_at(self, obs: np.ndarray) -> np.ndarray | None:
        if self._learner is None:
            return None
        return self._learner.q_values(obs)


class LinearQ2Predictor(_LinearRLBase):
    """Linear-Q with discrete-2 action space (long/short)."""
    base_name = "LinearQ-2"
    action_space_type = ACTION_DISCRETE_2


class LinearQ3Predictor(_LinearRLBase):
    """Linear-Q with discrete-3 action space (long/short/flat)."""
    base_name = "LinearQ-3"
    action_space_type = ACTION_DISCRETE_3
