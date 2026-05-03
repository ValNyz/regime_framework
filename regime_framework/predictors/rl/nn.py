"""NN approximator for RL predictors via Stable-Baselines3.

Three concrete predictors:
  - DQN-2  : SB3 DQN, action space discrete-2 (long/short, no flat)
  - DQN-3  : SB3 DQN, action space discrete-3 (long/short/flat)
  - SAC    : SB3 SAC, action space continuous

Multi-coin training is handled in commit 3 via SB3's vec_env. This commit
implements mono-coin only — vec_env wiring is a small follow-up that
swaps the env construction in `_train_approximator` for `make_vec_env`.

FT mode keeps the SB3 model object across fit() calls and continues
learning with reset_num_timesteps=False. The replay buffer (DQN) /
rollout buffer (SAC) is also preserved.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import RLBasePredictor
from .env import (
    RegimeTradingEnv,
    ACTION_DISCRETE_2,
    ACTION_DISCRETE_3,
    ACTION_CONTINUOUS,
)


class _NNRLBase(RLBasePredictor):
    """Shared NN-approximator logic (model construction, fit/predict via SB3)."""
    approximator_kind = "nn"

    def __init__(
        self,
        finetune: bool = False,
        transaction_cost: float = 0.0,
        flat_threshold: float = 0.05,
        total_timesteps: int = 100000,
        ft_steps_scale: float = 0.5,
        learning_rate: float = 5e-4,
        buffer_size: int = 50000,
        gamma: float = 0.99,
        net_arch: tuple[int, ...] = (64, 32),
        verbose: int = 0,
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
        self.learning_rate = float(learning_rate)
        self.buffer_size = int(buffer_size)
        self.gamma = float(gamma)
        self.net_arch = list(net_arch)
        self.verbose = int(verbose)
        self.seed = int(seed)
        self._algo: Any = None  # SB3 model — created on first fit, reused for FT

    def _has_prior_state(self) -> bool:
        return self._algo is not None

    def _make_algo(self, env):
        """Pick DQN (discrete) vs SAC (continuous) based on action space.

        Imports SB3 lazily so non-RL runs don't pay the import cost.
        """
        try:
            from stable_baselines3 import DQN, SAC
        except ImportError as e:
            raise ImportError(
                "stable-baselines3 not installed. Install with: "
                "pip install 'regime-framework[rl]' or pip install stable-baselines3"
            ) from e
        policy_kwargs = {"net_arch": self.net_arch}
        if self.action_space_type == ACTION_CONTINUOUS:
            return SAC(
                "MlpPolicy", env,
                learning_rate=self.learning_rate,
                buffer_size=self.buffer_size,
                gamma=self.gamma,
                policy_kwargs=policy_kwargs,
                verbose=self.verbose,
                seed=self.seed,
                device="auto",
            )
        return DQN(
            "MlpPolicy", env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            gamma=self.gamma,
            policy_kwargs=policy_kwargs,
            verbose=self.verbose,
            seed=self.seed,
            device="auto",
        )

    def _train_approximator(
        self,
        envs_data: list[tuple[np.ndarray, np.ndarray]],
        total_timesteps: int,
        warm: bool,
    ) -> None:
        # Multi-coin: build a SubprocVecEnv (or DummyVecEnv) wrapping N coin
        # envs in parallel. The agent collects transitions from all coins
        # interleaved — better generalization than serial concatenation.
        env = self._build_vec_env(envs_data)
        # SB3's built-in tqdm progress bar — same level of detail the user gets
        # from MLP/GRU/TST training in the rest of the framework.
        kind = "FT" if (warm and self._algo is not None) else "cold"
        print(f"      {self.name} training ({kind}): {total_timesteps} timesteps × {len(envs_data)} env(s)")
        if warm and self._algo is not None:
            # FT: continue training from existing weights + replay buffer
            self._algo.set_env(env)
            self._algo.learn(
                total_timesteps=total_timesteps,
                reset_num_timesteps=False,
                progress_bar=True,
            )
        else:
            self._algo = self._make_algo(env)
            self._algo.learn(total_timesteps=total_timesteps, progress_bar=True)

    def _build_vec_env(self, envs_data: list[tuple[np.ndarray, np.ndarray]]):
        """Build SB3 vec_env from N coin envs. Single coin → DummyVecEnv of 1.

        Uses DummyVecEnv (single-process) rather than SubprocVecEnv: our envs
        are pure Python + numpy and the bottleneck is the policy forward pass
        on GPU, not env stepping. Subprocess overhead would dominate.
        """
        from stable_baselines3.common.vec_env import DummyVecEnv

        def _make(features: np.ndarray, closes: np.ndarray):
            def _f():
                return self._build_env(features, closes)
            return _f

        env_fns = [_make(f, c) for (f, c) in envs_data]
        return DummyVecEnv(env_fns)

    def _act(self, obs: np.ndarray):
        if self._algo is None:
            raise RuntimeError(f"{self.name}: predict() called before fit()")
        action, _ = self._algo.predict(obs, deterministic=True)
        return action


class DQN2Predictor(_NNRLBase):
    """SB3 DQN with discrete-2 action space (long/short, always in)."""
    base_name = "DQN-2"
    action_space_type = ACTION_DISCRETE_2


class DQN3Predictor(_NNRLBase):
    """SB3 DQN with discrete-3 action space (long/short/flat)."""
    base_name = "DQN-3"
    action_space_type = ACTION_DISCRETE_3


class SACPredictor(_NNRLBase):
    """SB3 SAC with continuous action space (signed position size in [-1,+1])."""
    base_name = "SAC"
    action_space_type = ACTION_CONTINUOUS
