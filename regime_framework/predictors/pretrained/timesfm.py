"""TimesFM (Google, 2024) — decoder-only transformer for forecasting.

Two API variants supported:
  - Modern PyTorch backend (timesfm >= 1.3 with [torch] extra) — preferred,
    works on Python 3.12. Install: `pip install timesfm[torch]`.
  - Legacy JAX backend (timesfm 1.0 with paxml/lingvo) — only works on Python <3.12.

The wrapper auto-detects which API is available.
"""
from __future__ import annotations

import numpy as np

from .base import BasePretrainedPredictor


class TimesFMPredictor(BasePretrainedPredictor):
    name = "TimesFM-2.0"
    MODEL_ID = "google/timesfm-2.0-500m-pytorch"
    _default_context_len = 512
    _default_horizon = 24
    _supports_embedding = False

    def _load_model(self) -> None:
        # Try modern torch entry points (PyPI 'timesfm' 1.0 is JAX-only and broken
        # on Py3.12 — install from git for the torch backend).
        for entry_path in (
            ("timesfm", "TimesFm_2p0_500M_torch"),
            ("timesfm.timesfm_torch", "TimesFmTorch"),
            ("timesfm.torch", "TimesFmTorch"),
        ):
            try:
                mod = __import__(entry_path[0], fromlist=[entry_path[1]])
                cls = getattr(mod, entry_path[1])
                self._model = cls.from_pretrained(self.MODEL_ID)
                self._api = "torch_v2"
                return
            except (ImportError, AttributeError):
                continue
            except Exception:
                continue

        # Fallback: legacy JAX API (timesfm 1.0 on Python <3.12)
        try:
            from timesfm import TimesFm, TimesFmHparams, TimesFmCheckpoint
        except ImportError as e:
            raise ImportError(
                "timesfm torch backend not available. The PyPI 'timesfm' package "
                "is JAX-only and incompatible with Python 3.12. Install the torch "
                "backend from git:\n"
                "  pip install 'timesfm[torch] @ git+https://github.com/google-research/timesfm.git'\n"
                "If that fails on your env, skip TimesFM with --pretrained chronos_bolt_base ..."
            ) from e

        backend = "gpu" if self.device_str == "cuda" else "cpu"
        self._model = TimesFm(
            hparams=TimesFmHparams(
                backend=backend,
                per_core_batch_size=32,
                horizon_len=self._default_horizon,
                context_len=self.context_len,
                input_patch_len=32,
                output_patch_len=128,
                num_layers=50,
                model_dims=1280,
            ),
            checkpoint=TimesFmCheckpoint(huggingface_repo_id=self.MODEL_ID),
        )
        self._api = "jax_v1"

    def _forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        ctx = context.astype(np.float32)
        if self._api == "torch_v2":
            # PyTorch API: model.forecast(inputs=[1-D array]) → (point_forecast, quantiles)
            point_fc, _ = self._model.forecast(
                horizon=horizon,
                inputs=[ctx],
                # freq=[0] means high-frequency (e.g. hourly)
            )
            return np.asarray(point_fc[0][:horizon])

        # JAX legacy API
        forecasts, _ = self._model.forecast(inputs=[ctx], freq=[0])
        return np.asarray(forecasts[0][:horizon])
