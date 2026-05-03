"""TimesFM (Google, 2024) — decoder-only transformer for forecasting.

HF Hub: google/timesfm-2.0-500m-pytorch
Package: pip install timesfm
"""
from __future__ import annotations

import numpy as np

from .base import BasePretrainedPredictor


class TimesFMPredictor(BasePretrainedPredictor):
    name = "TimesFM-2.0"
    MODEL_ID = "google/timesfm-2.0-500m-pytorch"
    _default_context_len = 512
    _default_horizon = 24
    _supports_embedding = False  # embedding extraction not in public API

    def _load_model(self) -> None:
        try:
            import timesfm  # noqa: F401
            from timesfm import TimesFm, TimesFmHparams, TimesFmCheckpoint
        except ImportError as e:
            raise ImportError(
                "timesfm not installed. Run: pip install timesfm"
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
            checkpoint=TimesFmCheckpoint(
                huggingface_repo_id=self.MODEL_ID,
            ),
        )

    def _forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        # TimesFM expects a list of 1-D arrays
        forecasts, _quantiles = self._model.forecast(
            inputs=[context.astype(np.float32)],
            freq=[0],  # 0 = high-frequency (e.g. hourly)
        )
        return np.asarray(forecasts[0][:horizon])
