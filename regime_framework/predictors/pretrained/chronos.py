"""Chronos / Chronos-Bolt wrappers (Amazon, NeurIPS '24).

Uses the `chronos-forecasting` package (pip install chronos-forecasting).
Falls back to a clear error message if not installed.
"""
from __future__ import annotations

import numpy as np
import torch

from .base import BasePretrainedPredictor


class _ChronosBase(BasePretrainedPredictor):
    family = "pretrained"
    _supports_embedding = True
    _default_context_len = 512

    def _load_model(self) -> None:
        try:
            from chronos import BaseChronosPipeline
        except ImportError as e:
            raise ImportError(
                "chronos-forecasting not installed. Run: "
                "pip install chronos-forecasting"
            ) from e
        self._model = BaseChronosPipeline.from_pretrained(
            self.MODEL_ID,
            device_map=self.device_str,
            torch_dtype=torch.bfloat16 if self.device_str == "cuda" else torch.float32,
        )

    def _forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        ctx_t = torch.tensor(context, dtype=torch.float32)
        forecast = self._model.predict(ctx_t, prediction_length=horizon)
        # forecast shape (B=1, num_samples, H) for original Chronos; or (B, H) for Bolt
        arr = forecast.cpu().numpy() if hasattr(forecast, "cpu") else np.asarray(forecast)
        # take median across samples if 3D
        if arr.ndim == 3:
            return np.median(arr[0], axis=0)
        return arr[0]

    def _embed(self, context: np.ndarray) -> np.ndarray:
        # Chronos exposes encoder embeddings via .embed
        ctx_t = torch.tensor(context, dtype=torch.float32)
        try:
            emb, _ = self._model.embed(ctx_t)
            arr = emb.cpu().float().numpy() if hasattr(emb, "cpu") else np.asarray(emb)
            # mean-pool over the sequence dim if 3D (B, L, D)
            if arr.ndim == 3:
                return arr[0].mean(axis=0)
            return arr.flatten()
        except Exception:
            # Fallback: derive a simple summary from the forecast
            fc = self._forecast(context, self._default_horizon)
            return np.concatenate([fc, [fc.mean(), fc.std(), fc[-1] - fc[0]]]).astype(np.float32)


class ChronosBoltBasePredictor(_ChronosBase):
    name = "Chronos-Bolt-Base"
    MODEL_ID = "amazon/chronos-bolt-base"


class ChronosLargePredictor(_ChronosBase):
    name = "Chronos-T5-Large"
    MODEL_ID = "amazon/chronos-t5-large"
