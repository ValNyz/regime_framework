"""Toto (Datadog, late 2024) — observability-focused multivariate foundation model.

HF Hub: Datadog/Toto-Open-Base-1.0
"""
from __future__ import annotations

import numpy as np
import torch

from .base import BasePretrainedPredictor


class TotoPredictor(BasePretrainedPredictor):
    name = "Toto-Open-Base"
    MODEL_ID = "Datadog/Toto-Open-Base-1.0"
    _default_context_len = 512
    _default_horizon = 24
    _supports_embedding = False

    def _load_model(self) -> None:
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as e:
            raise ImportError("transformers required for Toto") from e
        self._model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.device_str == "cuda" else torch.float32,
        ).to(self.device_str)
        self._model.train(False)

    def _forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        device = torch.device(self.device_str)
        # Toto expects (B, T, D) with D = num_variates (1 for univariate)
        x = torch.tensor(context, dtype=torch.float32).reshape(1, -1, 1).to(device)
        try:
            with torch.no_grad():
                out = self._model.forecast(x, prediction_length=horizon)
            arr = out.cpu().float().numpy() if hasattr(out, "cpu") else np.asarray(out)
            # Expected shape (B=1, H, D=1) → flatten
            return arr.reshape(-1)[:horizon]
        except AttributeError:
            # Fallback if .forecast not available — try .generate-like API
            with torch.no_grad():
                out = self._model(x)
            return out[0, -horizon:, 0].float().cpu().numpy()
