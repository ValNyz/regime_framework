"""TimeMoE (Tsinghua, 2024) — Mixture-of-Experts foundation model.

HF Hub: Maple728/TimeMoE-200M
Loaded via transformers.AutoModelForCausalLM with trust_remote_code=True.
"""
from __future__ import annotations

import numpy as np
import torch

from .base import BasePretrainedPredictor


class TimeMoEPredictor(BasePretrainedPredictor):
    name = "TimeMoE-200M"
    MODEL_ID = "Maple728/TimeMoE-200M"
    _default_context_len = 512
    _default_horizon = 24
    _supports_embedding = True

    def _load_model(self) -> None:
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as e:
            raise ImportError("transformers required for TimeMoE") from e
        self._model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.device_str == "cuda" else torch.float32,
        ).to(self.device_str)
        self._model.train(False)

    def _forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        device = torch.device(self.device_str)
        # Normalize context (TimeMoE recommends instance norm)
        mean = float(np.mean(context))
        std = float(np.std(context) + 1e-8)
        ctx_norm = (context - mean) / std
        x = torch.tensor(ctx_norm, dtype=torch.float32).reshape(1, -1).to(device)
        with torch.no_grad():
            out = self._model.generate(x, max_new_tokens=horizon)
        # Slice the new tokens
        forecast_norm = out[0, -horizon:].float().cpu().numpy()
        return forecast_norm * std + mean

    def _embed(self, context: np.ndarray) -> np.ndarray:
        device = torch.device(self.device_str)
        mean = float(np.mean(context))
        std = float(np.std(context) + 1e-8)
        x = torch.tensor((context - mean) / std, dtype=torch.float32).reshape(1, -1).to(device)
        with torch.no_grad():
            out = self._model(x, output_hidden_states=True)
        hs = out.hidden_states[-1]    # (1, L, D)
        return hs[0].mean(dim=0).float().cpu().numpy()
