"""MOIRAI / MOIRAI-MoE (Salesforce, 2024-2025).

Package: pip install uni2ts
Hub: Salesforce/moirai-1.0-R-large, Salesforce/moirai-moe-1.0-R-base
"""
from __future__ import annotations

import numpy as np
import torch

from .base import BasePretrainedPredictor


class _MoiraiBase(BasePretrainedPredictor):
    family = "pretrained"
    _supports_embedding = False
    _default_context_len = 512
    _default_horizon = 24

    def _load_model(self) -> None:
        try:
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        except ImportError as e:
            raise ImportError(
                "uni2ts not installed. Run: pip install uni2ts"
            ) from e
        self._module = MoiraiModule.from_pretrained(self.MODEL_ID)
        self._MoiraiForecast = MoiraiForecast

    def _forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        device = torch.device(self.device_str)
        forecast = self._MoiraiForecast(
            module=self._module,
            prediction_length=horizon,
            context_length=len(context),
            patch_size="auto",
            num_samples=20,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        # Tensor formats expected: (B=1, L=context_len, T=1)
        past = torch.from_numpy(context.astype(np.float32)).reshape(1, -1, 1).to(device)
        observed_mask = torch.ones_like(past).bool()
        sample_id = torch.zeros(past.shape[1], dtype=torch.long, device=device)
        time_id = torch.arange(past.shape[1], dtype=torch.long, device=device)
        forecast.to(device)
        with torch.no_grad():
            samples = forecast(
                past_target=past,
                past_observed_target=observed_mask,
                past_is_pad=torch.zeros_like(time_id).bool(),
            )  # (B, num_samples, prediction_length, target_dim)
        return samples[0].median(dim=0).values[:, 0].cpu().numpy()


class MoiraiLargePredictor(_MoiraiBase):
    name = "MOIRAI-1.0-R-Large"
    MODEL_ID = "Salesforce/moirai-1.0-R-large"


class MoiraiMoEBasePredictor(_MoiraiBase):
    name = "MOIRAI-MoE-Base"
    MODEL_ID = "Salesforce/moirai-moe-1.0-R-base"
