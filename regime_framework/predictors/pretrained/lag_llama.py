"""Lag-Llama (ServiceNow / Morgan Stanley, 2024) — Llama-style decoder for time series.

HF Hub: time-series-foundation-models/Lag-Llama
Package: pip install gluonts (and the lag-llama checkpoint loader)
"""
from __future__ import annotations

import numpy as np
import torch

from .base import BasePretrainedPredictor


class LagLlamaPredictor(BasePretrainedPredictor):
    name = "Lag-Llama"
    MODEL_ID = "time-series-foundation-models/Lag-Llama"
    _default_context_len = 1024
    _default_horizon = 24
    _supports_embedding = False

    def _load_model(self) -> None:
        try:
            from huggingface_hub import hf_hub_download
            ckpt = hf_hub_download(repo_id=self.MODEL_ID, filename="lag-llama.ckpt")
            from gluonts.torch.model.predictor import PyTorchPredictor  # noqa: F401
            from lag_llama.gluon.estimator import LagLlamaEstimator
        except ImportError as e:
            raise ImportError(
                "lag-llama not installed. Run: "
                "pip install gluonts && pip install git+https://github.com/time-series-foundation-models/lag-llama"
            ) from e
        device = torch.device(self.device_str)
        ckpt_data = torch.load(ckpt, map_location=device, weights_only=False)
        estimator_args = ckpt_data["hyper_parameters"]["model_kwargs"]
        self._estimator = LagLlamaEstimator(
            ckpt_path=ckpt,
            prediction_length=self._default_horizon,
            context_length=self.context_len,
            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            scaling=estimator_args["scaling"],
            time_feat=estimator_args["time_feat"],
            batch_size=1,
            num_parallel_samples=20,
            device=device,
        )
        self._lightning_module = self._estimator.create_lightning_module()
        self._predictor = self._estimator.create_predictor(
            self._estimator.create_transformation(),
            self._lightning_module,
        )

    def _forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        from gluonts.dataset.pandas import PandasDataset
        import pandas as pd
        # Build a tiny PandasDataset from the 1-D context
        df = pd.DataFrame({"target": context}, index=pd.date_range("2020-01-01", periods=len(context), freq="h"))
        ds = PandasDataset(df, target="target")
        forecast_iter = self._predictor.predict(ds, num_samples=20)
        forecast = next(iter(forecast_iter))
        return np.asarray(forecast.median)[:horizon]
