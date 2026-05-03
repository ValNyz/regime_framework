"""Pretrained time-series foundation model wrappers.

Each wrapper exposes the same BasePredictor interface. Two modes per model:
  - "zero_shot": forecast next H bars from past context, derive label from
    sign(forecast_mean - close_now)
  - "fine_tuned": extract embeddings from the model's last hidden layer (no
    weight updates), train a small head (LogReg or MLP) to classify regime

A6000-friendly defaults; CPU fallback when CUDA unavailable.
"""
from .base import BasePretrainedPredictor
from .chronos import ChronosBoltBasePredictor, ChronosLargePredictor
from .timesfm import TimesFMPredictor
from .moirai import MoiraiLargePredictor, MoiraiMoEBasePredictor
from .timemoe import TimeMoEPredictor
from .lag_llama import LagLlamaPredictor
from .toto import TotoPredictor


PRETRAINED_REGISTRY = {
    "chronos_bolt_base": ChronosBoltBasePredictor,
    "chronos_large": ChronosLargePredictor,
    "timesfm": TimesFMPredictor,
    "moirai_large": MoiraiLargePredictor,
    "moirai_moe": MoiraiMoEBasePredictor,
    "timemoe": TimeMoEPredictor,
    "lag_llama": LagLlamaPredictor,
    "toto": TotoPredictor,
}


__all__ = [
    "BasePretrainedPredictor",
    "PRETRAINED_REGISTRY",
    "ChronosBoltBasePredictor",
    "ChronosLargePredictor",
    "TimesFMPredictor",
    "MoiraiLargePredictor",
    "MoiraiMoEBasePredictor",
    "TimeMoEPredictor",
    "LagLlamaPredictor",
    "TotoPredictor",
]
