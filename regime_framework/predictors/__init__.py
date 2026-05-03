from .base import BasePredictor, PredictionResult
from .classical import (
    LogRegPredictor,
    RandomForestPredictor,
    GBMPredictor,           # kept importable but no longer auto-registered
    MLPPredictor,
    XGBoostPredictor,
)
from .rule_based import RegimeV3Predictor, RegimeV4EmaPredictor
from .deep_nets import DeepMLPPredictor, GRUPredictor, LSTMPredictor
from .transformer import TimeSeriesTransformerPredictor

__all__ = [
    "BasePredictor",
    "PredictionResult",
    "LogRegPredictor",
    "RandomForestPredictor",
    "GBMPredictor",
    "MLPPredictor",
    "XGBoostPredictor",
    "RegimeV3Predictor",
    "RegimeV4EmaPredictor",
    "DeepMLPPredictor",
    "GRUPredictor",
    "LSTMPredictor",
    "TimeSeriesTransformerPredictor",
]
