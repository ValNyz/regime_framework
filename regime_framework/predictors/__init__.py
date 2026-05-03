from .base import BasePredictor, PredictionResult
from .classical import (
    LogRegPredictor,
    RandomForestPredictor,
    GBMPredictor,           # kept importable, not auto-registered (slow)
    ExtraTreesPredictor,
    MLPPredictor,           # torch GPU MLP
    XGBoostPredictor,
    LightGBMPredictor,
)
from .rule_based import RegimeV3Predictor, RegimeV4EmaPredictor
from .deep_nets import GRUPredictor, LSTMPredictor
from .transformer import TimeSeriesTransformerPredictor

__all__ = [
    "BasePredictor",
    "PredictionResult",
    "LogRegPredictor",
    "RandomForestPredictor",
    "GBMPredictor",
    "ExtraTreesPredictor",
    "MLPPredictor",
    "XGBoostPredictor",
    "LightGBMPredictor",
    "RegimeV3Predictor",
    "RegimeV4EmaPredictor",
    "GRUPredictor",
    "LSTMPredictor",
    "TimeSeriesTransformerPredictor",
]
