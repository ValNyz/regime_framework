"""Rule-based regime classifiers ported from crypto_comparative_analysis.

These are deterministic — no training. They take an OHLCV df, compute
indicators, and assign one of {bull, bear, range, volatile} per bar. For the
binary benchmark (bull/bear), we collapse range/volatile into the closer
direction (or, more rigorously, drop those bars from the comparison since
the rule-based predictors can't say bull vs bear there).

Strategy here: for binary comparison, we map:
  - rule "bull" → "bull"
  - rule "bear" → "bear"
  - rule "range" / "volatile" → use sign of EMA20-EMA50 cross as tie-break
                                 (rule-based fallback, still no training)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BasePredictor


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the indicator set required by v3/v4ema classifiers."""
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]

    # EMAs
    for w in (8, 20, 21, 50, 200):
        out[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()

    # ATR(14)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(14).mean()

    # ADX(14) Wilder approximated by EMA
    up = high.diff()
    dn = -low.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0)
    out["di_plus"] = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / (out["atr"] + 1e-12)
    out["di_minus"] = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / (out["atr"] + 1e-12)
    dx = 100 * (out["di_plus"] - out["di_minus"]).abs() / (out["di_plus"] + out["di_minus"] + 1e-12)
    out["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - 100 / (1 + rs)

    # MACD
    macd_fast = close.ewm(span=12, adjust=False).mean()
    macd_slow = close.ewm(span=26, adjust=False).mean()
    macd = macd_fast - macd_slow
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = macd - macd_signal

    # Bollinger Bands
    bb_ma = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_upper"] = bb_ma + 2 * bb_std
    out["bb_lower"] = bb_ma - 2 * bb_std
    out["bb_middle"] = bb_ma

    return out


def _binarize(regime_series: pd.Series, df_with_ind: pd.DataFrame) -> np.ndarray:
    """Map {bull, bear, range, volatile, quiet, ...} → {bull, bear} via EMA20 vs EMA50.
    Used to make rule-based outputs comparable to binary benchmark labels."""
    e20 = df_with_ind["ema_20"]
    e50 = df_with_ind["ema_50"]
    fallback = np.where(e20.values > e50.values, "bull", "bear")
    out = regime_series.values.astype(object).copy()
    mask_unknown = ~np.isin(out, ["bull", "bear"])
    out[mask_unknown] = fallback[mask_unknown]
    return out


class _RuleBasedBase(BasePredictor):
    family = "rule_based"
    needs_features = False  # uses df directly

    def fit(self, X_train, y_train, dates_train, df_train):
        return self  # no training

    def predict(self, X_test, dates_test, df_test):
        df_ind = _compute_indicators(df_test)
        regime = self._classify(df_ind)
        return _binarize(regime, df_ind)

    def _classify(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class RegimeV3Predictor(_RuleBasedBase):
    """Multi-factor regime: ADX + DI + EMA alignment + ATR/BB volatility + RSI/MACD momentum.

    Ported verbatim from crypto_comparative_analysis/lib/generation/templates/base.py
    REGIME_DETECTION_BLOCK ('v3').
    """
    name = "regime_v3"

    REGIME_LOOKBACK = 100
    REGIME_ADX_THRESHOLD = 22

    def _classify(self, df: pd.DataFrame) -> pd.Series:
        lb = self.REGIME_LOOKBACK

        atr_pct = df["atr"].rolling(lb).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10), raw=False
        )
        bb_width = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        bb_width_pct = bb_width.rolling(lb).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10), raw=False
        )
        vol_score = (atr_pct + bb_width_pct) / 2

        ema_bull = ((df["ema_8"] > df["ema_21"]) & (df["ema_21"] > df["ema_50"])).astype(float)
        ema_bear = ((df["ema_8"] < df["ema_21"]) & (df["ema_21"] < df["ema_50"])).astype(float)
        adx_score = np.clip(df["adx"] / 40, 0, 1)
        # not used for classification but kept for parity:
        _trend_strength = adx_score * (ema_bull - ema_bear + 1) / 2

        rsi_bull = (df["rsi_14"] > 50).astype(float)
        macd_bull = (df["macd_hist"] > 0).astype(float)
        macd_rising = (df["macd_hist"] > df["macd_hist"].shift(1)).astype(float)
        momentum = (rsi_bull + macd_bull + macd_rising) / 3

        is_high_vol = vol_score > 0.6
        is_low_vol = vol_score < 0.3
        is_trending = df["adx"] > self.REGIME_ADX_THRESHOLD

        is_strong_bull = is_trending & (df["di_plus"] > df["di_minus"]) & (momentum > 0.6)
        is_strong_bear = is_trending & (df["di_minus"] > df["di_plus"]) & (momentum < 0.4)
        is_volatile = is_high_vol & ~is_strong_bull & ~is_strong_bear
        is_quiet_range = is_low_vol & ~is_trending

        conditions = [is_strong_bull, is_strong_bear, is_volatile, is_quiet_range]
        choices = ["bull", "bear", "volatile", "quiet"]
        regime = pd.Series(
            np.select(conditions, choices, default="range"),
            index=df.index, dtype=object,
        )
        return regime


class RegimeV4EmaPredictor(_RuleBasedBase):
    """EMA-alignment + ATR-percentile classifier (v4ema).

    bull  = close > ema_50 > ema_200 AND atr_pct < 0.7
    bear  = close < ema_50 < ema_200 AND atr_pct < 0.7
    volatile = atr_pct >= 0.7
    range = else
    """
    name = "regime_v4ema"
    REGIME_LOOKBACK = 100

    def _classify(self, df: pd.DataFrame) -> pd.Series:
        lb = self.REGIME_LOOKBACK
        atr_norm = df["atr"] / df["close"]
        atr_percentile = atr_norm.rolling(lb).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        bull_aligned = (df["close"] > df["ema_50"]) & (df["ema_50"] > df["ema_200"])
        bear_aligned = (df["close"] < df["ema_50"]) & (df["ema_50"] < df["ema_200"])
        conditions = [
            bull_aligned & (atr_percentile < 0.7),
            bear_aligned & (atr_percentile < 0.7),
            atr_percentile >= 0.7,
        ]
        regime = pd.Series(
            np.select(conditions, ["bull", "bear", "volatile"], default="range"),
            index=df.index, dtype=object,
        )
        return regime
