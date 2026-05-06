"""Technical features computed from OHLCV (past-only).

Categories:
- returns at multiple horizons
- realized volatility, vol regime z-score, vol ratios
- ATR%, ATR ratios
- EMA distance, EMA cross ratios, EMA stack score
- trend strength (mean/std of returns)
- RSI, MACD, ROC, Bollinger %B / width, Stochastic K/D
- range position, drawdown from rolling peak
- bars-since-high (trend persistence proxy)
- consecutive same-sign run, autocorr lag-1
- Hurst proxy, return shape (skew, kurt)
- volume z-score, volume momentum (if column exists)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    log_ret = np.log(close / close.shift(1))

    feat = pd.DataFrame(index=df.index)

    # Returns
    for h in (5, 10, 20, 50, 100, 200, 500):
        feat[f"ret_{h}"] = close.pct_change(h)

    # Realized vol
    for w in (10, 20, 50, 100, 200):
        feat[f"vol_{w}"] = log_ret.rolling(w).std() * np.sqrt(w)
    feat["vol_zscore_50"] = (
        (feat["vol_50"] - feat["vol_50"].rolling(500).mean())
        / (feat["vol_50"].rolling(500).std() + 1e-12)
    )
    feat["vol_ratio_20_200"] = feat["vol_20"] / feat["vol_200"].replace(0, np.nan)

    # ATR
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14).mean()
    atr50 = tr.rolling(50).mean()
    atr200 = tr.rolling(200).mean()
    feat["atr_pct_14"] = atr14 / close
    feat["atr_pct_50"] = atr50 / close
    feat["atr_ratio_14_50"] = atr14 / atr50.replace(0, np.nan)
    feat["atr_ratio_50_200"] = atr50 / atr200.replace(0, np.nan)
    # Rolling percentile of ATR — companion to YAML's `atr_min/atr_max` gate.
    feat["atr_pct_14_pctile_500"] = (atr14 / close).rolling(500, min_periods=50).rank(pct=True)

    # ADX(14) — trending strength. YAML's `use_antitrend / adx_max` gate
    # references this; exposing the continuous value lets the model learn
    # smooth regime-conditional behaviour.
    up = high.diff()
    dn = -low.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0)
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / (atr14 + 1e-12)
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / (atr14 + 1e-12)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    feat["adx_14"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    # Garman-Klass OHLC vol estimator — uses range AND open/close, ~5x more
    # efficient than close-only realized vol because it exploits intra-bar
    # extremes. SOTA in volatility forecasting.
    open_ = df["open"] if "open" in df.columns else close.shift(1)
    log_hl = np.log(high / low.replace(0, np.nan))
    log_co = np.log(close / open_.replace(0, np.nan))
    gk_step = 0.5 * log_hl**2 - (2.0 * np.log(2.0) - 1.0) * log_co**2
    feat["vol_gk_14"] = np.sqrt(gk_step.rolling(14).mean().clip(lower=0))
    feat["vol_gk_50"] = np.sqrt(gk_step.rolling(50).mean().clip(lower=0))

    # EMA distance
    for w in (10, 20, 50, 100, 200):
        ema = close.ewm(span=w, adjust=False).mean()
        feat[f"dist_ema_{w}"] = close / ema - 1

    # EMA cross + stack
    e10 = close.ewm(span=10, adjust=False).mean()
    e20 = close.ewm(span=20, adjust=False).mean()
    e50 = close.ewm(span=50, adjust=False).mean()
    e100 = close.ewm(span=100, adjust=False).mean()
    e200 = close.ewm(span=200, adjust=False).mean()
    feat["ema_20_50"] = e20 / e50 - 1
    feat["ema_50_200"] = e50 / e200 - 1
    feat["ema_20_200"] = e20 / e200 - 1
    feat["ema_stack"] = (
        (e10 > e20).astype(int)
        + (e20 > e50).astype(int)
        + (e50 > e100).astype(int)
        + (e100 > e200).astype(int)
    ) - 2

    # Trend strength
    for w in (14, 30, 60, 200):
        m = log_ret.rolling(w).mean()
        s = log_ret.rolling(w).std()
        feat[f"trend_strength_{w}"] = (m / s).replace([np.inf, -np.inf], np.nan)
    # Trend-strength absolute percentile — continuous "is the market currently
    # trending or ranging" feature. Companion to ADX antitrend gate.
    ts60 = feat["trend_strength_60"]
    feat["trend_strength_60_pctile_500"] = ts60.abs().rolling(500, min_periods=50).rank(pct=True)
    # Multi-scale trend agreement — sum of sign(ret_w) across 5/20/60/200 bars.
    # Range [-4, +4]: +4 = aligned bull across all timescales, -4 aligned bear,
    # near 0 = conflicting / no clear trend. Captures multi-scale alignment in
    # a single feature.
    feat["htf_trend_agreement"] = (
        np.sign(close.pct_change(5).fillna(0)) +
        np.sign(close.pct_change(20).fillna(0)) +
        np.sign(close.pct_change(60).fillna(0)) +
        np.sign(close.pct_change(200).fillna(0))
    )
    # Asymmetric volatility — std(neg returns) / std(pos returns) over 50 bars.
    # >1 = downside vol dominates (fear); <1 = upside dominates (euphoria).
    pos_ret = log_ret.where(log_ret > 0, np.nan)
    neg_ret = log_ret.where(log_ret < 0, np.nan)
    pos_vol = pos_ret.rolling(50, min_periods=10).std()
    neg_vol = neg_ret.rolling(50, min_periods=10).std()
    feat["vol_asym_50"] = (neg_vol / (pos_vol + 1e-12)).replace([np.inf, -np.inf], np.nan)
    # Range breakout (signed) — +1 if close just exited a 50-bar range upward,
    # -1 downward, 0 inside. Captures compression -> expansion transitions.
    rl_50 = low.rolling(50).min()
    rh_50 = high.rolling(50).max()
    breakout = pd.Series(0.0, index=df.index, dtype=np.float64)
    breakout[(close > rh_50.shift(1)).fillna(False)] = 1.0
    breakout[(close < rl_50.shift(1)).fillna(False)] = -1.0
    feat["range_breakout_50"] = breakout

    # RSI
    for n in (14, 28):
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(n).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(n).mean()
        rs = gain / loss.replace(0, np.nan)
        feat[f"rsi_{n}"] = 100 - 100 / (1 + rs)

    # MACD
    macd_fast = close.ewm(span=12, adjust=False).mean()
    macd_slow = close.ewm(span=26, adjust=False).mean()
    macd_line = macd_fast - macd_slow
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    feat["macd_line"] = macd_line / close
    feat["macd_signal"] = macd_signal / close
    feat["macd_hist"] = (macd_line - macd_signal) / close

    # ROC
    feat["roc_10"] = close.pct_change(10)
    feat["roc_50"] = close.pct_change(50)

    # Bollinger Bands
    bb_n = 20
    bb_ma = close.rolling(bb_n).mean()
    bb_std = close.rolling(bb_n).std()
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    feat["bb_pct"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-12)
    feat["bb_width"] = (bb_upper - bb_lower) / bb_ma.replace(0, np.nan)

    # Stochastic
    stoch_n = 14
    rl = low.rolling(stoch_n).min()
    rh = high.rolling(stoch_n).max()
    stoch_k = 100 * (close - rl) / (rh - rl + 1e-12)
    feat["stoch_k"] = stoch_k
    feat["stoch_d"] = stoch_k.rolling(3).mean()

    # Range position
    for w in (20, 50, 100, 200):
        rh_w = high.rolling(w).max()
        rl_w = low.rolling(w).min()
        feat[f"range_pos_{w}"] = (close - rl_w) / (rh_w - rl_w + 1e-12)

    # Drawdown from rolling peak
    for w in (50, 100, 200):
        peak = close.rolling(w).max()
        feat[f"dd_{w}"] = close / peak - 1

    # Bars since high
    def _bars_since_high(s: pd.Series, w: int) -> pd.Series:
        return s.rolling(w).apply(lambda x: float(w - 1 - np.argmax(x[::-1])), raw=True)
    feat["bars_since_high_100"] = _bars_since_high(close, 100)
    feat["bars_since_high_200"] = _bars_since_high(close, 200)

    # Consecutive run (signed)
    sign_ret = np.sign(log_ret)
    grp = (sign_ret != sign_ret.shift()).cumsum()
    feat["consec_run"] = (sign_ret.groupby(grp).cumcount() + 1) * sign_ret

    # Autocorr lag-1
    feat["autocorr_1_50"] = log_ret.rolling(50).corr(log_ret.shift(1))

    # Hurst proxy
    var5 = log_ret.rolling(100).apply(
        lambda x: float(np.var(np.diff(x, n=5))) if len(x) > 5 else np.nan, raw=True
    )
    var25 = log_ret.rolling(100).apply(
        lambda x: float(np.var(np.diff(x, n=25))) if len(x) > 25 else np.nan, raw=True
    )
    feat["hurst_proxy"] = np.log(var25 / var5 + 1e-12) / (2 * np.log(5))

    # Return shape
    feat["skew_50"] = log_ret.rolling(50).skew()
    feat["kurt_50"] = log_ret.rolling(50).kurt()
    feat["skew_200"] = log_ret.rolling(200).skew()

    # Volume
    if "volume" in df.columns:
        v = df["volume"].astype(float)
        v_ma = v.rolling(50).mean()
        v_std = v.rolling(50).std()
        feat["volume_zscore"] = (v - v_ma) / (v_std + 1e-12)
        feat["volume_ratio_20_100"] = v.rolling(20).mean() / v.rolling(100).mean().replace(0, np.nan)
        feat["volume_momentum"] = v.rolling(10).mean() / v.rolling(50).mean().replace(0, np.nan) - 1

    return feat
