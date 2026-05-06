"""Regime-signal features — synthesized 3-state regime classifiers.

This category exists specifically for HIGH-LEVEL meta-classifications that
synthesize multiple raw indicators into a single "what regime are we in"
verdict. Pure indicators (ADX, GK vol, asymmetric vol, etc.) live in
`technical.py`; funding-based features live in `funding.py`; macro and
cross-asset live in `external.py`.

The features here mirror the regime gates used by funding-based signal
families in the trading-signals YAML (e.g. signals_btc_opt_OOS_2026):
  - `use_btc_regime: true, btc_regime_allowed: [bull, range]`
The YAML's binary signals fire only when those gates are open;
`regime_btc` exposes the underlying classification as a continuous
+1/0/-1 ordinal plus 3 one-hot variants so the model can learn
regime-conditional behaviour without being limited to those binary
triggers.

`regime_vol` and `regime_dd` are companion classifiers for volatility
and drawdown state — same 3-state ordinal + one-hot pattern.

15 features:
  regime_btc + 3 one-hots  (bull / range / bear)
  regime_vol + 3 one-hots  (low / mid / high)
  regime_dd  + 3 one-hots  (recovering / normal / drawdown)
  regime_btc_bars_in       (bars since regime_btc last changed)
  regime_vol_bars_in       (bars since regime_vol last changed)
  regime_dd_bars_in        (bars since regime_dd last changed)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _bars_since_change(s: pd.Series) -> pd.Series:
    """Count of bars since `s` last changed value (1 at change, +1 each bar)."""
    changed = (s != s.shift(1)).cumsum()
    return s.groupby(changed).cumcount() + 1


def compute_regime_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build synthesized regime classifiers (BTC trend, vol, drawdown).

    All features are past-only by construction (rolling / ewm / diff over
    historical bars). No look-ahead.
    """
    close = df["close"]
    log_ret = np.log(close / close.shift(1))

    feat = pd.DataFrame(index=df.index)

    # ---- BTC trend regime classifier (3-state) ----
    # bull: EMA50/EMA200 > +2% AND positive 60-bar trend strength
    # bear: EMA50/EMA200 < -2% AND negative 60-bar trend strength
    # range/neutral: otherwise
    e50 = close.ewm(span=50, adjust=False).mean()
    e200 = close.ewm(span=200, adjust=False).mean()
    ema_50_200 = e50 / e200 - 1.0
    m60 = log_ret.rolling(60).mean()
    s60 = log_ret.rolling(60).std()
    trend_strength_60 = (m60 / (s60 + 1e-12)).replace([np.inf, -np.inf], np.nan)
    bull = (ema_50_200 > 0.02) & (trend_strength_60 > 0)
    bear = (ema_50_200 < -0.02) & (trend_strength_60 < 0)
    btc_reg = pd.Series(0, index=df.index, dtype=np.int8)
    btc_reg[bull.fillna(False)] = 1
    btc_reg[bear.fillna(False)] = -1
    feat["regime_btc"] = btc_reg.astype(np.float64)
    feat["regime_btc_is_bull"]  = (btc_reg ==  1).astype(np.float64)
    feat["regime_btc_is_range"] = (btc_reg ==  0).astype(np.float64)
    feat["regime_btc_is_bear"]  = (btc_reg == -1).astype(np.float64)
    feat["regime_btc_bars_in"] = _bars_since_change(btc_reg).astype(np.float64)

    # ---- Volatility regime classifier (3-state) ----
    # Realized-vol percentile (50-bar std × √50) ranked over 500 bars.
    vol_50 = log_ret.rolling(50).std() * np.sqrt(50)
    vol_pctile = vol_50.rolling(500, min_periods=50).rank(pct=True)
    vol_reg = pd.Series(0, index=df.index, dtype=np.int8)
    vol_reg[(vol_pctile < 0.33).fillna(False)] = -1
    vol_reg[(vol_pctile > 0.66).fillna(False)] =  1
    feat["regime_vol"] = vol_reg.astype(np.float64)
    feat["regime_vol_is_low"]  = (vol_reg == -1).astype(np.float64)
    feat["regime_vol_is_mid"]  = (vol_reg ==  0).astype(np.float64)
    feat["regime_vol_is_high"] = (vol_reg ==  1).astype(np.float64)
    feat["regime_vol_bars_in"] = _bars_since_change(vol_reg).astype(np.float64)

    # ---- Drawdown regime classifier (3-state) ----
    # Where are we in the rolling-200 drawdown cycle?
    #   recovering: dd_200 > -3%   (close to peak)
    #   normal:     -10% < dd_200 < -3%
    #   drawdown:   dd_200 < -10%  (deep correction)
    peak200 = close.rolling(200).max()
    dd_200 = close / peak200 - 1.0
    dd_reg = pd.Series(0, index=df.index, dtype=np.int8)
    dd_reg[(dd_200 > -0.03).fillna(False)] =  1
    dd_reg[(dd_200 < -0.10).fillna(False)] = -1
    feat["regime_dd"] = dd_reg.astype(np.float64)
    feat["regime_dd_is_recovering"] = (dd_reg ==  1).astype(np.float64)
    feat["regime_dd_is_normal"]     = (dd_reg ==  0).astype(np.float64)
    feat["regime_dd_is_drawdown"]   = (dd_reg == -1).astype(np.float64)
    feat["regime_dd_bars_in"] = _bars_since_change(dd_reg).astype(np.float64)

    return feat
