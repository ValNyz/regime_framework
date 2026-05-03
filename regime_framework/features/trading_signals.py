"""Encode trading signals as binary 0/1 features for the regime classifier.

Reads a YAML file listing signals (signal_type + direction + params), evaluates
each on the OHLCV dataframe, and produces one binary column per signal.

Self-contained: no dependency on crypto_comparative_analysis. The user can
optionally point to their CCA YAML files (same schema).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _attach_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the indicators referenced by signal conditions (RSI, BB, VWAP, ATR, ADX)."""
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]

    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi"] = 100 - 100 / (1 + rs)

    # Bollinger Bands(20, 2)
    bb_ma = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_upper"] = bb_ma + 2 * bb_std
    out["bb_lower"] = bb_ma - 2 * bb_std
    out["bb_mid"] = bb_ma

    # ATR(14) and ATR%
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(14).mean()
    out["atr_pct"] = out["atr"] / close

    # ADX(14)
    up = high.diff()
    dn = -low.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0)
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / (out["atr"] + 1e-12)
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / (out["atr"] + 1e-12)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    out["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    # VWAP rolling 24h + zscore
    typical = (high + low + close) / 3
    vol = out["volume"] if "volume" in out.columns else pd.Series(1.0, index=out.index)
    cum_pv = (typical * vol).rolling(24, min_periods=1).sum()
    cum_v = vol.rolling(24, min_periods=1).sum()
    vwap = cum_pv / (cum_v + 1e-12)
    out["vwap"] = vwap
    dev = (close - vwap) / (vwap + 1e-12)
    out["vwap_zscore"] = (dev - dev.rolling(48).mean()) / (dev.rolling(48).std() + 1e-12)

    # MACD
    macd_fast = close.ewm(span=12, adjust=False).mean()
    macd_slow = close.ewm(span=26, adjust=False).mean()
    out["macd"] = macd_fast - macd_slow
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()

    # EMAs
    for w in (20, 50, 200):
        out[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()

    return out


def _eval_signal(df: pd.DataFrame, signal_type: str, direction: str, params: dict) -> pd.Series:
    """Evaluate a signal definition into a 0/1 Series.

    Supported signal_types:
      vwap_zscore, rsi, bb_extreme, bull_climax, bear_climax, trend_weakening,
      ema_cross, macd_cross
    """
    n = len(df)
    out = pd.Series(0, index=df.index, dtype=np.int8)

    if signal_type == "vwap_zscore":
        z = df["vwap_zscore"]
        thr = float(params.get("vwap_zscore_threshold", 2.0))
        if direction == "long":
            out.loc[z < -thr] = 1
        elif direction == "short":
            out.loc[z > thr] = 1

    elif signal_type == "rsi":
        r = df["rsi"]
        if direction == "long":
            out.loc[r < float(params.get("rsi_long_threshold", 30))] = 1
        elif direction == "short":
            out.loc[r > float(params.get("rsi_short_threshold", 70))] = 1

    elif signal_type == "bb_extreme":
        c = df["close"]
        if direction == "long":
            out.loc[c < df["bb_lower"]] = 1
        elif direction == "short":
            out.loc[c > df["bb_upper"]] = 1

    elif signal_type == "bear_climax":
        # Big down candle followed by reversal up
        ret = df["close"].pct_change()
        atrp = df.get("atr_pct", pd.Series(0.0, index=df.index))
        cond = (ret < -3 * atrp) & (ret.shift(-1) > 0)
        out.loc[cond.fillna(False)] = 1

    elif signal_type == "bull_climax":
        ret = df["close"].pct_change()
        atrp = df.get("atr_pct", pd.Series(0.0, index=df.index))
        cond = (ret > 3 * atrp) & (ret.shift(-1) < 0)
        out.loc[cond.fillna(False)] = 1

    elif signal_type == "trend_weakening":
        adx = df.get("adx", pd.Series(0.0, index=df.index))
        # ADX falling for 5+ consecutive bars
        falling = (adx.diff() < 0).rolling(5).sum() == 5
        out.loc[falling.fillna(False)] = 1

    elif signal_type == "ema_cross":
        fast = float(params.get("fast", 20))
        slow = float(params.get("slow", 50))
        ef = df["close"].ewm(span=fast, adjust=False).mean()
        es = df["close"].ewm(span=slow, adjust=False).mean()
        if direction == "long":
            out.loc[(ef > es) & (ef.shift(1) <= es.shift(1))] = 1
        elif direction == "short":
            out.loc[(ef < es) & (ef.shift(1) >= es.shift(1))] = 1

    elif signal_type == "macd_cross":
        m = df["macd"]
        s = df["macd_signal"]
        if direction == "long":
            out.loc[(m > s) & (m.shift(1) <= s.shift(1))] = 1
        elif direction == "short":
            out.loc[(m < s) & (m.shift(1) >= s.shift(1))] = 1

    # Other types (funding, ratio_btc/eth_*) are intentionally skipped — those
    # rely on external data that's already exposed via the external feature
    # set (target_funding, target_cross_ratio, etc.).
    return out


def compute_trading_signal_features(
    df: pd.DataFrame,
    yaml_path: Path | None,
) -> pd.DataFrame:
    """Compute one binary 0/1 feature per (signal name, direction) from a YAML file.

    YAML format:
        any_top_level_key:
          - name: "my_signal"
            signal_type: vwap_zscore
            direction: long
            params:
              vwap_zscore_threshold: 2.0

    Args:
        df: OHLCV frame
        yaml_path: path to signals YAML (resolved by config.py)

    Returns:
        DataFrame with columns named like `sig_<signal_name>_<direction>`.
    """
    if yaml_path is None or not Path(yaml_path).exists():
        return pd.DataFrame(index=df.index)

    try:
        data = yaml.safe_load(Path(yaml_path).read_text())
    except Exception as e:
        print(f"  WARN: failed to parse {yaml_path}: {e}")
        return pd.DataFrame(index=df.index)

    df_with_ind = _attach_indicators(df)

    signal_entries = []
    for group_name, entries in (data or {}).items():
        if not isinstance(entries, list):
            continue
        for sig in entries:
            if not sig.get("enabled", True):
                continue
            signal_entries.append(sig)

    print(f"  trading_signals: {len(signal_entries)} signals from {Path(yaml_path).name}")

    out = pd.DataFrame(index=df.index)
    for sig in signal_entries:
        name = sig.get("name", "unnamed")
        direction = sig.get("direction", "long")
        col_name = f"sig_{name}_{direction}"
        try:
            col = _eval_signal(df_with_ind, sig["signal_type"], direction, sig.get("params", {}) or {})
            out[col_name] = col.astype(np.int8)
        except KeyError as e:
            print(f"    WARN: {col_name} skipped (missing field: {e})")
        except Exception as e:
            print(f"    WARN: {col_name} skipped ({e})")

    return out
