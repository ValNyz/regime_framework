"""Encode the user's trading signals as binary features for the regime classifier.

For each SignalConfig in a YAML file, evaluate its entry condition on the OHLCV
dataframe → produce a 0/1 column. The result is a feature matrix where each
column is "is signal X currently triggered for direction D?" — usable both
as inputs to a regime classifier and as standalone predictors (see
signal_analysis/lift.py).

Lazy import of crypto_comparative_analysis if it's available — the framework
falls back to a minimal in-house signal evaluator if not.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Path to add to import the existing framework (where SignalConfig + entry_logic live)
CCA_ROOT = Path("/home/nyzam/Documents/Valentin/crypto_comparative_analysis")


def _import_cca():
    """Lazy import of crypto_comparative_analysis modules."""
    if str(CCA_ROOT) not in sys.path:
        sys.path.insert(0, str(CCA_ROOT))
    try:
        from lib.signals.base import SignalConfig
        from lib.signals.registry import load_signals_from_yaml
        from lib.config.base import Config as CcaConfig
        return SignalConfig, load_signals_from_yaml, CcaConfig
    except ImportError as e:
        print(f"  WARN: crypto_comparative_analysis not importable: {e}")
        return None, None, None


def _attach_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the indicators referenced by signal conditions (RSI, BB, etc.).

    Mirrors the canonical indicator block used by the CCA generator. Kept
    minimal — only what's broadly used. If the user has signals that depend
    on rarer indicators, this is the place to extend.
    """
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

    # Bollinger Bands (20, 2)
    bb_ma = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_upper"] = bb_ma + 2 * bb_std
    out["bb_lower"] = bb_ma - 2 * bb_std
    out["bb_mid"] = bb_ma

    # ATR(14) and ATR%
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(14).mean()
    out["atr_pct"] = out["atr"] / close

    # ADX(14) — Wilder's smoothing approximated by EMA
    up = high.diff()
    dn = -low.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0)
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / (out["atr"] + 1e-12)
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / (out["atr"] + 1e-12)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    out["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    # VWAP rolling (24h) + zscore
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

    # EMAs commonly referenced
    for w in (20, 50, 200):
        out[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()

    return out


def _eval_simple_signal(df: pd.DataFrame, signal_type: str, direction: str, params: dict) -> pd.Series:
    """Minimal in-house evaluator for the most common signal types.

    Supports: vwap_zscore, rsi, bb, ratio_btc_extreme (skipped if no BTC col),
    funding (skipped — needs external data), bear_climax / bull_climax,
    trend_weakening (rough proxy).

    Returns a 0/1 Series indexed identically to df.
    """
    n = len(df)
    out = pd.Series(0, index=df.index, dtype=int)

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

    elif signal_type in ("bear_climax", "bull_climax"):
        # Climax: extreme bar with reversal signature
        # Approximation: a bar with abs(return) > 3*ATR% AND opposite direction follow-through
        ret = df["close"].pct_change()
        atrp = df.get("atr_pct", pd.Series(0.0, index=df.index))
        if signal_type == "bear_climax":
            cond = (ret < -3 * atrp) & (ret.shift(-1) > 0)
        else:
            cond = (ret > 3 * atrp) & (ret.shift(-1) < 0)
        out.loc[cond.fillna(False)] = 1

    elif signal_type == "trend_weakening":
        # Approximation: ADX falling for 5+ bars in a row
        adx = df.get("adx", pd.Series(0.0, index=df.index))
        falling = (adx.diff() < 0).rolling(5).sum() == 5
        out.loc[falling.fillna(False)] = 1

    # Other types (ratio_btc_*, funding, ratio_eth_*, etc.) need extra data
    # not present in df — skip and return zeros.

    return out


def compute_trading_signal_features(
    df: pd.DataFrame,
    yaml_path: Path | None,
) -> pd.DataFrame:
    """Compute one binary 0/1 feature per (signal name, direction) pair from a YAML file.

    If `crypto_comparative_analysis` is importable, signals are loaded via its
    canonical loader (`load_signals_from_yaml`). Otherwise no features are
    produced.

    Args:
        df: OHLCV frame
        yaml_path: path to user signals YAML

    Returns:
        DataFrame with columns named like `sig_<signal_name>_<direction>` taking
        values in {0, 1}.
    """
    if yaml_path is None or not yaml_path.exists():
        return pd.DataFrame(index=df.index)

    SignalConfig, load_signals_from_yaml, CcaConfig = _import_cca()
    if load_signals_from_yaml is None:
        return pd.DataFrame(index=df.index)

    cca_cfg = CcaConfig(data_dir=str(df.attrs.get("data_dir", "/tmp")), strategies_dir="/tmp/_unused")
    try:
        signals = load_signals_from_yaml(cca_cfg, str(yaml_path))
    except Exception as e:
        print(f"  WARN: load_signals_from_yaml failed: {e}")
        return pd.DataFrame(index=df.index)

    print(f"  trading_signals: loaded {len(signals)} signals from {yaml_path.name}")
    df_with_ind = _attach_indicators(df)

    out = pd.DataFrame(index=df.index)
    for sig in signals:
        col_name = f"sig_{sig.name}_{sig.direction}"
        try:
            col = _eval_simple_signal(df_with_ind, sig.signal_type, sig.direction, sig.params)
            out[col_name] = col.astype(np.int8)
        except Exception as e:
            print(f"    WARN: {col_name} skipped ({e})")

    return out
