"""Encode trading signals as binary 0/1 features for the regime classifier.

Compatible with crypto_comparative_analysis YAML grids:
  - Group-prefix-based signal_type inference (funding_*, combo_*, technical_*)
  - Parameter grid expansion (cartesian product on list-valued params)
  - Built-in evaluators for: vwap_zscore, rsi, bb_extreme, bull/bear_climax,
    trend_weakening, ema_cross, macd_cross, funding (if funding rate provided)
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Group-prefix → default signal_type (mirrors crypto_comparative_analysis
# lib/signals/registry.py _signal_type_for_category)
# ---------------------------------------------------------------------------
def _signal_type_for_group(category: str) -> str | None:
    if category.startswith("funding"):
        return "funding"
    if category.startswith("combo"):
        return "combo"
    return None  # use entry's signal_type field


# ---------------------------------------------------------------------------
# Parameter grid expansion
# ---------------------------------------------------------------------------
def _expand_grid(params: dict | None) -> list[dict]:
    """Expand list-valued params into a cartesian product of single-value dicts."""
    if not params:
        return [{}]
    grid_keys = []
    grid_vals = []
    fixed: dict = {}
    for k, v in params.items():
        if isinstance(v, list):
            grid_keys.append(k)
            grid_vals.append(v)
        else:
            fixed[k] = v
    if not grid_keys:
        return [fixed]
    out = []
    for combo in itertools.product(*grid_vals):
        d = dict(fixed)
        for k, val in zip(grid_keys, combo):
            d[k] = val
        out.append(d)
    return out


def _format_name(template: str, params: dict) -> str:
    """Substitute {param_name} placeholders with values; sanitize for filenames."""
    try:
        return template.format(**params).replace(".", "_")
    except KeyError:
        return template.replace("{", "").replace("}", "")


# ---------------------------------------------------------------------------
# Indicators required by signal evaluators
# ---------------------------------------------------------------------------
def _attach_indicators(df: pd.DataFrame, funding: pd.Series | None = None) -> pd.DataFrame:
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

    # Bollinger
    bb_ma = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_upper"] = bb_ma + 2 * bb_std
    out["bb_lower"] = bb_ma - 2 * bb_std
    out["bb_mid"] = bb_ma

    # ATR
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(14).mean()
    out["atr_pct"] = out["atr"] / close

    # ADX
    up = high.diff()
    dn = -low.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0)
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / (out["atr"] + 1e-12)
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / (out["atr"] + 1e-12)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    out["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    # VWAP zscore (24h rolling, 48h dev window)
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

    # Funding (optional)
    if funding is not None:
        out["funding_rate"] = funding.values

    return out


# ---------------------------------------------------------------------------
# Signal evaluators
# ---------------------------------------------------------------------------
def _eval_signal(df: pd.DataFrame, signal_type: str, direction: str, params: dict) -> pd.Series:
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
        # Causal: detect AT BAR t that bar t-1 had a big down move (> 3 ATR%)
        # and this bar (t) is reversing up. Original used ret.shift(-1) which
        # peeked at the next bar — that was a look-ahead bias bug.
        ret = df["close"].pct_change()
        atrp = df.get("atr_pct", pd.Series(0.0, index=df.index))
        cond = (ret.shift(1) < -3 * atrp.shift(1)) & (ret > 0)
        out.loc[cond.fillna(False)] = 1

    elif signal_type == "bull_climax":
        # Causal counterpart of bear_climax — see comment above.
        ret = df["close"].pct_change()
        atrp = df.get("atr_pct", pd.Series(0.0, index=df.index))
        cond = (ret.shift(1) > 3 * atrp.shift(1)) & (ret < 0)
        out.loc[cond.fillna(False)] = 1

    elif signal_type == "trend_weakening":
        adx = df.get("adx", pd.Series(0.0, index=df.index))
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
        m = df["macd"]; s = df["macd_signal"]
        if direction == "long":
            out.loc[(m > s) & (m.shift(1) <= s.shift(1))] = 1
        elif direction == "short":
            out.loc[(m < s) & (m.shift(1) >= s.shift(1))] = 1

    elif signal_type == "funding":
        # Funding rate z-score over `lookback` bars; trigger when |z| > zscore threshold.
        # CCA convention: funding_long means funding extremely negative (shorts crowded → squeeze up).
        if "funding_rate" not in df.columns:
            return out  # silently zero — caller should warn
        fr = df["funding_rate"].astype(float)
        lb = int(params.get("lookback", 336))
        thr = float(params.get("zscore", 2.0))
        z = (fr - fr.rolling(lb).mean()) / (fr.rolling(lb).std() + 1e-12)
        if direction in ("long", "both"):
            out.loc[z < -thr] = 1
        if direction == "short":
            out.loc[z > thr] = 1
        if direction == "both":
            # Use 2-sided trigger; downstream lift will pick which side correlates with bull/bear
            out.loc[z.abs() > thr] = 1

    # Other types (combo, ratio_*) skipped — handled via external features
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def compute_trading_signal_features(
    df: pd.DataFrame,
    yaml_path: Path | None,
    funding: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute one binary 0/1 column per (signal grid point, direction).

    Args:
        df: OHLCV frame
        yaml_path: path to a signals YAML (CCA-compatible format)
        funding: optional funding rate Series aligned to df index — enables
                 the `funding` signal_type. If None, funding signals are zero.
    """
    if yaml_path is None or not Path(yaml_path).exists():
        return pd.DataFrame(index=df.index)

    try:
        data = yaml.safe_load(Path(yaml_path).read_text())
    except Exception as e:
        print(f"  WARN: failed to parse {yaml_path}: {e}")
        return pd.DataFrame(index=df.index)

    df_with_ind = _attach_indicators(df, funding=funding)

    # Collect (group_name, entry) pairs and apply group-prefix signal_type override
    expanded: list[tuple[str, str, str, dict]] = []  # (name, signal_type, direction, params)
    n_grid_total = 0
    for group_name, entries in (data or {}).items():
        if not isinstance(entries, list):
            continue
        forced_type = _signal_type_for_group(str(group_name))
        for sig in entries:
            if not sig.get("enabled", True):
                continue
            sig_type = forced_type or sig.get("signal_type")
            if sig_type is None:
                print(f"    WARN: '{sig.get('name', '?')}' skipped (no signal_type in entry nor group prefix)")
                continue
            direction = sig.get("direction", "long")
            base_name = sig.get("name", "unnamed")
            params_raw = sig.get("params", {}) or {}
            for params in _expand_grid(params_raw):
                name = _format_name(base_name, params)
                expanded.append((name, sig_type, direction, params))
                n_grid_total += 1

    print(
        f"  trading_signals: {n_grid_total} signals from {Path(yaml_path).name} "
        f"(after grid expansion)"
    )

    out = pd.DataFrame(index=df.index)
    for name, sig_type, direction, params in expanded:
        col_name = f"sig_{name}_{direction}"
        try:
            col = _eval_signal(df_with_ind, sig_type, direction, params)
            if col.sum() == 0 and sig_type in ("funding", "combo"):
                # Silently drop unsupported funding signals when no funding data
                continue
            out[col_name] = col.astype(np.int8)
        except KeyError as e:
            print(f"    WARN: {col_name} skipped (missing indicator: {e})")
        except Exception as e:
            print(f"    WARN: {col_name} skipped ({e})")

    return out
