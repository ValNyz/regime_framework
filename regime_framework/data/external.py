"""External data sources (FNG, ETF flows, DXY, VIX) + cross-asset OHLCV
loaded as features merged backward onto the main OHLCV frame.

Funding-rate features (target / BTC / ETH funding) live in their own
category — see `regime_framework/features/funding.py`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .alignment import merge_backward, merge_no_lookahead
from .loaders import load_parquet_or_feather


# Default file names inside the user's external/ directory.
# Funding files (binance_funding_BTC/ETH) live in features/funding.py.
DEFAULT_FILES = {
    "fng": "fng_daily.parquet",
    "etf_btc": "etf_flows_btc.parquet",
    "etf_eth": "etf_flows_eth.parquet",
    "dxy": "yfinance_DXY.parquet",
    "vix": "yfinance_VIX.parquet",
}


def _add_cross_features(
    feat: pd.DataFrame,
    df: pd.DataFrame,
    cross_path: Path,
    cross_name: str,
) -> None:
    """Append the standard ~13 cross-asset features for one reference coin."""
    cross = load_parquet_or_feather(cross_path)
    merged = merge_no_lookahead(
        df, cross,
        {"close": "_xc", "high": "_xh", "low": "_xl"},
    )
    cc = merged["_xc"]
    cross_log_ret = np.log(cc / cc.shift(1))
    for h in (5, 24, 72, 168, 720):
        feat[f"{cross_name}_ret_{h}"] = cc.pct_change(h).values
    for w in (24, 72, 168):
        feat[f"{cross_name}_vol_{w}"] = (cross_log_ret.rolling(w).std() * np.sqrt(w)).values
    for w in (50, 200):
        ema = cc.ewm(span=w, adjust=False).mean()
        feat[f"{cross_name}_dist_ema_{w}"] = (cc / ema - 1).values
    feat[f"target_{cross_name}_ratio"] = (df["close"].values / cc.values)
    ratio = pd.Series(feat[f"target_{cross_name}_ratio"], index=df.index)
    feat[f"target_{cross_name}_ratio_ret_24"] = ratio.pct_change(24).values
    feat[f"target_{cross_name}_ratio_ret_168"] = ratio.pct_change(168).values
    target_log_ret = np.log(df["close"] / df["close"].shift(1))
    feat[f"target_{cross_name}_corr_72"] = target_log_ret.rolling(72).corr(cross_log_ret).values
    feat[f"target_{cross_name}_corr_168"] = target_log_ret.rolling(168).corr(cross_log_ret).values


def load_external_features(
    df: pd.DataFrame,
    external_dir: Path | None,
    cross_ohlcv_path: Path | None = None,
    cross_name: str = "cross",
    extra_cross_paths: list[tuple[Path, str]] | None = None,
) -> pd.DataFrame:
    """Compute external feature matrix aligned to df (cross-asset + macro).

    Args:
        df: main OHLCV frame (must have 'date' and 'close')
        external_dir: directory containing macro files (FNG, ETF, DXY, VIX)
        cross_ohlcv_path: cross-asset OHLCV reference for relative-strength features
        cross_name: prefix-friendly name for the cross asset (eth, btc, ...)
        extra_cross_paths: optional list of (path, name) tuples — each adds the
            same ~13 features under its own prefix. Used by the multi-cross
            mechanism (cross_assets: in YAML).

    All features are past-only by construction. Funding-rate features
    (target / BTC / ETH) live in their own category — see
    `regime_framework/features/funding.py`.
    """
    feat = pd.DataFrame(index=df.index)

    # ----- Cross-asset OHLCV (relative strength + correlation) -----
    if cross_ohlcv_path is not None and cross_ohlcv_path.exists():
        try:
            _add_cross_features(feat, df, cross_ohlcv_path, cross_name)
        except Exception as e:
            print(f"  WARN: cross asset OHLCV skipped: {e}")

    # ----- Multi-cross side-channel features -----
    seen_names = {cross_name}
    for path, name in (extra_cross_paths or []):
        if name in seen_names:
            print(f"  WARN: extra cross '{name}' duplicates an existing prefix — skipped")
            continue
        if not path.exists():
            print(f"  WARN: extra cross missing, skipped: {path}")
            continue
        try:
            _add_cross_features(feat, df, path, name)
            seen_names.add(name)
        except Exception as e:
            print(f"  WARN: extra cross '{name}' skipped: {e}")

    if external_dir is None or not external_dir.exists():
        return feat

    # ----- Fear & Greed (daily) -----
    fng_path = external_dir / DEFAULT_FILES["fng"]
    if fng_path.exists():
        try:
            fng = load_parquet_or_feather(fng_path)
            merged = merge_backward(df, fng, ["fng_value"], prefix="fng")
            fv = merged["fng_fng_value"].astype(float)
            feat["fng_value"] = fv.values
            feat["fng_zscore_30"] = ((fv - fv.rolling(30).mean()) / (fv.rolling(30).std() + 1e-12)).values
            feat["fng_zscore_90"] = ((fv - fv.rolling(90).mean()) / (fv.rolling(90).std() + 1e-12)).values
            feat["fng_change_7"] = fv.diff(7 * 24).values
        except Exception as e:
            print(f"  WARN: FNG skipped: {e}")

    # ----- ETF flows BTC -----
    etf_btc = external_dir / DEFAULT_FILES["etf_btc"]
    if etf_btc.exists():
        try:
            etf = load_parquet_or_feather(etf_btc)
            merged = merge_backward(df, etf, ["flow_usd_m", "cumulative_usd_m"], prefix="etfbtc")
            fl = merged["etfbtc_flow_usd_m"].astype(float)
            cu = merged["etfbtc_cumulative_usd_m"].astype(float)
            feat["etf_btc_flow"] = fl.values
            feat["etf_btc_cum"] = cu.values
            feat["etf_btc_flow_mean_7"] = fl.rolling(7 * 24).mean().values
            feat["etf_btc_flow_zscore_30"] = (
                (fl - fl.rolling(30 * 24).mean()) / (fl.rolling(30 * 24).std() + 1e-12)
            ).values
        except Exception as e:
            print(f"  WARN: ETF BTC skipped: {e}")

    # ----- ETF flows ETH -----
    etf_eth = external_dir / DEFAULT_FILES["etf_eth"]
    if etf_eth.exists():
        try:
            etf = load_parquet_or_feather(etf_eth)
            merged = merge_backward(df, etf, ["flow_usd_m"], prefix="etfeth")
            fl = merged["etfeth_flow_usd_m"].astype(float)
            feat["etf_eth_flow"] = fl.values
            feat["etf_eth_flow_mean_7"] = fl.rolling(7 * 24).mean().values
        except Exception as e:
            print(f"  WARN: ETF ETH skipped: {e}")

    # ----- DXY -----
    dxy_path = external_dir / DEFAULT_FILES["dxy"]
    if dxy_path.exists():
        try:
            dxy = load_parquet_or_feather(dxy_path)
            merged = merge_backward(df, dxy, ["close"], prefix="dxy")
            dc = merged["dxy_close"].astype(float)
            feat["dxy_close"] = dc.values
            feat["dxy_ret_5d"] = dc.pct_change(5 * 24).values
            feat["dxy_ret_30d"] = dc.pct_change(30 * 24).values
            feat["dxy_zscore_90"] = (
                (dc - dc.rolling(90 * 24).mean()) / (dc.rolling(90 * 24).std() + 1e-12)
            ).values
        except Exception as e:
            print(f"  WARN: DXY skipped: {e}")

    # ----- VIX -----
    vix_path = external_dir / DEFAULT_FILES["vix"]
    if vix_path.exists():
        try:
            vix = load_parquet_or_feather(vix_path)
            merged = merge_backward(df, vix, ["close"], prefix="vix")
            vc = merged["vix_close"].astype(float)
            feat["vix_close"] = vc.values
            feat["vix_ret_5d"] = vc.pct_change(5 * 24).values
            feat["vix_zscore_30"] = ((vc - vc.rolling(30 * 24).mean()) / (vc.rolling(30 * 24).std() + 1e-12)).values
            feat["vix_zscore_90"] = ((vc - vc.rolling(90 * 24).mean()) / (vc.rolling(90 * 24).std() + 1e-12)).values
        except Exception as e:
            print(f"  WARN: VIX skipped: {e}")

    # External features have heterogeneous start dates (ETF flows from 2024,
    # DXY/VIX/FNG/funding from 2018-2019). Fill NaN with 0 so a row isn't
    # dropped just because one source's history is shorter than another's.
    # Semantically: 0 = "neutral / no info" (correct for z-scores, returns,
    # ratios; for absolute levels like vix_close it's effectively a missing-data
    # sentinel that the classifier learns to ignore).
    feat = feat.fillna(0.0)

    return feat
