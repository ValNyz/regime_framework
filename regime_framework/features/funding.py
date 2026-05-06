"""Funding-rate features (target asset native + cross-asset BTC/ETH).

Split out of `external.py` so funding signals form their own toggleable
feature category. The trading-signals YAML (e.g. signals_btc_opt_OOS_2026)
relies heavily on funding-rate z-scores at multiple lookbacks; this module
exposes the underlying continuous values + rolling stats so the model can
also learn from funding-rate dynamics directly, not only via the binary
trigger signals.

Past-only by construction: same-cadence joins use `merge_no_lookahead`
(1-bar shift), slow-cadence (8h funding) uses `merge_backward`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data.alignment import merge_backward, merge_no_lookahead
from ..data.loaders import load_parquet_or_feather


# Default file names inside the user's external/ directory
DEFAULT_FILES = {
    "btc_funding": "binance_funding_BTCUSDT.parquet",
    "eth_funding": "binance_funding_ETHUSDT.parquet",
}


def compute_funding_features(
    df: pd.DataFrame,
    target_funding_path: Path | None = None,
    external_dir: Path | None = None,
) -> pd.DataFrame:
    """Compute funding-rate feature matrix aligned to df.

    Args:
        df: main OHLCV frame (must have 'date')
        target_funding_path: native funding rate of the target asset (1h)
        external_dir: directory containing BTC / ETH funding parquets

    All features are past-only by construction.
    """
    feat = pd.DataFrame(index=df.index)

    # ----- Target asset funding (1h native, same-cadence join) -----
    if target_funding_path is not None and target_funding_path.exists():
        try:
            fund = load_parquet_or_feather(target_funding_path)
            fund_col = "open" if "open" in fund.columns else "funding_rate"
            merged = merge_no_lookahead(df, fund, {fund_col: "fund_rate"})
            fr = merged["fund_rate"].astype(float)
            feat["target_funding"] = fr.values
            for w in (24, 72, 168, 720):
                feat[f"target_funding_mean_{w}"] = fr.rolling(w).mean().values
                feat[f"target_funding_zscore_{w}"] = (
                    (fr - fr.rolling(w).mean()) / (fr.rolling(w).std() + 1e-12)
                ).values
            feat["target_funding_cum_168"] = fr.rolling(168).sum().values
            feat["target_funding_cum_720"] = fr.rolling(720).sum().values
        except Exception as e:
            print(f"  WARN: target funding skipped ({target_funding_path.name}): {e}")

    if external_dir is None or not external_dir.exists():
        return feat.fillna(0.0)

    # ----- BTC funding (8h sparse, slow-cadence backward join) -----
    btc_fund = external_dir / DEFAULT_FILES["btc_funding"]
    if btc_fund.exists():
        try:
            bfund = load_parquet_or_feather(btc_fund)
            merged = merge_backward(df, bfund, ["funding_rate"], prefix="btcfund")
            fr = merged["btcfund_funding_rate"].astype(float)
            feat["btc_funding"] = fr.values
            feat["btc_funding_mean_21"] = fr.rolling(21).mean().values
            feat["btc_funding_zscore_21"] = (
                (fr - fr.rolling(21).mean()) / (fr.rolling(21).std() + 1e-12)
            ).values
            feat["btc_funding_zscore_90"] = (
                (fr - fr.rolling(90).mean()) / (fr.rolling(90).std() + 1e-12)
            ).values
            feat["btc_funding_cum_21"] = fr.rolling(21).sum().values
        except Exception as e:
            print(f"  WARN: BTC funding skipped: {e}")

    # ----- ETH funding -----
    eth_fund = external_dir / DEFAULT_FILES["eth_funding"]
    if eth_fund.exists():
        try:
            efund = load_parquet_or_feather(eth_fund)
            merged = merge_backward(df, efund, ["funding_rate"], prefix="ethfund")
            fr = merged["ethfund_funding_rate"].astype(float)
            feat["eth_funding"] = fr.values
            feat["eth_funding_zscore_21"] = (
                (fr - fr.rolling(21).mean()) / (fr.rolling(21).std() + 1e-12)
            ).values
        except Exception as e:
            print(f"  WARN: ETH funding skipped: {e}")

    return feat.fillna(0.0)
