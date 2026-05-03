"""OHLCV / parquet / feather loaders with consistent dtype handling."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .alignment import force_ns_utc


def load_ohlcv(path: Path | str) -> pd.DataFrame:
    """Load an OHLCV feather/parquet file. Returns columns: date, open, high,
    low, close, volume (volume optional). `date` is always tz-aware UTC ns.
    """
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix == ".feather":
        df = pd.read_feather(p)
    else:
        raise ValueError(f"Unsupported extension {p.suffix} for OHLCV: {p}")

    if "timestamp" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"timestamp": "date"})
    if "date" not in df.columns:
        raise ValueError(f"OHLCV missing 'date' column: {p}")
    df["date"] = force_ns_utc(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def load_parquet_or_feather(path: Path | str) -> pd.DataFrame:
    """Generic loader for sidecar files (funding, FNG, ETF, DXY, VIX).
    Normalises 'timestamp' column to 'date' and forces ns-UTC.
    """
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix == ".feather":
        df = pd.read_feather(p)
    else:
        raise ValueError(f"Unsupported extension {p.suffix}: {p}")
    if "timestamp" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"timestamp": "date"})
    if "date" in df.columns:
        df["date"] = force_ns_utc(df["date"])
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df
