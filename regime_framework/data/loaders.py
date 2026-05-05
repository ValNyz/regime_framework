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

    Funding-rate caveat: Binance funding rates are stamped at the funding
    INTERVAL BOUNDARY (00:00, 08:00, 16:00 UTC for most pairs). Whether
    the source file stamps `date` at the START of the funding window
    (rate published, applies forward) or at the END (rate settles, applied
    backward) is convention-dependent. Caller code must compensate if
    needed — this loader passes timestamps through unchanged. Use
    `audit_funding_convention(df)` below to spot-check.
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


def audit_funding_convention(
    funding_df: pd.DataFrame,
    interval_hours: int = 8,
) -> str:
    """Diagnose whether a funding-rate frame uses window-start or window-end
    timestamps. Returns one of: "start", "end", "uncertain".

    Heuristic: Binance funding rates settle at 00:00 / 08:00 / 16:00 UTC.
    If the loaded timestamps land exactly on those hours, the file uses
    window-start convention (e.g. row stamped 08:00 covers 08:00-16:00).
    If they're offset by `interval_hours` (e.g. land on 16:00 / 00:00 /
    08:00, where 16:00 settles the 08:00-16:00 window), it's window-end.

    Print the result and let the user decide whether to apply a shift.
    """
    if "date" not in funding_df.columns or len(funding_df) < 5:
        return "uncertain"
    sample = funding_df["date"].iloc[:50]
    hours = sample.dt.hour.unique()
    canonical_starts = set(range(0, 24, interval_hours))           # {0, 8, 16}
    canonical_ends = {(h + interval_hours) % 24 for h in canonical_starts}
    if set(hours).issubset(canonical_starts):
        return "start"
    if set(hours).issubset(canonical_ends) and not set(hours).issubset(canonical_starts):
        return "end"
    return "uncertain"
