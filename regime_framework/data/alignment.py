"""Past-only alignment helpers for joining external data onto a main OHLCV frame."""
from __future__ import annotations

import pandas as pd


def force_ns_utc(series: pd.Series) -> pd.Series:
    """Coerce a datetime series to ns-UTC precision (required for merge_asof
    when source frames may have us- or ms-precision parquet timestamps).
    """
    return pd.to_datetime(series, utc=True, errors="coerce").astype("datetime64[ns, UTC]")


def merge_backward(
    main: pd.DataFrame,
    ext: pd.DataFrame,
    cols: list[str],
    prefix: str,
    on: str = "date",
) -> pd.DataFrame:
    """merge_asof(direction='backward'): at time t, use latest ext value with
    date <= t. Past-only by construction.

    Returns a 1-column-per-source frame indexed identically to `main`.
    """
    keep = [on] + [c for c in cols if c in ext.columns]
    sub = ext[keep].copy()
    rename_map = {c: f"{prefix}_{c}" for c in cols if c in sub.columns}
    sub = sub.rename(columns=rename_map)
    main_sorted = main[[on]].sort_values(on)
    out = pd.merge_asof(
        main_sorted,
        sub.sort_values(on),
        on=on,
        direction="backward",
    )
    return out
