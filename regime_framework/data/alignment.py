"""Past-only alignment helpers for joining external data onto a main OHLCV frame."""
from __future__ import annotations

import pandas as pd


def force_ns_utc(series: pd.Series) -> pd.Series:
    """Coerce a datetime series to ns-UTC precision (required for merge_asof
    when source frames may have us- or ms-precision parquet timestamps).
    """
    return pd.to_datetime(series, utc=True, errors="coerce").astype("datetime64[ns, UTC]")


def _main_bar_delta(main: pd.DataFrame, on: str = "date") -> pd.Timedelta | None:
    """Median time delta of `main[on]`. Returns None when undefined (n < 2)."""
    if len(main) < 2:
        return None
    s = pd.Series(main[on].sort_values().values)
    deltas = s.diff().dropna()
    if deltas.empty:
        return None
    return pd.Timedelta(deltas.median())


def merge_backward(
    main: pd.DataFrame,
    ext: pd.DataFrame,
    cols: list[str],
    prefix: str,
    on: str = "date",
) -> pd.DataFrame:
    """merge_asof(direction='backward'): at time t, use latest ext value with
    date <= t. Past-only by construction WHEN ext is at a STRICTLY SLOWER
    cadence than main (e.g. 8h funding onto 1h OHLCV, daily FNG, weekly
    ETF). For same-cadence joins use `merge_no_lookahead` instead — at
    same cadence the natural backward join picks ext[t] which on bar-OPEN
    timestamps is unobserved at row t.

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


def merge_no_lookahead(
    main: pd.DataFrame,
    ext: pd.DataFrame,
    rename_map: dict[str, str],
    *,
    on: str = "date",
) -> pd.DataFrame:
    """Backward merge for SAME-cadence sources with lookahead protection.

    Use whenever `ext` is at the same bar cadence as `main` (e.g. 1h
    funding onto 1h OHLCV, 1h cross-asset OHLCV onto 1h main). Shifts
    ext[on] forward by one main-bar median delta before merging, so the
    join returns ext[t-1]'s value at row t — the most recent observable
    bar at row t's OPEN.

    Without this shift, `merge_asof(direction='backward')` joins ext[t]
    onto main[t]. On Binance-style bar-OPEN timestamps, ext's bar-t close
    hasn't happened yet at time t → 1-bar lookahead inflates synth_gain
    and dir_kappa.

    Args:
        main: target frame, must have column `on`
        ext: source frame to pull columns from
        rename_map: {src_col_in_ext: dst_col_in_output}; explicit so callers
            don't need a post-merge `.rename(...)` cleanup
        on: time column (default "date")

    Returns DataFrame aligned to `main` with the renamed columns. Slow-
    cadence sources should use `merge_backward` instead — those are
    naturally protected since ext[t-k] (k>=1) is always observable.
    """
    src_cols = list(rename_map.keys())
    sub = ext[[on] + src_cols].rename(columns=rename_map).copy()
    bar_delta = _main_bar_delta(main, on=on)
    if bar_delta is not None:
        sub[on] = sub[on] + bar_delta
    main_sorted = main[[on]].sort_values(on)
    return pd.merge_asof(
        main_sorted, sub.sort_values(on),
        on=on, direction="backward",
    )
