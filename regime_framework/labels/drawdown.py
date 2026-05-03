"""Drawdown-from-peak labels with hysteresis. Simple alternative to trend-scan.

Bull when price is at-time-highs (within `bull_top_pct` of rolling N-bar peak).
Bear when drawdown from rolling peak exceeds `bear_dd_pct`.
Range otherwise. Hysteresis: regime change requires `min_run_bars` consecutive
bars in the new state.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseLabeller


class DrawdownLabeller(BaseLabeller):
    name = "drawdown"

    def __init__(
        self,
        peak_window: int = 24 * 30,    # 30 days at 1h
        bull_top_pct: float = 0.05,    # within 5% of recent peak = bull
        bear_dd_pct: float = 0.20,     # 20% DD = bear
        min_run_bars: int = 12,        # hysteresis
    ) -> None:
        self.peak_window = int(peak_window)
        self.bull_top_pct = float(bull_top_pct)
        self.bear_dd_pct = float(bear_dd_pct)
        self.min_run_bars = int(min_run_bars)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)
        peak = close.rolling(self.peak_window, min_periods=1).max()
        dd = close / peak - 1.0
        raw = pd.Series("range", index=df.index, dtype=object)
        raw[dd >= -self.bull_top_pct] = "bull"
        raw[dd <= -self.bear_dd_pct] = "bear"

        # Apply hysteresis: don't switch unless new regime persists min_run_bars
        out = raw.copy()
        cur = raw.iloc[0]
        run = 1
        for i in range(1, len(raw)):
            if raw.iloc[i] == cur:
                out.iloc[i] = cur
                run = 0
                continue
            run += 1
            if run >= self.min_run_bars:
                cur = raw.iloc[i]
                out.iloc[i] = cur
                run = 0
            else:
                out.iloc[i] = cur

        return out.rename("label")
