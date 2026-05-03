"""Triple-barrier labels (López de Prado ch. 3) — vol-scaled forward barriers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseLabeller


class TripleBarrierLabeller(BaseLabeller):
    name = "triple_barrier"

    def __init__(self, horizon: int = 48, alpha: float = 1.5, vol_lookback: int = 48) -> None:
        self.horizon = int(horizon)
        self.alpha = float(alpha)
        self.vol_lookback = int(vol_lookback)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].to_numpy()
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        n = len(close)
        log_close = np.log(close)
        log_ret = np.diff(log_close, prepend=log_close[0])
        sigma = pd.Series(log_ret).rolling(self.vol_lookback).std().to_numpy() * np.sqrt(self.vol_lookback)

        labels = np.full(n, "", dtype=object)
        for t in range(self.vol_lookback, n - self.horizon):
            sig = sigma[t]
            if np.isnan(sig) or sig <= 0:
                continue
            c0 = close[t]
            upper = c0 * (1.0 + self.alpha * sig)
            lower = c0 * (1.0 - self.alpha * sig)
            fh = high[t + 1 : t + 1 + self.horizon]
            fl = low[t + 1 : t + 1 + self.horizon]
            uh = np.where(fh >= upper)[0]
            lh = np.where(fl <= lower)[0]
            ut = uh[0] if len(uh) else None
            lt = lh[0] if len(lh) else None
            if ut is None and lt is None:
                labels[t] = "range"
            elif ut is None:
                labels[t] = "bear"
            elif lt is None:
                labels[t] = "bull"
            else:
                labels[t] = "bull" if ut < lt else "bear"
        return pd.Series(labels, index=df.index, name="label")
