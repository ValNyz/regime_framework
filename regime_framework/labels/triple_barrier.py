"""Triple-barrier labels (López de Prado ch. 3) — vol-scaled forward barriers.

At each bar t:
  upper = close[t] * (1 + tp_mult * σ_t)
  lower = close[t] * (1 - sl_mult * σ_t)
  σ_t   = rolling std of log returns over vol_lookback bars
  Look forward up to `horizon` bars. Whichever barrier is hit first sets
  the label:
    - upper hit first → 'bull'
    - lower hit first → 'bear'
    - neither hit (timeout) → '' (unlabelled — bar is dropped from the
      training/eval set, no false signal injected).

Adaptive in time horizon: in calm markets, prices take many bars to cross
±σ; in volatile markets, barriers get touched fast. The label is then a
direct trading question: 'taking a position now with a stop at ±σ, which
side gets hit first within the next H bars?'
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseLabeller


class TripleBarrierLabeller(BaseLabeller):
    name = "triple_barrier"

    def __init__(
        self,
        horizon: int = 720,        # 30 days at 1h timeframe
        tp_mult: float = 2.0,      # take-profit barrier in σ units
        sl_mult: float | None = None,  # stop-loss barrier in σ units; None = same as tp_mult
        vol_lookback: int = 168,   # 1 week at 1h timeframe for σ estimate
        # Backward compat alias: older configs used 'alpha' for symmetric barriers.
        alpha: float | None = None,
    ) -> None:
        self.horizon = int(horizon)
        if alpha is not None:
            tp_mult = float(alpha)
        self.tp_mult = float(tp_mult)
        self.sl_mult = float(sl_mult) if sl_mult is not None else float(tp_mult)
        self.vol_lookback = int(vol_lookback)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].to_numpy()
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        n = len(close)
        log_close = np.log(close)
        log_ret = np.diff(log_close, prepend=log_close[0])
        # σ_t = rolling std of log returns scaled to the lookback period.
        sigma = pd.Series(log_ret).rolling(self.vol_lookback).std().to_numpy() * np.sqrt(self.vol_lookback)

        labels = np.full(n, "", dtype=object)
        for t in range(self.vol_lookback, n - self.horizon):
            sig = sigma[t]
            if np.isnan(sig) or sig <= 0:
                continue
            c0 = close[t]
            upper = c0 * (1.0 + self.tp_mult * sig)
            lower = c0 * (1.0 - self.sl_mult * sig)
            fh = high[t + 1 : t + 1 + self.horizon]
            fl = low[t + 1 : t + 1 + self.horizon]
            uh = np.where(fh >= upper)[0]
            lh = np.where(fl <= lower)[0]
            ut = uh[0] if len(uh) else None
            lt = lh[0] if len(lh) else None
            # Timeout: leave label '' (unlabelled) — drops bar from training.
            # Cleaner than tagging 'range' which would be a third class that
            # the binary framework (LABEL_ORDER = bull/bear) can't handle.
            if ut is None and lt is None:
                continue
            elif ut is None:
                labels[t] = "bear"
            elif lt is None:
                labels[t] = "bull"
            else:
                labels[t] = "bull" if ut < lt else "bear"
        return pd.Series(labels, index=df.index, name="label")
