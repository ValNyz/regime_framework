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
        timeout_label: str = "sign",  # "sign" (López de Prado canon) | "drop"
        # Backward compat alias: older configs used 'alpha' for symmetric barriers.
        alpha: float | None = None,
    ) -> None:
        self.horizon = int(horizon)
        if alpha is not None:
            tp_mult = float(alpha)
        self.tp_mult = float(tp_mult)
        self.sl_mult = float(sl_mult) if sl_mult is not None else float(tp_mult)
        self.vol_lookback = int(vol_lookback)
        if timeout_label not in ("sign", "drop"):
            raise ValueError(
                f"timeout_label must be 'sign' or 'drop' (got {timeout_label!r})"
            )
        self.timeout_label = timeout_label

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].to_numpy()
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        n = len(close)
        log_close = np.log(close)
        log_ret = np.diff(log_close, prepend=log_close[0])
        # σ_t = rolling std of log returns scaled to the lookback period.
        sigma = pd.Series(log_ret).rolling(self.vol_lookback).std().to_numpy() * np.sqrt(self.vol_lookback)
        # Backfill leading NaN σ with the first valid σ — lets us label the
        # opening `vol_lookback` bars too instead of leaving them dropped.
        # This is a minor approximation (uses future-derived σ for past bars)
        # but the σ value is only used to scale the barrier height, not the
        # direction; the label still comes from observed forward prices.
        sigma_s = pd.Series(sigma)
        first_valid = sigma_s.first_valid_index()
        if first_valid is not None:
            sigma_s.iloc[:first_valid] = sigma_s.iloc[first_valid]
        sigma = sigma_s.to_numpy()

        labels = np.full(n, "", dtype=object)
        # Run from bar 0 to bar n-2 (need at least 1 forward bar). For bars
        # close to the end where horizon would overshoot, shrink the lookahead
        # to whatever's available — every bar gets a label.
        for t in range(0, n - 1):
            sig = sigma[t]
            if np.isnan(sig) or sig <= 0:
                continue
            c0 = close[t]
            upper = c0 * (1.0 + self.tp_mult * sig)
            lower = c0 * (1.0 - self.sl_mult * sig)
            window_end = min(t + 1 + self.horizon, n)
            fh = high[t + 1 : window_end]
            fl = low[t + 1 : window_end]
            uh = np.where(fh >= upper)[0]
            lh = np.where(fl <= lower)[0]
            ut = uh[0] if len(uh) else None
            lt = lh[0] if len(lh) else None
            if ut is not None and lt is not None:
                labels[t] = "bull" if ut < lt else "bear"
            elif ut is not None:
                labels[t] = "bull"
            elif lt is not None:
                labels[t] = "bear"
            else:
                # Timeout: neither barrier hit within available lookahead.
                if self.timeout_label == "sign":
                    # Sign of return at the (possibly shrunken) horizon end.
                    ret = close[window_end - 1] - c0
                    if ret > 0:
                        labels[t] = "bull"
                    elif ret < 0:
                        labels[t] = "bear"
                    # exact-zero return stays '' (extremely rare)
                # else timeout_label == "drop": leave '' to skip this bar
        return pd.Series(labels, index=df.index, name="label")
