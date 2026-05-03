"""Trend-scanning labels (López de Prado, Advances in FML, ch. 3.6) — binary version.

For each bar t, scan forward windows of length L in L_range, fitting a linear
regression close ~ alpha + beta·time on log(close)[t : t+L]. Find L* that
maximises |t-stat(beta)|. Label = sign(beta at L*) when |t| > t_threshold,
else "range" (which is rare with t_threshold=0).

OPTIONAL: temporal hysteresis. Without it, every bar's label is independent
and the result is noisy (a single counter-trend pullback inside a macro bull
flips the label to bear for that bar). With hysteresis, once a label is set
to bull, the next bars stay bull until the t-stat is strongly negative
(|t| > strong_threshold) for `hysteresis_bars` consecutive bars. Symmetric
for bear. Result: macro-trend regimes with realistic durations (weeks-months
on BTC 1h), at the cost of slower transition detection.

Closed-form vectorised regression; hysteresis is a single pass O(n).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseLabeller


class TrendScanLabeller(BaseLabeller):
    name = "trend_scan"

    def __init__(
        self,
        L_range: list[int] | None = None,
        t_threshold: float = 0.0,
        hysteresis_bars: int = 0,
        strong_threshold: float = 2.0,
    ) -> None:
        self.L_range = L_range or [72, 120, 168, 240, 336, 480, 720, 1080]
        self.t_threshold = float(t_threshold)
        self.hysteresis_bars = int(hysteresis_bars)
        self.strong_threshold = float(strong_threshold)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].to_numpy(dtype=float)
        n = len(close)
        log_close = np.log(close)

        best_tstat = np.full(n, np.nan)

        for L in self.L_range:
            if L >= n - 1:
                continue
            x = np.arange(L, dtype=float)
            x_mean = x.mean()
            x_dev = x - x_mean
            SS_x = float((x_dev ** 2).sum())

            beta_num = np.convolve(log_close, x_dev[::-1], mode="valid")
            beta = beta_num / SS_x

            roll_y_sum = np.convolve(log_close, np.ones(L), mode="valid")
            roll_y2_sum = np.convolve(log_close ** 2, np.ones(L), mode="valid")
            y_mean = roll_y_sum / L
            SS_y = roll_y2_sum - L * (y_mean ** 2)

            SSR = np.maximum(SS_y - (beta ** 2) * SS_x, 1e-15)
            sigma2 = SSR / max(L - 2, 1)
            SE_beta = np.sqrt(sigma2 / SS_x)
            tstat = beta / np.maximum(SE_beta, 1e-15)

            full = np.full(n, np.nan)
            full[: len(tstat)] = tstat

            abs_full = np.abs(full)
            abs_best = np.abs(best_tstat)
            update = (~np.isnan(abs_full)) & (np.isnan(abs_best) | (abs_full > abs_best))
            best_tstat = np.where(update, full, best_tstat)

        labels = np.full(n, "", dtype=object)
        for t in range(n):
            v = best_tstat[t]
            if np.isnan(v):
                continue
            if v > self.t_threshold:
                labels[t] = "bull"
            elif v < -self.t_threshold:
                labels[t] = "bear"
            else:
                labels[t] = "range"

        # Apply hysteresis if requested
        if self.hysteresis_bars > 0:
            labels = self._apply_hysteresis(labels, best_tstat)

        return pd.Series(labels, index=df.index, name="label")

    def _apply_hysteresis(self, labels: np.ndarray, tstat: np.ndarray) -> np.ndarray:
        """Smooth label flips: stay in current regime until the OPPOSITE direction
        is confirmed by |t| > strong_threshold for `hysteresis_bars` consecutive bars.
        """
        out = labels.copy()
        cur = ""
        contra_count = 0
        for t in range(len(out)):
            v = tstat[t]
            raw = out[t]
            if raw == "" or np.isnan(v):
                # Skip unlabelled bars; keep current regime
                if cur:
                    out[t] = cur
                contra_count = 0
                continue

            if cur == "":
                cur = raw
                contra_count = 0
                continue

            if raw == cur:
                contra_count = 0
                out[t] = cur
                continue

            # Opposite direction candidate — count consecutive STRONG opposite t-stats
            is_strong = abs(v) > self.strong_threshold
            opposite_consistent = (raw != cur) and is_strong
            if opposite_consistent:
                contra_count += 1
                if contra_count >= self.hysteresis_bars:
                    cur = raw
                    contra_count = 0
                    out[t] = cur
                else:
                    out[t] = cur  # stay in current regime
            else:
                # Weak opposite signal — reset counter, stay in current
                contra_count = 0
                out[t] = cur
        return out
