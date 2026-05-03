"""Trend-scanning labels (López de Prado, Advances in FML, ch. 3.6) — binary version.

For each bar t, scan forward windows of length L in L_range, fitting a linear
regression close ~ alpha + beta·time on log(close)[t : t+L]. Find L* that
maximises |t-stat(beta)|. Label = sign(beta at L*) when |t| > t_threshold,
else "range" (which is rare with t_threshold=0).

Closed-form vectorised: rolling-sum operations, no nested loops over windows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseLabeller


class TrendScanLabeller(BaseLabeller):
    name = "trend_scan"

    def __init__(self, L_range: list[int] | None = None, t_threshold: float = 0.0) -> None:
        self.L_range = L_range or [72, 120, 168, 240, 336, 480, 720, 1080]
        self.t_threshold = float(t_threshold)

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

        return pd.Series(labels, index=df.index, name="label")
