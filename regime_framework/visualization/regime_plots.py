"""Plot generators: 3 variants for labels, 3 for predictions.

(A) price + colored background spans + step regime panel
(B) synthetic equity curve from labels (perfect-regime trader) vs actual price
(C) multicolor price line, color = regime
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.dates import date2num
import numpy as np
import pandas as pd

from ..config import LABEL_COLORS, LABEL_ORDER, RunConfig


def denoise_labels(labels: pd.Series, window: int = 24) -> pd.Series:
    """Smooth a regime label series by rolling mode (most frequent in window)."""
    s = labels.copy()
    s[s == ""] = np.nan
    code_map = {c: i for i, c in enumerate(LABEL_ORDER)}
    inv_map = {i: c for c, i in code_map.items()}
    coded = s.map(code_map).astype(float)

    def _mode(arr):
        v = arr[~np.isnan(arr)]
        if len(v) == 0:
            return np.nan
        vals_, counts_ = np.unique(v.astype(int), return_counts=True)
        return float(vals_[counts_.argmax()])

    smoothed = coded.rolling(window, center=True, min_periods=1).apply(_mode, raw=True)
    out = smoothed.map(lambda x: inv_map.get(int(x), "") if pd.notna(x) else "")
    return pd.Series(out.values, index=labels.index, dtype=object)


def _compute_runs(df: pd.DataFrame, smooth: pd.Series) -> list[tuple[str, object, object]]:
    mask = smooth.values != ""
    if not mask.any():
        return []
    dates = pd.to_datetime(df["date"].values)
    labs = smooth.values
    runs = []
    cur_label = None
    cur_start = None
    for i in range(len(labs)):
        if not mask[i]:
            if cur_label is not None:
                runs.append((cur_label, cur_start, dates[i - 1]))
                cur_label = None
            continue
        if labs[i] != cur_label:
            if cur_label is not None:
                runs.append((cur_label, cur_start, dates[i - 1]))
            cur_label = labs[i]
            cur_start = dates[i]
    if cur_label is not None:
        runs.append((cur_label, cur_start, dates[-1]))
    return runs


def _legend_handles() -> list:
    return [mpatches.Patch(color=LABEL_COLORS[l], label=l, alpha=0.7) for l in LABEL_ORDER if l in LABEL_COLORS]


def _plot_A(df, runs, out_path, title_suffix: str, split_dt=None) -> None:
    fig, (ax_price, ax_regime) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]},
    )
    dates = pd.to_datetime(df["date"].values)
    closes = df["close"].values
    ax_price.plot(dates, closes, color="black", linewidth=0.7)
    ax_price.set_yscale("log")
    ax_price.set_ylabel("close (log)")
    ax_price.set_title(f"(A) [{title_suffix}] regime — price + colored background")
    ax_price.grid(True, alpha=0.3)
    if split_dt is not None:
        ax_price.axvline(split_dt, color="blue", linestyle="--", linewidth=1.2, alpha=0.7)

    y_levels = {l: i for i, l in enumerate(LABEL_ORDER)}
    for lab, s_dt, e_dt in runs:
        ax_price.axvspan(s_dt, e_dt, color=LABEL_COLORS.get(lab, "gray"), alpha=0.18, lw=0)
        if lab in y_levels:
            ax_regime.hlines(
                y_levels[lab], s_dt, e_dt,
                color=LABEL_COLORS.get(lab, "gray"), linewidth=5,
            )
    ax_regime.set_yticks(list(y_levels.values()))
    ax_regime.set_yticklabels(list(y_levels.keys()))
    ax_regime.set_ylabel("regime")
    ax_regime.set_ylim(-0.5, len(LABEL_ORDER) - 0.5)
    ax_regime.grid(True, alpha=0.3)
    ax_price.legend(handles=_legend_handles(), loc="upper left", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_B(df, smooth, runs, out_path, title_suffix: str, split_dt=None) -> None:
    dates = pd.to_datetime(df["date"].values)
    closes = df["close"].values
    log_ret = np.zeros(len(closes))
    log_ret[1:] = np.log(closes[1:] / closes[:-1])
    sign = np.zeros(len(smooth))
    sign[smooth.values == "bull"] = +1.0
    sign[smooth.values == "bear"] = -1.0
    synth_log_eq = np.cumsum(sign * log_ret)
    synth_eq = np.exp(synth_log_eq) * float(closes[0])

    fig, ax = plt.subplots(figsize=(14, 6))
    for lab, s_dt, e_dt in runs:
        ax.axvspan(s_dt, e_dt, color=LABEL_COLORS.get(lab, "gray"), alpha=0.10, lw=0)
    ax.plot(dates, closes, color="black", linewidth=0.7, label="close (actual)")
    ax.plot(dates, synth_eq, color="#1f77b4", linewidth=1.5, label="synth equity (long bull / short bear)")
    ax.set_yscale("log")
    ax.set_ylabel("price / equity (log)")
    ax.set_title(f"(B) [{title_suffix}] synthetic regime-trader equity vs price")
    ax.grid(True, alpha=0.3)
    if split_dt is not None:
        ax.axvline(split_dt, color="blue", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_C(df, smooth, runs, out_path, title_suffix: str, split_dt=None) -> None:
    dates_dt = pd.to_datetime(df["date"].values)
    dates_num = date2num(dates_dt)
    closes = df["close"].values
    labs = smooth.values

    pts = np.column_stack([dates_num, closes])
    segments = np.stack([pts[:-1], pts[1:]], axis=1)
    seg_colors = []
    for i in range(len(labs) - 1):
        lab = labs[i] if labs[i] != "" else (labs[i + 1] if i + 1 < len(labs) else "")
        seg_colors.append(LABEL_COLORS.get(lab, "#cccccc"))

    fig, ax = plt.subplots(figsize=(14, 6))
    lc = LineCollection(segments, colors=seg_colors, linewidths=1.0)
    ax.add_collection(lc)
    ax.set_xlim(dates_num.min(), dates_num.max())
    valid = closes[~np.isnan(closes.astype(float))]
    ax.set_ylim(valid.min() * 0.9, valid.max() * 1.1)
    ax.set_yscale("log")
    ax.xaxis_date()
    ax.set_ylabel("close (log)")
    ax.set_title(f"(C) [{title_suffix}] price line — color = regime")
    ax.grid(True, alpha=0.3)
    if split_dt is not None:
        ax.axvline(date2num(split_dt), color="blue", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.legend(handles=_legend_handles(), loc="upper left", framealpha=0.9)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def save_label_plots(df: pd.DataFrame, labels: pd.Series, out_dir: Path, cfg: RunConfig) -> None:
    smooth = denoise_labels(labels, window=24)
    runs = _compute_runs(df, smooth)
    suffix = f"labels-{cfg.target}-{cfg.timeframe}"
    _plot_A(df, runs, out_dir / "A_labels_background.png", suffix)
    _plot_B(df, smooth, runs, out_dir / "B_labels_synth.png", suffix)
    _plot_C(df, smooth, runs, out_dir / "C_labels_multicolor.png", suffix)


def save_prediction_plots(
    df: pd.DataFrame, predictions: pd.Series, out_dir: Path, cfg: RunConfig,
    predictor_name: str, split_dt=None,
) -> None:
    smooth = denoise_labels(predictions, window=24)
    runs = _compute_runs(df, smooth)
    suffix = f"pred-{predictor_name}-{cfg.target}-{cfg.timeframe}"
    _plot_A(df, runs, out_dir / "A_predictions_background.png", suffix, split_dt)
    _plot_B(df, smooth, runs, out_dir / "B_predictions_synth.png", suffix, split_dt)
    _plot_C(df, smooth, runs, out_dir / "C_predictions_multicolor.png", suffix, split_dt)
