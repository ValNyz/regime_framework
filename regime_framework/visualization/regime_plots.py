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
import numpy as np
import pandas as pd

from ..config import LABEL_COLORS, LABEL_ORDER, RunConfig


def denoise_labels(labels: pd.Series, window: int = 168) -> pd.Series:
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

    # Causal smoothing (center=False): each bar's smoothed value uses only its
    # own past `window` bars. Earlier center=True peeked ~window/2 bars into
    # the future, which made the synth-equity plot look better than tradeable
    # because trades fired on look-ahead-smoothed labels.
    smoothed = coded.rolling(window, center=False, min_periods=1).apply(_mode, raw=True)
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


def _plot_B(
    df, raw_labels, smooth, runs, out_path, title_suffix: str, split_dt=None,
) -> None:
    """raw_labels drive the equity curve (matches the synth_gain metric);
    smooth drives the colored regime bands (visual only). Earlier code used
    smooth for both, which made the plot diverge from the console gain metric
    by up to a 7-day lag (denoise window).
    """
    from ..evaluation.metrics import synth_equity_curve
    dates = pd.to_datetime(df["date"].values)
    closes = df["close"].values
    synth_eq, _ = synth_equity_curve(closes, raw_labels.values)

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


# _plot_C (price line color-coded by regime) was removed — duplicated info
# already shown by plot A's regime bands and plot B's equity curve.


def plot_stitched_oos_equity(
    df: pd.DataFrame,
    folds: list[dict],
    out_path: Path,
    title_suffix: str,
    denoise_window: int = 168,
) -> None:
    """Stitched OOS synth equity across CV folds — best predictor per fold.

    folds: list of dicts with keys
        - test_index: pd.Index of bars in this fold's test set
        - predictions: np.ndarray of label strings aligned with test_index
        - predictor_name: str (for legend)
        - kappa: float (for color intensity)
        - test_start: str date (for boundary marker)
    """
    from ..evaluation.metrics import synth_equity_curve
    closes = df["close"].values.astype(float)
    dates = pd.to_datetime(df["date"].values)

    # Build a single label series spanning all folds' test windows
    out = pd.Series("", index=df.index, dtype=object)
    for fold in folds:
        idx = fold["test_index"]
        preds = fold["predictions"]
        if len(preds) == len(idx):
            out.loc[idx] = preds

    # Use raw stitched predictions (no smoothing) — matches the per-fold
    # synth_gain metric. Smoothing was a 7-day visual cleanup that diverged
    # the equity curve from the reported numbers.
    synth_eq, _ = synth_equity_curve(closes, out.values)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(dates, closes, color="black", linewidth=0.7, alpha=0.4, label="close (actual)")
    ax.plot(dates, synth_eq, color="#1f77b4", linewidth=1.5,
            label="OOS synth equity (best predictor per fold, stitched)")

    # Fold boundary lines + per-fold annotation (predictor name + kappa)
    for i, fold in enumerate(folds):
        idx = fold["test_index"]
        if len(idx) == 0:
            continue
        start_dt = pd.to_datetime(df.loc[idx[0], "date"])
        ax.axvline(start_dt, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        # Annotate at the top of the fold
        ax.text(
            start_dt, ax.get_ylim()[1] * 0.95 if False else closes.max(),
            f"  {fold['predictor_name']}\n  κ={fold['kappa']:+.3f}",
            fontsize=7, color="gray", verticalalignment="top",
        )

    ax.set_yscale("log")
    ax.set_ylabel("price / equity (log)")
    ax.set_title(
        f"(B-stitched) [{title_suffix}] OOS synth equity across {len(folds)} folds"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_synth_equity_multi(
    df: pd.DataFrame,
    predictions_by_name: dict[str, pd.Series],
    out_path: Path,
    title_suffix: str,
    split_dt=None,
    denoise_window: int = 168,
) -> None:
    """Overlay synthetic regime-trader equity curves for multiple predictors.

    Each predictor's predictions → smoothed → long_when_bull/short_when_bear/flat
    → cumulative equity. All curves on the same log y-axis with the actual price
    in dim black for reference.
    """
    from ..evaluation.metrics import synth_equity_curve
    closes = df["close"].values.astype(float)
    dates = pd.to_datetime(df["date"].values)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(dates, closes, color="black", linewidth=0.7, alpha=0.35, label="close (actual)")

    cmap = plt.get_cmap("tab10")
    items = sorted(predictions_by_name.items())  # stable color assignment
    for i, (name, preds) in enumerate(items):
        try:
            # Use raw predictions (no smoothing) so the equity curves match
            # the synth_gain console metric. Earlier 168-bar smoothing made
            # the multi plot diverge from the per-fold gain numbers.
            preds_arr = preds if isinstance(preds, np.ndarray) else np.asarray(preds)
            synth_eq, _ = synth_equity_curve(closes, preds_arr)
            ax.plot(dates, synth_eq, color=cmap(i % 10), linewidth=1.2,
                    alpha=0.9, label=name)
        except Exception:
            continue

    ax.set_yscale("log")
    ax.set_ylabel("price / equity (log)")
    ax.set_title(f"(B-multi) [{title_suffix}] synth equity per predictor")
    if split_dt is not None:
        ax.axvline(split_dt, color="blue", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8, ncols=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_regime_step_multi(
    df: pd.DataFrame,
    predictions_by_name: dict[str, pd.Series],
    out_path: Path,
    title_suffix: str,
    split_dt=None,
    denoise_window: int = 168,
) -> None:
    """Plot price (top) + one row per predictor showing its bull/bear track (bottom)."""
    closes = df["close"].values.astype(float)
    dates = pd.to_datetime(df["date"].values)
    n_pred = len(predictions_by_name)
    if n_pred == 0:
        return

    fig = plt.figure(figsize=(14, 4 + 0.6 * n_pred))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, max(0.3 * n_pred, 1.0)])
    ax_price = fig.add_subplot(gs[0])
    ax_panel = fig.add_subplot(gs[1], sharex=ax_price)

    ax_price.plot(dates, closes, color="black", linewidth=0.7)
    ax_price.set_yscale("log")
    ax_price.set_ylabel("close (log)")
    ax_price.set_title(f"(A-multi) [{title_suffix}] regime track per predictor")
    ax_price.grid(True, alpha=0.3)
    if split_dt is not None:
        ax_price.axvline(split_dt, color="blue", linestyle="--", linewidth=1.2, alpha=0.7)

    items = sorted(predictions_by_name.items())
    yticks = []
    yticklabels = []
    for i, (name, preds) in enumerate(items):
        try:
            smooth = denoise_labels(preds, window=denoise_window)
            runs = _compute_runs(df, smooth)
            for lab, s_dt, e_dt in runs:
                col = LABEL_COLORS.get(lab, "gray")
                ax_panel.hlines(i, s_dt, e_dt, color=col, linewidth=6)
            yticks.append(i)
            yticklabels.append(name)
        except Exception:
            yticks.append(i)
            yticklabels.append(f"{name} (err)")

    ax_panel.set_yticks(yticks)
    ax_panel.set_yticklabels(yticklabels)
    ax_panel.set_ylim(-0.5, n_pred - 0.5)
    ax_panel.grid(True, alpha=0.3)
    if split_dt is not None:
        ax_panel.axvline(split_dt, color="blue", linestyle="--", linewidth=1.2, alpha=0.7)
    ax_price.legend(handles=_legend_handles(), loc="upper left", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def save_label_plots(df: pd.DataFrame, labels: pd.Series, out_dir: Path, cfg: RunConfig) -> None:
    smooth = denoise_labels(labels, window=168)
    runs = _compute_runs(df, smooth)
    suffix = f"labels-{cfg.target}-{cfg.timeframe}"
    _plot_A(df, runs, out_dir / "A_labels_background.png", suffix)
    # Equity uses raw labels (matches metric); regime bands use smoothed.
    _plot_B(df, labels, smooth, runs, out_dir / "B_labels_synth.png", suffix)


def save_prediction_plots(
    df: pd.DataFrame, predictions: pd.Series, out_dir: Path, cfg: RunConfig,
    predictor_name: str, split_dt=None,
) -> None:
    smooth = denoise_labels(predictions, window=168)
    runs = _compute_runs(df, smooth)
    suffix = f"pred-{predictor_name}-{cfg.target}-{cfg.timeframe}"
    _plot_A(df, runs, out_dir / "A_predictions_background.png", suffix, split_dt)
    # Equity uses raw predictions (matches synth_gain console metric);
    # regime bands keep the 7-day smoothing for visual cleanliness.
    _plot_B(df, predictions, smooth, runs, out_dir / "B_predictions_synth.png", suffix, split_dt)
