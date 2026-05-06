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


def _detect_label_span(labels) -> tuple[int, int] | None:
    """Find first and last bar indices where a label is non-empty.

    Used by plots to auto-zoom on the prediction window: if predictions
    cover only the test slice (typical for prediction plots), the rest of
    the timeline shows empty labels and gets cropped out. For label plots
    (full-history labels), every bar is non-empty so this returns the full
    range — effectively a no-op zoom.
    """
    if labels is None:
        return None
    arr = np.asarray(labels)
    if arr.size == 0:
        return None
    mask = arr != ""
    if not mask.any():
        return None
    nz = np.where(mask)[0]
    return int(nz[0]), int(nz[-1])


def _maybe_zoom(ax, dates, labels, padding_bars: int = 0) -> None:
    """If `labels` covers a strict subset of `dates`, restrict ax xlim to it.
    Use `padding_bars` to add a small left buffer (e.g. show some training
    context before the test split).
    """
    span = _detect_label_span(labels)
    if span is None:
        return
    first, last = span
    if first == 0 and last == len(dates) - 1:
        return  # full range — no zoom needed
    lo = max(0, first - padding_bars)
    hi = min(len(dates) - 1, last)
    ax.set_xlim(dates[lo], dates[hi])


def _apply_xlim(ax, xlim_dates: tuple | None) -> bool:
    """If xlim_dates=(start, end) is provided, set ax xlim and return True.
    Used by per-fold plots to override auto-zoom and show the broader OOS
    timeline (first fold start → last fold end) instead of just one fold.
    """
    if xlim_dates is None:
        return False
    start, end = xlim_dates
    ax.set_xlim(start, end)
    return True


def _rebase_equity(
    synth_eq: np.ndarray,
    closes: np.ndarray,
    dates: np.ndarray,
    anchor_dt,
) -> np.ndarray:
    """Rescale equity so equity[anchor_pos] == closes[anchor_pos].

    Without rebasing, synth_equity_curve always starts at closes[0] (the
    earliest price in the data — could be 2019). Per-fold plots show a
    much later window (e.g. 2024+) where the displayed price level is
    very different from the cumulative equity level. Rebasing aligns the
    curve with the price at the fold start so the equity reads naturally
    against the price line.
    """
    if anchor_dt is None or len(synth_eq) == 0:
        return synth_eq
    pos = int(np.searchsorted(dates, np.datetime64(pd.to_datetime(anchor_dt))))
    pos = max(0, min(pos, len(synth_eq) - 1))
    if synth_eq[pos] == 0 or np.isnan(synth_eq[pos]):
        return synth_eq
    return synth_eq * (float(closes[pos]) / float(synth_eq[pos]))


def _plot_A(df, runs, out_path, title_suffix: str, split_dt=None, labels=None,
            xlim_dates: tuple | None = None) -> None:
    fig, (ax_price, ax_regime) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]},
    )
    dates = pd.to_datetime(df["date"].values)
    closes = df["close"].values
    # X-axis priority: explicit xlim_dates > auto-zoom from labels > full range.
    # Per-fold plots pass xlim_dates = full OOS span (first fold start, last
    # fold end) so each fold's plot shows the broader CV context.
    if not _apply_xlim(ax_price, xlim_dates) and labels is not None:
        _maybe_zoom(ax_price, dates, labels)
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
    xlim_dates: tuple | None = None,
    transaction_cost: float = 0.0,
    long_only: bool = False,
) -> None:
    """raw_labels drive the equity curve (matches the synth_gain metric);
    smooth drives the colored regime bands (visual only). Earlier code used
    smooth for both, which made the plot diverge from the console gain metric
    by up to a 7-day lag (denoise window).
    """
    from ..evaluation.metrics import synth_equity_curve
    dates = pd.to_datetime(df["date"].values)
    closes = df["close"].values
    synth_eq, _ = synth_equity_curve(
        closes, raw_labels.values,
        transaction_cost=transaction_cost, long_only=long_only,
    )
    # Rebase equity at the fold start (split_dt) — or at the visible window
    # start if no split_dt — so the curve reads naturally against the price
    # line on screen (instead of starting at the dataset's 2019 price level).
    anchor = split_dt if split_dt is not None else (xlim_dates[0] if xlim_dates else None)
    synth_eq = _rebase_equity(synth_eq, closes, dates.values, anchor)
    # Mask the equity to NaN outside the prediction window so flat segments
    # (where the strategy has no position) are not drawn — cleaner Y-axis
    # cadrage when xlim_dates is wider than the prediction span.
    raw_arr = raw_labels.values if hasattr(raw_labels, "values") else np.asarray(raw_labels)
    pred_mask = raw_arr != ""
    if pred_mask.any() and not pred_mask.all():
        synth_eq_plot = synth_eq.astype(float).copy()
        synth_eq_plot[~pred_mask] = np.nan
    else:
        synth_eq_plot = synth_eq

    fig, ax = plt.subplots(figsize=(14, 6))
    for lab, s_dt, e_dt in runs:
        ax.axvspan(s_dt, e_dt, color=LABEL_COLORS.get(lab, "gray"), alpha=0.10, lw=0)
    ax.plot(dates, closes, color="black", linewidth=0.7, label="close (actual)")
    ax.plot(dates, synth_eq_plot, color="#1f77b4", linewidth=1.5, label="synth equity (long bull / short bear)")
    # X-axis priority: explicit xlim_dates > auto-zoom from raw_labels.
    if not _apply_xlim(ax, xlim_dates):
        _maybe_zoom(ax, dates, raw_labels.values if hasattr(raw_labels, "values") else raw_labels)
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
    folds_per_predictor: dict[str, list[dict]],
    out_path: Path,
    title_suffix: str,
    denoise_window: int = 168,
    fold_test_indices: list | None = None,
    transaction_cost: float = 0.0,
    long_only: bool = False,
) -> None:
    """Stitched OOS synth equity across CV folds — top-N predictors overlaid.

    folds_per_predictor: dict mapping predictor_name → list of fold dicts.
        Each fold dict has:
          - test_index: pd.Index of bars in this fold's test set
          - predictions: np.ndarray of label strings aligned with test_index
        All predictors must share the same fold structure (same test_index
        values across predictors), but their predictions can differ.

    fold_test_indices: optional list of all fold test_index objects in order
        (one per fold) — used to draw fold boundary lines on the plot.
        If None, derived from the first predictor's folds.
    """
    from ..evaluation.metrics import synth_equity_curve
    closes = df["close"].values.astype(float)
    dates = pd.to_datetime(df["date"].values)

    if not folds_per_predictor:
        return

    # Determine the OOS span — first bar of fold 0 to last bar of last fold.
    # Use the first predictor's folds; all should agree on fold structure.
    first_preds_folds = next(iter(folds_per_predictor.values()))
    oos_first = min((f["test_index"][0] for f in first_preds_folds if len(f["test_index"])), default=None)
    oos_last = max((f["test_index"][-1] for f in first_preds_folds if len(f["test_index"])), default=None)
    if oos_first is not None and oos_last is not None:
        oos_slice = (df.index >= oos_first) & (df.index <= oos_last)
        first_pos = int(np.where(oos_slice)[0][0])
    else:
        oos_slice = np.ones(len(df), dtype=bool)
        first_pos = 0

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(dates[oos_slice], closes[oos_slice], color="black", linewidth=0.7,
            alpha=0.4, label="close (actual)")

    # One line per predictor — distinct colors via tab10. Legend shows each
    # predictor's total gain over the stitched OOS span.
    cmap = plt.get_cmap("tab10")
    # Anchor date: start of the OOS span — equity curves all start at the
    # actual OOS-start price so they're directly comparable to each other
    # and to the price line.
    anchor_dt = dates[first_pos] if oos_first is not None else None
    for i, (pred_name, pred_folds) in enumerate(folds_per_predictor.items()):
        out = pd.Series("", index=df.index, dtype=object)
        for fold in pred_folds:
            idx = fold["test_index"]
            preds = fold["predictions"]
            if len(preds) == len(idx):
                out.loc[idx] = preds
        synth_eq, _ = synth_equity_curve(
            closes, out.values, transaction_cost=transaction_cost, long_only=long_only,
        )
        synth_eq = _rebase_equity(synth_eq, closes, dates.values, anchor_dt)
        # Total gain over the stitched OOS span (last visible / first visible − 1).
        last_visible_pos = int(np.where(oos_slice)[0][-1])
        gain = (synth_eq[last_visible_pos] / synth_eq[first_pos]) - 1.0 if synth_eq[first_pos] else float("nan")
        gain_str = f"{gain*100:+.1f}%" if not np.isnan(gain) else "n/a"
        ax.plot(
            dates[oos_slice], synth_eq[oos_slice],
            color=cmap(i % 10), linewidth=1.5, alpha=0.9,
            label=f"{pred_name} ({gain_str})",
        )

    # Fold boundary lines (no per-fold annotation since multiple predictors).
    boundary_indices = fold_test_indices if fold_test_indices is not None else [
        f["test_index"] for f in first_preds_folds
    ]
    for idx in boundary_indices:
        if len(idx) == 0:
            continue
        start_dt = pd.to_datetime(df.loc[idx[0], "date"])
        ax.axvline(start_dt, color="gray", linestyle=":", linewidth=0.6, alpha=0.4)

    ax.set_yscale("log")
    ax.set_ylabel("price / equity (log)")
    n_folds = len(boundary_indices)
    n_preds = len(folds_per_predictor)
    ax.set_title(
        f"(B-stitched) [{title_suffix}] OOS synth equity — top {n_preds} predictors over {n_folds} folds"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
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
    xlim_dates: tuple | None = None,
    transaction_cost: float = 0.0,
    long_only: bool = False,
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
    # Track the union span across all predictors for the auto-zoom.
    sample_preds = None
    # Rebase anchor: split_dt if provided (per-fold plots), else xlim start.
    anchor_dt = split_dt if split_dt is not None else (xlim_dates[0] if xlim_dates else None)
    for i, (name, preds) in enumerate(items):
        try:
            # Use raw predictions (no smoothing) so the equity curves match
            # the synth_gain console metric. Earlier 168-bar smoothing made
            # the multi plot diverge from the per-fold gain numbers.
            preds_arr = preds if isinstance(preds, np.ndarray) else np.asarray(preds)
            if sample_preds is None:
                sample_preds = preds_arr
            synth_eq, _ = synth_equity_curve(
                closes, preds_arr, transaction_cost=transaction_cost, long_only=long_only,
            )
            synth_eq = _rebase_equity(synth_eq, closes, dates.values, anchor_dt)
            # Mask flat segments outside the prediction window for clean Y-axis.
            pred_mask = preds_arr != ""
            if pred_mask.any() and not pred_mask.all():
                synth_eq_plot = synth_eq.astype(float).copy()
                synth_eq_plot[~pred_mask] = np.nan
            else:
                synth_eq_plot = synth_eq
            ax.plot(dates, synth_eq_plot, color=cmap(i % 10), linewidth=1.2,
                    alpha=0.9, label=name)
        except Exception:
            continue
    # X-axis priority: explicit xlim_dates > auto-zoom from any predictor.
    if not _apply_xlim(ax, xlim_dates) and sample_preds is not None:
        _maybe_zoom(ax, dates, sample_preds)

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
    xlim_dates: tuple | None = None,
) -> None:
    """Plot price (top) + one row per predictor showing its bull/bear track (bottom).

    xlim_dates: optional (start, end) — clamps both the price axis and the
    panel axis to this window. If None, falls back to auto-zoom on the
    union of predictor spans.
    """
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
    sample_preds = None
    for i, (name, preds) in enumerate(items):
        try:
            if sample_preds is None:
                sample_preds = preds.values if hasattr(preds, "values") else np.asarray(preds)
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
    # X-axis: explicit xlim_dates > auto-zoom from any predictor.
    if not _apply_xlim(ax_price, xlim_dates) and sample_preds is not None:
        _maybe_zoom(ax_price, dates, sample_preds)
    ax_price.legend(handles=_legend_handles(), loc="upper left", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def save_label_plots(
    df: pd.DataFrame, labels: pd.Series, out_dir: Path, cfg: RunConfig,
    xlim_dates: tuple | None = None,
) -> None:
    """Save label A/B plots. xlim_dates: optional (start, end) — clamps the
    X-axis to a specific window. Used to align label plots' OOS window with
    the prediction plots (so B_labels_synth.png shows the 'perfect-regime
    trader' equity over exactly the same period as the predictor plots).
    """
    smooth = denoise_labels(labels, window=168)
    runs = _compute_runs(df, smooth)
    suffix = f"labels-{cfg.target}-{cfg.timeframe}"
    _plot_A(df, runs, out_dir / "A_labels_background.png", suffix,
            xlim_dates=xlim_dates)
    # Equity uses raw labels (matches metric); regime bands use smoothed.
    _plot_B(df, labels, smooth, runs, out_dir / "B_labels_synth.png", suffix,
            xlim_dates=xlim_dates)


def save_prediction_plots(
    df: pd.DataFrame, predictions: pd.Series, out_dir: Path, cfg: RunConfig,
    predictor_name: str, split_dt=None,
) -> None:
    smooth = denoise_labels(predictions, window=168)
    runs = _compute_runs(df, smooth)
    suffix = f"pred-{predictor_name}-{cfg.target}-{cfg.timeframe}"
    # Pass labels=predictions so _plot_A zooms to the prediction window.
    _plot_A(df, runs, out_dir / "A_predictions_background.png", suffix, split_dt,
            labels=predictions)
    # Equity uses raw predictions (matches synth_gain console metric);
    # regime bands keep the 7-day smoothing for visual cleanliness.
    # _plot_B auto-zooms via raw_labels (predictions has empty bars in train).
    _plot_B(df, predictions, smooth, runs, out_dir / "B_predictions_synth.png", suffix, split_dt)
