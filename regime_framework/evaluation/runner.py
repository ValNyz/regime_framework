"""End-to-end benchmark runner.

Loads data → labels → features → split → loops over all configured predictors
→ collects metrics → produces a consolidated table + plots.
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from ..config import RunConfig, RESULTS_DIR, PLOTS_DIR, ExtraCoinSpec
from ..data.loaders import load_ohlcv
from ..data.conventions import DataRoot
from ..labels import get_labeller
from ..features.pipeline import FeaturePipeline
from ..predictors.base import BasePredictor, PredictionResult
from ..predictors.classical import (
    LogRegPredictor, RandomForestPredictor, ExtraTreesPredictor,
    MLPPredictor, XGBoostPredictor, LightGBMPredictor,
)
from ..predictors.rule_based import RegimeV3Predictor, RegimeV4EmaPredictor
from ..predictors.deep_nets import GRUPredictor, LSTMPredictor
from ..predictors.transformer import TimeSeriesTransformerPredictor
from ..predictors.pretrained import PRETRAINED_REGISTRY
from ..signal_analysis import rank_signals
from ..visualization.regime_plots import save_label_plots, save_prediction_plots
from .metrics import evaluate
from .splits import time_aware_split, walk_forward_splits, leave_one_out_splits


console = Console()


def _build_predictors(cfg: RunConfig) -> list[BasePredictor]:
    """Instantiate all predictors enabled in the config."""
    out: list[BasePredictor] = []
    families = set(cfg.predictors.families)

    if "classical" in families:
        out += [
            LogRegPredictor(),
            RandomForestPredictor(),
            ExtraTreesPredictor(),  # randomized splits, RF cousin
            MLPPredictor(),         # torch GPU, BN+GELU+Dropout, ~84k params
            XGBoostPredictor(),
            LightGBMPredictor(),    # leaf-wise GBDT, ~2-3x faster than XGB
        ]
    if "rule_based" in families:
        out += [RegimeV3Predictor(), RegimeV4EmaPredictor()]
    if "deep_nets" in families:
        out += [GRUPredictor(), LSTMPredictor()]
    if "transformer" in families:
        out += [TimeSeriesTransformerPredictor()]
    if "pretrained" in families:
        for model_name in cfg.predictors.pretrained_models:
            cls = PRETRAINED_REGISTRY.get(model_name)
            if cls is None:
                console.print(f"[yellow]WARN: unknown pretrained model: {model_name}[/yellow]")
                continue
            for mode in cfg.predictors.pretrained_modes:
                # Skip fine_tuned for models that don't support embeddings
                if mode == "fine_tuned" and not getattr(cls, "_supports_embedding", False):
                    continue
                p = cls(mode=mode, horizon=cfg.predictors.forecast_horizon)
                p.name = f"{cls.name}-{mode}"
                out.append(p)
    return out


class BenchmarkRunner:
    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        self.results: list[PredictionResult] = []
        self.label_distribution: dict[str, int] = {}
        self.feature_count: int = 0
        self.df: pd.DataFrame | None = None
        self.labels: pd.Series | None = None
        self.X: pd.DataFrame | None = None
        self.y: pd.Series | None = None
        self.dates: pd.Series | None = None

    def run(self) -> dict:
        cfg = self.cfg
        RESULTS_DIR.mkdir(exist_ok=True, parents=True)
        PLOTS_DIR.mkdir(exist_ok=True, parents=True)

        console.rule(f"[bold cyan]regime_framework — target={cfg.target} venue={cfg.venue} TF={cfg.timeframe}")

        # ----- 1. Data -----
        console.print(f"[bold]Loading OHLCV[/bold] from {cfg.paths.ohlcv}")
        df = load_ohlcv(cfg.paths.ohlcv)
        console.print(f"  {len(df)} bars  {df['date'].iloc[0]} → {df['date'].iloc[-1]}")
        self.df = df

        # ----- 2. Labels -----
        labeller = get_labeller(
            cfg.label.method,
            L_range=cfg.label.L_range,
            t_threshold=cfg.label.t_threshold,
            hysteresis_bars=cfg.label.hysteresis_bars,
            strong_threshold=cfg.label.strong_threshold,
            horizon=cfg.label.horizon,
            alpha=cfg.label.alpha,
        )
        console.print(f"[bold]Labelling[/bold] ({labeller.name})")
        labels = labeller.compute(df)
        counts = labels[labels != ""].value_counts()
        self.label_distribution = counts.to_dict()
        for k, v in counts.items():
            console.print(f"  {k:8s} {v:>8d}  ({100*v/counts.sum():.1f}%)")
        self.labels = labels

        # ----- 3. Save label plots immediately -----
        console.print("[bold]Saving label plots[/bold]")
        save_label_plots(df, labels, PLOTS_DIR, cfg)

        # ----- 4. Features -----
        console.print("[bold]Computing features[/bold]")
        pipeline = FeaturePipeline(
            use_technical=cfg.features.use_technical,
            use_external=cfg.features.use_external,
            use_trading_signals=cfg.features.use_trading_signals,
            trading_signals_yaml=cfg.features.trading_signals_yaml,
            target_funding_path=cfg.paths.funding,
            cross_ohlcv_path=cfg.paths.cross_ohlcv,
            cross_name=cfg.paths.cross_name,
            external_dir=cfg.paths.external_dir,
            drop_nan_rows=cfg.features.drop_nan_rows,
        )
        X, y, dates = pipeline.build(df, labels)
        self.X, self.y, self.dates = X, y, dates
        self.feature_count = X.shape[1]
        console.print(f"  Final dataset: {len(X)} rows × {X.shape[1]} cols")

        # ----- 5. Split: single or cross-validation -----
        purge = cfg.purge_bars
        # Backward-compat: walk_forward_folds → cv_folds with mode walk_forward
        if cfg.split.walk_forward_folds and not cfg.split.cv_folds:
            cfg.split.cv_folds = cfg.split.walk_forward_folds
            cfg.split.cv_mode = "walk_forward"

        if cfg.split.cv_folds and cfg.split.cv_folds > 0:
            return self._run_cv(cfg, df, X, y, dates, purge)
        else:
            return self._run_single_split(cfg, df, labels, X, y, dates, purge)

    def _run_single_split(self, cfg, df, labels, X, y, dates, purge: int) -> dict:
        X_tr, y_tr, d_tr, X_te, y_te, d_te = time_aware_split(
            X, y, dates,
            train_fraction=cfg.split.train_fraction,
            purge_bars=purge,
        )
        df_tr = df.loc[X_tr.index]
        df_te = df.loc[X_te.index]

        # Multi-coin: replace train with stacked target+extras (test stays target-only)
        if cfg.training.extra_coins:
            X_tr_stacked, y_tr_stacked, d_tr_stacked = self._stack_with_target(
                cfg, X_tr, y_tr, d_tr,
            )
            # Align test set columns with train (test must have same coin_id one-hot cols)
            X_te = X_te.copy()
            for col in X_tr_stacked.columns:
                if col.startswith("coin_") and col not in X_te.columns:
                    X_te[col] = 1.0 if col == f"coin_{cfg.target}" else 0.0
            X_te = X_te[X_tr_stacked.columns]   # exact match
            X_tr, y_tr, d_tr = X_tr_stacked, y_tr_stacked, d_tr_stacked
            df_tr = df.iloc[:0]   # not used by ML predictors; rule-based use df_te
        console.print(
            f"  Train: {len(X_tr):>6} bars ({d_tr.iloc[0].date()} → {d_tr.iloc[-1].date()})"
        )
        console.print(
            f"  Test : {len(X_te):>6} bars ({d_te.iloc[0].date()} → {d_te.iloc[-1].date()})"
        )
        console.print(f"  Purge gap: {purge} bars")

        console.print("[bold]Signal analysis (lift + MI on train)[/bold]")
        try:
            ranking = rank_signals(X_tr, y_tr)
            top10 = ranking.head(10)
            self._print_signal_ranking(top10)
            ranking.to_csv(RESULTS_DIR / "signal_ranking.csv", index=False)
        except Exception as e:
            console.print(f"[yellow]signal_analysis skipped: {e}[/yellow]")

        predictors = _build_predictors(cfg)
        console.print(f"[bold]Running {len(predictors)} predictors[/bold]")
        baseline_acc = float((y_te.values == y_tr.value_counts().idxmax()).mean())
        console.print(f"  Baseline (always-predict majority): acc={baseline_acc:.3f}")

        results, per_predictor_predictions = self._fit_eval_loop(
            predictors, X_tr, y_tr, d_tr, df_tr, X_te, y_te, d_te, df_te
        )
        self.results = results
        self._print_summary(baseline_acc)
        self._save_summary_csv()

        if results:
            best = max(results, key=lambda r: r.kappa if not np.isnan(r.kappa) else -2)
            best_pred = self._build_full_pred_series(
                df, labels, X.index, X_tr.index, X_te.index,
                per_predictor_predictions[best.name],
            )
            console.print(
                f"[bold]Best predictor: {best.name}[/bold] "
                f"(κ={best.kappa:+.3f}, acc={best.accuracy:.3f}) — saving prediction plots"
            )
            save_prediction_plots(
                df, best_pred, PLOTS_DIR, cfg,
                predictor_name=best.name,
                split_dt=d_te.iloc[0],
            )
            # Multi-classifier overlay (single-split)
            if len(per_predictor_predictions) > 1:
                self._save_fold_plot_multi(
                    cfg, df, "single", 0, per_predictor_predictions, X_te, d_te,
                )
            # Feature importance from the best predictor
            best_obj = next((p for p in predictors if p.name == best.name), None)
            if best_obj is not None:
                self._print_and_save_importance(best_obj, X_te, y_te)

        return {
            "results": [r.__dict__ for r in results],
            "label_distribution": self.label_distribution,
            "feature_count": self.feature_count,
            "baseline_acc": baseline_acc,
        }
        # ----- end of _run_single_split -----

    # -------------------------------------------------------------------
    # Helpers shared by single-split and walk-forward
    # -------------------------------------------------------------------
    def _fit_eval_loop(
        self,
        predictors: list[BasePredictor],
        X_tr, y_tr, d_tr, df_tr,
        X_te, y_te, d_te, df_te,
    ) -> tuple[list[PredictionResult], dict[str, np.ndarray]]:
        results: list[PredictionResult] = []
        per_predictor_predictions: dict[str, np.ndarray] = {}
        for predictor in predictors:
            with console.status(f"[bold green]{predictor.name}[/bold green]"):
                t0 = time.time()
                try:
                    predictor.fit(X_tr, y_tr, d_tr, df_tr)
                    pred = predictor.predict(X_te, d_te, df_te)
                    pred_arr = np.asarray(pred)
                    res = evaluate(
                        name=predictor.name,
                        family=predictor.family,
                        y_true=np.asarray(y_te.values),
                        y_pred=pred_arr,
                        metadata={"elapsed_sec": round(time.time() - t0, 2)},
                    )
                    results.append(res)
                    per_predictor_predictions[predictor.name] = pred_arr
                    console.print(
                        f"  [green]✔[/green] {predictor.name:35s} "
                        f"acc={res.accuracy:.3f} κ={res.kappa:+.3f} F1={res.f1_macro:.3f} "
                        f"({res.metadata['elapsed_sec']}s)"
                    )
                except Exception as e:
                    console.print(f"  [red]✘[/red] {predictor.name}: {e}")
                    traceback.print_exc()
        return results, per_predictor_predictions

    def _run_cv(self, cfg, df, X, y, dates, purge: int) -> dict:
        n_folds = int(cfg.split.cv_folds)
        modes = ["walk_forward", "leave_one_out"] if cfg.split.cv_mode == "both" else [cfg.split.cv_mode]

        all_aggr: dict[str, pd.DataFrame] = {}
        for mode in modes:
            per_fold_df = self._run_cv_single_mode(cfg, df, X, y, dates, purge, mode, n_folds)
            if per_fold_df is not None and not per_fold_df.empty:
                all_aggr[mode] = per_fold_df

        # If two modes ran, print a side-by-side comparison
        if len(all_aggr) == 2:
            self._print_cv_comparison(all_aggr)

        return {"modes": list(all_aggr.keys()), "n_folds": n_folds}

    def _run_cv_single_mode(self, cfg, df, X, y, dates, purge: int, mode: str, n_folds: int) -> pd.DataFrame | None:
        if mode == "leave_one_out":
            console.print(
                f"\n[bold]Leave-one-out CV[/bold] — {n_folds} folds, purge={purge} bars on each side"
            )
            split_iter = leave_one_out_splits(n=len(X), n_folds=n_folds, purge_bars=purge)
        else:
            console.print(
                f"\n[bold]Walk-forward CV[/bold] — {n_folds} folds, "
                f"min_train={cfg.split.min_train_fraction:.0%}, purge={purge} bars"
            )
            split_iter = walk_forward_splits(
                n=len(X),
                n_folds=n_folds,
                purge_bars=purge,
                min_train_fraction=cfg.split.min_train_fraction,
            )

        per_fold_rows: list[dict] = []
        # Keep ALL predictors' predictions per fold (used for the stitched plot
        # at the end, where we pick the predictor with the best MEAN kappa across folds).
        all_fold_preds: list[dict] = []

        for train_idx, test_idx, fold_id in split_iter:
            X_tr = X.iloc[train_idx]
            y_tr = y.iloc[train_idx]
            d_tr = dates.iloc[train_idx]
            df_tr = df.loc[X_tr.index]
            X_te = X.iloc[test_idx]
            y_te = y.iloc[test_idx]
            d_te = dates.iloc[test_idx]
            df_te = df.loc[X_te.index]

            # Multi-coin: stack target+extras into the train fold
            if cfg.training.extra_coins:
                X_tr_stacked, y_tr_stacked, d_tr_stacked = self._stack_with_target(
                    cfg, X_tr, y_tr, d_tr,
                )
                # Filter extra-coin training data to bars BEFORE the fold's test start
                # to prevent leakage (extra coins' future shouldn't be used to train fold k)
                cutoff = d_te.iloc[0]
                keep = d_tr_stacked < cutoff
                X_tr_stacked = X_tr_stacked.loc[keep].reset_index(drop=True)
                y_tr_stacked = y_tr_stacked.loc[keep].reset_index(drop=True)
                d_tr_stacked = d_tr_stacked.loc[keep].reset_index(drop=True)
                # Align test columns
                X_te = X_te.copy()
                for col in X_tr_stacked.columns:
                    if col.startswith("coin_") and col not in X_te.columns:
                        X_te[col] = 1.0 if col == f"coin_{cfg.target}" else 0.0
                X_te = X_te[X_tr_stacked.columns]
                X_tr, y_tr, d_tr = X_tr_stacked, y_tr_stacked, d_tr_stacked
                df_tr = df.iloc[:0]

            console.rule(
                f"[bold cyan]Fold {fold_id+1}/{n_folds} ({mode}): "
                f"train={len(X_tr)} | "
                f"test={len(X_te)} ({d_te.iloc[0].date()}→{d_te.iloc[-1].date()})"
            )

            baseline_acc = float((y_te.values == y_tr.value_counts().idxmax()).mean())
            predictors = _build_predictors(cfg)
            if not predictors:
                console.print(
                    f"[red]No predictors built! Configured families: {cfg.predictors.families}. "
                    f"Valid: classical, rule_based, deep_nets, transformer, pretrained.[/red]"
                )
                return None
            fold_results, fold_predictions = self._fit_eval_loop(
                predictors, X_tr, y_tr, d_tr, df_tr, X_te, y_te, d_te, df_te
            )

            for r in fold_results:
                per_fold_rows.append({
                    "fold": fold_id, "predictor": r.name, "family": r.family,
                    "accuracy": r.accuracy, "kappa": r.kappa, "f1_macro": r.f1_macro,
                    "n_test": r.n_test, "elapsed_sec": r.metadata.get("elapsed_sec", 0),
                    "test_start": str(d_te.iloc[0].date()),
                    "test_end": str(d_te.iloc[-1].date()),
                    "baseline_acc": baseline_acc,
                })

            # Save prediction plots for this fold's best predictor + multi overlay
            if fold_results:
                best_fold = max(
                    fold_results, key=lambda r: r.kappa if not np.isnan(r.kappa) else -2
                )
                if best_fold.name in fold_predictions:
                    self._save_fold_plot(
                        cfg, df, mode, fold_id, best_fold,
                        X_te, d_te, fold_predictions[best_fold.name],
                    )
                # Keep ALL predictors' predictions for this fold — used to
                # build the stitched plot using the GLOBAL best predictor.
                all_fold_preds.append({
                    "fold_id": fold_id,
                    "test_index": X_te.index,
                    "predictions": dict(fold_predictions),  # name -> ndarray
                })
                if len(fold_predictions) > 1:
                    self._save_fold_plot_multi(
                        cfg, df, mode, fold_id, fold_predictions,
                        X_te, d_te,
                    )

        if not per_fold_rows:
            console.print("[red]No folds produced — check min_train_fraction / data length[/red]")
            return None

        per_fold_df = pd.DataFrame(per_fold_rows)
        per_fold_df["cv_mode"] = mode
        per_fold_df.to_csv(RESULTS_DIR / f"cv_{mode}_per_fold.csv", index=False)
        self._print_cv_summary(per_fold_df, mode)

        # Stitched OOS synth equity: pick the predictor with the best MEAN
        # kappa across all folds, then concatenate ITS predictions over all
        # fold test windows. Realistic: a single model deployed across the
        # full OOS timeline (vs the per-fold-best which would require an
        # oracle to know which model to use when).
        if len(all_fold_preds) >= 2 and not per_fold_df.empty:
            mean_kappa = per_fold_df.groupby("predictor")["kappa"].mean()
            best_name = str(mean_kappa.idxmax())
            best_mean = float(mean_kappa[best_name])
            stitched_folds = []
            for fp in all_fold_preds:
                if best_name not in fp["predictions"]:
                    continue
                fold_kappa_row = per_fold_df[
                    (per_fold_df["fold"] == fp["fold_id"])
                    & (per_fold_df["predictor"] == best_name)
                ]
                fold_kappa = float(fold_kappa_row["kappa"].iloc[0]) if not fold_kappa_row.empty else float("nan")
                stitched_folds.append({
                    "fold_id": fp["fold_id"],
                    "test_index": fp["test_index"],
                    "predictions": fp["predictions"][best_name],
                    "predictor_name": best_name,
                    "kappa": fold_kappa,
                })
            try:
                from ..visualization.regime_plots import plot_stitched_oos_equity
                plot_stitched_oos_equity(
                    df, stitched_folds,
                    PLOTS_DIR / f"B_stitched_oos_{mode}.png",
                    title_suffix=f"{cfg.target}-{cfg.timeframe}-{mode}-{best_name}-meanK{best_mean:+.3f}",
                )
                console.print(
                    f"[dim]Stitched OOS plot: B_stitched_oos_{mode}.png "
                    f"(best by mean κ: {best_name}, mean_κ={best_mean:+.3f}, {len(stitched_folds)} folds)[/dim]"
                )
            except Exception as e:
                console.print(f"[yellow]Stitched plot failed: {e}[/yellow]")

        return per_fold_df

    def _print_cv_comparison(self, all_aggr: dict[str, pd.DataFrame]) -> None:
        # Build per-mode aggregated views
        wf = all_aggr.get("walk_forward")
        lo = all_aggr.get("leave_one_out")
        if wf is None or lo is None:
            return

        def _agg(d: pd.DataFrame) -> pd.DataFrame:
            return d.groupby(["predictor", "family"])["kappa"].agg(["mean", "std"]).reset_index()

        wf_a = _agg(wf).rename(columns={"mean": "wf_kappa_mean", "std": "wf_kappa_std"})
        lo_a = _agg(lo).rename(columns={"mean": "lo_kappa_mean", "std": "lo_kappa_std"})
        merged = wf_a.merge(lo_a, on=["predictor", "family"], how="outer")
        merged["delta"] = merged["lo_kappa_mean"] - merged["wf_kappa_mean"]
        merged = merged.sort_values("wf_kappa_mean", ascending=False)
        merged.to_csv(RESULTS_DIR / "cv_comparison.csv", index=False)

        table = Table(title="CV mode comparison — κ_mean ± std (walk-forward vs leave-one-out)")
        table.add_column("predictor")
        table.add_column("family")
        table.add_column("walk-forward κ", justify="right")
        table.add_column("leave-one-out κ", justify="right")
        table.add_column("Δ (loo - wf)", justify="right")
        for _, r in merged.iterrows():
            wfm = r.get("wf_kappa_mean", float("nan"))
            wfs = r.get("wf_kappa_std", float("nan"))
            lom = r.get("lo_kappa_mean", float("nan"))
            los = r.get("lo_kappa_std", float("nan"))
            delta = r["delta"] if pd.notna(r["delta"]) else float("nan")
            color = "green" if pd.notna(delta) and delta > 0 else "red" if pd.notna(delta) and delta < 0 else ""
            d_str = f"[{color}]{delta:+.3f}[/{color}]" if color else f"{delta:+.3f}"
            table.add_row(
                str(r["predictor"]), str(r["family"]),
                f"{wfm:+.3f} ±{wfs:.3f}" if pd.notna(wfm) else "n/a",
                f"{lom:+.3f} ±{los:.3f}" if pd.notna(lom) else "n/a",
                d_str,
            )
        console.print(table)
        console.print(
            "[dim]Δ > 0 means leave-one-out is easier than walk-forward "
            "(future-info advantage). Large Δ = predictor relies on regime stability.[/dim]"
        )

    def _print_cv_summary(self, per_fold: pd.DataFrame, mode: str) -> None:
        agg = per_fold.groupby(["predictor", "family"]).agg(
            kappa_mean=("kappa", "mean"),
            kappa_std=("kappa", "std"),
            kappa_min=("kappa", "min"),
            kappa_max=("kappa", "max"),
            acc_mean=("accuracy", "mean"),
            f1_mean=("f1_macro", "mean"),
            n_folds=("fold", "count"),
        ).reset_index().sort_values("kappa_mean", ascending=False)

        agg["cv_mode"] = mode
        agg.to_csv(RESULTS_DIR / f"cv_{mode}_aggregated.csv", index=False)

        title = f"{mode.replace('_', '-')} aggregate — sorted by mean κ"
        table = Table(title=title)
        for col in ("predictor", "family", "kappa_mean", "kappa_std", "kappa_min", "kappa_max", "acc_mean", "f1_mean", "n_folds"):
            table.add_column(col, justify="right" if col not in ("predictor", "family") else "left")
        for _, r in agg.iterrows():
            color = ""
            km = float(r["kappa_mean"])
            if km > 0.4:
                color = "green"
            elif km > 0.2:
                color = "yellow"
            elif km < 0:
                color = "red"
            kstr = f"[{color}]{km:+.3f}[/{color}]" if color else f"{km:+.3f}"
            table.add_row(
                str(r["predictor"]), str(r["family"]),
                kstr, f"{r['kappa_std']:.3f}", f"{r['kappa_min']:+.3f}", f"{r['kappa_max']:+.3f}",
                f"{r['acc_mean']:.3f}", f"{r['f1_mean']:.3f}", str(int(r["n_folds"])),
            )
        console.print(table)

    # -------------------------------------------------------------------
    @staticmethod
    def _build_full_pred_series(
        df: pd.DataFrame, labels: pd.Series,
        used_index: pd.Index, train_index: pd.Index, test_index: pd.Index,
        test_predictions: np.ndarray,
    ) -> pd.Series:
        """Build a label series spanning df: train labels (true), test predictions."""
        out = pd.Series("", index=df.index, dtype=object)
        # Train: use true labels
        out.loc[train_index] = labels.loc[train_index]
        # Test: use predictions (aligned by position to test_index)
        if len(test_predictions) == len(test_index):
            out.loc[test_index] = test_predictions
        return out

    def _print_signal_ranking(self, top10: pd.DataFrame) -> None:
        if top10.empty:
            return
        table = Table(title="Top 10 features by combined predictive power", show_lines=False)
        col_order = ["signal", "feature_kind", "n_triggers", "lift_bull", "lift_bear", "mi", "combined_score"]
        for col in col_order:
            if col in top10.columns:
                table.add_column(col)
        def _nz_int(v) -> int:
            try:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return 0
                return int(v)
            except (TypeError, ValueError):
                return 0

        def _nz_float(v) -> float:
            try:
                f = float(v)
                return f if np.isfinite(f) else float("nan")
            except (TypeError, ValueError):
                return float("nan")

        for _, row in top10.iterrows():
            cells = [str(row["signal"])]
            if "feature_kind" in top10.columns:
                cells.append(str(row.get("feature_kind", "")))
            cells.extend([
                f"{_nz_int(row.get('n_triggers'))}",
                f"{_nz_float(row.get('lift_bull')):.2f}",
                f"{_nz_float(row.get('lift_bear')):.2f}",
                f"{_nz_float(row.get('mi')):.4f}",
                f"{_nz_float(row.get('combined_score')):+.2f}",
            ])
            table.add_row(*cells)
        console.print(table)

    def _print_summary(self, baseline_acc: float) -> None:
        if not self.results:
            console.print("[yellow]No predictor results.[/yellow]")
            return
        ordered = sorted(self.results, key=lambda r: -(r.kappa if not np.isnan(r.kappa) else -2))
        table = Table(title="Predictor benchmark — sorted by Cohen's κ")
        table.add_column("rank", style="dim", justify="right")
        table.add_column("predictor", style="bold")
        table.add_column("family")
        table.add_column("acc", justify="right")
        table.add_column("κ", justify="right")
        table.add_column("F1", justify="right")
        table.add_column("n_test", justify="right")
        table.add_column("elapsed", justify="right")
        # baseline row
        table.add_row("--", "[dim]baseline[/dim]", "--", f"{baseline_acc:.3f}", "0.000", "--", "--", "--")
        for i, r in enumerate(ordered, 1):
            color = ""
            if r.kappa > 0.4:
                color = "green"
            elif r.kappa > 0.2:
                color = "yellow"
            elif r.kappa < 0:
                color = "red"
            kappa_str = f"[{color}]{r.kappa:+.3f}[/{color}]" if color else f"{r.kappa:+.3f}"
            table.add_row(
                str(i), r.name, r.family,
                f"{r.accuracy:.3f}",
                kappa_str,
                f"{r.f1_macro:.3f}",
                str(r.n_test),
                f"{r.metadata.get('elapsed_sec', 0):.1f}s",
            )
        console.print(table)

    # -------------------------------------------------------------------
    # Multi-coin training data
    # -------------------------------------------------------------------
    def _build_coin_data(
        self, cfg: RunConfig, coin: ExtraCoinSpec,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
        """Build (X, y, dates, df) for one extra coin using main config defaults."""
        if cfg.data_root is None:
            raise RuntimeError(
                "data_root not set in config — required to resolve extra coin paths. "
                "Add `data_root: ~/regime_data` to the preset."
            )
        venue = coin.venue or cfg.venue
        quote = coin.quote or cfg.quote
        settle = coin.settle or cfg.settle
        timeframe = coin.timeframe or cfg.timeframe

        root = DataRoot(
            data_root=cfg.data_root,
            venue=venue,
            target=coin.target,
            quote=quote,
            settle=settle,
            timeframe=timeframe,
        )
        if not root.ohlcv().exists():
            raise FileNotFoundError(f"OHLCV missing for extra coin: {root.ohlcv()}")

        df = load_ohlcv(root.ohlcv())
        labeller = get_labeller(
            cfg.label.method,
            L_range=cfg.label.L_range,
            t_threshold=cfg.label.t_threshold,
            hysteresis_bars=cfg.label.hysteresis_bars,
            strong_threshold=cfg.label.strong_threshold,
            horizon=cfg.label.horizon,
            alpha=cfg.label.alpha,
        )
        labels = labeller.compute(df)
        pipeline = FeaturePipeline(
            use_technical=cfg.features.use_technical,
            use_external=cfg.features.use_external,
            use_trading_signals=cfg.features.use_trading_signals,
            trading_signals_yaml=cfg.features.trading_signals_yaml,
            target_funding_path=root.funding() if root.funding().exists() else None,
            cross_ohlcv_path=root.cross_ohlcv() if root.cross_ohlcv().exists() else None,
            cross_name=root.cross_name(),
            external_dir=cfg.paths.external_dir,
            drop_nan_rows=cfg.features.drop_nan_rows,
        )
        X, y, dates = pipeline.build(df, labels)
        return X, y, dates, df

    def _stack_with_target(
        self,
        cfg: RunConfig,
        X_target: pd.DataFrame,
        y_target: pd.Series,
        dates_target: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Concatenate target + extra coins for training. Adds coin_id one-hot
        columns. Returns combined (X, y, dates).
        """
        if not cfg.training.extra_coins:
            return X_target, y_target, dates_target

        all_X: list[pd.DataFrame] = []
        all_y: list[pd.Series] = []
        all_dates: list[pd.Series] = []
        coin_ids: list[str] = []

        # Target coin first
        Xt = X_target.copy()
        Xt["coin_id"] = cfg.target
        all_X.append(Xt)
        all_y.append(y_target.copy())
        all_dates.append(dates_target.copy())
        coin_ids.append(cfg.target)

        # Extra coins
        for coin in cfg.training.extra_coins:
            try:
                console.print(f"[cyan]  loading extra coin: {coin.target}[/cyan]")
                Xc, yc, dc, _ = self._build_coin_data(cfg, coin)
                Xc = Xc.copy()
                Xc["coin_id"] = coin.target
                # Align columns with target
                missing_in_target = set(Xc.columns) - set(Xt.columns) - {"coin_id"}
                missing_in_extra = set(Xt.columns) - set(Xc.columns) - {"coin_id"}
                if missing_in_target or missing_in_extra:
                    # Use intersection only
                    common = sorted((set(Xt.columns) & set(Xc.columns)) | {"coin_id"})
                    Xt = Xt[common]
                    all_X[0] = Xt   # update target's stored frame
                    Xc = Xc[common]
                all_X.append(Xc)
                all_y.append(yc)
                all_dates.append(dc)
                coin_ids.append(coin.target)
                console.print(f"    {coin.target}: {len(Xc):>6} bars")
            except Exception as e:
                console.print(f"[yellow]    {coin.target} skipped: {e}[/yellow]")

        # One-hot encode coin_id
        X_combined = pd.concat(all_X, axis=0, ignore_index=False)
        y_combined = pd.concat(all_y, axis=0, ignore_index=False)
        dates_combined = pd.concat(all_dates, axis=0, ignore_index=False)
        if cfg.training.add_coin_id_feature:
            ohe = pd.get_dummies(X_combined["coin_id"], prefix="coin").astype(float)
            X_combined = pd.concat([X_combined.drop(columns="coin_id"), ohe], axis=1)
        else:
            X_combined = X_combined.drop(columns="coin_id")

        # Sort by date for clean walk-forward semantics
        order = dates_combined.argsort()
        X_combined = X_combined.iloc[order].reset_index(drop=True)
        y_combined = y_combined.iloc[order].reset_index(drop=True)
        dates_combined = dates_combined.iloc[order].reset_index(drop=True)

        n_coins = len(coin_ids)
        n_total = len(X_combined)
        console.print(
            f"[bold cyan]Multi-coin train data: {n_coins} coins, {n_total} total bars[/bold cyan]"
        )
        return X_combined, y_combined, dates_combined

    def _save_fold_plot_multi(
        self, cfg: RunConfig, df: pd.DataFrame, mode: str, fold_id: int,
        fold_predictions: dict, X_te: pd.DataFrame, d_te: pd.Series,
    ) -> None:
        """Save multi-classifier plots for one CV fold (B-multi synth equity overlay,
        A-multi step panel)."""
        from ..visualization.regime_plots import plot_synth_equity_multi, plot_regime_step_multi

        preds_dict: dict[str, pd.Series] = {}
        for name, pred_arr in fold_predictions.items():
            s = pd.Series("", index=df.index, dtype=object)
            if len(pred_arr) == len(X_te):
                s.loc[X_te.index] = pred_arr
            preds_dict[name] = s

        suffix = f"{cfg.target}-{cfg.timeframe}-{mode}-fold{fold_id+1}"
        split_dt = d_te.iloc[0]
        try:
            plot_synth_equity_multi(
                df, preds_dict,
                PLOTS_DIR / f"B_multi_{mode}_fold{fold_id+1}.png",
                suffix, split_dt,
            )
            plot_regime_step_multi(
                df, preds_dict,
                PLOTS_DIR / f"A_multi_{mode}_fold{fold_id+1}.png",
                suffix, split_dt,
            )
            console.print(
                f"      [dim]multi-classifier plots: A/B_multi_{mode}_fold{fold_id+1}.png "
                f"({len(preds_dict)} predictors)[/dim]"
            )
        except Exception as e:
            console.print(f"      [yellow]multi-plot save failed: {e}[/yellow]")

    def _save_fold_plot(
        self, cfg: RunConfig, df: pd.DataFrame, mode: str, fold_id: int,
        best: PredictionResult, X_te: pd.DataFrame, d_te: pd.Series,
        test_predictions: np.ndarray,
    ) -> None:
        """Save the 3 prediction plots for one CV fold's best predictor.

        Plot files are suffixed with mode + fold to disambiguate across runs.
        """
        # Build a label series spanning the full df:
        #   - bars in the test window: predictor's predictions
        #   - other bars: empty string (denoiser will skip them)
        out = pd.Series("", index=df.index, dtype=object)
        if len(test_predictions) == len(X_te):
            out.loc[X_te.index] = test_predictions

        # We want a fold-specific filename — temporarily swap PLOT paths then restore
        from ..visualization.regime_plots import (
            denoise_labels, _compute_runs, _plot_A, _plot_B, _plot_C,
        )
        smooth = denoise_labels(out, window=168)   # 7 days at 1h TF
        runs = _compute_runs(df, smooth)
        suffix = f"pred-{best.name}-{cfg.target}-{cfg.timeframe}-{mode}-fold{fold_id+1}"
        out_dir = PLOTS_DIR
        split_dt = d_te.iloc[0]
        try:
            _plot_A(df, runs, out_dir / f"A_pred_{mode}_fold{fold_id+1}.png", suffix, split_dt)
            _plot_B(df, smooth, runs, out_dir / f"B_pred_{mode}_fold{fold_id+1}.png", suffix, split_dt)
            _plot_C(df, smooth, runs, out_dir / f"C_pred_{mode}_fold{fold_id+1}.png", suffix, split_dt)
            console.print(
                f"      [dim]plots saved: A/B/C_pred_{mode}_fold{fold_id+1}.png "
                f"(predictor={best.name}, κ={best.kappa:+.3f})[/dim]"
            )
        except Exception as e:
            console.print(f"      [yellow]plot save failed: {e}[/yellow]")

    def _print_and_save_importance(self, predictor: BasePredictor, X_te, y_te) -> None:
        try:
            with console.status(f"[cyan]Computing feature importance ({predictor.name})..."):
                imp = predictor.feature_importances(X_te, y_te, n_repeats=3)
        except Exception as e:
            console.print(f"[yellow]feature_importances skipped ({predictor.name}): {e}[/yellow]")
            return
        if imp is None or len(imp) == 0:
            console.print(f"[dim]feature_importances unavailable for {predictor.name}.[/dim]")
            return

        # Save full ranking
        out_csv = RESULTS_DIR / f"feature_importance_{predictor.name}.csv"
        imp_df = imp.reset_index().rename(columns={"index": "feature"})
        imp_df.columns = ["feature", "importance"]
        imp_df.to_csv(out_csv, index=False)

        top = imp.head(20)
        max_imp = float(top.iloc[0]) if len(top) and top.iloc[0] != 0 else 1.0
        table = Table(title=f"Top 20 features — {predictor.name}")
        table.add_column("rank", justify="right", style="dim")
        table.add_column("feature")
        table.add_column("importance", justify="right")
        table.add_column("bar")
        for rank, (feat, val) in enumerate(top.items(), 1):
            bar_len = int(round(20 * float(val) / max_imp)) if max_imp > 0 else 0
            bar = "█" * max(bar_len, 0)
            table.add_row(str(rank), str(feat), f"{val:.4f}", bar)
        console.print(table)
        console.print(f"[dim]Full ranking → {out_csv}[/dim]")

    def _save_summary_csv(self) -> None:
        df = pd.DataFrame([
            {
                "name": r.name, "family": r.family,
                "accuracy": r.accuracy, "kappa": r.kappa, "f1_macro": r.f1_macro,
                "n_test": r.n_test,
                "elapsed_sec": r.metadata.get("elapsed_sec", 0),
            } for r in self.results
        ])
        df.to_csv(RESULTS_DIR / "predictor_summary.csv", index=False)
