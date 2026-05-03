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

from ..config import RunConfig, RESULTS_DIR, PLOTS_DIR
from ..data.loaders import load_ohlcv
from ..labels import get_labeller
from ..features.pipeline import FeaturePipeline
from ..predictors.base import BasePredictor, PredictionResult
from ..predictors.classical import (
    LogRegPredictor, RandomForestPredictor, GBMPredictor, MLPPredictor, XGBoostPredictor,
)
from ..predictors.rule_based import RegimeV3Predictor, RegimeV4EmaPredictor
from ..predictors.deep_nets import DeepMLPPredictor, GRUPredictor, LSTMPredictor
from ..predictors.transformer import TimeSeriesTransformerPredictor
from ..predictors.pretrained import PRETRAINED_REGISTRY
from ..signal_analysis import rank_signals
from ..visualization.regime_plots import save_label_plots, save_prediction_plots
from .metrics import evaluate
from .splits import time_aware_split, walk_forward_splits


console = Console()


def _build_predictors(cfg: RunConfig) -> list[BasePredictor]:
    """Instantiate all predictors enabled in the config."""
    out: list[BasePredictor] = []
    families = set(cfg.predictors.families)

    if "classical" in families:
        out += [
            LogRegPredictor(),
            RandomForestPredictor(),
            GBMPredictor(),
            MLPPredictor(),
            XGBoostPredictor(),
        ]
    if "rule_based" in families:
        out += [RegimeV3Predictor(), RegimeV4EmaPredictor()]
    if "deep_nets" in families:
        out += [DeepMLPPredictor(), GRUPredictor(), LSTMPredictor()]
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

        # ----- 5. Split: single or walk-forward -----
        purge = cfg.purge_bars
        if cfg.split.walk_forward_folds and cfg.split.walk_forward_folds > 0:
            return self._run_walk_forward(cfg, df, X, y, dates, purge)
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

    def _run_walk_forward(self, cfg, df, X, y, dates, purge: int) -> dict:
        n_folds = int(cfg.split.walk_forward_folds)
        console.print(
            f"[bold]Walk-forward CV[/bold] — {n_folds} folds, "
            f"min_train={cfg.split.min_train_fraction:.0%}, purge={purge} bars"
        )

        # Per-fold results: rows = (fold, predictor), cols = metrics
        per_fold_rows: list[dict] = []
        all_results_aggr: dict[str, list[PredictionResult]] = {}

        for train_idx, test_idx, fold_id in walk_forward_splits(
            n=len(X),
            n_folds=n_folds,
            purge_bars=purge,
            min_train_fraction=cfg.split.min_train_fraction,
        ):
            X_tr = X.iloc[train_idx]
            y_tr = y.iloc[train_idx]
            d_tr = dates.iloc[train_idx]
            df_tr = df.loc[X_tr.index]
            X_te = X.iloc[test_idx]
            y_te = y.iloc[test_idx]
            d_te = dates.iloc[test_idx]
            df_te = df.loc[X_te.index]

            console.rule(
                f"[bold cyan]Fold {fold_id+1}/{n_folds}: "
                f"train={len(X_tr)} ({d_tr.iloc[0].date()}→{d_tr.iloc[-1].date()}) "
                f"| test={len(X_te)} ({d_te.iloc[0].date()}→{d_te.iloc[-1].date()})"
            )

            baseline_acc = float((y_te.values == y_tr.value_counts().idxmax()).mean())
            predictors = _build_predictors(cfg)
            fold_results, _ = self._fit_eval_loop(
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
                all_results_aggr.setdefault(r.name, []).append(r)

        if not per_fold_rows:
            console.print("[red]No folds produced — check min_train_fraction / data length[/red]")
            return {"results": [], "folds": []}

        per_fold_df = pd.DataFrame(per_fold_rows)
        per_fold_df.to_csv(RESULTS_DIR / "walk_forward_per_fold.csv", index=False)
        self._print_walk_forward_summary(per_fold_df)
        return {"folds": per_fold_rows, "n_folds_run": per_fold_df["fold"].nunique()}

    def _print_walk_forward_summary(self, per_fold: pd.DataFrame) -> None:
        # Aggregated table: mean ± std κ across folds
        agg = per_fold.groupby(["predictor", "family"]).agg(
            kappa_mean=("kappa", "mean"),
            kappa_std=("kappa", "std"),
            kappa_min=("kappa", "min"),
            kappa_max=("kappa", "max"),
            acc_mean=("accuracy", "mean"),
            f1_mean=("f1_macro", "mean"),
            n_folds=("fold", "count"),
        ).reset_index().sort_values("kappa_mean", ascending=False)

        agg.to_csv(RESULTS_DIR / "walk_forward_aggregated.csv", index=False)

        table = Table(title="Walk-forward aggregate — sorted by mean κ")
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
