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
from ..predictors.ensemble import EnsemblePredictor, ConfidenceEnsemblePredictor
from ..predictors.pretrained import PRETRAINED_REGISTRY
from ..signal_analysis import rank_signals
from ..visualization.regime_plots import save_label_plots, save_prediction_plots
from .metrics import evaluate
from .splits import (
    time_aware_split, walk_forward_splits, leave_one_out_splits,
    rolling_walk_forward_splits,
)


console = Console()


def _resolve_extra_cross_paths(
    cfg: RunConfig,
    root_override: "DataRoot | None" = None,
) -> list[tuple[Path, str]]:
    """Resolve cfg.cross_assets to (path, name) tuples for the FeaturePipeline.

    Each spec defaults missing fields to the run's root values (venue, quote,
    settle, market_type) so the minimal entry is `- target: SOL`. Names default
    to `target.lower()`. Missing files are reported by the loader, not here.
    """
    specs = getattr(cfg, "cross_assets", None) or []
    if not specs:
        return []
    if cfg.data_root is None:
        console.print(
            "[yellow]cross_assets: ignored — set `data_root:` to resolve paths.[/yellow]"
        )
        return []
    if root_override is not None:
        base = root_override
    else:
        base = DataRoot(
            data_root=cfg.data_root,
            venue=cfg.venue,
            target=cfg.target,
            quote=cfg.quote,
            settle=cfg.settle,
            timeframe=cfg.timeframe,
            market_type=cfg.market_type,
        )
    out: list[tuple[Path, str]] = []
    for spec in specs:
        path = base.extra_cross_ohlcv(
            target=spec.target,
            quote=spec.quote,
            settle=spec.settle,
            market_type=spec.market_type,
            venue=spec.venue,
        )
        name = (spec.name or spec.target).lower()
        out.append((path, name))
    return out


def _has_native_importance(predictor: BasePredictor) -> bool:
    """True iff predictor exposes cheap, native feature importance.

    Native = sklearn-style `feature_importances_` (trees, GBDT) or `coef_`
    (linear models). Excludes neural nets (MLP, GRU, LSTM, TST) which would
    fall back to slow permutation importance.
    """
    clf = getattr(predictor, "clf", None)
    if clf is None:
        return False
    return hasattr(clf, "feature_importances_") or hasattr(clf, "coef_")


def _build_predictors(cfg: RunConfig) -> list[BasePredictor]:
    """Instantiate all predictors enabled in the config.

    When `cfg.predictors.include_finetune` is True, also instantiate a `-FT`
    variant of each class whose `supports_finetune = True`. The same class
    produces both cold and FT instances depending on the constructor flag.
    """
    out: list[BasePredictor] = []
    families = set(cfg.predictors.families)
    add_ft = bool(cfg.predictors.include_finetune)

    def _add(cls, **kwargs):
        out.append(cls(**kwargs))
        if add_ft and getattr(cls, "supports_finetune", False):
            out.append(cls(finetune=True, **kwargs))

    if "classical" in families:
        _add(LogRegPredictor)
        _add(RandomForestPredictor)
        _add(ExtraTreesPredictor)
        _add(MLPPredictor)
        _add(XGBoostPredictor)
        _add(LightGBMPredictor)
    if "rule_based" in families:
        out += [RegimeV3Predictor(), RegimeV4EmaPredictor()]
    if "deep_nets" in families:
        _add(GRUPredictor)
        _add(LSTMPredictor)
    if "transformer" in families:
        _add(TimeSeriesTransformerPredictor)
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
    if "rl" in families:
        # RL predictors: 3 approximators × 2-3 action spaces. Each subclass'
        # action_space_type is fixed at the class level — we instantiate the
        # subset selected by cfg.predictors.rl.action_spaces.
        from ..predictors.rl import (
            DQN2Predictor, DQN3Predictor, SACPredictor,
            LinearQ2Predictor, LinearQ3Predictor,
            LGBQ2Predictor, LGBQ3Predictor,
            XGBQ2Predictor, XGBQ3Predictor,
            RFQ2Predictor, RFQ3Predictor,
            RidgeQ2Predictor, RidgeQ3Predictor,
            HuberQ2Predictor, HuberQ3Predictor,
            HistGBQ2Predictor, HistGBQ3Predictor,
            CatQ2Predictor, CatQ3Predictor,
        )
        rl_cfg = cfg.predictors.rl
        # Shared kwargs (every approximator gets these). total_timesteps is
        # per-approximator below since some approximators (RF) want a smaller
        # budget than the shared default.
        rl_shared = dict(
            transaction_cost=rl_cfg.transaction_cost,
            flat_threshold=rl_cfg.flat_threshold,
            ft_steps_scale=rl_cfg.ft_steps_scale,
            proba_temperature=rl_cfg.proba_temperature,
            seed=cfg.seed,
        )
        def _ts(override: int | None) -> int:
            """Per-approximator total_timesteps with shared fallback."""
            return int(override) if override is not None else int(rl_cfg.total_timesteps)
        # Per-approximator hyperparam blocks — built once, reused across action spaces
        nn_kw = dict(rl_shared, total_timesteps=_ts(rl_cfg.nn_total_timesteps),
            learning_rate=rl_cfg.nn_learning_rate,
            buffer_size=rl_cfg.nn_buffer_size,
            gamma=rl_cfg.nn_gamma, net_arch=tuple(rl_cfg.nn_net_arch),
            verbose=rl_cfg.nn_verbose)
        linear_kw = dict(rl_shared, total_timesteps=_ts(rl_cfg.linear_total_timesteps),
            learning_rate=rl_cfg.linear_learning_rate, gamma=rl_cfg.linear_gamma,
            epsilon_start=rl_cfg.linear_epsilon_start,
            epsilon_end=rl_cfg.linear_epsilon_end)
        lgb_kw = dict(rl_shared, total_timesteps=_ts(rl_cfg.lgb_total_timesteps),
            n_estimators=rl_cfg.lgb_n_estimators, max_depth=rl_cfg.lgb_max_depth,
            learning_rate=rl_cfg.lgb_learning_rate, gamma=rl_cfg.lgb_gamma,
            iterations=rl_cfg.lgb_iterations)
        xgb_kw = dict(rl_shared, total_timesteps=_ts(rl_cfg.xgb_total_timesteps),
            n_estimators=rl_cfg.xgb_n_estimators, max_depth=rl_cfg.xgb_max_depth,
            learning_rate=rl_cfg.xgb_learning_rate, gamma=rl_cfg.xgb_gamma,
            iterations=rl_cfg.xgb_iterations)
        rf_kw = dict(rl_shared, total_timesteps=_ts(rl_cfg.rf_total_timesteps),
            n_estimators=rl_cfg.rf_n_estimators, max_depth=rl_cfg.rf_max_depth,
            min_samples_leaf=rl_cfg.rf_min_samples_leaf, gamma=rl_cfg.rf_gamma,
            iterations=rl_cfg.rf_iterations)
        ridge_kw = dict(rl_shared, total_timesteps=_ts(rl_cfg.ridge_total_timesteps),
            alpha=rl_cfg.ridge_alpha, gamma=rl_cfg.ridge_gamma,
            iterations=rl_cfg.ridge_iterations)
        huber_kw = dict(rl_shared, total_timesteps=_ts(rl_cfg.huber_total_timesteps),
            alpha=rl_cfg.huber_alpha, epsilon=rl_cfg.huber_epsilon,
            max_iter=rl_cfg.huber_max_iter, tol=rl_cfg.huber_tol,
            gamma=rl_cfg.huber_gamma, iterations=rl_cfg.huber_iterations)
        histgb_kw = dict(rl_shared, total_timesteps=_ts(rl_cfg.histgb_total_timesteps),
            max_iter=rl_cfg.histgb_max_iter, max_depth=rl_cfg.histgb_max_depth,
            learning_rate=rl_cfg.histgb_learning_rate,
            min_samples_leaf=rl_cfg.histgb_min_samples_leaf,
            l2_regularization=rl_cfg.histgb_l2_regularization,
            gamma=rl_cfg.histgb_gamma, iterations=rl_cfg.histgb_iterations)
        cat_kw = dict(rl_shared, total_timesteps=_ts(rl_cfg.cat_total_timesteps),
            n_estimators=rl_cfg.cat_n_estimators, max_depth=rl_cfg.cat_max_depth,
            learning_rate=rl_cfg.cat_learning_rate, l2_leaf_reg=rl_cfg.cat_l2_leaf_reg,
            gamma=rl_cfg.cat_gamma, iterations=rl_cfg.cat_iterations)
        # Map action_space → list of (cls, kwargs)
        rl_classes = {
            "discrete-2": [
                (DQN2Predictor, nn_kw),
                (LinearQ2Predictor, linear_kw),
                (LGBQ2Predictor, lgb_kw),
                (XGBQ2Predictor, xgb_kw),
                (RFQ2Predictor, rf_kw),
                (RidgeQ2Predictor, ridge_kw),
                (HuberQ2Predictor, huber_kw),
                (HistGBQ2Predictor, histgb_kw),
                (CatQ2Predictor, cat_kw),
            ],
            "discrete-3": [
                (DQN3Predictor, nn_kw),
                (LinearQ3Predictor, linear_kw),
                (LGBQ3Predictor, lgb_kw),
                (XGBQ3Predictor, xgb_kw),
                (RFQ3Predictor, rf_kw),
                (RidgeQ3Predictor, ridge_kw),
                (HuberQ3Predictor, huber_kw),
                (HistGBQ3Predictor, histgb_kw),
                (CatQ3Predictor, cat_kw),
            ],
            "continuous": [
                (SACPredictor, nn_kw),
            ],
        }
        for action_space in rl_cfg.action_spaces:
            for cls, kw in rl_classes.get(action_space, []):
                out.append(cls(**kw))
                if add_ft and getattr(cls, "supports_finetune", False):
                    out.append(cls(finetune=True, **kw))

    # Auto-attach Ensemble + Ensemble-Conf whenever any probabilistic base
    # family is enabled. Both are just aggregators — make no sense without
    # bases. Ensemble = uniform soft vote; Ensemble-Conf = per-bar
    # max-proba-weighted vote (more confident base wins each bar).
    # RL predictors expose softmax(Q) as proba (one-hot fallback for
    # continuous / unfitted) so they participate in the ensemble like any
    # other probabilistic base. The Q-margin information naturally weights
    # the ConfidenceEnsemble — bars with one dominant action contribute
    # sharper proba, bars with similar Q-values contribute flatter proba.
    proba_families = {"classical", "deep_nets", "transformer", "rl"}
    if cfg.predictors.include_ensemble and (proba_families & families):
        global_normalize = bool(cfg.predictors.ensemble_normalize_proba)
        _add(EnsemblePredictor, proba_normalize=global_normalize)
        _add(ConfidenceEnsemblePredictor, proba_normalize=global_normalize)

        # Subset ensembles — one per entry in cfg.predictors.ensemble_groups.
        # Each produces Ensemble-{name} (and -FT) voting only over the named
        # bases. Optional fields:
        #   method: "uniform" (default) or "confidence"
        #   normalize_proba: bool, overrides cfg.predictors.ensemble_normalize_proba
        for group in cfg.predictors.ensemble_groups:
            gname = str(group.get("name", "")).strip()
            gbases = list(group.get("bases", []))
            method = str(group.get("method", "uniform")).strip().lower()
            normalize = bool(group.get("normalize_proba", global_normalize))
            if not gname or not gbases:
                console.print(f"[yellow]WARN: ignoring malformed ensemble_group: {group}[/yellow]")
                continue
            if method not in ("uniform", "confidence"):
                console.print(
                    f"[yellow]WARN: ensemble_group '{gname}' method='{method}' "
                    f"unknown, defaulting to 'uniform' (valid: uniform, confidence)[/yellow]"
                )
                method = "uniform"
            cls = ConfidenceEnsemblePredictor if method == "confidence" else EnsemblePredictor
            out.append(cls(
                bases_filter=gbases, name_suffix=f"-{gname}",
                proba_normalize=normalize,
            ))
            if add_ft:
                out.append(cls(
                    finetune=True, bases_filter=gbases, name_suffix=f"-{gname}",
                    proba_normalize=normalize,
                ))

    # Disabled list — match the final display name (post -FT and post-suffix).
    disabled = set(cfg.predictors.disabled or [])
    if disabled:
        before_names = {p.name for p in out}
        out = [p for p in out if p.name not in disabled]
        actually_disabled = sorted(disabled & before_names)
        not_found = sorted(disabled - before_names)
        if actually_disabled:
            console.print(
                f"[yellow]Disabled {len(actually_disabled)} predictors: "
                f"{actually_disabled}[/yellow]"
            )
        if not_found:
            console.print(
                f"[yellow]WARN: disabled names not in build (typos?): {not_found}[/yellow]"
            )

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
        # Mapping {feature_name: source_group} populated after the feature
        # pipeline runs. Used to annotate feature-importance tables.
        self.feature_sources: dict[str, str] = {}
        # Cache of extra-coin (X, y, dates) keyed by coin.target. Populated on
        # first call to _stack_with_target so CV folds don't re-run the entire
        # feature pipeline for ETH/SOL/etc on every fold.
        # Cache: coin.target -> (X, y, dates, df). df has 'close' price column,
        # required by multi-coin-aware predictors (RL agents need price series
        # for reward; pretrained fine_tuned needs them for embedding).
        self._extra_coin_cache: dict[str, tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]] = {}

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
        # Defer label plots until we know the OOS span (computed in
        # _run_cv_single_mode / _run_single_split after fold materialization).
        # This lets B_labels_synth.png clamp to the OOS window for direct
        # visual comparison with the prediction plots.
        self._label_plots_saved = False

        # ----- 4. Features -----
        console.print("[bold]Computing features[/bold]")
        extra_cross_paths = _resolve_extra_cross_paths(cfg)
        pipeline = FeaturePipeline(
            use_technical=cfg.features.use_technical,
            use_external=cfg.features.use_external,
            use_funding=cfg.features.use_funding,
            use_regime_signals=cfg.features.use_regime_signals,
            use_trading_signals=cfg.features.use_trading_signals,
            trading_signals_yaml=cfg.features.trading_signals_yaml,
            target_funding_path=cfg.paths.funding,
            cross_ohlcv_path=cfg.paths.cross_ohlcv,
            cross_name=cfg.paths.cross_name,
            extra_cross_paths=extra_cross_paths,
            external_dir=cfg.paths.external_dir,
            drop_nan_rows=cfg.features.drop_nan_rows,
        )
        X, y, dates = pipeline.build(df, labels)
        self.X, self.y, self.dates = X, y, dates
        self.feature_count = X.shape[1]
        self.feature_sources = dict(getattr(pipeline, "column_sources", {}))
        console.print(f"  Final dataset: {len(X)} rows × {X.shape[1]} cols")

        # ----- 5. Split: single or cross-validation -----
        purge = cfg.purge_bars
        # Backward-compat: walk_forward_folds → cv_folds with mode walk_forward
        if cfg.split.walk_forward_folds and not cfg.split.cv_folds:
            cfg.split.cv_folds = cfg.split.walk_forward_folds
            cfg.split.cv_mode = "walk_forward"

        # Rolling mode is always CV regardless of cv_folds (folds derive from
        # train/test window sizes). For other modes, cv_folds > 0 triggers CV.
        if cfg.split.cv_mode == "rolling" or (cfg.split.cv_folds and cfg.split.cv_folds > 0):
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

        # Save label plots clamped to the test (OOS) span — same X-axis as
        # the prediction plots for direct visual comparison.
        if cfg.plots.enabled and not getattr(self, "_label_plots_saved", False):
            console.print("[bold]Saving label plots[/bold]")
            oos_span = (pd.to_datetime(d_te.iloc[0]), pd.to_datetime(d_te.iloc[-1]))
            save_label_plots(df, labels, PLOTS_DIR, cfg, xlim_dates=oos_span)
            self._label_plots_saved = True

        # Signal analysis is a separate concern (its own `regime-run signals`
        # subcommand). The CV path already skips it; running it here too made
        # single-split runs gratuitously slower than CV runs. Now: only run
        # when the user explicitly invoked `signals` (empty families).
        if not cfg.predictors.families:
            console.print("[bold]Signal analysis (lift + MI on train)[/bold]")
            try:
                ranking = rank_signals(X_tr, y_tr)
                top10 = ranking.head(10)
                self._print_signal_ranking(top10)
                ranking.to_csv(RESULTS_DIR / "signal_ranking.csv", index=False)
            except Exception as e:
                console.print(f"[yellow]signal_analysis skipped: {e}[/yellow]")

        predictors = _build_predictors(cfg)
        # Single-split has no fold concept — drop FT variants (they would just
        # cold-start with no prior state and duplicate the cold rows).
        skipped_ft = [p.name for p in predictors if getattr(p, "is_finetune", False)]
        predictors = [p for p in predictors if not getattr(p, "is_finetune", False)]
        if skipped_ft:
            console.print(
                f"[yellow]Skipping FT variants in single-split mode "
                f"(no folds to warm-start from): {', '.join(skipped_ft)}[/yellow]"
            )
        console.print(f"[bold]Running {len(predictors)} predictors[/bold]")
        baseline_acc = float((y_te.values == y_tr.value_counts().idxmax()).mean())
        console.print(f"  Baseline (always-predict majority): acc={baseline_acc:.3f}")

        results, per_predictor_predictions = self._fit_eval_loop(
            predictors, X_tr, y_tr, d_tr, df_tr, X_te, y_te, d_te, df_te
        )
        self.results = results
        from .metrics import buy_and_hold_gain
        bh_gain_split = (
            buy_and_hold_gain(np.asarray(df_te["close"].values, dtype=np.float64))
            if "close" in df_te.columns else float("nan")
        )
        self._print_summary(baseline_acc, bh_gain_split)
        self._save_summary_csv()

        if results:
            best = max(results, key=lambda r: r.kappa if not np.isnan(r.kappa) else -2)
            best_pred = self._build_full_pred_series(
                df, labels, X.index, X_tr.index, X_te.index,
                per_predictor_predictions[best.name],
            )
            if cfg.plots.enabled:
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
            else:
                console.print(
                    f"[bold]Best predictor: {best.name}[/bold] "
                    f"(κ={best.kappa:+.3f}, acc={best.accuracy:.3f}) [dim](plots disabled)[/dim]"
                )
            # Feature importance from the best CLASSICAL predictor with native
            # importance (skip NNs — permutation is too slow and not native).
            if cfg.predictors.feature_importance:
                classical_ranked = sorted(
                    [r for r in results
                     if r.family == "classical" and not np.isnan(r.kappa)],
                    key=lambda r: r.kappa, reverse=True,
                )
                for r in classical_ranked:
                    cand = next((p for p in predictors if p.name == r.name), None)
                    if cand is not None and _has_native_importance(cand):
                        self._print_and_save_importance(cand, X_te, y_te)
                        break

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
        skip_fit: bool = False,
    ) -> tuple[list[PredictionResult], dict[str, np.ndarray]]:
        results: list[PredictionResult] = []
        per_predictor_predictions: dict[str, np.ndarray] = {}
        # Per-base predict_proba (n_test, n_classes); fed to ensemble predictors
        # in the second pass.
        per_predictor_probas: dict[str, np.ndarray] = {}

        # Two-pass: base predictors first, then ensembles (which depend on
        # base outputs). Maintain original predictor ordering otherwise.
        base_predictors = [p for p in predictors if not getattr(p, "is_ensemble", False)]
        ensemble_predictors = [p for p in predictors if getattr(p, "is_ensemble", False)]

        # Close prices for synth_gain — passed once per fold; aligned to y_te.
        closes_te = np.asarray(df_te["close"].values, dtype=np.float64) if "close" in df_te.columns else None

        long_only = (getattr(self.cfg, "market_type", "futures") == "spot")

        def _evaluate(predictor, pred_arr, t0):
            res = evaluate(
                name=predictor.name,
                family=predictor.family,
                y_true=np.asarray(y_te.values),
                y_pred=pred_arr,
                closes=closes_te,
                dates=np.asarray(d_te.values) if d_te is not None else None,
                timeframe=self.cfg.timeframe,
                transaction_cost=self.cfg.predictors.evaluation_transaction_cost,
                long_only=long_only,
                metadata={"elapsed_sec": round(time.time() - t0, 2)},
            )
            results.append(res)
            per_predictor_predictions[predictor.name] = pred_arr
            dk_str = (
                f"{res.dir_kappa:+.3f}" if not np.isnan(res.dir_kappa) else "  nan"
            )
            sh_str = f"{res.sharpe:+.2f}" if not np.isnan(res.sharpe) else " nan"
            dd_str = f"{res.max_dd*100:+.1f}%" if not np.isnan(res.max_dd) else "  nan"
            cm_str = (
                f"{res.n_positive_months}/{res.n_total_months}"
                if res.n_total_months else " 0/0"
            )
            console.print(
                f"  [green]✔[/green] {predictor.name:35s} "
                f"acc={res.accuracy:.3f} κ={res.kappa:+.3f} dκ={dk_str} "
                f"F1={res.f1_macro:.3f} "
                f"gain={res.synth_gain*100:+.1f}% sh={sh_str} dd={dd_str} "
                f"+m={cm_str} ({res.metadata['elapsed_sec']}s)"
            )

        # ---- Pass 1: base predictors (also try predict_proba for ensemble) ----
        for predictor in base_predictors:
            with console.status(f"[bold green]{predictor.name}[/bold green]"):
                t0 = time.time()
                try:
                    if not skip_fit:
                        predictor.fit(X_tr, y_tr, d_tr, df_tr)
                    pred_arr = np.asarray(predictor.predict(X_te, d_te, df_te))
                    _evaluate(predictor, pred_arr, t0)
                    # Store proba for ensemble (base may return None if unsupported).
                    try:
                        proba = predictor.predict_proba(X_te, d_te, df_te)
                        if proba is not None:
                            per_predictor_probas[predictor.name] = np.asarray(proba)
                    except Exception:
                        pass
                except Exception as e:
                    console.print(f"  [red]✘[/red] {predictor.name}: {e}")
                    traceback.print_exc()

        # ---- Pass 2: ensembles (fed with this fold's base probabilities) ----
        # IMPORTANT: per_predictor_probas is *only* populated in Pass 1 above —
        # ensembles read from it but never write back. So Ensemble-Conf cannot
        # ingest Ensemble's output and vice versa. Each ensemble averages only
        # genuine base predictors (classical / deep_nets / transformer).
        for predictor in ensemble_predictors:
            with console.status(f"[bold green]{predictor.name}[/bold green]"):
                t0 = time.time()
                try:
                    predictor.feed_base_probas(per_predictor_probas)
                    if not per_predictor_probas:
                        console.print(
                            f"  [yellow]⊘[/yellow] {predictor.name}: no base predict_proba "
                            f"available — skipping"
                        )
                        continue
                    predictor.fit(X_tr, y_tr, d_tr, df_tr)  # no-op
                    pred_arr = np.asarray(predictor.predict(X_te, d_te, df_te))
                    _evaluate(predictor, pred_arr, t0)
                    # Use the post-dedup base list (cold ensembles see cold
                    # bases only; FT ensembles see FT bases only) — not the
                    # raw runner-side dict which has both flavors.
                    eff_bases = predictor.get_effective_bases()
                    console.print(
                        f"      [dim]ensemble over {len(eff_bases)} base predictors: "
                        f"{', '.join(eff_bases)}[/dim]"
                    )
                except Exception as e:
                    console.print(f"  [red]✘[/red] {predictor.name}: {e}")
                    traceback.print_exc()

        return results, per_predictor_predictions

    def _run_cv(self, cfg, df, X, y, dates, purge: int) -> dict:
        n_folds = int(cfg.split.cv_folds)
        if cfg.split.cv_mode == "both":
            modes = ["walk_forward", "leave_one_out"]
        else:
            modes = [cfg.split.cv_mode]

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
        elif mode == "rolling":
            tw = int(cfg.split.train_window_bars)
            te = int(cfg.split.test_window_bars)
            step = int(cfg.split.step_bars) or te
            if tw <= 0 or te <= 0:
                console.print(
                    f"[red]rolling CV requires train_window_bars and test_window_bars > 0 "
                    f"(got tw={tw}, te={te})[/red]"
                )
                return None
            # cv_folds (-k) caps the number of consecutive folds — useful for
            # short runs without changing window or step. Without -k, the
            # iterator yields every fold that fits in the data.
            max_folds = int(cfg.split.cv_folds) if cfg.split.cv_folds and cfg.split.cv_folds > 0 else None
            est_folds_full = max(0, (len(X) - tw - purge - te) // step + 1)
            est_folds = min(est_folds_full, max_folds) if max_folds else est_folds_full
            cap_note = (
                f" (capped at {max_folds} via -k; data fits {est_folds_full})"
                if max_folds and est_folds_full > est_folds else ""
            )
            # Show RL-specific purge alongside the global one when "rl" is
            # enabled and uses a smaller gap — global purge protects the
            # train/test split for label-based predictors; RL gets a smaller
            # effective gap because its reward is one-bar lookahead only.
            rl_pb = getattr(cfg.split, "rl_purge_bars", None)
            purge_str = f"{purge}"
            if (
                "rl" in (cfg.predictors.families or [])
                and rl_pb is not None and rl_pb < purge
            ):
                purge_str = f"{purge} (rl={rl_pb})"
            console.print(
                f"\n[bold]Rolling-window CV[/bold] — train={tw} bars, test={te} bars, "
                f"step={step} bars, purge={purge_str}, est. {est_folds} folds{cap_note}"
            )
            split_iter = rolling_walk_forward_splits(
                n=len(X),
                train_window_bars=tw,
                test_window_bars=te,
                purge_bars=purge,
                step_bars=step,
            )
            if max_folds:
                # Take the LATEST max_folds (chronologically most recent) and
                # renumber the kept folds 0..k-1 so console display + CSVs
                # show clean Fold 1/18 ... 18/18 instead of stale chrono ids.
                # The chronological position is still recoverable from the
                # test_start / test_end date columns in the per_fold CSV.
                all_folds = list(split_iter)
                last = all_folds[-max_folds:]
                split_iter = iter([
                    (tr, te_idx, new_id) for new_id, (tr, te_idx, _) in enumerate(last)
                ])
            n_folds = est_folds
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

        # Materialize all folds before the loop to compute the full OOS span
        # (first fold's test_start → last fold's test_end). Per-fold plots use
        # this span as their X-axis so each plot shows the broader CV context
        # (where this fold sits in the timeline) instead of a 1-month sliver.
        all_fold_list = list(split_iter)
        oos_span_dates: tuple | None = None
        if all_fold_list:
            first_test_idx = all_fold_list[0][1][0]
            last_test_idx = all_fold_list[-1][1][-1]
            oos_span_dates = (
                pd.to_datetime(df.loc[first_test_idx, "date"]),
                pd.to_datetime(df.loc[last_test_idx, "date"]),
            )
        split_iter = iter(all_fold_list)

        # Save label plots NOW — once per run — clamped to the OOS span.
        # This gives B_labels_synth.png the same X-axis as the prediction
        # plots, making them directly visually comparable. Skipped when
        # plots are disabled or when this is the second mode of cv_mode=both.
        if cfg.plots.enabled and not getattr(self, "_label_plots_saved", False):
            console.print("[bold]Saving label plots[/bold]")
            save_label_plots(df, self.labels, PLOTS_DIR, cfg, xlim_dates=oos_span_dates)
            self._label_plots_saved = True

        # Build predictors ONCE before the fold loop so FT instances retain
        # their state (booster / model weights / forest) across folds. Cold
        # variants reset internally on each fit() so reusing them is safe.
        predictors = _build_predictors(cfg)
        if mode == "leave_one_out":
            # Warm-starting fold N from fold N-1 under LOO would leak fold N's
            # test data (which sat in fold N-1's train). Filter FT out.
            skipped_ft = [p.name for p in predictors if getattr(p, "is_finetune", False)]
            predictors = [p for p in predictors if not getattr(p, "is_finetune", False)]
            if skipped_ft:
                console.print(
                    f"[yellow]Skipping FT variants in leave_one_out mode "
                    f"(would leak): {', '.join(skipped_ft)}[/yellow]"
                )
        if not predictors:
            console.print(
                f"[red]No predictors built! Configured families: {cfg.predictors.families}. "
                f"Valid: classical, rule_based, deep_nets, transformer, pretrained.[/red]"
            )
            return None
        console.print(f"[bold]Running {len(predictors)} predictors[/bold]")

        per_fold_rows: list[dict] = []
        # Long-format per-month gain rows: one entry per (fold, predictor, month).
        monthly_rows: list[dict] = []
        # Keep ALL predictors' predictions per fold (used for the stitched plot
        # at the end, where we pick the predictor with the best MEAN kappa across folds).
        all_fold_preds: list[dict] = []
        # Keep the LAST fold's trained predictor instances + its test set so we
        # can query feature importance from the global best predictor at the end.
        last_fold_predictors: list[BasePredictor] | None = None
        last_fold_X_te = None
        last_fold_y_te = None

        # Refit cadence — see SplitConfig.retrain_every.
        retrain_every = int(getattr(cfg.split, "retrain_every", 1))
        if retrain_every <= 0:
            console.print(
                f"  [cyan]retrain_every=0[/cyan]: fitting predictors at fold 0 "
                f"only; subsequent folds reuse the trained model."
            )
        elif retrain_every > 1:
            console.print(
                f"  [cyan]retrain_every={retrain_every}[/cyan]: refit predictors "
                f"every {retrain_every} folds; reuse the trained model in between."
            )
        fold_idx_in_loop = 0

        for train_idx, test_idx, fold_id in split_iter:
            if retrain_every <= 0:
                skip_fit = fold_idx_in_loop > 0
            elif retrain_every == 1:
                skip_fit = False
            else:
                skip_fit = (fold_idx_in_loop % retrain_every) != 0
            fold_idx_in_loop += 1
            X_tr = X.iloc[train_idx]
            y_tr = y.iloc[train_idx]
            d_tr = dates.iloc[train_idx]
            df_tr = df.loc[X_tr.index]
            X_te = X.iloc[test_idx]
            y_te = y.iloc[test_idx]
            d_te = dates.iloc[test_idx]
            df_te = df.loc[X_te.index]

            # Capture target's pre-stacked data (before _stack_with_target
            # mixes in extras) — multi-coin-aware predictors (RL, pretrained
            # fine_tuned) need separate per-coin views, not the stacked one.
            target_pre_X = X_tr.copy()
            target_pre_y = y_tr.copy()
            target_pre_d = d_tr.copy()
            target_pre_df = df.loc[X_tr.index].copy()

            # RL-extended slice: RL only has 1-bar lookahead via reward (no
            # L_range label leakage), so the global purge wastes recent data.
            # Build an extended target slice ending closer to test_start that
            # we'll route to RL predictors only.
            rl_purge = cfg.split.rl_purge_bars
            rl_extension_bars = 0
            if (rl_purge is not None) and (rl_purge < purge):
                rl_extension_bars = int(purge - rl_purge)
                rl_train_end_pos = int(test_idx[0] - rl_purge)
                rl_train_positions = np.arange(int(train_idx[0]), rl_train_end_pos)
                rl_train_positions = rl_train_positions[rl_train_positions < len(X)]
                target_pre_X_rl = X.iloc[rl_train_positions].copy()
                target_pre_y_rl = y.iloc[rl_train_positions].copy()
                target_pre_d_rl = dates.iloc[rl_train_positions].copy()
                target_pre_df_rl = df.iloc[rl_train_positions].copy()
            else:
                target_pre_X_rl = target_pre_X
                target_pre_y_rl = target_pre_y
                target_pre_d_rl = target_pre_d
                target_pre_df_rl = target_pre_df

            # Multi-coin: stack target+extras into the train fold
            if cfg.training.extra_coins:
                X_tr_stacked, y_tr_stacked, d_tr_stacked = self._stack_with_target(
                    cfg, X_tr, y_tr, d_tr,
                )
                # Date-window the stacked train set:
                #   - rolling: clip extras to the SAME date range as target's
                #     fold window [d_tr_start, d_tr_end] so the rolling-window
                #     philosophy applies across all coins (constant span).
                #   - other modes (expanding / LOO): clip extras to bars BEFORE
                #     the test cutoff (no leakage); extras may span much more
                #     history than target since target's d_tr expands too.
                if mode == "rolling" and len(d_tr) >= 2:
                    d_start = d_tr.iloc[0]
                    d_end = d_tr.iloc[-1]
                    keep = (d_tr_stacked >= d_start) & (d_tr_stacked <= d_end)
                else:
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

            # Compute buy-and-hold over the test slice up-front so we can show
            # it in the fold header — gives a quick sense of which way the
            # market moved before any predictor ran.
            from .metrics import buy_and_hold_gain
            _closes_te = (
                np.asarray(df_te["close"].values, dtype=np.float64)
                if "close" in df_te.columns else None
            )
            bh_gain_fold = (
                buy_and_hold_gain(_closes_te) if _closes_te is not None else float("nan")
            )
            bh_str = f"{bh_gain_fold*100:+.1f}%" if not np.isnan(bh_gain_fold) else "nan"

            console.rule(
                f"[bold cyan]Fold {fold_id+1}/{n_folds} ({mode}): "
                f"train={len(X_tr)} | "
                f"test={len(X_te)} ({d_te.iloc[0].date()}→{d_te.iloc[-1].date()}) "
                f"| B&H={bh_str}"
            )

            baseline_acc = float((y_te.values == y_tr.value_counts().idxmax()).mean())

            # Multi-coin-aware predictors (RL agents, pretrained fine_tuned)
            # need per-coin price series, not the stacked X. Push them now,
            # before fit() is called inside _fit_eval_loop. Pass X_te.columns
            # as the canonical column set so each coin's X is aligned to the
            # exact feature set the predictor will see at predict time
            # (otherwise _stack_with_target's column-intersection drops some
            # target columns and the RL predictor fits with N features but
            # predicts with N-k features — shape mismatch crash).
            if cfg.training.extra_coins and not skip_fit:
                # Multi-coin push only happens at fold 0 in train_once mode —
                # subsequent folds skip both fit() and the multi-coin data
                # update so the predictor keeps using its initial training set.
                self._push_multi_coin_data_to_predictors(
                    predictors, cfg,
                    target_X=target_pre_X, target_y=target_pre_y,
                    target_d=target_pre_d, target_df=target_pre_df,
                    fold_d_start=target_pre_d.iloc[0],
                    fold_d_end=target_pre_d.iloc[-1],
                    expected_cols=list(X_te.columns),
                    target_X_rl=target_pre_X_rl, target_y_rl=target_pre_y_rl,
                    target_d_rl=target_pre_d_rl, target_df_rl=target_pre_df_rl,
                    fold_d_end_rl=target_pre_d_rl.iloc[-1],
                )

            fold_results, fold_predictions = self._fit_eval_loop(
                predictors, X_tr, y_tr, d_tr, df_tr, X_te, y_te, d_te, df_te,
                skip_fit=skip_fit,
            )
            # Capture this fold's trained instances for end-of-CV importance query.
            last_fold_predictors = predictors
            last_fold_X_te = X_te
            last_fold_y_te = y_te

            # bh_gain_fold + closes were already computed for the fold header
            # above — reuse them here. Synth-gain monthly bucketing imports
            # land lazily here since not every fold path needs them.
            from .metrics import synth_gain_by_month
            closes_te_arr = _closes_te

            for r in fold_results:
                per_fold_rows.append({
                    "fold": fold_id, "predictor": r.name, "family": r.family,
                    "accuracy": r.accuracy, "kappa": r.kappa, "f1_macro": r.f1_macro,
                    "dir_kappa": r.dir_kappa,
                    "synth_gain": r.synth_gain,
                    "sharpe": r.sharpe,
                    "max_dd": r.max_dd,
                    "calmar": r.calmar,
                    "profit_factor": r.profit_factor,
                    "n_positive_months": r.n_positive_months,
                    "n_total_months": r.n_total_months,
                    "bh_gain": bh_gain_fold,
                    "n_test": r.n_test, "elapsed_sec": r.metadata.get("elapsed_sec", 0),
                    "test_start": str(d_te.iloc[0].date()),
                    "test_end": str(d_te.iloc[-1].date()),
                    "baseline_acc": baseline_acc,
                })

            # Per-month gain per predictor — written to a long-format CSV at
            # end of CV so users can audit when each strategy made/lost money.
            if closes_te_arr is not None:
                for r in fold_results:
                    preds = fold_predictions.get(r.name)
                    if preds is None or len(preds) != len(closes_te_arr):
                        continue
                    monthly = synth_gain_by_month(
                        closes_te_arr, np.asarray(preds), d_te.values,
                    )
                    for month, gain in monthly.items():
                        monthly_rows.append({
                            "fold": fold_id, "predictor": r.name, "family": r.family,
                            "month": month, "gain": float(gain),
                        })

            # Refresh FT ensembles' weights for the next fold — softmax of this
            # fold's per-base kappas. Cold ensembles ignore this hook.
            base_kappas = {
                r.name: r.kappa for r in fold_results
                if r.family != "ensemble" and not np.isnan(r.kappa)
            }
            for p in predictors:
                if getattr(p, "is_ensemble", False):
                    p.update_prior_kappas(base_kappas)

            # Single-best per-fold plot is deferred to AFTER the loop so we can
            # use the GLOBAL top-1 predictor (by --rank-by) on every fold,
            # consistent with the stitched plot.
            if fold_results and cfg.plots.enabled:
                # Always accumulate fold predictions when plots are enabled —
                # the stitched OOS plot uses this regardless of per_fold.
                # Previously this was nested inside the per_fold gate, which
                # silently disabled the stitched plot when per_fold=false.
                all_fold_preds.append({
                    "fold_id": fold_id,
                    "test_index": X_te.index,
                    "X_te": X_te,
                    "d_te": d_te,
                    "predictions": dict(fold_predictions),  # name -> ndarray
                    "fold_results": list(fold_results),
                })

                # Per-fold-only outputs: multi-overlay plot + per-fold feature
                # importance. Both gated by per_fold.
                if cfg.plots.per_fold:
                    if len(fold_predictions) > 1:
                        self._save_fold_plot_multi(
                            cfg, df, mode, fold_id, fold_predictions,
                            X_te, d_te,
                            xlim_dates=oos_span_dates,
                        )

                    if cfg.predictors.feature_importance:
                        fold_classical = sorted(
                            [r for r in fold_results
                             if r.family == "classical" and not np.isnan(r.kappa)],
                            key=lambda r: r.kappa, reverse=True,
                        )
                        for r in fold_classical:
                            cand = next((p for p in predictors if p.name == r.name), None)
                            if cand is not None and _has_native_importance(cand):
                                self._print_and_save_importance(
                                    cand, X_te, y_te,
                                    suffix=f"{mode}_fold{fold_id+1}",
                                )
                                break

        if not per_fold_rows:
            console.print("[red]No folds produced — check min_train_fraction / data length[/red]")
            return None

        per_fold_df = pd.DataFrame(per_fold_rows)
        per_fold_df["cv_mode"] = mode
        per_fold_df.to_csv(RESULTS_DIR / f"cv_{mode}_per_fold.csv", index=False)
        if monthly_rows:
            monthly_df = pd.DataFrame(monthly_rows)
            monthly_df["cv_mode"] = mode
            monthly_df.to_csv(RESULTS_DIR / f"cv_{mode}_monthly_gain.csv", index=False)

        # Stitched monthly consistency — compute n_positive / n_total over the
        # CONCATENATED OOS series per predictor, replacing the per-fold sum
        # which double-counts months that span fold boundaries (e.g. a 30-day
        # fold spanning May 28 → June 27 contributes BOTH May and June, then
        # the next fold contributes June again).
        from .metrics import (
            synth_gain_by_month, consistency_positive_months,
            sharpe_ratio, periods_per_year,
            synth_equity_curve, max_drawdown, profit_factor, calmar_ratio,
            avg_excess_ratio, time_above_bh,
        )
        stitched_consistency: dict[str, tuple[int, int]] = {}
        stitched_sharpe: dict[str, float] = {}
        stitched_calmar: dict[str, float] = {}
        stitched_pf: dict[str, float] = {}
        stitched_max_dd: dict[str, float] = {}
        stitched_avg_excess: dict[str, float] = {}
        stitched_time_above_bh: dict[str, float] = {}
        stitched_gain: dict[str, float] = {}
        if all_fold_preds:
            cost = float(cfg.predictors.evaluation_transaction_cost)
            ppy = periods_per_year(cfg.timeframe)
            stitched_long_only = (getattr(cfg, "market_type", "futures") == "spot")
            all_pred_names: set[str] = set()
            for fp in all_fold_preds:
                all_pred_names.update(fp["predictions"].keys())
            for pname in all_pred_names:
                preds_list, dates_list, closes_list = [], [], []
                for fp in all_fold_preds:
                    if pname not in fp["predictions"]:
                        continue
                    preds_list.append(np.asarray(fp["predictions"][pname]))
                    dates_list.append(np.asarray(fp["d_te"].values))
                    if "close" in df.columns:
                        cl = df.loc[fp["test_index"], "close"].to_numpy(dtype=np.float64)
                        closes_list.append(cl)
                if not preds_list or not closes_list:
                    continue
                preds_concat = np.concatenate(preds_list)
                dates_concat = np.concatenate(dates_list)
                closes_concat = np.concatenate(closes_list)
                if len(preds_concat) != len(closes_concat):
                    continue
                monthly = synth_gain_by_month(
                    closes_concat, preds_concat, dates_concat,
                    transaction_cost=cost, long_only=stitched_long_only,
                )
                stitched_consistency[pname] = consistency_positive_months(monthly)
                # Stitched Sharpe: ONE estimate over all concatenated OOS bars
                # (vs mean(per-fold Sharpe) which is biased and noisy on short
                # folds). Same definition: mean(strategy_log_ret) / std × √ppy
                # but computed once on N_folds × test_window_bars samples
                # instead of averaged over short windows.
                stitched_sharpe[pname] = sharpe_ratio(
                    closes_concat, preds_concat,
                    periods_per_year=ppy, transaction_cost=cost,
                    long_only=stitched_long_only,
                )
                # Stitched DD over the full concatenated equity curve. Used
                # for Calmar AND replaces the maxDD column (which was the
                # worst PER-FOLD DD — different quantity, made the table
                # internally inconsistent: a strategy could show maxDD=-14%
                # while Calmar = gain/|stitched_dd| reflected -17%).
                eq_concat, gain_concat = synth_equity_curve(
                    closes_concat, preds_concat,
                    transaction_cost=cost, long_only=stitched_long_only,
                )
                dd_concat = max_drawdown(eq_concat)
                stitched_max_dd[pname] = dd_concat
                stitched_calmar[pname] = calmar_ratio(gain_concat, dd_concat)
                # Total gain over the stitched OOS span. Per-fold compounded
                # gain (used previously) misses cross-fold boundary returns
                # and over-counts entry costs (one per fold), so it can show
                # opposite sign from PF / Sharpe on the same data. Use the
                # stitched value here so all aggregate metrics agree.
                stitched_gain[pname] = gain_concat
                # Stitched Profit Factor over all OOS bars.
                stitched_pf[pname] = profit_factor(
                    closes_concat, preds_concat,
                    transaction_cost=cost, long_only=stitched_long_only,
                )
                # Path-dependent outperformance vs B&H — captures the area
                # between strategy equity and the price curve, normalized
                # by deployment length, plus the fraction of bars spent
                # above B&H. Endpoint `gain_total` only sees the final bar.
                stitched_avg_excess[pname] = avg_excess_ratio(
                    closes_concat, preds_concat,
                    transaction_cost=cost, long_only=stitched_long_only,
                )
                stitched_time_above_bh[pname] = time_above_bh(
                    closes_concat, preds_concat,
                    transaction_cost=cost, long_only=stitched_long_only,
                )

        self._print_cv_summary(
            per_fold_df, mode, cfg=cfg,
            stitched_consistency=stitched_consistency,
            stitched_sharpe=stitched_sharpe,
            stitched_calmar=stitched_calmar,
            stitched_pf=stitched_pf,
            stitched_max_dd=stitched_max_dd,
            stitched_avg_excess=stitched_avg_excess,
            stitched_time_above_bh=stitched_time_above_bh,
            stitched_gain=stitched_gain,
        )

        # Stitched OOS synth equity: pick the predictor with the best MEAN
        # kappa across all folds, then concatenate ITS predictions over all
        # fold test windows. Realistic: a single model deployed across the
        # full OOS timeline (vs the per-fold-best which would require an
        # oracle to know which model to use when).
        # Stitched OOS plot is a once-per-CV summary, gated only by the
        # master 'enabled' flag (NOT per_fold — it survives --no-fold-plots).
        if cfg.plots.enabled and len(all_fold_preds) >= 2 and not per_fold_df.empty:
            # Rank ALL predictors and pick the top N for overlay on the stitched plot.
            rank_by = getattr(cfg.predictors, "rank_by", "kappa")
            if rank_by in ("gain", "vs_bh"):
                ranking = per_fold_df.groupby("predictor")["synth_gain"].apply(
                    lambda s: float(np.prod(1.0 + s.dropna().values) - 1.0)
                )
            else:
                ranking = per_fold_df.groupby("predictor")["kappa"].mean()
            top_n = 5
            top_predictors = ranking.sort_values(ascending=False).head(top_n)
            # Build folds_per_predictor: dict[predictor_name → list of fold dicts]
            folds_per_predictor: dict[str, list[dict]] = {}
            for pred_name in top_predictors.index:
                pred_folds = []
                for fp in all_fold_preds:
                    if pred_name not in fp["predictions"]:
                        continue
                    pred_folds.append({
                        "test_index": fp["test_index"],
                        "predictions": fp["predictions"][pred_name],
                    })
                if pred_folds:
                    folds_per_predictor[pred_name] = pred_folds
            try:
                from ..visualization.regime_plots import plot_stitched_oos_equity
                # Suffix uses the #1 predictor; n folds derived inside the plot.
                best_name = str(top_predictors.index[0])
                best_mean = float(top_predictors.iloc[0])
                plot_stitched_oos_equity(
                    df, folds_per_predictor,
                    PLOTS_DIR / f"B_stitched_oos_{mode}.png",
                    title_suffix=f"{cfg.target}-{cfg.timeframe}-{mode}-{rank_by}",
                    transaction_cost=cfg.predictors.evaluation_transaction_cost,
                    long_only=stitched_long_only,
                )
                rank_str = {"kappa": "mean κ", "dir_kappa": "dir-κ", "gain": "gain_total", "vs_bh": "vs_BH"}.get(rank_by, "mean κ")
                top_summary = ", ".join(
                    f"{n}={v:+.3f}" for n, v in top_predictors.items()
                )
                console.print(
                    f"[dim]Stitched OOS plot: B_stitched_oos_{mode}.png "
                    f"(top {len(folds_per_predictor)} by {rank_str}: {top_summary})[/dim]"
                )
            except Exception as e:
                console.print(f"[yellow]Stitched plot failed: {e}[/yellow]")

            # ---- Deferred per-fold single-best plot ----
            # Use the GLOBAL top-1 predictor (by --rank-by) on EVERY fold so
            # the per-fold A/B plots are consistent across folds and aligned
            # with the stitched plot's #1 line.
            if cfg.plots.enabled and cfg.plots.per_fold:
                global_best_name = str(top_predictors.index[0])
                for fp in all_fold_preds:
                    if global_best_name not in fp["predictions"]:
                        continue
                    fold_results_for_fp = fp.get("fold_results", [])
                    fold_best_obj = next(
                        (r for r in fold_results_for_fp if r.name == global_best_name),
                        None,
                    )
                    if fold_best_obj is None:
                        continue
                    self._save_fold_plot(
                        cfg, df, mode, fp["fold_id"], fold_best_obj,
                        fp["X_te"], fp["d_te"], fp["predictions"][global_best_name],
                        xlim_dates=oos_span_dates,
                    )

        # Feature importance for the best CLASSICAL predictor with native
        # importance (computed on the last fold's test set — that fold has
        # the largest train window in walk-forward). Skip NNs.
        if (
            cfg.predictors.feature_importance
            and last_fold_predictors is not None
            and last_fold_X_te is not None
        ):
            classical_ranked = (
                per_fold_df[per_fold_df["family"] == "classical"]
                .groupby("predictor")["kappa"].mean()
                .sort_values(ascending=False)
            )
            for cand_name in classical_ranked.index:
                cand = next(
                    (p for p in last_fold_predictors if p.name == cand_name),
                    None,
                )
                if cand is not None and _has_native_importance(cand):
                    self._print_and_save_importance(
                        cand, last_fold_X_te, last_fold_y_te
                    )
                    break

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

    def _print_cv_summary(
        self,
        per_fold: pd.DataFrame,
        mode: str,
        cfg: RunConfig | None = None,
        stitched_consistency: dict[str, tuple[int, int]] | None = None,
        stitched_sharpe: dict[str, float] | None = None,
        stitched_calmar: dict[str, float] | None = None,
        stitched_pf: dict[str, float] | None = None,
        stitched_max_dd: dict[str, float] | None = None,
        stitched_avg_excess: dict[str, float] | None = None,
        stitched_time_above_bh: dict[str, float] | None = None,
        stitched_gain: dict[str, float] | None = None,
    ) -> None:
        from .metrics import compound_returns

        def _compound(series: pd.Series) -> float:
            return compound_returns(series.values)

        agg = per_fold.groupby(["predictor", "family"]).agg(
            kappa_mean=("kappa", "mean"),
            kappa_std=("kappa", "std"),
            dir_kappa_mean=("dir_kappa", "mean"),
            acc_mean=("accuracy", "mean"),
            f1_mean=("f1_macro", "mean"),
            gain_mean=("synth_gain", "mean"),
            gain_std=("synth_gain", "std"),
            gain_total=("synth_gain", _compound),
            sharpe_mean=("sharpe", "mean"),
            max_dd_min=("max_dd", "min"),  # min of negative DDs = worst DD
            calmar_mean=("calmar", "mean"),
            pf_mean=("profit_factor", "mean"),
            n_pos_months_sum=("n_positive_months", "sum"),
            n_total_months_sum=("n_total_months", "sum"),
            n_folds=("fold", "count"),
        ).reset_index()

        # Replace per-fold-summed month counts (which double-count months
        # spanning fold boundaries) with stitched-OOS unique month counts.
        if stitched_consistency:
            def _stitched_pos(name): return stitched_consistency.get(name, (0, 0))[0]
            def _stitched_tot(name): return stitched_consistency.get(name, (0, 0))[1]
            agg["n_pos_months_sum"] = agg["predictor"].map(_stitched_pos)
            agg["n_total_months_sum"] = agg["predictor"].map(_stitched_tot)

        # Replace mean-of-per-fold Sharpe (noisy + biased upward by short
        # window annualization) with the stitched-OOS Sharpe — one mean/std
        # estimate over the full concatenated test slice. The "true" Sharpe
        # the strategy would deliver if deployed across all fold windows.
        if stitched_sharpe:
            agg["sharpe_mean"] = agg["predictor"].map(
                lambda n: stitched_sharpe.get(n, float("nan"))
            )
        if stitched_calmar:
            agg["calmar_mean"] = agg["predictor"].map(
                lambda n: stitched_calmar.get(n, float("nan"))
            )
        if stitched_pf:
            agg["pf_mean"] = agg["predictor"].map(
                lambda n: stitched_pf.get(n, float("nan"))
            )
        if stitched_max_dd:
            agg["max_dd_min"] = agg["predictor"].map(
                lambda n: stitched_max_dd.get(n, float("nan"))
            )
        # Path-dependent outperformance vs B&H. NaN-fill rather than
        # column-skip so the table layout stays consistent.
        agg["avg_excess"] = agg["predictor"].map(
            lambda n: (stitched_avg_excess or {}).get(n, float("nan"))
        )
        agg["time_above_bh"] = agg["predictor"].map(
            lambda n: (stitched_time_above_bh or {}).get(n, float("nan"))
        )
        # Override gain_total with the stitched value so all aggregate
        # metrics (gain_total / Sharpe / Calmar / PF / maxDD / avg_exc / t>BH)
        # come from the same source. Per-fold compounded gain is biased
        # vs the others (misses cross-fold boundary returns, over-counts
        # entry costs) and could show opposite sign from PF on the same
        # row — see the LinearQ-3 anomaly: gain_total=+24%, PF=0.96.
        if stitched_gain:
            agg["gain_total"] = agg["predictor"].map(
                lambda n: stitched_gain.get(n, float("nan"))
            )

        # Buy-and-hold reference: same per fold, so de-dup by fold first.
        bh_per_fold = per_fold.drop_duplicates("fold").set_index("fold").get(
            "bh_gain", pd.Series(dtype=float)
        )
        bh_total = compound_returns(bh_per_fold.values) if len(bh_per_fold) else float("nan")
        bh_mean = float(bh_per_fold.mean()) if len(bh_per_fold) else float("nan")

        agg["gain_vs_bh"] = agg["gain_total"] - bh_total
        agg["cv_mode"] = mode

        # Sort by user-chosen criterion. Gain / vs_BH ranks differently from
        # κ when predictors are right on high-magnitude bars but noisy on
        # low-magnitude bars (common in strongly-directional markets).
        cfg_eff = cfg if cfg is not None else self.cfg
        rank_by = getattr(cfg_eff.predictors, "rank_by", "kappa")
        sort_col = {
            "kappa": "kappa_mean",
            "dir_kappa": "dir_kappa_mean",
            "gain": "gain_total",
            "vs_bh": "gain_vs_bh",
        }.get(rank_by, "kappa_mean")
        agg = agg.sort_values(sort_col, ascending=False)

        agg.to_csv(RESULTS_DIR / f"cv_{mode}_aggregated.csv", index=False)

        bh_std = float(bh_per_fold.std()) if len(bh_per_fold) > 1 else float("nan")
        rank_label = {
            "kappa": "mean κ", "dir_kappa": "dir-κ", "gain": "gain_total", "vs_bh": "vs_BH",
        }.get(rank_by, "mean κ")
        title = f"{mode.replace('_', '-')} aggregate — sorted by {rank_label} (B&H total = {bh_total*100:+.1f}%)"
        table = Table(title=title)
        for col in (
            "predictor", "family", "κ_mean", "κ_std", "dir-κ", "acc", "F1",
            "gain_mean", "gain_std", "gain_total",
            "Sharpe", "Calmar", "PF", "maxDD",
            "avg_exc", "t≥BH",
            "+m", "n_folds",
        ):
            table.add_column(col, justify="right" if col not in ("predictor", "family") else "left")

        # B&H reference row at the top.
        bh_std_str = f"[dim]{bh_std*100:.1f}%[/dim]" if not np.isnan(bh_std) else "[dim]--[/dim]"
        table.add_row(
            "[dim]Buy & Hold[/dim]", "[dim]reference[/dim]",
            "--", "--", "--", "--", "--",
            f"[dim]{bh_mean*100:+.1f}%[/dim]",
            bh_std_str,
            f"[dim]{bh_total*100:+.1f}%[/dim]",
            "[dim]--[/dim]", "[dim]--[/dim]", "[dim]--[/dim]", "[dim]--[/dim]",
            "[dim]0.0%[/dim]", "[dim]50.0%[/dim]",
            "[dim]--[/dim]",
            f"[dim]{int(len(bh_per_fold))}[/dim]",
        )

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
            dk = float(r["dir_kappa_mean"]) if not pd.isna(r["dir_kappa_mean"]) else float("nan")
            dk_str = f"{dk:+.3f}" if not np.isnan(dk) else "--"
            gain_total = float(r["gain_total"])
            # Color gain_total: green if beats B&H by 5%+, yellow if positive
            # but below B&H, red if negative. Using vs_bh as the secondary
            # cue keeps the table interpretable on bull/bear backdrops.
            gt_vs_bh = gain_total - bh_total
            if gain_total > 0 and gt_vs_bh > 0.05:
                gt_color = "green"
            elif gain_total > 0:
                gt_color = "yellow"
            else:
                gt_color = "red"
            gain_std_v = float(r["gain_std"]) if not pd.isna(r["gain_std"]) else float("nan")
            gstd_str = f"{gain_std_v*100:.1f}%" if not np.isnan(gain_std_v) else "--"
            sharpe = float(r["sharpe_mean"]) if not pd.isna(r["sharpe_mean"]) else float("nan")
            sh_str = f"{sharpe:+.2f}" if not np.isnan(sharpe) else "--"
            sh_color = "green" if sharpe > 1.0 else ("yellow" if sharpe > 0 else "red")
            calmar = float(r["calmar_mean"]) if not pd.isna(r["calmar_mean"]) else float("nan")
            calmar_str = f"{calmar:+.2f}" if not np.isnan(calmar) else "--"
            calmar_color = "green" if calmar > 1.0 else ("yellow" if calmar > 0 else "red")
            pf_v = float(r["pf_mean"]) if not pd.isna(r["pf_mean"]) else float("nan")
            if np.isinf(pf_v):
                pf_str = "∞"
                pf_color = "green"
            elif np.isnan(pf_v):
                pf_str = "--"
                pf_color = ""
            else:
                pf_str = f"{pf_v:.2f}"
                pf_color = "green" if pf_v > 1.5 else ("yellow" if pf_v > 1.0 else "red")
            max_dd = float(r["max_dd_min"]) if not pd.isna(r["max_dd_min"]) else float("nan")
            dd_str = f"{max_dd*100:+.1f}%" if not np.isnan(max_dd) else "--"
            avg_exc = float(r["avg_excess"]) if not pd.isna(r["avg_excess"]) else float("nan")
            avg_exc_str = f"{avg_exc*100:+.1f}%" if not np.isnan(avg_exc) else "--"
            avg_exc_color = "green" if avg_exc > 0.10 else ("yellow" if avg_exc > 0 else "red")
            t_above = float(r["time_above_bh"]) if not pd.isna(r["time_above_bh"]) else float("nan")
            t_above_str = f"{t_above*100:.1f}%" if not np.isnan(t_above) else "--"
            t_above_color = "green" if t_above > 0.6 else ("yellow" if t_above > 0.5 else "red")
            n_pos = int(r["n_pos_months_sum"]) if not pd.isna(r["n_pos_months_sum"]) else 0
            n_tot = int(r["n_total_months_sum"]) if not pd.isna(r["n_total_months_sum"]) else 0
            cm_str = f"{n_pos}/{n_tot}" if n_tot else "0/0"
            table.add_row(
                str(r["predictor"]), str(r["family"]),
                kstr, f"{r['kappa_std']:.3f}",
                dk_str,
                f"{r['acc_mean']:.3f}", f"{r['f1_mean']:.3f}",
                f"{r['gain_mean']*100:+.1f}%",
                gstd_str,
                f"[{gt_color}]{gain_total*100:+.1f}%[/{gt_color}]",
                f"[{sh_color}]{sh_str}[/{sh_color}]" if not np.isnan(sharpe) else "--",
                f"[{calmar_color}]{calmar_str}[/{calmar_color}]" if not np.isnan(calmar) else "--",
                f"[{pf_color}]{pf_str}[/{pf_color}]" if pf_color else pf_str,
                dd_str,
                f"[{avg_exc_color}]{avg_exc_str}[/{avg_exc_color}]" if not np.isnan(avg_exc) else "--",
                f"[{t_above_color}]{t_above_str}[/{t_above_color}]" if not np.isnan(t_above) else "--",
                cm_str,
                str(int(r["n_folds"])),
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

    def _print_summary(self, baseline_acc: float, bh_gain: float = float("nan")) -> None:
        if not self.results:
            console.print("[yellow]No predictor results.[/yellow]")
            return
        ordered = sorted(self.results, key=lambda r: -(r.kappa if not np.isnan(r.kappa) else -2))
        bh_str = f" (B&H = {bh_gain*100:+.1f}%)" if not np.isnan(bh_gain) else ""
        table = Table(title=f"Predictor benchmark — sorted by κ{bh_str}")
        table.add_column("rank", style="dim", justify="right")
        table.add_column("predictor", style="bold")
        table.add_column("family")
        table.add_column("acc", justify="right")
        table.add_column("κ", justify="right")
        table.add_column("dir-κ", justify="right")
        table.add_column("F1", justify="right")
        table.add_column("gain", justify="right")
        table.add_column("vs_BH", justify="right")
        table.add_column("n_test", justify="right")
        table.add_column("elapsed", justify="right")
        # baseline row
        table.add_row("--", "[dim]baseline[/dim]", "--",
                      f"{baseline_acc:.3f}", "0.000", "--", "--", "--", "--", "--", "--")
        # buy-and-hold reference
        if not np.isnan(bh_gain):
            table.add_row("--", "[dim]Buy & Hold[/dim]", "[dim]reference[/dim]",
                          "--", "--", "--", "--",
                          f"[dim]{bh_gain*100:+.1f}%[/dim]", "[dim]0.0%[/dim]",
                          "--", "--")
        for i, r in enumerate(ordered, 1):
            color = ""
            if r.kappa > 0.4:
                color = "green"
            elif r.kappa > 0.2:
                color = "yellow"
            elif r.kappa < 0:
                color = "red"
            kappa_str = f"[{color}]{r.kappa:+.3f}[/{color}]" if color else f"{r.kappa:+.3f}"
            dk_str = f"{r.dir_kappa:+.3f}" if not np.isnan(r.dir_kappa) else "--"
            gain_str = f"{r.synth_gain*100:+.1f}%" if not np.isnan(r.synth_gain) else "--"
            if not np.isnan(r.synth_gain) and not np.isnan(bh_gain):
                vs_bh = r.synth_gain - bh_gain
                vs_bh_color = "green" if vs_bh > 0 else "red"
                vs_bh_str = f"[{vs_bh_color}]{vs_bh*100:+.1f}%[/{vs_bh_color}]"
            else:
                vs_bh_str = "--"
            table.add_row(
                str(i), r.name, r.family,
                f"{r.accuracy:.3f}",
                kappa_str,
                dk_str,
                f"{r.f1_macro:.3f}",
                gain_str, vs_bh_str,
                str(r.n_test),
                f"{r.metadata.get('elapsed_sec', 0):.1f}s",
            )
        console.print(table)

    # -------------------------------------------------------------------
    # Multi-coin training data
    # -------------------------------------------------------------------
    def _push_multi_coin_data_to_predictors(
        self,
        predictors: list[BasePredictor],
        cfg: RunConfig,
        target_X: pd.DataFrame,
        target_y: pd.Series,
        target_d: pd.Series,
        target_df: pd.DataFrame,
        fold_d_start,
        fold_d_end,
        expected_cols: list | None = None,
        target_X_rl: pd.DataFrame | None = None,
        target_y_rl: pd.Series | None = None,
        target_d_rl: pd.Series | None = None,
        target_df_rl: pd.DataFrame | None = None,
        fold_d_end_rl=None,
    ) -> None:
        """Build per-coin (X, y, dates, df) views clipped to the fold's date
        range and push them onto every multi-coin-aware predictor in the list.
        Predictors that don't opt in (no `is_multi_coin_aware` attr) are skipped.

        Filters extras to the same date window as the target's fold so the
        multi-coin training stays consistent with the rolling-window
        philosophy (constant timespan across all coins).

        expected_cols (optional): canonical column set used at predict time.
        When set, every coin's X is reindexed to this column set so RL
        predictors fit and predict on the same feature dimension.

        target_X_rl / .._rl args (optional): when provided, RL-family
        predictors receive these extended slices instead of the default
        target_X. Lets RL skip the global L_range purge gap (no label
        leakage at RL's 1-bar reward horizon).
        """
        def _align_coin_X(X: pd.DataFrame, coin_name: str) -> pd.DataFrame:
            if expected_cols is None:
                return X
            X = X.reindex(columns=expected_cols, fill_value=0.0)
            ccol = f"coin_{coin_name}"
            if ccol in X.columns:
                X[ccol] = 1.0
            return X

        def _build(target_X, target_y, target_d, target_df, end_clip):
            """Build (target_data, extras_data) clipped to [fold_d_start, end_clip]."""
            tdata = {
                "X": _align_coin_X(target_X, cfg.target),
                "y": target_y, "dates": target_d, "df": target_df,
            }
            extras: dict[str, dict] = {}
            for coin in cfg.training.extra_coins:
                cached = self._extra_coin_cache.get(coin.target)
                if cached is None:
                    continue
                Xc, yc, dc, dfc = cached
                keep = (dc >= fold_d_start) & (dc <= end_clip)
                if not keep.any():
                    continue
                Xc_w = Xc.loc[keep].reset_index(drop=True)
                yc_w = yc.loc[keep].reset_index(drop=True)
                dc_w = dc.loc[keep].reset_index(drop=True)
                df_keep = (dfc["date"] >= fold_d_start) & (dfc["date"] <= end_clip)
                dfc_w = dfc.loc[df_keep].reset_index(drop=True)
                if len(Xc_w) >= 2 and len(dfc_w) >= 2 and len(Xc_w) == len(dfc_w):
                    extras[coin.target] = {
                        "X": _align_coin_X(Xc_w, coin.target),
                        "y": yc_w, "dates": dc_w, "df": dfc_w,
                    }
            return tdata, extras

        target_data, extras_data = _build(target_X, target_y, target_d, target_df, fold_d_end)

        # RL-specific dataset (extended end clip) — built only if all RL slices
        # were supplied. Otherwise fall back to the default dataset.
        rl_supplied = (
            target_X_rl is not None and target_y_rl is not None
            and target_d_rl is not None and target_df_rl is not None
            and fold_d_end_rl is not None
        )
        if rl_supplied:
            target_data_rl, extras_data_rl = _build(
                target_X_rl, target_y_rl, target_d_rl, target_df_rl, fold_d_end_rl,
            )
        else:
            target_data_rl, extras_data_rl = target_data, extras_data

        # Push, routing per-family.
        for p in predictors:
            if not getattr(p, "is_multi_coin_aware", False):
                continue
            if getattr(p, "family", "") == "rl":
                p.set_multi_coin_data(target_data_rl, extras_data_rl)
            else:
                p.set_multi_coin_data(target_data, extras_data)

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
        market_type = coin.market_type or getattr(cfg, "market_type", "futures")

        root = DataRoot(
            data_root=cfg.data_root,
            venue=venue,
            target=coin.target,
            quote=quote,
            settle=settle,
            timeframe=timeframe,
            market_type=market_type,
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
            cross_ohlcv_path=(
                root.cross_ohlcv()
                if root.cross_ohlcv() is not None and root.cross_ohlcv().exists()
                else None
            ),
            cross_name=root.cross_name() or "cross",
            extra_cross_paths=_resolve_extra_cross_paths(cfg, root_override=root),
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
                cached = self._extra_coin_cache.get(coin.target)
                if cached is None:
                    console.print(f"[cyan]  loading extra coin: {coin.target}[/cyan]")
                    Xc, yc, dc, dfc = self._build_coin_data(cfg, coin)
                    self._extra_coin_cache[coin.target] = (Xc, yc, dc, dfc)
                else:
                    console.print(f"[dim]  cached extra coin: {coin.target}[/dim]")
                    Xc, yc, dc, dfc = cached
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
            # Tag the new one-hot columns as coin_id sources
            for c in ohe.columns:
                self.feature_sources[c] = "coin_id"
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
        xlim_dates: tuple | None = None,
    ) -> None:
        """Save multi-classifier plots for one CV fold (B-multi synth equity overlay,
        A-multi step panel). xlim_dates: optional (start, end) to widen X-axis to
        the full OOS span across all folds (lets each fold's plot show its
        position in the broader CV timeline)."""
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
                xlim_dates=xlim_dates,
                transaction_cost=cfg.predictors.evaluation_transaction_cost,
                long_only=(getattr(cfg, "market_type", "futures") == "spot"),
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
        xlim_dates: tuple | None = None,
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
            denoise_labels, _compute_runs, _plot_A, _plot_B,
        )
        smooth = denoise_labels(out, window=168)   # 7 days at 1h TF
        runs = _compute_runs(df, smooth)
        suffix = f"pred-{best.name}-{cfg.target}-{cfg.timeframe}-{mode}-fold{fold_id+1}"
        out_dir = PLOTS_DIR
        split_dt = d_te.iloc[0]
        try:
            # When xlim_dates is supplied (full OOS span), it overrides auto-zoom.
            _plot_A(df, runs, out_dir / f"A_pred_{mode}_fold{fold_id+1}.png", suffix,
                    split_dt, labels=out, xlim_dates=xlim_dates)
            # _plot_B takes (raw_labels, smooth, runs) — raw drives the equity
            # curve (matches synth_gain metric), smooth drives the regime bands.
            _plot_B(df, out, smooth, runs, out_dir / f"B_pred_{mode}_fold{fold_id+1}.png",
                    suffix, split_dt, xlim_dates=xlim_dates,
                    transaction_cost=cfg.predictors.evaluation_transaction_cost,
                    long_only=(getattr(cfg, "market_type", "futures") == "spot"))
            console.print(
                f"      [dim]plots saved: A/B_pred_{mode}_fold{fold_id+1}.png "
                f"(predictor={best.name}, κ={best.kappa:+.3f})[/dim]"
            )
        except Exception as e:
            console.print(f"      [yellow]plot save failed: {e}[/yellow]")

    def _print_and_save_importance(
        self, predictor: BasePredictor, X_te, y_te, suffix: str = "",
    ) -> None:
        try:
            with console.status(f"[cyan]Computing feature importance ({predictor.name})..."):
                imp = predictor.feature_importances(X_te, y_te, n_repeats=3)
        except Exception as e:
            console.print(f"[yellow]feature_importances skipped ({predictor.name}): {e}[/yellow]")
            return
        if imp is None or len(imp) == 0:
            console.print(f"[dim]feature_importances unavailable for {predictor.name}.[/dim]")
            return

        # Resolve source group for each feature (technical / external / signals
        # / coin_id / unknown).
        def _src(name: str) -> str:
            return self.feature_sources.get(name, "unknown")

        # Save full ranking with source column
        sfx = f"_{suffix}" if suffix else ""
        out_csv = RESULTS_DIR / f"feature_importance_{predictor.name}{sfx}.csv"
        imp_df = imp.reset_index().rename(columns={"index": "feature"})
        imp_df.columns = ["feature", "importance"]
        imp_df["source"] = imp_df["feature"].map(_src)
        imp_df = imp_df[["feature", "source", "importance"]]
        imp_df.to_csv(out_csv, index=False)

        top = imp.head(20)
        max_imp = float(top.iloc[0]) if len(top) and top.iloc[0] != 0 else 1.0
        title_sfx = f" ({suffix})" if suffix else ""
        table = Table(title=f"Top 20 features — {predictor.name}{title_sfx}")
        table.add_column("rank", justify="right", style="dim")
        table.add_column("feature")
        table.add_column("source", style="cyan")
        table.add_column("importance", justify="right")
        table.add_column("bar")
        for rank, (feat, val) in enumerate(top.items(), 1):
            bar_len = int(round(20 * float(val) / max_imp)) if max_imp > 0 else 0
            bar = "█" * max(bar_len, 0)
            table.add_row(str(rank), str(feat), _src(str(feat)), f"{val:.4f}", bar)
        console.print(table)

        # Per-source aggregation (sum of importance) — top-of-table summary
        if len(imp_df) > 0:
            agg = (
                imp_df.groupby("source")["importance"].sum()
                .sort_values(ascending=False)
            )
            total = float(agg.sum()) or 1.0
            agg_table = Table(title=f"Importance by source — {predictor.name}{title_sfx}")
            agg_table.add_column("source", style="cyan")
            agg_table.add_column("sum", justify="right")
            agg_table.add_column("share", justify="right")
            for src, val in agg.items():
                agg_table.add_row(str(src), f"{float(val):.4f}", f"{100*float(val)/total:5.1f}%")
            console.print(agg_table)

        console.print(f"[dim]Full ranking → {out_csv}[/dim]")

    def _save_summary_csv(self) -> None:
        df = pd.DataFrame([
            {
                "name": r.name, "family": r.family,
                "accuracy": r.accuracy, "kappa": r.kappa, "f1_macro": r.f1_macro,
                "dir_kappa": r.dir_kappa,
                "synth_gain": r.synth_gain,
                "n_test": r.n_test,
                "elapsed_sec": r.metadata.get("elapsed_sec", 0),
            } for r in self.results
        ])
        df.to_csv(RESULTS_DIR / "predictor_summary.csv", index=False)
