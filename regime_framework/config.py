"""Central configuration with YAML-loadable presets.

A `RunConfig` describes one full benchmark run: which target asset, which
TF, which paths, which labelling method, which feature sets, which predictors.

Presets live in `configs/presets/*.yaml` — one preset per (asset, venue, TF).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge — override wins on leaf values, dicts merge recursively."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR = REPO_ROOT / "plots"


@dataclass
class DataPaths:
    """Filesystem paths used by data loaders.

    Two ways to populate this:
      1. Explicit paths (each field set manually)
      2. Auto-resolved from a DataRoot — set `data_root` + venue/asset fields
         in the YAML and the resolver fills the rest.
    """
    ohlcv: Path
    funding: Path | None = None
    cross_ohlcv: Path | None = None
    cross_name: str = "cross"
    external_dir: Path | None = None

    @classmethod
    def from_data_root(
        cls,
        data_root: str | Path,
        venue: str,
        target: str,
        quote: str,
        settle: str,
        timeframe: str,
        cross_target: str | None = None,
        cross_quote: str | None = None,
        cross_settle: str | None = None,
    ) -> "DataPaths":
        from .data.conventions import DataRoot
        root = DataRoot(
            data_root=Path(data_root).expanduser(),
            venue=venue, target=target, quote=quote, settle=settle,
            timeframe=timeframe,
            cross_target=cross_target,
            cross_quote=cross_quote,
            cross_settle=cross_settle,
        )
        return cls(
            ohlcv=root.ohlcv(),
            funding=root.funding(),
            cross_ohlcv=root.cross_ohlcv(),
            cross_name=root.cross_name(),
            external_dir=root.external_dir(),
        )


@dataclass
class LabelConfig:
    method: str = "trend_scan"           # "trend_scan" | "triple_barrier" | "drawdown"
    L_range: list[int] = field(default_factory=lambda: [72, 120, 168, 240, 336, 480, 720, 1080])
    t_threshold: float = 0.0             # 0 = pure sign (binary bull/bear)
    # trend_scan hysteresis (0 = disabled)
    hysteresis_bars: int = 0
    strong_threshold: float = 2.0
    # triple_barrier extras (unused for trend_scan)
    # horizon: max forward bars to look for a barrier hit (timeout if none).
    # tp_mult/sl_mult: barrier distance from entry in σ units. σ is the rolling
    #   std of log-returns over vol_lookback bars. Asymmetric (tp != sl) =
    #   different risk/reward profile.
    horizon: int = 720                   # 30d at 1h
    tp_mult: float = 2.0                 # take-profit barrier in σ units
    sl_mult: float | None = None         # stop-loss; None = same as tp_mult
    vol_lookback: int = 168              # 1 week at 1h
    # On vertical-barrier timeout (no tp/sl hit), how to resolve:
    #   "sign" (default): label = sign(close[t+horizon] - close[t])  — canonical
    #   "drop": leave bar unlabelled (excluded from training)
    timeout_label: str = "sign"
    # Backward-compat alias (older configs used 'alpha' for symmetric barriers).
    alpha: float | None = None


@dataclass
class FeatureConfig:
    # Categories — each is independently togglable via its `use_*` flag.
    # technical:      pure OHLCV-derived (returns, vol, ATR, EMA, ADX,
    #                 GK vol, asymmetric vol, HTF agreement, breakout, etc.)
    # external:       cross-asset OHLCV + macro (FNG, ETF, DXY, VIX)
    # funding:        funding-rate features (target / BTC / ETH funding)
    # regime_signals: synthesized 3-state regime classifiers
    #                 (BTC trend, vol, drawdown — mirror YAML's regime gates)
    # trading_signals: binary 0/1 columns from a CCA-format signals YAML
    use_technical: bool = True
    use_external: bool = True
    use_funding: bool = True
    use_regime_signals: bool = True
    use_trading_signals: bool = False
    trading_signals_yaml: Path | None = None
    drop_nan_rows: bool = True


@dataclass
class SplitConfig:
    train_fraction: float = 0.80
    purge_bars: int | None = None        # if None, defaults to max(L_range)
    cv_folds: int = 0                    # 0 = single split; >0 = K-fold CV
    cv_mode: str = "walk_forward"        # "walk_forward" (expanding) | "leave_one_out"
                                         # | "rolling" (fixed-window sliding) | "both"
    min_train_fraction: float = 0.40     # walk-forward: fold-0 train size
    # Backward-compat alias (older configs may still use this)
    walk_forward_folds: int = 0
    # Rolling-window CV (cv_mode = "rolling"). Train on a fixed-size past
    # window, test on the next test_window_bars, slide forward by step_bars.
    # Bars depend on timeframe — at 1h: 1 month ≈ 730, 6 months ≈ 4380.
    # cv_folds is ignored in rolling mode (fold count = how many fit between).
    train_window_bars: int = 0           # 0 = required for rolling mode
    test_window_bars: int = 0            # 0 = required for rolling mode
    step_bars: int = 0                   # 0 = defaults to test_window_bars
    # Per-family purge override. Default global purge_bars=max(L_range) protects
    # classical/transformer predictors from forward-label leakage (trend_scan
    # looks ahead L bars to label bar t). RL only learns from (state, reward)
    # where reward is t→t+1 log return — no L-bar lookahead, so the big purge
    # wastes recent data. Set rl_purge_bars=1 (default) to use a minimal
    # 1-bar purge for RL training data only. None = use global purge_bars.
    rl_purge_bars: int | None = 1
    # Refit cadence. 1 (default) = refit each fold (per-fold-retrain baseline).
    # N > 1 = refit every N folds, reuse the model on the in-between folds —
    # e.g. retrain_every=3 with monthly folds = quarterly retrain. 0 = fit
    # only at fold 0 and never again ("train once and deploy"). Lets you find
    # the right cadence between "always retrain" (noisy adaptation) and
    # "never retrain" (stable but stale). Multi-coin data is also only
    # pushed on the folds that actually train.
    retrain_every: int = 1


@dataclass
class ExtraCoinSpec:
    """One additional coin to include in training data only.

    Tests still happen on the target coin. Each extra coin contributes its
    own (features, labels) computed independently — same FeaturePipeline,
    different OHLCV/funding/cross paths. A `coin_id` one-hot feature is added
    so the classifier can learn coin-specific biases.

    Defaults follow the main target's venue/quote/settle/timeframe when None.
    """
    target: str
    venue: str | None = None
    quote: str | None = None
    settle: str | None = None
    timeframe: str | None = None


@dataclass
class TrainingConfig:
    extra_coins: list[ExtraCoinSpec] = field(default_factory=list)
    add_coin_id_feature: bool = True


@dataclass
class RLConfig:
    """Hyperparameters for the rl predictor family. Each approximator can
    override the shared ones via its dedicated sub-block (nn / linear / lgb).
    """
    # Shared across approximators
    total_timesteps: int = 100000      # RL training budget per fold (default for all
                                       # approximators; override per-approximator below)
    # Per-side trading cost in log-return units. Default 5 bps matches Binance
    # futures taker (~0.05% per side; round-trip flip from +1 to -1 costs
    # 2 × 0.0005 = 10 bps). Set to 0 only for trade-cost-blind sanity runs;
    # otherwise the agent learns to flip aggressively on noise and the
    # reported synth_gain inflates vs realistic execution. Spot venues:
    # ~0.001. Always-maker (post-only) execution: ~0.0001.
    transaction_cost: float = 0.0005
    flat_threshold: float = 0.05       # continuous action → flat if |action| < this
    ft_steps_scale: float = 0.5        # FT mode: total_timesteps × this
    # Temperature for predict_proba's softmax over per-action Q-values. None
    # = auto-calibrate from std(Q) across the test slice (recommended). Set
    # a positive float to force sharper (small T) or flatter (large T) proba.
    # Affects ensembling: with raw Q-values softmax produces near one-hot
    # proba which collapses ConfidenceEnsemble to plain Ensemble.
    proba_temperature: float | None = None
    # Action-space variants to instantiate (omit any to skip those predictors)
    action_spaces: list[str] = field(default_factory=lambda: [
        "discrete-2", "discrete-3", "continuous"
    ])
    # Approximator-specific (not all keys apply to every approximator).
    # Each approximator may override total_timesteps (None = inherit the shared
    # value above). Useful when an expensive approximator needs less budget.
    nn_total_timesteps: int | None = None
    nn_learning_rate: float = 5e-4
    nn_buffer_size: int = 50000
    nn_gamma: float = 0.99
    nn_net_arch: list[int] = field(default_factory=lambda: [64, 32])
    nn_verbose: int = 0
    linear_total_timesteps: int | None = None
    linear_learning_rate: float = 1e-3
    linear_gamma: float = 0.99
    linear_epsilon_start: float = 1.0
    linear_epsilon_end: float = 0.05
    lgb_total_timesteps: int | None = None
    lgb_n_estimators: int = 200
    lgb_max_depth: int = 6
    lgb_learning_rate: float = 0.05
    lgb_gamma: float = 0.99
    lgb_iterations: int = 20
    xgb_total_timesteps: int | None = None
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_gamma: float = 0.99
    xgb_iterations: int = 20
    rf_total_timesteps: int | None = None
    rf_n_estimators: int = 200
    rf_max_depth: int | None = None      # None = unlimited (sklearn default)
    rf_min_samples_leaf: int = 1
    rf_gamma: float = 0.99
    rf_iterations: int = 20
    ridge_total_timesteps: int | None = None
    ridge_alpha: float = 1.0             # L2 regularization strength
    ridge_gamma: float = 0.99
    ridge_iterations: int = 20
    huber_total_timesteps: int | None = None
    # Bumped alpha from sklearn's 1e-4 default: stronger L2 makes the loss
    # strictly convex even when features are correlated, so lbfgs converges
    # in fewer iterations. Trade-off is slightly more bias, fine for FQI.
    huber_alpha: float = 1e-3
    huber_epsilon: float = 1.35
    huber_max_iter: int = 200          # lbfgs iterations (sklearn default)
    huber_tol: float = 1e-3            # looser than sklearn 1e-5
    huber_gamma: float = 0.99
    huber_iterations: int = 20
    histgb_total_timesteps: int | None = None
    histgb_max_iter: int = 200
    histgb_max_depth: int = 6
    histgb_learning_rate: float = 0.05
    histgb_min_samples_leaf: int = 20
    histgb_l2_regularization: float = 0.0
    histgb_gamma: float = 0.99
    histgb_iterations: int = 20
    cat_total_timesteps: int | None = None
    cat_n_estimators: int = 200
    cat_max_depth: int = 6
    cat_learning_rate: float = 0.05
    cat_l2_leaf_reg: float = 3.0
    cat_gamma: float = 0.99
    cat_iterations: int = 20


def _extract_predictor_kwargs(d: dict) -> dict:
    """Translate the YAML 'predictors' block into PredictorConfig kwargs.

    The 'rl' sub-block is wrapped into an RLConfig before instantiation.
    Other keys pass through unchanged.
    """
    if not d:
        return {}
    out = dict(d)
    if "rl" in out and isinstance(out["rl"], dict):
        out["rl"] = RLConfig(**out["rl"])
    return out


@dataclass
class PredictorConfig:
    families: list[str] = field(default_factory=lambda: [
        "classical", "rule_based", "deep_nets", "transformer", "pretrained"
    ])
    pretrained_models: list[str] = field(default_factory=lambda: [
        "chronos_bolt_base",
        "chronos_large",
        "timesfm",
        "moirai_large",
        "moirai_moe",
        "timemoe",
        "lag_llama",
        "toto",
    ])
    pretrained_modes: list[str] = field(default_factory=lambda: ["zero_shot", "fine_tuned"])
    fine_tune_head: str = "logreg"       # "logreg" | "mlp"
    forecast_horizon: int = 24           # zero-shot forecasting horizon
    # When True, also instantiate fine-tune variants (LightGBM-FT, XGBoost-FT,
    # MLP-FT, GRU-FT, LSTM-FT, TST-FT) that warm-start from the previous CV
    # fold's weights. Walk-forward only; auto-skipped under leave_one_out and
    # single-split (would leak / has no fold concept).
    include_finetune: bool = False
    # When True, auto-attach Ensemble + (if include_finetune) Ensemble-FT to
    # any run that has at least one probabilistic base family enabled
    # (classical / deep_nets / transformer). Independent of `families` list.
    include_ensemble: bool = True
    # Subset ensembles. Each entry produces Ensemble-{name} (cold) + (if
    # include_finetune) Ensemble-{name}-FT, voting only over the named bases.
    # Example:
    #   ensemble_groups:
    #     - name: trees
    #       bases: [RandomForest, ExtraTrees]
    #     - name: lr_trees
    #       bases: [LogReg, RandomForest, ExtraTrees]
    # `bases` matches the cold-variant base_name; FT versions are auto-paired.
    ensemble_groups: list[dict] = field(default_factory=list)
    # Final-name match-list of predictors to exclude from the run. Uses the
    # display name including any -FT / -name_suffix. Example:
    #   disabled: [MLP-FT, Ensemble-trees-FT]
    # Skips those predictors entirely; same family / FT defaults otherwise.
    disabled: list[str] = field(default_factory=list)
    # When True, ensemble predictors quantile-normalize each base's proba to
    # uniform[0.5, 1.0] in max(proba) before averaging. Equalizes vote
    # influence across heterogeneous families (e.g. classical's near-one-hot
    # proba vs RL's softer softmax). Per-group `normalize_proba` overrides
    # this default for individual ensemble_groups entries.
    ensemble_normalize_proba: bool = False
    # When False, skip per-fold and end-of-CV feature importance computation.
    # Useful when you only care about predictor metrics (κ / gain) and want
    # cleaner console output / faster runs (no permutation fallback overhead
    # on the rare classical predictor that lacks native importance).
    feature_importance: bool = True
    # Criterion used to sort the CV aggregate table and to select the
    # "best predictor" for the stitched OOS plot. One of:
    #   "kappa"  — Cohen's κ (default; classification skill)
    #   "gain"   — gain_total (compounded synth_gain across folds)
    #   "vs_bh"  — gain_total minus B&H gain (excess over buy-and-hold)
    # Useful when κ and gain disagree (common in strongly-directional markets
    # where "always long" beats κ-skilled predictors on synth_gain).
    rank_by: str = "kappa"
    # Cost deducted from synth_gain / Sharpe / monthly-gain at evaluation
    # time, on every position change (entry / exit / flip). Default matches
    # RL training cost (Binance futures taker, one side; round-trip flip =
    # 10 bps). Without this, evaluation reports pre-cost equity which won't
    # reproduce in live execution. Set 0 only for trade-cost-blind sanity.
    evaluation_transaction_cost: float = 0.0005
    # RL approximator hyperparams. See RLConfig for each field.
    rl: RLConfig = field(default_factory=RLConfig)


@dataclass
class PlotConfig:
    """Two-level plot output control:

    enabled=False kills ALL plots (label, prediction, fold, multi, stitched).
    Useful when you only want metrics (e.g. on a CI runner with no display).

    per_fold=False keeps the once-per-run plots (label plots, end-of-CV
    stitched OOS) but drops the per-fold plots (best-predictor plots and
    multi-overlay plots, which scale with fold count). With 50+ rolling
    folds, per-fold plots dominate the plots/ directory; disabling them
    keeps the summary visualizations without the clutter.
    """
    enabled: bool = True
    per_fold: bool = True


@dataclass
class RunConfig:
    target: str = "BTC"
    venue: str = "binance"
    quote: str = "USDT"
    settle: str = "USDT"
    timeframe: str = "1h"
    data_root: Path | None = None        # for resolving extra coin paths
    paths: DataPaths = field(default_factory=lambda: DataPaths(ohlcv=Path("/dev/null")))
    label: LabelConfig = field(default_factory=LabelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    predictors: PredictorConfig = field(default_factory=PredictorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    plots: PlotConfig = field(default_factory=PlotConfig)
    seed: int = 42

    @classmethod
    def from_preset(cls, preset_name: str) -> "RunConfig":
        """Load a preset YAML, deep-merged on top of configs/default.yaml.

        Presets only need to specify what differs from defaults (typically
        `target`, `venue`, `quote`, `settle`, `timeframe`, `data_root`).
        """
        preset_path = CONFIG_DIR / "presets" / f"{preset_name}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(f"Preset not found: {preset_path}")

        default_path = CONFIG_DIR / "default.yaml"
        merged_data: dict = {}
        if default_path.exists():
            merged_data = yaml.safe_load(default_path.read_text()) or {}
        preset_data = yaml.safe_load(preset_path.read_text()) or {}
        merged_data = _deep_merge(merged_data, preset_data)

        # Reuse from_yaml logic by writing the merged dict to a temp file would
        # be wasteful — call the dict-based loader directly.
        return cls._from_dict(merged_data, source=preset_path)

    @classmethod
    def from_yaml(cls, path: Path) -> "RunConfig":
        """Load a single YAML (no default merge). Use from_preset() for merging."""
        data = yaml.safe_load(path.read_text())
        return cls._from_dict(data, source=path)

    @classmethod
    def _from_dict(cls, data: dict, source: Path | None = None) -> "RunConfig":
        target = data.get("target", "BTC")
        venue = data.get("venue", "binance")
        quote = data.get("quote", "USDT")
        settle = data.get("settle", quote)
        timeframe = data.get("timeframe", "1h")

        # Two ways to specify paths in the YAML:
        #   (a) "data_root" + asset/venue fields → auto-resolve via DataRoot conventions
        #   (b) "paths" with explicit fields → use as-is (override)
        paths_block = data.get("paths", {})
        data_root = data.get("data_root")

        # Skip auto-resolve if data_root is the placeholder /tmp/missing.feather (default)
        is_placeholder = (
            (paths_block.get("ohlcv") or "").startswith("/tmp/missing")
        )
        if is_placeholder:
            paths_block = {}
        if data_root and not paths_block:
            cross = data.get("cross", {}) or {}
            resolved_paths = DataPaths.from_data_root(
                data_root=data_root,
                venue=venue,
                target=target,
                quote=data.get("quote", "USDT"),
                settle=data.get("settle", data.get("quote", "USDT")),
                timeframe=timeframe,
                cross_target=cross.get("target"),
                cross_quote=cross.get("quote"),
                cross_settle=cross.get("settle"),
            )
        else:
            # Explicit paths block (legacy / override mode)
            if not paths_block:
                raise ValueError(
                    f"{source}: must provide either `data_root:` (auto-resolve) or `paths:` (explicit)."
                )
            resolved_paths = DataPaths(
                ohlcv=Path(paths_block["ohlcv"]).expanduser(),
                funding=Path(paths_block["funding"]).expanduser() if paths_block.get("funding") else None,
                cross_ohlcv=Path(paths_block["cross_ohlcv"]).expanduser() if paths_block.get("cross_ohlcv") else None,
                cross_name=paths_block.get("cross_name", "cross"),
                external_dir=Path(paths_block["external_dir"]).expanduser() if paths_block.get("external_dir") else None,
            )
        # Resolve trading_signals_yaml: if relative, resolve under CONFIG_DIR
        feat_block = dict(data.get("features", {}))
        ts_yaml = feat_block.get("trading_signals_yaml")
        if ts_yaml:
            p = Path(str(ts_yaml)).expanduser()
            if not p.is_absolute():
                p = CONFIG_DIR / p
            feat_block["trading_signals_yaml"] = p

        # Parse training (multi-coin extra training data)
        train_block = data.get("training", {}) or {}
        extra_specs = []
        for ec in (train_block.get("extra_coins") or []):
            extra_specs.append(ExtraCoinSpec(
                target=ec["target"],
                venue=ec.get("venue"),
                quote=ec.get("quote"),
                settle=ec.get("settle"),
                timeframe=ec.get("timeframe"),
            ))
        training_cfg = TrainingConfig(
            extra_coins=extra_specs,
            add_coin_id_feature=bool(train_block.get("add_coin_id_feature", True)),
        )

        return cls(
            target=target,
            venue=venue,
            quote=quote,
            settle=settle,
            timeframe=timeframe,
            data_root=Path(data_root).expanduser() if data_root else None,
            paths=resolved_paths,
            label=LabelConfig(**data.get("label", {})),
            features=FeatureConfig(**feat_block),
            split=SplitConfig(**data.get("split", {})),
            predictors=PredictorConfig(**_extract_predictor_kwargs(data.get("predictors", {}))),
            training=training_cfg,
            plots=PlotConfig(**data.get("plots", {})),
            seed=data.get("seed", 42),
        )

    @property
    def purge_bars(self) -> int:
        if self.split.purge_bars is not None:
            return self.split.purge_bars
        return max(self.label.L_range) if self.label.L_range else self.label.horizon


LABEL_ORDER = ["bull", "bear"]
LABEL_COLORS = {"bull": "#2ca02c", "bear": "#d62728"}
