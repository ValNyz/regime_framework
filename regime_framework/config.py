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
    horizon: int = 48
    alpha: float = 1.5


@dataclass
class FeatureConfig:
    use_technical: bool = True
    use_external: bool = True
    use_trading_signals: bool = False
    trading_signals_yaml: Path | None = None
    drop_nan_rows: bool = True


@dataclass
class SplitConfig:
    train_fraction: float = 0.80
    purge_bars: int | None = None        # if None, defaults to max(L_range)
    cv_folds: int = 0                    # 0 = single split; >0 = K-fold CV
    cv_mode: str = "walk_forward"        # "walk_forward" | "leave_one_out"
    min_train_fraction: float = 0.40     # walk-forward: fold-0 train size
    # Backward-compat alias (older configs may still use this)
    walk_forward_folds: int = 0


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
            predictors=PredictorConfig(**data.get("predictors", {})),
            training=training_cfg,
            seed=data.get("seed", 42),
        )

    @property
    def purge_bars(self) -> int:
        if self.split.purge_bars is not None:
            return self.split.purge_bars
        return max(self.label.L_range) if self.label.L_range else self.label.horizon


LABEL_ORDER = ["bull", "bear"]
LABEL_COLORS = {"bull": "#2ca02c", "bear": "#d62728"}
