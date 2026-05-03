"""regime_framework CLI — typer + rich.

Commands:
  run [PRESET]        Run the full benchmark on a preset
  signals [PRESET]    Just signal analysis (lift + MI on labels)
  pretrained [PRESET] Just the pretrained models (zero-shot)
  presets             List available presets
"""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import CONFIG_DIR, RunConfig

app = typer.Typer(
    name="regime",
    help="Modular benchmark of time-series regime classification approaches.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    preset: str = typer.Argument(..., help="Preset name (e.g. btc_binance_1h)"),
    families: list[str] = typer.Option(
        ["classical", "rule_based", "deep_nets", "transformer", "pretrained"],
        "--family", "-f",
        help="Predictor families: classical, rule_based, deep_nets, transformer, pretrained. "
             "Accepts repeated flags (-f classical -f rule_based) OR comma-separated "
             "(-f classical,rule_based).",
    ),
    pretrained: list[str] | None = typer.Option(
        None, "--pretrained", "-p",
        help="Override pretrained models list. If unset, uses preset config.",
    ),
    skip_pretrained: bool = typer.Option(
        False, "--skip-pretrained", help="Skip all foundation models (faster).",
    ),
    cv_folds: int = typer.Option(
        0, "--cv-folds", "-k",
        help="Number of CV folds. 0 = single split (default). In 'rolling' mode, "
             "caps at this many folds — the LATEST N consecutive folds are kept "
             "(window + step stay fixed; fold_id reflects original chronological "
             "position in the data).",
    ),
    cv_mode: str = typer.Option(
        "walk_forward", "--cv-mode",
        help="CV mode: walk_forward (expanding), leave_one_out, both, or "
             "rolling (fixed train + sliding test window).",
    ),
    min_train_fraction: float = typer.Option(
        0.40, "--min-train-fraction",
        help="Walk-forward only: fold-0 train size (fraction of total).",
    ),
    train_window_bars: int = typer.Option(
        0, "--train-window-bars",
        help="Rolling mode: training window size in bars. Defaults to 4380 "
             "(6mo at 1h timeframe) when unset.",
    ),
    test_window_bars: int = typer.Option(
        0, "--test-window-bars",
        help="Rolling mode: test window size in bars. Defaults to 730 "
             "(1mo at 1h timeframe) when unset.",
    ),
    step_bars: int = typer.Option(
        0, "--step-bars",
        help="Rolling mode: slide step in bars. 0 = same as test window "
             "(non-overlapping consecutive tests).",
    ),
    feature_importance: bool = typer.Option(
        None, "--feature-importance/--no-feature-importance",
        help="Compute + display feature importance for the best classical "
             "predictor (per fold + end of CV). When unset, the preset's "
             "predictors.feature_importance wins. CLI overrides preset only "
             "when the flag is explicitly passed.",
    ),
    plots: bool = typer.Option(
        None, "--plots/--no-plots",
        help="Master plot switch. --no-plots kills ALL plot output (label, "
             "prediction, fold, multi, stitched). Defaults to preset value.",
    ),
    fold_plots: bool = typer.Option(
        None, "--fold-plots/--no-fold-plots",
        help="Per-fold plots only (best-predictor + multi-overlay per CV "
             "fold). --no-fold-plots keeps once-per-run summary plots "
             "(label, stitched OOS) but drops the per-fold ones — useful "
             "when running with 50+ rolling folds.",
    ),
    rank_by: str = typer.Option(
        None, "--rank-by",
        help="Sort the CV summary table and pick the 'best' predictor for "
             "the stitched OOS plot by: 'kappa' (default — classification "
             "skill), 'gain' (compounded synth_gain across folds), or "
             "'vs_bh' (gain_total minus B&H). Use 'gain' or 'vs_bh' when "
             "you care about money rather than per-bar agreement.",
    ),
):
    """Run the full benchmark on a preset config."""
    # Support both repeated -f and comma-separated lists
    flat_families: list[str] = []
    for f in families:
        flat_families.extend([x.strip() for x in str(f).split(",") if x.strip()])
    valid = {"classical", "rule_based", "deep_nets", "transformer", "pretrained"}
    bad = [f for f in flat_families if f not in valid]
    if bad:
        raise typer.BadParameter(
            f"Unknown family/families: {bad}. Valid: {sorted(valid)}"
        )

    cfg = RunConfig.from_preset(preset)
    cfg.predictors.families = flat_families
    if skip_pretrained and "pretrained" in cfg.predictors.families:
        cfg.predictors.families.remove("pretrained")
    if pretrained:
        cfg.predictors.pretrained_models = list(pretrained)
    valid_modes = ("walk_forward", "leave_one_out", "both", "rolling")
    if cv_mode == "rolling":
        # Rolling mode is always CV; cv_folds derives from window sizes.
        cfg.split.cv_mode = "rolling"
        cfg.split.cv_folds = max(1, cv_folds)  # >0 just to enter the CV path
        cfg.split.train_window_bars = train_window_bars or cfg.split.train_window_bars or 4380
        cfg.split.test_window_bars = test_window_bars or cfg.split.test_window_bars or 730
        cfg.split.step_bars = step_bars or cfg.split.step_bars
    elif cv_folds > 0:
        if cv_mode not in valid_modes:
            raise typer.BadParameter(
                f"--cv-mode must be one of {valid_modes} (got {cv_mode!r})"
            )
        cfg.split.cv_folds = cv_folds
        cfg.split.cv_mode = cv_mode
        cfg.split.min_train_fraction = float(min_train_fraction)

    # CLI flags override preset only when explicitly passed (None = unset).
    if feature_importance is not None:
        cfg.predictors.feature_importance = bool(feature_importance)
    if plots is not None:
        cfg.plots.enabled = bool(plots)
    if fold_plots is not None:
        cfg.plots.per_fold = bool(fold_plots)
    if rank_by is not None:
        if rank_by not in ("kappa", "gain", "vs_bh"):
            raise typer.BadParameter(
                f"--rank-by must be one of: kappa, gain, vs_bh (got {rank_by!r})"
            )
        cfg.predictors.rank_by = rank_by

    from .evaluation.runner import BenchmarkRunner
    runner = BenchmarkRunner(cfg)
    runner.run()


@app.command()
def signals(preset: str = typer.Argument(..., help="Preset name")):
    """Run only the signal analysis (lift + MI)."""
    cfg = RunConfig.from_preset(preset)
    # Force signal analysis only — disable all predictors
    cfg.predictors.families = []
    from .evaluation.runner import BenchmarkRunner
    runner = BenchmarkRunner(cfg)
    runner.run()


@app.command()
def pretrained(
    preset: str = typer.Argument(..., help="Preset name"),
    zero_shot_only: bool = typer.Option(False, "--zero-shot-only", help="Skip fine-tuned head"),
):
    """Run only pretrained foundation models."""
    cfg = RunConfig.from_preset(preset)
    cfg.predictors.families = ["pretrained"]
    if zero_shot_only:
        cfg.predictors.pretrained_modes = ["zero_shot"]
    from .evaluation.runner import BenchmarkRunner
    runner = BenchmarkRunner(cfg)
    runner.run()


@app.command()
def presets():
    """List available presets."""
    preset_dir = CONFIG_DIR / "presets"
    if not preset_dir.exists():
        console.print(f"[yellow]No presets directory found at {preset_dir}[/yellow]")
        return
    table = Table(title=f"Available presets in {preset_dir}")
    table.add_column("name", style="cyan")
    table.add_column("file")
    for p in sorted(preset_dir.glob("*.yaml")):
        table.add_row(p.stem, p.name)
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
