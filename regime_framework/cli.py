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
        help="Predictor families to run. Repeat the flag for multiple values.",
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
        help="Number of cross-validation folds. 0 = single train/test split (default).",
    ),
    cv_mode: str = typer.Option(
        "walk_forward", "--cv-mode",
        help="CV mode: walk_forward (expanding window, live-style), "
             "leave_one_out (each fold tested with rest as train, future-info), "
             "or both (run both and compare).",
    ),
    min_train_fraction: float = typer.Option(
        0.40, "--min-train-fraction",
        help="Walk-forward only: fold-0 train size (fraction of total).",
    ),
):
    """Run the full benchmark on a preset config."""
    cfg = RunConfig.from_preset(preset)
    cfg.predictors.families = list(families)
    if skip_pretrained and "pretrained" in cfg.predictors.families:
        cfg.predictors.families.remove("pretrained")
    if pretrained:
        cfg.predictors.pretrained_models = list(pretrained)
    if cv_folds > 0:
        if cv_mode not in ("walk_forward", "leave_one_out", "both"):
            raise typer.BadParameter(
                f"--cv-mode must be one of: walk_forward, leave_one_out, both (got {cv_mode!r})"
            )
        cfg.split.cv_folds = cv_folds
        cfg.split.cv_mode = cv_mode
        cfg.split.min_train_fraction = float(min_train_fraction)

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
