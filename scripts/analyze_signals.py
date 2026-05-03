#!/usr/bin/env python3
"""Run only the signal analysis (lift + MI) on a preset.

Useful when you want to understand which features (technical, external, or
your trading signals) carry the most regime-predictive information, without
training any model.
"""
from __future__ import annotations

import typer

from regime_framework.config import RunConfig
from regime_framework.evaluation.runner import BenchmarkRunner


def main(
    preset: str = typer.Argument("btc_binance_1h", help="Preset name"),
    use_trading_signals: bool = typer.Option(False, "--with-trading-signals"),
    yaml_path: str | None = typer.Option(None, "--signals-yaml"),
):
    cfg = RunConfig.from_preset(preset)
    cfg.predictors.families = []  # signal analysis only, no predictor training
    if use_trading_signals:
        cfg.features.use_trading_signals = True
        if yaml_path:
            from pathlib import Path
            cfg.features.trading_signals_yaml = Path(yaml_path)
    runner = BenchmarkRunner(cfg)
    runner.run()


if __name__ == "__main__":
    typer.run(main)
