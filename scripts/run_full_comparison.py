#!/usr/bin/env python3
"""End-to-end benchmark on a preset.

Wraps `regime-run run <preset>` for users who prefer a script entry point.
"""
from __future__ import annotations

import sys

import typer

from regime_framework.config import RunConfig
from regime_framework.evaluation.runner import BenchmarkRunner


def main(
    preset: str = typer.Argument("btc_binance_1h", help="Preset name"),
    skip_pretrained: bool = typer.Option(False, "--skip-pretrained"),
):
    cfg = RunConfig.from_preset(preset)
    if skip_pretrained and "pretrained" in cfg.predictors.families:
        cfg.predictors.families.remove("pretrained")
    runner = BenchmarkRunner(cfg)
    runner.run()


if __name__ == "__main__":
    typer.run(main)
