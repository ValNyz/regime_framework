#!/usr/bin/env python3
"""Benchmark only the pretrained foundation models on a preset."""
from __future__ import annotations

import typer

from regime_framework.config import RunConfig
from regime_framework.evaluation.runner import BenchmarkRunner


def main(
    preset: str = typer.Argument("btc_binance_1h", help="Preset name"),
    zero_shot_only: bool = typer.Option(False, "--zero-shot-only"),
):
    cfg = RunConfig.from_preset(preset)
    cfg.predictors.families = ["pretrained"]
    if zero_shot_only:
        cfg.predictors.pretrained_modes = ["zero_shot"]
    runner = BenchmarkRunner(cfg)
    runner.run()


if __name__ == "__main__":
    typer.run(main)
