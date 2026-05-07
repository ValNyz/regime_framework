"""Freqtrade-based backtesting of regime predictor outputs.

Public API consumed by `regime-run backtest`:
  - dump_stitched_predictions: persist stitched OOS labels + closes to feather + manifest.
  - render_strategy: produce a self-contained freqtrade IStrategy file from a Template.
  - build_freqtrade_config / write_freqtrade_config: minimal config.json builder.
  - run_backtest: spawn `freqtrade backtesting` via subprocess (list-of-args, shell=False).
  - parse_backtest_result: extract metrics from the resulting backtest-result-*.zip.
  - format_side_by_side: rich.Table comparing framework vs freqtrade metrics.

This module duplicates a small amount of strategy-template code from the user's
crypto_comparative_analysis project intentionally — they live in different git
repos and we keep the dependency direction one-way (regime_framework standalone).
"""
