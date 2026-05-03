# regime_framework

Modular benchmark of time-series regime classification approaches on crypto OHLCV.

## What it does

Given OHLCV data of one target asset (e.g. BTC/USDT 1h on Binance), the
framework:

1. **Labels** every bar as `bull` or `bear` using trend-scanning
   (López de Prado, Advances in FML, ch. 3.6) — adaptive forward-window
   regression, sign of the most-significant slope.
2. **Builds features** from the data (technical, external macro, cross-asset,
   funding, and the user's own trading signals).
3. **Benchmarks predictors** that try to predict the regime label from past
   data only:
   - Classical ML: LogReg, RandomForest, GBM, MLP, XGBoost.
   - Rule-based (ported from `crypto_comparative_analysis`): regime_v3,
     regime_v4ema.
   - Deep nets: DeepMLP, GRU, LSTM, in-house TimeSeriesTransformer.
   - Pretrained foundation models: Chronos-Bolt-Base, Chronos-Large,
     TimesFM-2.0, MOIRAI, MOIRAI-MoE, TimeMoE, Lag-Llama, Toto.
4. **Analyses signals as predictors**: for each user trading signal
   (vwap_zscore, funding extremes, etc.), measures lift, mutual information,
   and conditional accuracy — answers "which signals are intrinsically
   regime-informative".
5. **Reports**: consolidated comparison table + plots.

## Quickstart

```bash
pip install -e .
python scripts/run_full_comparison.py --preset btc_binance_1h
```

## Layout

```
regime_framework/
├── data/             OHLCV, external (FNG/ETF/DXY/VIX), funding loaders
├── labels/           trend-scan, triple-barrier, drawdown
├── features/         technical, external, trading_signals
├── predictors/       classical, deep_nets, transformer, rule_based, pretrained/
├── signal_analysis/  lift, MI, ranker
├── evaluation/       metrics, splits, runner
├── visualization/    plots
└── cli.py
```
