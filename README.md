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
# Edit configs/presets/btc_binance_1h.yaml: set `data_root: /path/to/your/data`
regime-run run btc_binance_1h
```

## Pretrained model dependencies (optional, à la carte)

Each foundation model needs its own package — install only what you'll run:

```bash
pip install chronos-forecasting       # Chronos-Bolt-Base + Chronos-T5-Large
pip install 'timesfm[torch]'          # TimesFM-2.0 (PyTorch backend, works on Py3.12)
pip install uni2ts                    # MOIRAI + MOIRAI-MoE
pip install gluonts                 # required by Lag-Llama
pip install git+https://github.com/time-series-foundation-models/lag-llama
# TimeMoE: loaded via transformers.AutoModelForCausalLM, no extra install
# Toto: disabled by default — Datadog checkpoint requires their custom loader
```

If a model isn't installed, the framework prints a clear `ImportError` and
moves on (other predictors still run).

## Data layout (auto-resolved from `data_root`)

```
{data_root}/
├── binance/futures/
│   ├── BTC_USDT_USDT-1h-futures.feather
│   ├── BTC_USDT_USDT-1h-funding_rate.feather
│   ├── ETH_USDT_USDT-1h-futures.feather
│   └── ...
├── hyperliquid/futures/
│   ├── HYPE_USDC_USDC-1h-futures.feather
│   ├── HYPE_USDC_USDC-1h-funding_rate.feather
│   └── ...
└── external/
    ├── binance_funding_BTCUSDT.parquet
    ├── etf_flows_btc.parquet
    ├── fng_daily.parquet
    ├── yfinance_DXY.parquet
    └── yfinance_VIX.parquet
```

A preset only specifies `target`, `venue`, `quote`, `settle`, `timeframe`, `data_root` — the framework derives all 4 file paths automatically. Override individual paths via the `paths:` block if your layout differs.

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
