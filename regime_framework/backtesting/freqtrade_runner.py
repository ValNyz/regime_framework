"""Build freqtrade config + spawn `freqtrade backtesting` + parse results.

`freqtrade` must be on PATH. The framework does not declare it as a dep
(install separately: `pip install freqtrade`). We invoke it with a list-of-
args + shell=False; user-controlled values are passed as separate argv
entries — never concatenated into a string.

Pair derivation:
  - binance + spot     -> BTC/USDT
  - binance + futures  -> BTC/USDT:USDT
  - hyperliquid + spot -> BTC/USDC
  - hyperliquid + futures -> BTC/USDC:USDC

Fee convention:
  cfg.predictors.evaluation_transaction_cost is the per-flip cost in the
  framework's stitched metrics (one cost on every position change).
  freqtrade's `fee` is per-side (entry AND exit) — same arithmetic on a flip.
  We pass it through unchanged unless cfg.backtest.fee overrides.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any


_VENUE_TO_FT_EXCHANGE = {
    "binance": "binance",
    "hyperliquid": "hyperliquid",
}


def derive_pair(target: str, quote: str, settle: str, market_type: str) -> str:
    """Return the freqtrade pair string for this asset."""
    if market_type == "futures":
        return f"{target}/{quote}:{settle}"
    return f"{target}/{quote}"


def derive_stake_currency(cfg) -> str:
    """Default stake currency = settle for futures, quote for spot."""
    if cfg.backtest.stake_currency:
        return cfg.backtest.stake_currency
    return cfg.settle if cfg.market_type == "futures" else cfg.quote


def build_freqtrade_config(
    cfg,
    *,
    predictor_name: str,
    datadir: Path,
    user_data_dir: Path,
) -> dict[str, Any]:
    """Assemble a minimal freqtrade backtesting config dict from cfg."""
    pair = cfg.backtest.pair or derive_pair(
        cfg.target, cfg.quote, cfg.settle, cfg.market_type,
    )
    fee = cfg.backtest.fee
    if fee is None:
        fee = float(cfg.predictors.evaluation_transaction_cost)
    exchange_name = _VENUE_TO_FT_EXCHANGE.get(cfg.venue, cfg.venue)

    config: dict[str, Any] = {
        "max_open_trades": int(cfg.backtest.max_open_trades),
        "stake_currency": derive_stake_currency(cfg),
        "stake_amount": "unlimited",
        "tradable_balance_ratio": 0.99,
        "dry_run": True,
        "dry_run_wallet": float(cfg.backtest.dry_run_wallet),
        "fiat_display_currency": "USD",
        "timeframe": cfg.timeframe,
        "trading_mode": cfg.market_type,
        "fee": float(fee),
        "dataformat_ohlcv": "feather",
        "datadir": str(Path(datadir).resolve()),
        "user_data_dir": str(Path(user_data_dir).resolve()),
        "exchange": {
            "name": exchange_name,
            "key": "",
            "secret": "",
            "ccxt_config": {},
            "ccxt_async_config": {},
            "pair_whitelist": [pair],
            "pair_blacklist": [],
        },
        "entry_pricing": {
            "price_side": "same",
            "use_order_book": False,
            "order_book_top": 1,
        },
        "exit_pricing": {
            "price_side": "same",
            "use_order_book": False,
            "order_book_top": 1,
        },
        "pairlists": [{"method": "StaticPairList"}],
        # tag: written by regime_framework; helpful to audit which run produced this
        "regime_framework_predictor": predictor_name,
    }
    if cfg.market_type == "futures":
        config["margin_mode"] = "isolated"
    return config


def write_freqtrade_config(config: dict[str, Any], path: Path) -> Path:
    """Pretty-print the config dict to JSON at `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2))
    return path


def run_backtest(
    *,
    strategy_class: str,
    strategies_dir: Path,
    config_path: Path,
    user_data_dir: Path,
    datadir: Path,
    timerange: str,
    export_path: Path,
) -> int:
    """Spawn `freqtrade backtesting` synchronously. Returns its exit code.

    Streams freqtrade's stdout/stderr to the parent terminal directly
    (no buffering — long backtests print incremental progress).

    Raises FileNotFoundError when freqtrade is not on PATH.
    Raises RuntimeError when freqtrade exits non-zero.
    """
    if shutil.which("freqtrade") is None:
        raise FileNotFoundError(
            "`freqtrade` not found on PATH. Install it with:\n"
            "    pip install freqtrade\n"
            "regime_framework does not depend on freqtrade by default."
        )
    Path(export_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "freqtrade", "backtesting",
        "--config", str(Path(config_path).resolve()),
        "--strategy", strategy_class,
        "--strategy-path", str(Path(strategies_dir).resolve()),
        "--user-data-dir", str(Path(user_data_dir).resolve()),
        "--datadir", str(Path(datadir).resolve()),
        "--timerange", timerange,
        "--export", "trades",
        "--export-filename", str(Path(export_path).resolve()),
    ]
    print("  $ " + " ".join(cmd))
    # shell=False (default): list-of-args is safe — no shell interpolation.
    completed = subprocess.run(cmd, shell=False, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"freqtrade backtesting exited with code {completed.returncode}. "
            f"Re-run the command above to inspect the full stderr."
        )
    return completed.returncode


def parse_backtest_result(export_path: Path, strategy_class: str) -> dict[str, Any]:
    """Read freqtrade's exported result JSON and return key headline metrics.

    Freqtrade's --export-filename writes `<name>.json` (a JSON pointer file)
    plus `<name>.zip` containing the actual results. The shape varies by
    freqtrade version; we look for the strategy stats under several known
    paths and return None for fields that don't exist.
    """
    export_path = Path(export_path)
    candidates: list[Path] = []
    if export_path.suffix == ".json" and export_path.exists():
        candidates.append(export_path)
    zip_path = export_path.with_suffix(".zip")
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if name.endswith(".json"):
                    extracted = export_path.parent / name
                    extracted.parent.mkdir(parents=True, exist_ok=True)
                    extracted.write_bytes(zf.read(name))
                    candidates.append(extracted)

    payload = None
    for cand in candidates:
        try:
            payload = json.loads(cand.read_text())
            break
        except (json.JSONDecodeError, FileNotFoundError):
            continue

    if payload is None:
        raise RuntimeError(
            f"Could not locate freqtrade backtest result JSON near {export_path}."
        )

    strat = (payload.get("strategy") or {}).get(strategy_class) or {}
    # fall back to the first strategy block if the class name doesn't match
    if not strat and payload.get("strategy"):
        strat = next(iter(payload["strategy"].values()))

    def _get(*keys, default=None):
        for k in keys:
            v = strat.get(k)
            if v is not None:
                return v
        return default

    return {
        "profit_total": _get("profit_total"),
        "profit_total_pct": _get("profit_total_pct", "profit_total_percentage"),
        "sharpe": _get("sharpe", "sharpe_ratio"),
        "sortino": _get("sortino", "sortino_ratio"),
        "max_drawdown_pct": _get(
            "max_drawdown_account",
            "max_drawdown",
            "max_relative_drawdown",
        ),
        "total_trades": _get("total_trades", "trade_count"),
        "expectancy": _get("expectancy"),
    }
