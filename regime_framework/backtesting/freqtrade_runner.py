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
    breakdown: str | None = None,
) -> int:
    """Spawn `freqtrade backtesting` synchronously. Returns its exit code.

    Streams freqtrade's stdout/stderr to the parent terminal directly
    (no buffering — long backtests print incremental progress).

    breakdown: when set, passes `--breakdown <value>` (day | week | month |
    year) so freqtrade prints a per-period summary table at the end of the
    run AND embeds the breakdown into the result JSON (parsed downstream).

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
        # Note: --export-filename is deprecated since freqtrade 2026.x; the
        # backtest result lands under <user_data_dir>/backtest_results/ with
        # a timestamped name, then resolved by parse_backtest_result via
        # .last_result.json (see _find_freqtrade_result fallback chain).
    ]
    if breakdown:
        cmd += ["--breakdown", breakdown]
    print("  $ " + " ".join(cmd))
    # shell=False (default): list-of-args is safe — no shell interpolation.
    completed = subprocess.run(cmd, shell=False, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"freqtrade backtesting exited with code {completed.returncode}. "
            f"Re-run the command above to inspect the full stderr."
        )
    return completed.returncode


def _find_freqtrade_result(export_path: Path) -> Path | None:
    """Locate the freqtrade backtest-result JSON written by the most recent run.

    Strategy (in order):
      1. The explicit export_path (works on freqtrade versions where
         --export-filename still has an effect).
      2. `<results_dir>/.last_result.json` — freqtrade's pointer to the
         latest result, written every backtest. We follow `latest_backtest`
         to the actual zip / json.
      3. The most recent `backtest-result-*.zip` in <results_dir>, by mtime.
    """
    export_path = Path(export_path)
    results_dir = export_path.parent

    # 1. Explicit path (legacy freqtrade)
    if export_path.exists():
        return export_path
    zip_path = export_path.with_suffix(".zip")
    if zip_path.exists():
        return zip_path

    # 2. .last_result.json pointer
    last_result = results_dir / ".last_result.json"
    if last_result.exists():
        try:
            ptr = json.loads(last_result.read_text())
            latest_name = ptr.get("latest_backtest")
            if latest_name:
                latest = results_dir / latest_name
                if latest.exists():
                    return latest
                # .last_result.json sometimes points to the .json sibling;
                # try the .zip too.
                zip_sibling = latest.with_suffix(".zip")
                if zip_sibling.exists():
                    return zip_sibling
        except (json.JSONDecodeError, OSError):
            pass

    # 3. Most recent backtest-result-*.zip
    candidates = sorted(
        results_dir.glob("backtest-result-*.zip"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    return None


def parse_backtest_result(export_path: Path, strategy_class: str) -> dict[str, Any]:
    """Read freqtrade's exported result JSON and return key headline metrics.

    Locates the result file via _find_freqtrade_result (handles old explicit
    path, freqtrade 2026's .last_result.json pointer, and bare timestamped
    zips as fallback). The shape varies by freqtrade version; we look for
    strategy stats under several known keys and return None for missing ones.
    """
    located = _find_freqtrade_result(Path(export_path))
    if located is None:
        results_dir = Path(export_path).parent
        listing = "\n  ".join(sorted(p.name for p in results_dir.glob("*"))[:20])
        raise RuntimeError(
            f"Could not locate freqtrade backtest result near {export_path}.\n"
            f"Looked in {results_dir}; first 20 entries:\n  {listing or '(empty)'}"
        )

    payload = None
    if located.suffix == ".zip":
        # Freqtrade zip contains backtest-result-<TS>.json (the real payload)
        # plus _config.json + _market_change.feather + the strategy source.
        with zipfile.ZipFile(located) as zf:
            for name in zf.namelist():
                if name.endswith(".json") and "config" not in name and "meta" not in name:
                    payload = json.loads(zf.read(name).decode("utf-8"))
                    break
    elif located.suffix == ".json":
        try:
            payload = json.loads(located.read_text())
        except (json.JSONDecodeError, OSError):
            payload = None

    if payload is None:
        raise RuntimeError(
            f"Located {located} but could not extract a JSON payload from it."
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

    def _to_fraction(v):
        """Normalize freqtrade percentages to fractions (e.g. 168.25 -> 1.6825).

        Freqtrade is inconsistent across versions: some fields store
        percentages (168.25 means 168.25%), others store fractions (0.0899
        means 8.99%). We compare to the framework's stitched metrics, which
        are uniformly stored as fractions, so we coerce.

        Heuristic: |v| > 1 -> percent units -> divide by 100. |v| <= 1 ->
        already a fraction. False positives only happen for absurdly large
        fractional gains (>100%) — but those would print weirdly anyway.
        """
        if v is None:
            return None
        v = float(v)
        return v / 100.0 if abs(v) > 1.0 else v

    # Periodic breakdown — present when `--breakdown <unit>` was passed.
    # Freqtrade stores it as `periodic_breakdown` (newer) or
    # `daily_profit` / `weekly_profit` / `monthly_profit` (older).
    breakdown = _get("periodic_breakdown")
    if not breakdown:
        # Older shape — gather any per-period buckets that exist
        breakdown = {
            unit: _get(f"{unit}_profit") or _get(f"{unit}ly_profit")
            for unit in ("day", "week", "month", "year")
        }
        breakdown = {k: v for k, v in breakdown.items() if v}

    # Profit total: try several aliases. Freqtrade 2026 typically stores
    # the percentage version as `profit_total` (already a fraction) or
    # `profit_total_pct` (already a fraction in newer schema).
    profit_total_pct = _to_fraction(_get(
        "profit_total",            # often the fraction in 2026.x
        "profit_total_pct",
        "profit_total_percentage",
        "profit_total_long_pct",
        "total_profit_pct",
    ))

    # Drawdown: framework's convention is NEGATIVE (e.g. -0.1354). Freqtrade
    # often reports as POSITIVE magnitude (8.99%). We force negative sign.
    max_dd_raw = _get(
        "max_drawdown_account",
        "max_drawdown",
        "max_relative_drawdown",
        "max_drawdown_abs",
    )
    max_dd = _to_fraction(max_dd_raw)
    if max_dd is not None and max_dd > 0:
        max_dd = -max_dd

    # Calmar: prefer freqtrade's own field; fall back to gain / |dd| (matches
    # the framework's calmar_ratio() convention — non-annualized).
    calmar = _get("calmar", "calmar_ratio")
    if calmar is None and profit_total_pct is not None and max_dd not in (None, 0):
        calmar = profit_total_pct / abs(max_dd)

    return {
        "profit_total": _get("profit_total_abs", "profit_abs"),  # absolute USDC
        "profit_total_pct": profit_total_pct,                     # fraction
        "sharpe": _get("sharpe", "sharpe_ratio"),
        "sortino": _get("sortino", "sortino_ratio"),
        "calmar": calmar,
        "cagr": _get("cagr"),
        "max_drawdown_pct": max_dd,                                # fraction, negative
        "total_trades": _get("total_trades", "trade_count", "trades"),
        "expectancy": _get("expectancy"),
        "starting_balance": _get("starting_balance"),
        "final_balance": _get("final_balance"),
        "periodic_breakdown": breakdown or {},
    }
