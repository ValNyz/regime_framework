"""Filesystem layout conventions for crypto OHLCV / funding / external data.

Centralizes the naming scheme so a preset only needs to specify
(target, venue, quote, settle, timeframe, data_root) — paths are derived.

Conventions (relative to data_root):
  binance/futures/{ASSET}_{QUOTE}_{SETTLE}-{TF}-futures.feather
  binance/futures/{ASSET}_{QUOTE}_{SETTLE}-{TF}-funding_rate.feather
  hyperliquid/futures/{ASSET}_{QUOTE}_{SETTLE}-{TF}-futures.feather
  hyperliquid/futures/{ASSET}_{QUOTE}_{SETTLE}-{TF}-funding_rate.feather
  external/binance_funding_{SYMBOL}.parquet     (SYMBOL = ASSET+QUOTE, e.g. BTCUSDT)
  external/etf_flows_{btc|eth}.parquet
  external/fng_daily.parquet
  external/yfinance_DXY.parquet
  external/yfinance_VIX.parquet

If you have a non-standard layout, override the resolved paths at config time
via DataPaths(...) — the resolver is opt-in via DataRoot.resolve().
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# Default cross-asset reference per target (used if `cross.target` not set in preset)
DEFAULT_CROSS = {
    "BTC": "ETH",
    "ETH": "BTC",
    "HYPE": "BTC",
    "SOL": "BTC",
    "ENA": "BTC",
    "SUI": "BTC",
    "PURR": "BTC",
}


@dataclass
class DataRoot:
    """Root directory + venue/asset/timeframe → resolves all canonical paths.

    Example:
        root = DataRoot(
            data_root="/data",
            venue="binance",
            target="BTC", quote="USDT", settle="USDT",
            timeframe="1h",
        )
        root.ohlcv()         # /data/binance/futures/BTC_USDT_USDT-1h-futures.feather
        root.funding()       # /data/binance/futures/BTC_USDT_USDT-1h-funding_rate.feather
        root.cross_ohlcv()   # /data/binance/futures/ETH_USDT_USDT-1h-futures.feather
        root.external_dir()  # /data/external
    """
    data_root: Path
    venue: str                    # "binance" | "hyperliquid"
    target: str                   # asset symbol, e.g. "BTC"
    quote: str                    # "USDT" | "USDC"
    settle: str                   # usually same as quote
    timeframe: str                # "1h" | "30m" | ...
    cross_target: str | None = None      # auto-resolved if None
    cross_quote: str | None = None       # defaults to quote
    cross_settle: str | None = None      # defaults to settle

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root).expanduser()
        if self.cross_target is None:
            self.cross_target = DEFAULT_CROSS.get(self.target.upper(), "BTC")
        if self.cross_quote is None:
            self.cross_quote = self.quote
        if self.cross_settle is None:
            self.cross_settle = self.settle

    # ---------------------------------------------------------------------
    def venue_dir(self) -> Path:
        return self.data_root / self.venue / "futures"

    def _ohlcv(self, asset: str, quote: str, settle: str) -> Path:
        return self.venue_dir() / f"{asset}_{quote}_{settle}-{self.timeframe}-futures.feather"

    def _funding(self, asset: str, quote: str, settle: str) -> Path:
        return self.venue_dir() / f"{asset}_{quote}_{settle}-{self.timeframe}-funding_rate.feather"

    # ---------------------------------------------------------------------
    def ohlcv(self) -> Path:
        return self._ohlcv(self.target, self.quote, self.settle)

    def funding(self) -> Path:
        return self._funding(self.target, self.quote, self.settle)

    def cross_ohlcv(self) -> Path:
        return self._ohlcv(self.cross_target or "BTC", self.cross_quote or self.quote, self.cross_settle or self.settle)

    def cross_name(self) -> str:
        return (self.cross_target or "btc").lower()

    def external_dir(self) -> Path:
        return self.data_root / "external"

    # ---------------------------------------------------------------------
    def assert_minimal_layout(self) -> list[str]:
        """Return a list of warnings about missing optional files (does not raise)."""
        warnings = []
        if not self.ohlcv().exists():
            warnings.append(f"OHLCV missing: {self.ohlcv()}")
        if not self.funding().exists():
            warnings.append(f"funding missing (skipped): {self.funding()}")
        if not self.cross_ohlcv().exists():
            warnings.append(f"cross OHLCV missing (skipped): {self.cross_ohlcv()}")
        if not self.external_dir().exists():
            warnings.append(f"external dir missing (skipped): {self.external_dir()}")
        return warnings
