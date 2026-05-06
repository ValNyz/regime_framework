"""Filesystem layout conventions for crypto OHLCV / funding / external data.

Centralizes the naming scheme so a preset only needs to specify
(target, venue, quote, settle, timeframe, data_root) — paths are derived.

Conventions (relative to data_root):
  futures (default):
    {VENUE}/futures/{ASSET}_{QUOTE}_{SETTLE}-{TF}-futures.feather
    {VENUE}/futures/{ASSET}_{QUOTE}_{SETTLE}-{TF}-funding_rate.feather
  spot (Freqtrade-compatible — file at venue root, no settle):
    {VENUE}/{ASSET}_{QUOTE}-{TF}.feather
    (no funding for spot)

Same VENUE serves both market types — futures live under `futures/`, spot at
the venue root. Examples:
  binance/futures/BTC_USDT_USDT-1h-futures.feather       (linear futures)
  binance/BTC_USDT-1h.feather                            (spot, same venue)
  hyperliquid/futures/BTC_USDC_USDC-1h-futures.feather   (futures)
  hyperliquid/BTC_USDC-1h.feather                        (spot, same venue)
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


# Cross-asset is opt-in. Add a `cross:` block (dict for one coin, list for
# many) in the YAML to activate the 13-feature relative-strength block per
# cross. All cross specs go through DataRoot.coin_ohlcv() — there is no
# longer a "primary" vs "extra" cross distinction.


@dataclass
class DataRoot:
    """Root directory + venue/asset/timeframe -> resolves all canonical paths.

    Example:
        root = DataRoot(
            data_root="/data",
            venue="binance",
            target="BTC", quote="USDT", settle="USDT",
            timeframe="1h",
        )
        root.ohlcv()              # /data/binance/futures/BTC_USDT_USDT-1h-futures.feather
        root.funding()            # /data/binance/futures/BTC_USDT_USDT-1h-funding_rate.feather
        root.coin_ohlcv("ETH")    # /data/binance/futures/ETH_USDT_USDT-1h-futures.feather
        root.external_dir()       # /data/external
    """
    data_root: Path
    venue: str                    # "binance" | "hyperliquid"
    target: str                   # asset symbol, e.g. "BTC"
    quote: str                    # "USDT" | "USDC"
    settle: str                   # usually same as quote (ignored for spot)
    timeframe: str                # "1h" | "30m" | ...
    market_type: str = "futures"  # "futures" | "spot" — controls path layout

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root).expanduser()

    # ---------------------------------------------------------------------
    def _venue_dir(self, venue: str, market_type: str) -> Path:
        if market_type == "spot":
            return self.data_root / venue
        return self.data_root / venue / "futures"

    def venue_dir(self) -> Path:
        return self._venue_dir(self.venue, self.market_type)

    def _ohlcv_path(
        self, venue: str, asset: str, quote: str, settle: str, market_type: str,
    ) -> Path:
        vdir = self._venue_dir(venue, market_type)
        if market_type == "spot":
            # Freqtrade convention: {venue}/{ASSET}_{QUOTE}-{TF}.feather (no settle)
            return vdir / f"{asset}_{quote}-{self.timeframe}.feather"
        return vdir / f"{asset}_{quote}_{settle}-{self.timeframe}-futures.feather"

    def _funding_path(
        self, venue: str, asset: str, quote: str, settle: str, market_type: str,
    ) -> Path:
        vdir = self._venue_dir(venue, market_type)
        if market_type == "spot":
            # Spot has no funding rate — return a non-existent path so the
            # downstream `.exists()` check skips funding cleanly.
            return vdir / f"{asset}_{quote}-{self.timeframe}-funding_rate-NOT-AVAILABLE-FOR-SPOT.feather"
        return vdir / f"{asset}_{quote}_{settle}-{self.timeframe}-funding_rate.feather"

    # ---------------------------------------------------------------------
    def ohlcv(self) -> Path:
        return self._ohlcv_path(
            self.venue, self.target, self.quote, self.settle, self.market_type,
        )

    def funding(self) -> Path:
        return self._funding_path(
            self.venue, self.target, self.quote, self.settle, self.market_type,
        )

    def coin_ohlcv(
        self,
        target: str,
        quote: str | None = None,
        settle: str | None = None,
        market_type: str | None = None,
        venue: str | None = None,
    ) -> Path:
        """Resolve OHLCV path for an arbitrary cross-coin spec.

        Each field defaults to the run's root value when unset, so the
        minimal spec is just `target`. Used uniformly for every entry in
        the unified `cross:` list — there is no longer a "primary" vs
        "extra" distinction.
        """
        return self._ohlcv_path(
            venue or self.venue,
            target,
            quote or self.quote,
            settle or self.settle,
            market_type or self.market_type,
        )

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
        if not self.external_dir().exists():
            warnings.append(f"external dir missing (skipped): {self.external_dir()}")
        return warnings
