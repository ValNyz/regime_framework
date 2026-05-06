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


# Cross-asset is now opt-in. Set `cross.target:` in the YAML to activate the
# 13-feature relative-strength block; leave it unset for a target-only run.
# (Previously this map auto-filled cross_target from the run's target —
# silently injecting ETH/BTC features into every preset. Removed because it
# made feature inventories non-obvious.)


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
    settle: str                   # usually same as quote (ignored for spot)
    timeframe: str                # "1h" | "30m" | ...
    market_type: str = "futures"  # "futures" | "spot" — controls path layout
    cross_target: str | None = None             # auto-resolved if None
    cross_quote: str | None = None              # defaults to quote
    cross_settle: str | None = None             # defaults to settle
    cross_market_type: str | None = None        # defaults to market_type (root)
    cross_venue: str | None = None              # defaults to venue (root)

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root).expanduser()
        # Cross fields are filled only when cross_target is explicitly provided.
        # Leaving cross_target=None disables the historical cross block entirely
        # (cross_ohlcv() returns None, cross_name() returns None).
        if self.cross_target is not None:
            if self.cross_quote is None:
                self.cross_quote = self.quote
            if self.cross_settle is None:
                self.cross_settle = self.settle
            if self.cross_market_type is None:
                self.cross_market_type = self.market_type
            if self.cross_venue is None:
                self.cross_venue = self.venue

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

    def cross_ohlcv(self) -> Path | None:
        if self.cross_target is None:
            return None
        return self._ohlcv_path(
            self.cross_venue or self.venue,
            self.cross_target,
            self.cross_quote or self.quote,
            self.cross_settle or self.settle,
            self.cross_market_type or self.market_type,
        )

    def cross_name(self) -> str | None:
        if self.cross_target is None:
            return None
        return self.cross_target.lower()

    def extra_cross_ohlcv(
        self,
        target: str,
        quote: str | None = None,
        settle: str | None = None,
        market_type: str | None = None,
        venue: str | None = None,
    ) -> Path:
        """Resolve OHLCV path for an arbitrary additional cross asset.

        Defaults each field to the corresponding root value if unset, so
        a CrossAssetSpec(target="SOL") in the YAML inherits venue/quote/
        settle/market_type from the run config without restating them.
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
        cross_path = self.cross_ohlcv()
        if cross_path is not None and not cross_path.exists():
            warnings.append(f"cross OHLCV missing (skipped): {cross_path}")
        if not self.external_dir().exists():
            warnings.append(f"external dir missing (skipped): {self.external_dir()}")
        return warnings
