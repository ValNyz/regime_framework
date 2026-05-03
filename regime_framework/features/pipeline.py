"""Feature pipeline composer.

Combines technical, external, and trading-signal features into one matrix,
optionally drops bars with NaN. Returns a (X, label_aligned, dates) tuple
ready for predictors.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import pandas as pd

from .technical import compute_technical_features
from .external import compute_external_features
from .trading_signals import compute_trading_signal_features


@dataclass
class FeaturePipeline:
    use_technical: bool = True
    use_external: bool = True
    use_trading_signals: bool = False
    trading_signals_yaml: Path | None = None
    target_funding_path: Path | None = None
    cross_ohlcv_path: Path | None = None
    cross_name: str = "cross"
    external_dir: Path | None = None
    drop_nan_rows: bool = True

    def build(self, df: pd.DataFrame, labels: pd.Series) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Build the combined feature matrix.

        Side effect: populates `self.column_sources`, a dict mapping every
        feature name to its origin group ("technical", "external", "signals").
        Consumed by the runner to annotate feature-importance reports.

        Returns:
            X: feature DataFrame (rows aligned, NaN dropped if configured)
            y: label Series aligned to X.index
            dates: pd.Series of dates aligned to X.index (for plotting / split)
        """
        feats: list[pd.DataFrame] = []
        self.column_sources: dict[str, str] = {}

        if self.use_technical:
            tech = compute_technical_features(df)
            feats.append(tech)
            for c in tech.columns:
                self.column_sources[c] = "technical"
            print(f"  technical: {tech.shape[1]} features")

        if self.use_external:
            ext = compute_external_features(
                df,
                external_dir=self.external_dir,
                target_funding_path=self.target_funding_path,
                cross_ohlcv_path=self.cross_ohlcv_path,
                cross_name=self.cross_name,
            )
            feats.append(ext)
            for c in ext.columns:
                self.column_sources[c] = "external"
            print(f"  external:  {ext.shape[1]} features")

        if self.use_trading_signals:
            # Make funding rate available to funding-type signals
            funding_series = None
            if self.target_funding_path is not None and self.target_funding_path.exists():
                try:
                    from ..data.loaders import load_parquet_or_feather
                    fund = load_parquet_or_feather(self.target_funding_path)
                    fund_col = "open" if "open" in fund.columns else "funding_rate"
                    merged = pd.merge_asof(
                        df[["date"]].sort_values("date"),
                        fund[["date", fund_col]].rename(columns={fund_col: "fr"}).sort_values("date"),
                        on="date", direction="backward",
                    )
                    funding_series = merged["fr"].astype(float)
                    funding_series.index = df.index
                except Exception as e:
                    print(f"  WARN: failed to load funding for trading signals: {e}")
            ts = compute_trading_signal_features(df, self.trading_signals_yaml, funding=funding_series)
            feats.append(ts)
            for c in ts.columns:
                self.column_sources[c] = "signals"
            print(f"  signals:   {ts.shape[1]} usable features")

        if not feats:
            raise ValueError("No feature group enabled in FeaturePipeline.")

        X = pd.concat(feats, axis=1)
        print(f"  TOTAL:     {X.shape[1]} features")

        # Align with labels — keep only labelled rows
        mask_labelled = labels.values != ""
        idx_labelled = df.index[mask_labelled]
        X = X.loc[idx_labelled]
        y = labels.loc[idx_labelled]
        dates = df.loc[idx_labelled, "date"]

        if self.drop_nan_rows:
            valid = ~X.isna().any(axis=1)
            X = X.loc[valid]
            y = y.loc[X.index]
            dates = dates.loc[X.index]

        return X, y, dates
