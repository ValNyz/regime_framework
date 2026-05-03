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

        Returns:
            X: feature DataFrame (rows aligned, NaN dropped if configured)
            y: label Series aligned to X.index
            dates: pd.Series of dates aligned to X.index (for plotting / split)
        """
        feats: list[pd.DataFrame] = []

        if self.use_technical:
            tech = compute_technical_features(df)
            feats.append(tech)
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
            print(f"  external:  {ext.shape[1]} features")

        if self.use_trading_signals:
            ts = compute_trading_signal_features(df, self.trading_signals_yaml)
            feats.append(ts)
            print(f"  signals:   {ts.shape[1]} features")

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
