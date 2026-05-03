"""Base interface for label generators."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseLabeller(ABC):
    """Compute regime labels from OHLCV data.

    Subclasses must implement `compute(df)` returning a Series of object dtype
    with values in {"bull", "bear", ...}. Empty string "" means unlabelled
    (e.g. boundary bars where the forward window is incomplete).
    """

    name: str = "base"

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Return a label Series indexed identically to df."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
