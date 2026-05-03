"""Re-export of data.external for the features namespace.

Keeps the conceptual split: feature builders live in `features/`, data loaders
live in `data/`. The feature pipeline composes them.
"""
from __future__ import annotations

from ..data.external import load_external_features as compute_external_features

__all__ = ["compute_external_features"]
