"""Common base for pretrained-model predictors.

Subclasses must implement:
  - `_load_model()`        — load the HF / package model lazily on first use
  - `_forecast(context, horizon)` — forecast from a 1-D price array, return forecast array
  - (optional) `_embed(context)`  — return a (D,) embedding of the past window

Both modes implemented at this base level:
  - zero_shot: forecast next horizon → label = sign(mean(forecast) - close_now)
  - fine_tuned: rolling embeddings on train / test → small classifier head

Subclass tips:
  - Set `class_var MODEL_ID` to the HF Hub repo id
  - Override `_default_context_len` if the model has a hard context cap
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ..base import BasePredictor
from ...config import LABEL_ORDER


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasePretrainedPredictor(BasePredictor):
    family = "pretrained"

    # Subclasses override
    MODEL_ID: str = ""
    _default_context_len: int = 512
    _default_horizon: int = 24
    _supports_embedding: bool = False  # set True in subclass if _embed implemented

    def __init__(
        self,
        mode: str = "zero_shot",       # "zero_shot" | "fine_tuned"
        context_len: Optional[int] = None,
        horizon: Optional[int] = None,
        head: str = "logreg",          # only used if mode == "fine_tuned"
        device: Optional[str] = None,
        embed_subsample: int = 4,      # compute embedding every N bars (zero-shot uses every bar)
    ) -> None:
        self.mode = mode
        self.context_len = int(context_len or self._default_context_len)
        self.horizon = int(horizon or self._default_horizon)
        self.head = head
        self.device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_subsample = int(embed_subsample)
        self._model = None
        self._head_clf = None
        self._head_scaler: StandardScaler | None = None

    # ----- subclass interface ------------------------------------------------
    @abstractmethod
    def _load_model(self) -> None:
        """Load the underlying model into self._model."""

    @abstractmethod
    def _forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        """Forecast `horizon` future values from a 1-D `context` array."""

    def _embed(self, context: np.ndarray) -> np.ndarray:
        """Optional: return a 1-D embedding of `context`. Default: not supported."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement _embed")

    # ----- common implementation ---------------------------------------------
    def _ensure_model(self) -> None:
        if self._model is None:
            print(f"      loading {self.name} ({self.MODEL_ID}) on {self.device_str}...")
            self._load_model()

    def _zero_shot_predict(self, df: pd.DataFrame) -> np.ndarray:
        """Slide the context over every bar t, forecast horizon, label by SLOPE sign.

        Comparing mean(forecast) vs close_now is biased by the asset's intrinsic
        drift (e.g. BTC has +5%/year drift → almost every horizon ends above
        close_now → always 'bull'). Instead we fit a linear trend on the forecast
        path itself and label by sign of slope: if the forecast is rising over
        the horizon → bull, falling → bear. This is asset-drift-invariant.
        """
        self._ensure_model()
        close = df["close"].to_numpy(dtype=np.float64)
        n = len(close)
        out = np.full(n, "", dtype=object)
        x = np.arange(self.horizon, dtype=np.float64)
        x_dev = x - x.mean()
        SS_x = float((x_dev ** 2).sum())
        for t in tqdm(range(self.context_len, n), desc=f"      {self.name}-zs", leave=False):
            ctx = close[t - self.context_len : t]
            try:
                fc = self._forecast(ctx, self.horizon)
                fc = np.asarray(fc, dtype=np.float64)[: self.horizon]
                if len(fc) < 2:
                    out[t] = "bull"
                    continue
                # Regress forecast on time → slope sign = predicted direction
                y_dev = fc - fc.mean()
                slope = float((x_dev[: len(fc)] * y_dev).sum()) / max(SS_x, 1e-12)
                # Normalise slope by current price for invariance across assets
                norm_slope = slope / max(close[t - 1], 1e-12)
                out[t] = "bull" if norm_slope > 0 else "bear"
            except Exception as e:
                out[t] = "bull"
                if t == self.context_len:
                    print(f"        WARN: forecast error: {e}")
        return out

    def _embed_series(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (idx, embeddings) where idx[k] is the bar position at which embedding k was computed."""
        if not self._supports_embedding:
            raise RuntimeError(f"{self.name} does not support embedding extraction")
        self._ensure_model()
        if df is None or "close" not in df.columns:
            raise ValueError(
                f"{self.name} _embed_series: df missing 'close' column"
            )
        close = df["close"].to_numpy(dtype=np.float64)
        n = len(close)
        if n <= self.context_len:
            raise ValueError(
                f"{self.name} _embed_series: only {n} bars provided but context_len="
                f"{self.context_len} required. In multi-coin training mode the runner "
                f"sets df_train empty (no single price series across stacked coins) — "
                f"pretrained fine_tuned predictors are not compatible with multi-coin "
                f"runs. Disable them via cfg.predictors.disabled or use a non-multi-coin run."
            )
        idxs = list(range(self.context_len, n, self.embed_subsample))
        if not idxs:
            raise ValueError(
                f"{self.name} _embed_series: no embedding indices generated "
                f"(n={n}, context_len={self.context_len}, step={self.embed_subsample})"
            )
        embs = []
        for t in tqdm(idxs, desc=f"      {self.name}-emb", leave=False):
            ctx = close[t - self.context_len : t]
            try:
                e = self._embed(ctx)
                embs.append(e)
            except Exception as ex:
                if t == idxs[0]:
                    print(f"        WARN: embed error: {ex}")
                embs.append(np.zeros(self._default_context_len, dtype=np.float32))
        return np.array(idxs), np.stack(embs).astype(np.float32)

    def _fit_head(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._head_scaler = StandardScaler()
        Xs = self._head_scaler.fit_transform(X_train)
        if self.head == "logreg":
            self._head_clf = LogisticRegression(max_iter=2000, C=1.0)
        else:
            from sklearn.neural_network import MLPClassifier
            self._head_clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42)
        self._head_clf.fit(Xs, y_train)

    def _predict_head(self, X_test: np.ndarray) -> np.ndarray:
        Xs = self._head_scaler.transform(X_test) if self._head_scaler is not None else X_test
        return self._head_clf.predict(Xs)

    # ----- BasePredictor interface -------------------------------------------
    def fit(self, X_train, y_train, dates_train, df_train):
        if self.mode == "zero_shot":
            return self  # nothing to train

        # fine_tuned: extract embeddings on train, train head
        idxs, embs = self._embed_series(df_train)
        # Map sampled embeddings to labels at the same bar position
        cls_to_idx = {c: i for i, c in enumerate(LABEL_ORDER)}
        y_arr = np.array([cls_to_idx[v] for v in y_train.values], dtype=np.int64)
        # df_train shares index with y_train; idxs are positional
        positions_to_label = idxs[idxs < len(y_arr)]
        y_subset = y_arr[positions_to_label]
        X_subset = embs[: len(positions_to_label)]
        self._fit_head(X_subset, y_subset)
        return self

    def predict(self, X_test, dates_test, df_test):
        if self.mode == "zero_shot":
            return self._zero_shot_predict(df_test)

        # fine_tuned: extract embeddings on test, predict via head, propagate
        idxs, embs = self._embed_series(df_test)
        pred_int = self._predict_head(embs)
        idx_to_cls = {i: c for i, c in enumerate(LABEL_ORDER)}
        out = np.full(len(df_test), "", dtype=object)
        for i, t in enumerate(idxs):
            if t < len(out):
                out[t] = idx_to_cls[int(pred_int[i])]
        # Forward-fill empty positions
        last = ""
        for i in range(len(out)):
            if out[i] == "":
                out[i] = last if last else "bull"
            else:
                last = out[i]
        return out
