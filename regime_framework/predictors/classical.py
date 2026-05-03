"""Classical predictors: LogReg, RandomForest, ExtraTrees, GBM, MLP, XGBoost,
LightGBM. Each predictor accepts a `finetune` flag — when True, the runner
warm-starts it from the previous CV fold's state via the predictor's
`_warm_fit` method. Walk-forward CV only (the runner filters FT instances out
under leave_one_out / single-split because warm-starting would leak).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.amp import GradScaler, autocast  # type: ignore[attr-defined]
from torch.utils.data import DataLoader, TensorDataset

from .base import BasePredictor
from ..config import LABEL_ORDER


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_torch(seed: int = 42) -> None:
    """Make torch + CUDA initialization deterministic so cold and FT variants
    of MLP/GRU/LSTM/TST start from identical weights at fold 1. Called before
    model construction and DataLoader creation in every fit() so the shuffle
    order is also reproducible — leaves only warm-start as the variable when
    comparing cold vs FT κ at fold N+1.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _reorder_proba_to_label_order(p: np.ndarray, src_classes: list) -> np.ndarray:
    """Reorder a (n, k) probability matrix so its columns match LABEL_ORDER.
    Used to align sklearn classifier outputs (which order columns by their
    classes_ attribute, typically alphabetical) with the framework convention.
    """
    out = np.zeros((p.shape[0], len(LABEL_ORDER)), dtype=p.dtype)
    for src_idx, cls in enumerate(src_classes):
        if cls in LABEL_ORDER:
            tgt_idx = LABEL_ORDER.index(cls)
            out[:, tgt_idx] = p[:, src_idx]
    return out


# ---------------------------------------------------------------------------
# Sklearn-style base: cold/warm-fit dispatcher + shared predict/importance
# ---------------------------------------------------------------------------
class _SklearnBase(BasePredictor):
    family = "classical"
    needs_scaling = False
    base_name: str = ""
    supports_finetune: bool = False  # subclass flips True if it has a real _warm_fit

    def __init__(self, finetune: bool = False) -> None:
        self.finetune = bool(finetune)
        self.is_finetune = self.finetune
        self.name = self.base_name + ("-FT" if self.finetune else "")
        self.scaler: StandardScaler | None = None
        # `clf` is typed Any: subclasses assign different sklearn classifier
        # types, no shared interface in the stubs.
        self.clf: Any = None
        self._fitted = False

    # Public lifecycle: dispatches between cold and warm.
    def fit(self, X_train, y_train, dates_train, df_train):
        if self.finetune and self._has_prior_state():
            return self._warm_fit(X_train, y_train, dates_train, df_train)
        return self._cold_fit(X_train, y_train, dates_train, df_train)

    def _cold_fit(self, X_train, y_train, dates_train, df_train):
        if self.needs_scaling:
            self.scaler = StandardScaler()
            Xs = self.scaler.fit_transform(X_train)
        else:
            Xs = X_train.values
        self.clf.fit(Xs, y_train.values)
        self._fitted = True
        return self

    def _warm_fit(self, X_train, y_train, dates_train, df_train):
        """Default: degrade to cold. Override in subclasses that truly support
        warm-start (RandomForest/ExtraTrees via warm_start=True, LogReg via
        SAGA solver)."""
        return self._cold_fit(X_train, y_train, dates_train, df_train)

    def _has_prior_state(self) -> bool:
        return self._fitted

    def predict(self, X_test, dates_test, df_test):
        if self.needs_scaling and self.scaler is not None:
            Xs = self.scaler.transform(X_test)
        else:
            Xs = X_test.values
        return self.clf.predict(Xs)

    def predict_proba(self, X_test, dates_test, df_test):
        """Return probabilities reordered to LABEL_ORDER columns. None if the
        underlying classifier doesn't expose predict_proba.
        """
        if not hasattr(self.clf, "predict_proba"):
            return None
        if self.needs_scaling and self.scaler is not None:
            Xs = self.scaler.transform(X_test)
        else:
            Xs = X_test.values
        p = self.clf.predict_proba(Xs)
        return _reorder_proba_to_label_order(p, list(self.clf.classes_))

    def feature_importances(self, X_test, y_test, n_repeats=3, random_state=42):
        """Native sklearn importances when available; else permutation fallback."""
        if hasattr(self.clf, "feature_importances_"):
            imp = np.asarray(self.clf.feature_importances_, dtype=float)
            return pd.Series(imp, index=X_test.columns, name="importance").sort_values(ascending=False)
        if hasattr(self.clf, "coef_"):
            coef = np.asarray(self.clf.coef_, dtype=float)
            imp = np.abs(coef).mean(axis=0) if coef.ndim == 2 else np.abs(coef)
            return pd.Series(imp, index=X_test.columns, name="importance").sort_values(ascending=False)
        return super().feature_importances(X_test, y_test, n_repeats, random_state)


# ---------------------------------------------------------------------------
# Linear & tree ensembles
# ---------------------------------------------------------------------------
class LogRegPredictor(_SklearnBase):
    """Multinomial logistic regression. With finetune=True, enables
    warm_start=True so the lbfgs optimizer continues from prior coef_ instead
    of restarting (same solver as cold — lbfgs supports warm_start in modern
    sklearn). Cost overhead at fold 1 is zero; fold 2+ converges faster.
    """
    base_name = "LogReg"
    needs_scaling = True
    supports_finetune = True

    def __init__(self, finetune: bool = False) -> None:
        super().__init__(finetune=finetune)
        self.clf = LogisticRegression(max_iter=2000, warm_start=finetune)

    def _warm_fit(self, X_train, y_train, dates_train, df_train):
        # warm_start=True keeps coef_/intercept_ as the lbfgs starting point.
        assert self.scaler is not None, "scaler should be fit on first cold call"
        Xs = self.scaler.fit_transform(X_train)  # rescale on new fold data
        self.clf.fit(Xs, y_train.values)
        return self


class RandomForestPredictor(_SklearnBase):
    """RandomForest. With finetune=True, sklearn's warm_start=True keeps prior
    trees; each fold ADDS n_per_fold new trees grown on the new fold's data.
    Ensemble vote gradually shifts toward recent regimes while retaining
    older-regime knowledge in the kept trees.
    """
    base_name = "RandomForest"
    supports_finetune = True

    def __init__(self, finetune: bool = False, n_per_fold: int = 100) -> None:
        super().__init__(finetune=finetune)
        self.n_per_fold = int(n_per_fold)
        self.clf = RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=42, n_jobs=-1,
            warm_start=finetune,
        )

    def _warm_fit(self, X_train, y_train, dates_train, df_train):
        self.clf.n_estimators += self.n_per_fold
        self.clf.fit(X_train.values, y_train.values)
        return self


class ExtraTreesPredictor(_SklearnBase):
    """Extremely Randomized Trees: random thresholds reduce variance on noisy
    financial features. ~1.5x faster than RandomForest, comparable κ. With
    finetune=True, behaves like RandomForestPredictor in FT mode.
    """
    base_name = "ExtraTrees"
    supports_finetune = True

    def __init__(self, finetune: bool = False, n_per_fold: int = 100) -> None:
        super().__init__(finetune=finetune)
        self.n_per_fold = int(n_per_fold)
        self.clf = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            warm_start=finetune,
        )

    def _warm_fit(self, X_train, y_train, dates_train, df_train):
        self.clf.n_estimators += self.n_per_fold
        self.clf.fit(X_train.values, y_train.values)
        return self


class GBMPredictor(_SklearnBase):
    """sklearn GradientBoostingClassifier — kept importable but slow; not in
    the default classical roster. No FT support (sklearn GBM warm_start adds
    trees but doesn't refit cleanly across distribution shifts).
    """
    base_name = "GBM"
    supports_finetune = False

    def __init__(self, finetune: bool = False) -> None:
        super().__init__(finetune=finetune)
        self.clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, random_state=42
        )


# ---------------------------------------------------------------------------
# PyTorch MLP (GPU)
# ---------------------------------------------------------------------------
class _MLPNet(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: tuple[int, ...] = (256, 128, 64), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPPredictor(BasePredictor):
    """PyTorch GPU MLP. Hidden 256-128-64, BN+GELU+Dropout, class-weighted CE,
    Adam, 40 epochs by default. With finetune=True, retains self.model across
    folds and continues training with fewer epochs at a smaller LR; the
    StandardScaler is refit each fold (input distribution drifts in crypto).
    """
    base_name = "MLP"
    family = "classical"
    supports_finetune = True

    def __init__(
        self, hidden=(256, 128, 64), dropout=0.2, epochs=40, batch_size=4096, lr=1e-3,
        finetune: bool = False, ft_epochs: int | None = None, ft_lr_scale: float = 0.5,
    ) -> None:
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.finetune = bool(finetune)
        self.is_finetune = self.finetune
        self.name = self.base_name + ("-FT" if self.finetune else "")
        self.ft_epochs = int(ft_epochs) if ft_epochs is not None else max(self.epochs // 4, 5)
        self.ft_lr_scale = float(ft_lr_scale)
        self.scaler: StandardScaler | None = None
        self.model: _MLPNet | None = None

    def fit(self, X_train, y_train, dates_train, df_train):
        if self.finetune and self.model is not None:
            return self._warm_fit(X_train, y_train, dates_train, df_train)
        return self._cold_fit(X_train, y_train, dates_train, df_train)

    def _build_loader_and_weights(self, X_train, y_train):
        device = _device()
        self.scaler = StandardScaler()
        Xs = np.asarray(self.scaler.fit_transform(X_train.values), dtype=np.float32)
        cls_to_idx = {c: i for i, c in enumerate(LABEL_ORDER)}
        y = np.array([cls_to_idx[v] for v in y_train.values], dtype=np.int64)
        cls_counts = np.array([(y == i).sum() for i in range(len(LABEL_ORDER))], dtype=np.float32)
        w = (1.0 / np.maximum(cls_counts, 1))
        w = w * (len(LABEL_ORDER) / w.sum())
        weight = torch.from_numpy(w).float().to(device)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(Xs), torch.from_numpy(y)),
            batch_size=self.batch_size, shuffle=True,
        )
        return device, Xs, loader, weight

    def _train(self, loader, weight, device, lr: float, epochs: int, log_prefix: str):
        assert self.model is not None, "_train called before model build"
        use_amp = device.type == "cuda"
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=weight)
        amp_scaler = GradScaler("cuda") if use_amp else None
        self.model.train()
        log_every = max(1, epochs // 4)
        for epoch in range(epochs):
            tot = 0.0; n = 0
            for xb, yb in loader:
                xb = xb.to(device); yb = yb.to(device)
                optimizer.zero_grad()
                with autocast("cuda", enabled=use_amp):
                    logits = self.model(xb)
                    loss = criterion(logits, yb)
                if amp_scaler is not None:
                    amp_scaler.scale(loss).backward()
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                tot += float(loss.item()); n += 1
            if (epoch + 1) % log_every == 0 or epoch == 0:
                print(f"      {log_prefix} epoch {epoch+1}/{epochs} loss={tot/max(n,1):.4f}")

    def _cold_fit(self, X_train, y_train, dates_train, df_train):
        _seed_torch()
        device, Xs, loader, weight = self._build_loader_and_weights(X_train, y_train)
        self.model = _MLPNet(Xs.shape[1], len(LABEL_ORDER), self.hidden, self.dropout).to(device)
        self._train(loader, weight, device, lr=self.lr, epochs=self.epochs, log_prefix=self.name)
        return self

    def _warm_fit(self, X_train, y_train, dates_train, df_train):
        _seed_torch()
        device, _, loader, weight = self._build_loader_and_weights(X_train, y_train)
        # Keep self.model weights — refit only scaler + continue training.
        self._train(loader, weight, device, lr=self.lr * self.ft_lr_scale,
                    epochs=self.ft_epochs, log_prefix=self.name)
        return self

    def predict(self, X_test, dates_test, df_test):
        assert self.scaler is not None and self.model is not None, "predict() called before fit()"
        device = _device()
        use_amp = device.type == "cuda"
        Xs = np.asarray(self.scaler.transform(X_test.values), dtype=np.float32)
        idx_to_cls = {i: c for i, c in enumerate(LABEL_ORDER)}
        self.model.train(False)
        with torch.no_grad(), autocast("cuda", enabled=use_amp):
            logits = self.model(torch.from_numpy(Xs).to(device))
            pred = torch.argmax(logits, dim=1).cpu().numpy()
        return np.array([idx_to_cls[int(i)] for i in pred], dtype=object)

    def predict_proba(self, X_test, dates_test, df_test):
        assert self.scaler is not None and self.model is not None, "predict_proba() called before fit()"
        device = _device()
        use_amp = device.type == "cuda"
        Xs = np.asarray(self.scaler.transform(X_test.values), dtype=np.float32)
        self.model.train(False)
        with torch.no_grad(), autocast("cuda", enabled=use_amp):
            logits = self.model(torch.from_numpy(Xs).to(device))
            proba = torch.softmax(logits.float(), dim=1).cpu().numpy()
        return proba.astype(np.float32)


# ---------------------------------------------------------------------------
# XGBoost / LightGBM (gradient-boosted decision trees with native warm-start)
# ---------------------------------------------------------------------------
class _GBDTBase(BasePredictor):
    """Shared scaffolding for XGBoost and LightGBM wrappers — both use int
    label encoding and dispatch fit() to cold/warm paths.
    """
    family = "classical"
    base_name: str = ""
    supports_finetune = True

    def __init__(self, finetune: bool = False) -> None:
        self.finetune = bool(finetune)
        self.is_finetune = self.finetune
        self.name = self.base_name + ("-FT" if self.finetune else "")
        self.cls_to_idx = {c: i for i, c in enumerate(LABEL_ORDER)}
        self.idx_to_cls = {i: c for c, i in self.cls_to_idx.items()}
        self.clf: Any = None  # subclass instantiates concrete XGB/LGB classifier

    def fit(self, X_train, y_train, dates_train, df_train):
        if self.finetune and self._has_prior_state():
            return self._warm_fit(X_train, y_train, dates_train, df_train)
        return self._cold_fit(X_train, y_train, dates_train, df_train)

    def _y_int(self, y_train: pd.Series) -> np.ndarray:
        return np.array([self.cls_to_idx[v] for v in y_train.values], dtype=np.int64)

    def predict(self, X_test, dates_test, df_test):
        pred_int = self.clf.predict(X_test)
        return np.array([self.idx_to_cls[int(i)] for i in pred_int], dtype=object)

    def predict_proba(self, X_test, dates_test, df_test):
        """XGBoost/LightGBM return columns indexed by int class id, which by
        construction matches LABEL_ORDER (we set cls_to_idx = enumerate(LABEL_ORDER)).
        """
        if not hasattr(self.clf, "predict_proba"):
            return None
        return np.asarray(self.clf.predict_proba(X_test), dtype=np.float32)

    def feature_importances(self, X_test, y_test, n_repeats=3, random_state=42):
        if hasattr(self.clf, "feature_importances_"):
            imp = np.asarray(self.clf.feature_importances_, dtype=float)
            return pd.Series(imp, index=X_test.columns, name="importance").sort_values(ascending=False)
        return super().feature_importances(X_test, y_test, n_repeats, random_state)


class XGBoostPredictor(_GBDTBase):
    base_name = "XGBoost"

    def __init__(self, finetune: bool = False) -> None:
        from xgboost import XGBClassifier
        super().__init__(finetune=finetune)
        self.clf = XGBClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        )
        self._prev_booster = None

    def _has_prior_state(self) -> bool:
        return self._prev_booster is not None

    def _cold_fit(self, X_train, y_train, dates_train, df_train):
        self.clf.fit(X_train, self._y_int(y_train))
        if self.finetune:
            self._prev_booster = self.clf.get_booster()
        return self

    def _warm_fit(self, X_train, y_train, dates_train, df_train):
        self.clf.fit(X_train, self._y_int(y_train), xgb_model=self._prev_booster)
        self._prev_booster = self.clf.get_booster()
        return self


class LightGBMPredictor(_GBDTBase):
    """LightGBM multiclass GBDT. Leaf-wise growth — typically 2-3x faster than
    XGBoost on wide tabular feature sets. With finetune=True, warm-starts from
    the previous fold's booster via init_model=.
    """
    base_name = "LightGBM"

    def __init__(self, finetune: bool = False) -> None:
        from lightgbm import LGBMClassifier
        super().__init__(finetune=finetune)
        self.clf = LGBMClassifier(
            n_estimators=600,
            num_leaves=63,
            max_depth=-1,
            learning_rate=0.05,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            objective="multiclass",
            num_class=len(LABEL_ORDER),
            verbosity=-1,
        )

    def _has_prior_state(self) -> bool:
        return getattr(self.clf, "booster_", None) is not None

    def _cold_fit(self, X_train, y_train, dates_train, df_train):
        self.clf.fit(X_train, self._y_int(y_train))
        return self

    def _warm_fit(self, X_train, y_train, dates_train, df_train):
        self.clf.fit(X_train, self._y_int(y_train), init_model=self.clf.booster_)
        return self
