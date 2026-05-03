"""Classical predictors: LogReg, RandomForest, GBM, MLP (torch GPU), XGBoost."""
from __future__ import annotations

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
from torch.utils.data import DataLoader, TensorDataset

from .base import BasePredictor
from ..config import LABEL_ORDER


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _SklearnBase(BasePredictor):
    family = "classical"
    needs_scaling = False

    def __init__(self) -> None:
        self.scaler: StandardScaler | None = None
        self.clf = None  # set by subclass

    def fit(self, X_train, y_train, dates_train, df_train):
        if self.needs_scaling:
            self.scaler = StandardScaler()
            Xs = self.scaler.fit_transform(X_train)
        else:
            Xs = X_train.values
        self.clf.fit(Xs, y_train.values)
        return self

    def predict(self, X_test, dates_test, df_test):
        if self.needs_scaling and self.scaler is not None:
            Xs = self.scaler.transform(X_test)
        else:
            Xs = X_test.values
        return self.clf.predict(Xs)

    def feature_importances(self, X_test, y_test, n_repeats=3, random_state=42):
        """Native sklearn importances when available; else permutation fallback."""
        if hasattr(self.clf, "feature_importances_"):
            imp = np.asarray(self.clf.feature_importances_, dtype=float)
            return pd.Series(imp, index=X_test.columns, name="importance").sort_values(ascending=False)
        if hasattr(self.clf, "coef_"):
            coef = np.asarray(self.clf.coef_, dtype=float)
            # Multi-class: take L1 norm across classes; binary: take abs value
            imp = np.abs(coef).mean(axis=0) if coef.ndim == 2 else np.abs(coef)
            return pd.Series(imp, index=X_test.columns, name="importance").sort_values(ascending=False)
        return super().feature_importances(X_test, y_test, n_repeats, random_state)


class LogRegPredictor(_SklearnBase):
    name = "LogReg"
    needs_scaling = True

    def __init__(self) -> None:
        super().__init__()
        self.clf = LogisticRegression(max_iter=2000)


class RandomForestPredictor(_SklearnBase):
    name = "RandomForest"

    def __init__(self) -> None:
        super().__init__()
        self.clf = RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=42, n_jobs=-1
        )


class GBMPredictor(_SklearnBase):
    name = "GBM"

    def __init__(self) -> None:
        super().__init__()
        self.clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, random_state=42
        )


class ExtraTreesPredictor(_SklearnBase):
    """Extremely Randomized Trees: random thresholds reduce variance on noisy
    financial features. Typically ~1.5x faster than RandomForest on the same
    n_estimators, with comparable or slightly better OOS kappa.
    """
    name = "ExtraTrees"

    def __init__(self) -> None:
        super().__init__()
        self.clf = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )


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
    Adam, 40 epochs. ~84k params for ~160 features.
    """
    name = "MLP"
    family = "classical"

    def __init__(self, hidden=(256, 128, 64), dropout=0.2, epochs=40, batch_size=4096, lr=1e-3) -> None:
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.scaler: StandardScaler | None = None
        self.model: _MLPNet | None = None

    def fit(self, X_train, y_train, dates_train, df_train):
        device = _device()
        use_amp = device.type == "cuda"
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X_train.values).astype(np.float32)
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
        self.model = _MLPNet(Xs.shape[1], len(LABEL_ORDER), self.hidden, self.dropout).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss(weight=weight)
        amp_scaler = torch.amp.GradScaler("cuda") if use_amp else None

        self.model.train()
        for epoch in range(self.epochs):
            tot = 0.0
            n = 0
            for xb, yb in loader:
                xb = xb.to(device); yb = yb.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
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
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"      MLP epoch {epoch+1}/{self.epochs} loss={tot/max(n,1):.4f}")
        return self

    def predict(self, X_test, dates_test, df_test):
        device = _device()
        use_amp = device.type == "cuda"
        Xs = self.scaler.transform(X_test.values).astype(np.float32)
        idx_to_cls = {i: c for i, c in enumerate(LABEL_ORDER)}
        self.model.train(False)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            logits = self.model(torch.from_numpy(Xs).to(device))
            pred = torch.argmax(logits, dim=1).cpu().numpy()
        return np.array([idx_to_cls[int(i)] for i in pred], dtype=object)


class XGBoostPredictor(BasePredictor):
    name = "XGBoost"
    family = "classical"

    def __init__(self) -> None:
        from xgboost import XGBClassifier
        from ..config import LABEL_ORDER
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
        self.cls_to_idx = {c: i for i, c in enumerate(LABEL_ORDER)}
        self.idx_to_cls = {i: c for c, i in self.cls_to_idx.items()}

    def fit(self, X_train, y_train, dates_train, df_train):
        y_int = np.array([self.cls_to_idx[v] for v in y_train.values], dtype=np.int64)
        # Pass DataFrame directly so XGBoost preserves feature names end-to-end.
        self.clf.fit(X_train, y_int)
        return self

    def predict(self, X_test, dates_test, df_test):
        pred_int = self.clf.predict(X_test)
        return np.array([self.idx_to_cls[int(i)] for i in pred_int], dtype=object)

    def feature_importances(self, X_test, y_test, n_repeats=3, random_state=42):
        if hasattr(self.clf, "feature_importances_"):
            imp = np.asarray(self.clf.feature_importances_, dtype=float)
            return pd.Series(imp, index=X_test.columns, name="importance").sort_values(ascending=False)
        return super().feature_importances(X_test, y_test, n_repeats, random_state)


class LightGBMPredictor(BasePredictor):
    """LightGBM multiclass GBDT. Leaf-wise growth -> typically 2-3x faster
    than XGBoost on wide tabular feature sets, often matching or beating its
    accuracy. Uses int label encoding like the XGBoost wrapper.
    """
    name = "LightGBM"
    family = "classical"

    def __init__(self) -> None:
        from lightgbm import LGBMClassifier
        from ..config import LABEL_ORDER
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
        self.cls_to_idx = {c: i for i, c in enumerate(LABEL_ORDER)}
        self.idx_to_cls = {i: c for c, i in self.cls_to_idx.items()}

    def fit(self, X_train, y_train, dates_train, df_train):
        y_int = np.array([self.cls_to_idx[v] for v in y_train.values], dtype=np.int64)
        # Pass DataFrame directly so LightGBM preserves feature names end-to-end
        # (avoids sklearn warning about feature-name mismatch on predict).
        self.clf.fit(X_train, y_int)
        return self

    def predict(self, X_test, dates_test, df_test):
        pred_int = self.clf.predict(X_test)
        return np.array([self.idx_to_cls[int(i)] for i in pred_int], dtype=object)

    def feature_importances(self, X_test, y_test, n_repeats=3, random_state=42):
        if hasattr(self.clf, "feature_importances_"):
            imp = np.asarray(self.clf.feature_importances_, dtype=float)
            return pd.Series(imp, index=X_test.columns, name="importance").sort_values(ascending=False)
        return super().feature_importances(X_test, y_test, n_repeats, random_state)
