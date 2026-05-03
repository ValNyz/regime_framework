"""Classical sklearn / xgboost predictors."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .base import BasePredictor


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


class MLPPredictor(_SklearnBase):
    name = "MLP"
    needs_scaling = True

    def __init__(self) -> None:
        super().__init__()
        self.clf = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=400,
            random_state=42,
            early_stopping=False,
        )


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
        self.clf.fit(X_train.values, y_int)
        return self

    def predict(self, X_test, dates_test, df_test):
        pred_int = self.clf.predict(X_test.values)
        return np.array([self.idx_to_cls[int(i)] for i in pred_int], dtype=object)

    def feature_importances(self, X_test, y_test, n_repeats=3, random_state=42):
        if hasattr(self.clf, "feature_importances_"):
            imp = np.asarray(self.clf.feature_importances_, dtype=float)
            return pd.Series(imp, index=X_test.columns, name="importance").sort_values(ascending=False)
        return super().feature_importances(X_test, y_test, n_repeats, random_state)
