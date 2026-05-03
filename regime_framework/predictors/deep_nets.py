"""Deep neural net predictors: GRU, LSTM (PyTorch).

The MLP variant (formerly DeepMLPPredictor) has been moved to classical.py
as MLPPredictor since it operates on flat feature vectors like the other
classical predictors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .base import BasePredictor
from ..config import LABEL_ORDER


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _SeqRNN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_layers: int, n_classes: int, kind: str = "lstm", dropout: float = 0.2):
        super().__init__()
        if kind == "lstm":
            self.rnn = nn.LSTM(in_dim, hidden, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        elif kind == "gru":
            self.rnn = nn.GRU(in_dim, hidden, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        else:
            raise ValueError(kind)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


def _make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    n = len(X) - seq_len + 1
    if n <= 0:
        raise ValueError(f"Data too short for seq_len={seq_len}")
    seqs = np.lib.stride_tricks.sliding_window_view(X, (seq_len, X.shape[1])).squeeze(1)
    seqs = seqs[:n]
    targets = y[seq_len - 1:]
    return seqs, targets


class _SeqPredictor(BasePredictor):
    family = "deep"
    kind: str = "lstm"
    def __init__(self, seq_len=64, hidden=128, n_layers=2, epochs=20, batch_size=256, lr=1e-3, dropout=0.2):
        self.seq_len = int(seq_len)
        self.hidden = int(hidden)
        self.n_layers = int(n_layers)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.dropout = float(dropout)
        self.scaler: StandardScaler | None = None
        self.model: _SeqRNN | None = None

    def fit(self, X_train, y_train, dates_train, df_train):
        device = _device()
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X_train.values).astype(np.float32)
        cls_to_idx = {c: i for i, c in enumerate(LABEL_ORDER)}
        y = np.array([cls_to_idx[v] for v in y_train.values], dtype=np.int64)
        seqs, targets = _make_sequences(Xs, y, self.seq_len)

        cls_counts = np.array([(targets == i).sum() for i in range(len(LABEL_ORDER))], dtype=np.float32)
        w = (1.0 / np.maximum(cls_counts, 1))
        w = w * (len(LABEL_ORDER) / w.sum())
        weight = torch.from_numpy(w).float().to(device)

        loader = DataLoader(
            TensorDataset(torch.from_numpy(seqs.astype(np.float32)), torch.from_numpy(targets)),
            batch_size=self.batch_size, shuffle=True,
        )
        self.model = _SeqRNN(Xs.shape[1], self.hidden, self.n_layers, len(LABEL_ORDER), kind=self.kind, dropout=self.dropout).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss(weight=weight)

        self.model.train()
        for epoch in range(self.epochs):
            tot = 0.0
            n = 0
            for xb, yb in loader:
                xb = xb.to(device); yb = yb.to(device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                tot += float(loss.item()); n += 1
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"      {self.name} epoch {epoch+1}/{self.epochs} loss={tot/max(n,1):.4f}")
        return self

    def predict(self, X_test, dates_test, df_test):
        device = _device()
        Xs = self.scaler.transform(X_test.values).astype(np.float32)
        cls_to_idx = {c: i for i, c in enumerate(LABEL_ORDER)}
        idx_to_cls = {i: c for i, c in enumerate(LABEL_ORDER)}
        # We need labels too to build sequences with the right alignment;
        # we synthesize zeros and only use seqs.
        dummy_y = np.zeros(len(Xs), dtype=np.int64)
        seqs, _ = _make_sequences(Xs, dummy_y, self.seq_len)

        self.model.train(False)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(seqs.astype(np.float32)).to(device))
            pred = torch.argmax(logits, dim=1).cpu().numpy()

        out = np.full(len(Xs), "", dtype=object)
        out[self.seq_len - 1:] = [idx_to_cls[int(i)] for i in pred]
        # First seq_len-1 entries: forward-fill with first valid prediction
        if len(out) > self.seq_len - 1:
            out[: self.seq_len - 1] = out[self.seq_len - 1]
        return out


class GRUPredictor(_SeqPredictor):
    name = "GRU"
    kind = "gru"


class LSTMPredictor(_SeqPredictor):
    name = "LSTM"
    kind = "lstm"
