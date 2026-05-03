"""Deep neural net predictors: GRU, LSTM (PyTorch).

Each accepts a `finetune` flag — when True, warm-starts from prior CV fold
weights instead of rebuilding self.model on every fit. Walk-forward only.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.amp import GradScaler, autocast  # type: ignore[attr-defined]
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


def _make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
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
    base_name: str = ""
    supports_finetune = True

    def __init__(
        self, seq_len=64, hidden=128, n_layers=2, epochs=20, batch_size=1024,
        lr=1e-3, dropout=0.2,
        finetune: bool = False, ft_epochs: int | None = None, ft_lr_scale: float = 0.5,
    ):
        self.seq_len = int(seq_len)
        self.hidden = int(hidden)
        self.n_layers = int(n_layers)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.dropout = float(dropout)
        self.finetune = bool(finetune)
        self.is_finetune = self.finetune
        self.name = self.base_name + ("-FT" if self.finetune else "")
        self.ft_epochs = int(ft_epochs) if ft_epochs is not None else max(self.epochs // 4, 5)
        self.ft_lr_scale = float(ft_lr_scale)
        self.scaler: StandardScaler | None = None
        self.model: _SeqRNN | None = None

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
        seqs, targets = _make_sequences(Xs, y, self.seq_len)
        cls_counts = np.array([(targets == i).sum() for i in range(len(LABEL_ORDER))], dtype=np.float32)
        w = (1.0 / np.maximum(cls_counts, 1))
        w = w * (len(LABEL_ORDER) / w.sum())
        weight = torch.from_numpy(w).float().to(device)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(seqs.astype(np.float32)), torch.from_numpy(targets)),
            batch_size=self.batch_size, shuffle=True,
        )
        return device, Xs, loader, weight

    def _train(self, loader, weight, device, lr: float, epochs: int):
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
                print(f"      {self.name} epoch {epoch+1}/{epochs} loss={tot/max(n,1):.4f}")

    def _cold_fit(self, X_train, y_train, dates_train, df_train):
        device, Xs, loader, weight = self._build_loader_and_weights(X_train, y_train)
        self.model = _SeqRNN(Xs.shape[1], self.hidden, self.n_layers, len(LABEL_ORDER),
                             kind=self.kind, dropout=self.dropout).to(device)
        self._train(loader, weight, device, lr=self.lr, epochs=self.epochs)
        return self

    def _warm_fit(self, X_train, y_train, dates_train, df_train):
        device, _, loader, weight = self._build_loader_and_weights(X_train, y_train)
        self._train(loader, weight, device, lr=self.lr * self.ft_lr_scale, epochs=self.ft_epochs)
        return self

    def predict(self, X_test, dates_test, df_test):
        assert self.scaler is not None and self.model is not None, "predict() called before fit()"
        device = _device()
        use_amp = device.type == "cuda"
        Xs = np.asarray(self.scaler.transform(X_test.values), dtype=np.float32)
        idx_to_cls = {i: c for i, c in enumerate(LABEL_ORDER)}
        dummy_y = np.zeros(len(Xs), dtype=np.int64)
        seqs, _ = _make_sequences(Xs, dummy_y, self.seq_len)

        self.model.train(False)
        with torch.no_grad(), autocast("cuda", enabled=use_amp):
            logits = self.model(torch.from_numpy(seqs.astype(np.float32)).to(device))
            pred = torch.argmax(logits, dim=1).cpu().numpy()

        out = np.full(len(Xs), "", dtype=object)
        out[self.seq_len - 1:] = [idx_to_cls[int(i)] for i in pred]
        if len(out) > self.seq_len - 1:
            out[: self.seq_len - 1] = out[self.seq_len - 1]
        return out

    def predict_proba(self, X_test, dates_test, df_test):
        assert self.scaler is not None and self.model is not None, "predict_proba() called before fit()"
        device = _device()
        use_amp = device.type == "cuda"
        Xs = np.asarray(self.scaler.transform(X_test.values), dtype=np.float32)
        dummy_y = np.zeros(len(Xs), dtype=np.int64)
        seqs, _ = _make_sequences(Xs, dummy_y, self.seq_len)

        self.model.train(False)
        with torch.no_grad(), autocast("cuda", enabled=use_amp):
            logits = self.model(torch.from_numpy(seqs.astype(np.float32)).to(device))
            proba = torch.softmax(logits.float(), dim=1).cpu().numpy()

        out = np.zeros((len(Xs), len(LABEL_ORDER)), dtype=np.float32)
        out[self.seq_len - 1:] = proba
        if len(out) > self.seq_len - 1:
            out[: self.seq_len - 1] = out[self.seq_len - 1]
        return out


class GRUPredictor(_SeqPredictor):
    base_name = "GRU"
    kind = "gru"


class LSTMPredictor(_SeqPredictor):
    base_name = "LSTM"
    kind = "lstm"
