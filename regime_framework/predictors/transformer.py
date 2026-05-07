"""In-house TimeSeriesTransformer (TST) for regime classification.

Architecture:
  - Linear projection from input feature dim to d_model
  - Sinusoidal positional encoding
  - N stacked encoder blocks (multi-head self-attention + feed-forward)
  - Attention pooling over the sequence (learnable query)
  - Linear classifier head

Tuned for ~1-10M params; suitable for A6000 fine-tuning in seconds.
"""
from __future__ import annotations

import math

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


def _seed_torch(seed: int = 42) -> None:
    """Deterministic init so cold and FT variants share fold-1 starting weights."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):  # x: (B, L, D)
        return x + self.pe[:, : x.size(1)]


class _AttentionPool(nn.Module):
    """Single-query attention pool over the sequence dimension."""
    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) / math.sqrt(d_model))
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)

    def forward(self, x):
        q = self.query.expand(x.size(0), -1, -1)
        out, _ = self.attn(q, x, x, need_weights=False)
        return out.squeeze(1)


class _TST(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_hidden: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = _PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_hidden,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        # enable_nested_tensor=False: the nested-tensor fast path is not
        # compatible with norm_first=True (pre-LN), so torch emits a warning
        # at construction. We use pre-LN intentionally (more stable gradients);
        # explicitly disabling the option silences the warning without changing
        # behavior — it would have been disabled internally anyway.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False,
        )
        self.pool = _AttentionPool(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def forward(self, x):  # x: (B, L, F)
        h = self.pos(self.proj(x))
        h = self.encoder(h)
        h = self.pool(h)
        return self.head(h)


def _make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    n = len(X) - seq_len + 1
    if n <= 0:
        raise ValueError(f"Data too short for seq_len={seq_len}")
    seqs = np.lib.stride_tricks.sliding_window_view(X, (seq_len, X.shape[1])).squeeze(1)
    seqs = seqs[:n]
    targets = y[seq_len - 1:]
    return seqs, targets


class TimeSeriesTransformerPredictor(BasePredictor):
    base_name = "TST"
    family = "transformer"
    supports_finetune = True

    def __init__(
        self,
        seq_len: int = 96,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_hidden: int = 1024,
        dropout: float = 0.1,
        epochs: int = 30,
        batch_size: int = 1024,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        finetune: bool = False,
        ft_epochs: int | None = None,
        ft_lr_scale: float = 0.5,
        show_progress: bool = True,
    ) -> None:
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.ff_hidden = int(ff_hidden)
        self.dropout = float(dropout)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.finetune = bool(finetune)
        self.is_finetune = self.finetune
        self.name = self.base_name + ("-FT" if self.finetune else "")
        self.ft_epochs = int(ft_epochs) if ft_epochs is not None else max(self.epochs // 4, 5)
        self.ft_lr_scale = float(ft_lr_scale)
        self.show_progress = bool(show_progress)
        self.scaler: StandardScaler | None = None
        self.model: _TST | None = None

    def fit(self, X_train, y_train, dates_train, df_train):
        if self.finetune and self.model is not None:
            return self._warm_fit(X_train, y_train, dates_train, df_train)
        return self._cold_fit(X_train, y_train, dates_train, df_train)

    def _build_loader_and_weights(self, X_train, y_train):
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
        return device, Xs, loader, weight

    def _train(self, loader, weight, device, lr: float, epochs: int):
        assert self.model is not None, "_train called before model build"
        use_amp = device.type == "cuda"
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss(weight=weight)
        amp_scaler = GradScaler("cuda") if use_amp else None
        self.model.train()
        log_every = max(1, epochs // 6)
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
                    amp_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                tot += float(loss.item()); n += 1
            scheduler.step()
            if self.show_progress and ((epoch + 1) % log_every == 0 or epoch == 0):
                print(f"      {self.name} epoch {epoch+1}/{epochs} loss={tot/max(n,1):.4f} lr={scheduler.get_last_lr()[0]:.2e}")

    def _cold_fit(self, X_train, y_train, dates_train, df_train):
        _seed_torch()
        device, Xs, loader, weight = self._build_loader_and_weights(X_train, y_train)
        self.model = _TST(
            in_dim=Xs.shape[1],
            n_classes=len(LABEL_ORDER),
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            ff_hidden=self.ff_hidden,
            dropout=self.dropout,
        ).to(device)
        n_params = sum(p.numel() for p in self.model.parameters())
        if self.show_progress:
            print(f"      {self.name} parameters: {n_params/1e6:.2f}M  (device={device})")
        self._train(loader, weight, device, lr=self.lr, epochs=self.epochs)
        return self

    def _warm_fit(self, X_train, y_train, dates_train, df_train):
        _seed_torch()
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
        preds = []
        with torch.no_grad(), autocast("cuda", enabled=use_amp):
            # batched inference (seqs may be large)
            for i in range(0, len(seqs), self.batch_size):
                batch = torch.from_numpy(seqs[i : i + self.batch_size].astype(np.float32)).to(device)
                logits = self.model(batch)
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        pred_int = np.concatenate(preds)

        out = np.full(len(Xs), "", dtype=object)
        out[self.seq_len - 1:] = [idx_to_cls[int(i)] for i in pred_int]
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
        proba_chunks = []
        with torch.no_grad(), autocast("cuda", enabled=use_amp):
            for i in range(0, len(seqs), self.batch_size):
                batch = torch.from_numpy(seqs[i : i + self.batch_size].astype(np.float32)).to(device)
                logits = self.model(batch)
                proba_chunks.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        probas = np.concatenate(proba_chunks, axis=0)

        out = np.zeros((len(Xs), len(LABEL_ORDER)), dtype=np.float32)
        out[self.seq_len - 1:] = probas
        if len(out) > self.seq_len - 1:
            out[: self.seq_len - 1] = out[self.seq_len - 1]
        return out
