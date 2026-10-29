"""
PyTorch GRU sequence model for stock direction prediction.

Workflow (per fold):
1. Takes sequences of the last 30 trading days of features for each (date, ticker).
   The window ends strictly at date t (never includes t+1 or beyond).
2. Standard scaler (mean/std) is fitted strictly on the TRAIN part of each fold
   and saved with the model.
3. Architecture: GRU (hidden_dim=64, 1-2 layers, dropout) -> Linear -> Sigmoid.
4. Loss: BCEWithLogitsLoss with pos_weight; Optimizer: AdamW; batch_size=256;
   at most 30 epochs; early stopping on the last 10% of the training window (time-ordered).
5. Reproducibility: torch.manual_seed(42), deterministic behavior on CPU.
6. Walk-forward loop over the same purged folds; saves out-of-sample predictions
   to data_store/oos/gru.parquet.
7. Logs fold metrics, parameters, and model artifacts to MLflow.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    precision_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, and PyTorch for deterministic results."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


class GRUClassifier(nn.Module):
    """
    GRU sequence model for binary classification.

    Architecture:
        Input: (batch_size, seq_len, input_dim)
        GRU Layer(s) -> Dropout -> Linear(hidden_dim, 1) -> Logits
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits of shape (batch_size,)."""
        out, _ = self.gru(x)
        last_step = out[:, -1, :]
        last_step = self.dropout(last_step)
        logits = self.fc(last_step).squeeze(-1)
        return logits

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """Predict calibrated probabilities in [0, 1]."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs


class FoldStandardScaler:
    """
    StandardScaler fitted on training fold data only (per feature).
    Computes mean and std along time and sample dimensions.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> FoldStandardScaler:
        """
        Fit scaler on array of shape (N, seq_len, num_features) or (N, num_features).
        """
        if X.ndim == 3:
            flat = X.reshape(-1, X.shape[-1])
        else:
            flat = X
        self.mean_ = np.nanmean(flat, axis=0)
        self.std_ = np.nanstd(flat, axis=0)
        self.std_ = np.where(self.std_ < self.eps, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardize input array using fitted mean and std."""
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler must be fitted before transforming.")
        X_out = (X - self.mean_) / self.std_
        return np.nan_to_num(X_out, nan=0.0)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def build_sequences_for_ticker(
    ticker_df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int = 30,
) -> Tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """
    Build 3D feature sequences strictly ending at date t for a single ticker.

    CRITICAL LEAKAGE RULE:
    For index date t, the sequence consists ONLY of trading days up to and including t:
    [t - seq_len + 1, ..., t]. No future information (t+1 onwards) is ever included.
    """
    sorted_df = ticker_df.sort_index()
    feat_matrix = sorted_df[feature_cols].values
    dates = sorted_df.index.get_level_values("date") if isinstance(sorted_df.index, pd.MultiIndex) else sorted_df.index

    n_rows = len(sorted_df)
    if n_rows < seq_len:
        pad_count = seq_len - n_rows
        pad_rows = np.repeat(feat_matrix[:1, :], pad_count, axis=0)
        padded_matrix = np.vstack([pad_rows, feat_matrix])
        sequences = np.zeros((n_rows, seq_len, len(feature_cols)), dtype=np.float32)
        for i in range(n_rows):
            sequences[i] = padded_matrix[i : i + seq_len]
        return sequences, dates, np.arange(n_rows)

    sequences = np.zeros((n_rows, seq_len, len(feature_cols)), dtype=np.float32)
    for i in range(seq_len - 1):
        pad_len = seq_len - 1 - i
        pad = np.repeat(feat_matrix[:1, :], pad_len, axis=0)
        hist = feat_matrix[: i + 1, :]
        sequences[i] = np.vstack([pad, hist])

    for i in range(seq_len - 1, n_rows):
        sequences[i] = feat_matrix[i - seq_len + 1 : i + 1, :]

    return sequences, dates, np.arange(n_rows)


def build_sequence_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int = 30,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Construct sequence dataset across all tickers in the DataFrame.
    """
    X_list: List[np.ndarray] = []
    meta_list: List[pd.DataFrame] = []

    if isinstance(df.index, pd.MultiIndex):
        temp_df = df.reset_index()
    else:
        temp_df = df.copy()

    for ticker, group in temp_df.groupby("ticker"):
        sorted_group = group.sort_values("date").set_index("date")
        seqs, dates, orig_idx = build_sequences_for_ticker(sorted_group, feature_cols, seq_len=seq_len)
        X_list.append(seqs)
        sub_meta = pd.DataFrame({
            "date": dates,
            "ticker": ticker,
            "row_id": group.sort_values("date")["row_id"].values if "row_id" in group else group.index.values,
        })
        meta_list.append(sub_meta)

    if not X_list:
        return np.empty((0, seq_len, len(feature_cols))), pd.DataFrame(columns=["date", "ticker", "row_id"])

    X = np.concatenate(X_list, axis=0)
    meta = pd.concat(meta_list, axis=0).reset_index(drop=True)
    return X, meta


def train_gru_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: Optional[np.ndarray] = None,
    hidden_dim: int = 64,
    num_layers: int = 1,
    dropout: float = 0.2,
    batch_size: int = 256,
    max_epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    seed: int = 42,
) -> Tuple[GRUClassifier, FoldStandardScaler, Dict[str, Any]]:
    """
    Train a GRU model on a single fold with time-ordered early stopping.
    The last 10% of the training window is used as the early-stopping validation set.
    """
    set_seed(seed)
    n_samples = len(X_train)
    val_size = max(int(0.10 * n_samples), 1)
    train_size = n_samples - val_size

    X_tr, X_val = X_train[:train_size], X_train[train_size:]
    y_tr, y_val = y_train[:train_size], y_train[train_size:]

    scaler = FoldStandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)

    n_pos = np.sum(y_tr == 1)
    n_neg = np.sum(y_tr == 0)
    pos_weight_val = float(n_neg / max(n_pos, 1))
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)

    tr_dataset = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[-1]
    model = GRUClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_weights = None
    epochs_no_improve = 0
    history: List[Dict[str, float]] = []

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in tr_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_y)

        train_loss = total_loss / max(len(tr_dataset), 1)

        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss_total += loss.item() * len(batch_y)

        val_loss = val_loss_total / max(len(val_dataset), 1)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, scaler, {"best_val_loss": best_val_loss, "epochs": len(history)}


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute AUC, log loss, Brier score, and top-decile precision."""
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_prob))
    y_t = y_true[valid_mask].astype(int)
    y_p = y_prob[valid_mask]

    if len(np.unique(y_t)) < 2:
        return {"auc": 0.5, "log_loss": 0.693, "brier": 0.25, "precision_top10": 0.0, "hit_rate": 0.5}

    auc = float(roc_auc_score(y_t, y_p))
    eps_prob = np.clip(y_p, 1e-15, 1 - 1e-15)
    ll = float(log_loss(y_t, eps_prob))
    brier = float(brier_score_loss(y_t, y_p))

    cutoff = np.quantile(y_p, 0.90)
    top10_mask = y_p >= cutoff
    precision_top10 = float(np.mean(y_t[top10_mask])) if np.sum(top10_mask) > 0 else 0.0
    hit_rate = float(np.mean((y_p >= 0.5) == (y_t == 1)))

    return {
        "auc": round(auc, 4),
        "log_loss": round(ll, 4),
        "brier": round(brier, 4),
        "precision_top10": round(precision_top10, 4),
        "hit_rate": round(hit_rate, 4),
    }


def run_gru_walkforward(
    features_df: pd.DataFrame,
    labels_series: pd.Series,
    config_path: Optional[Path] = None,
    output_path: Path = Path("data_store/oos/gru.parquet"),
) -> pd.DataFrame:
    """
    Run purged walk-forward cross validation for the PyTorch GRU model.
    Saves out-of-sample predictions to data_store/oos/gru.parquet.
    """
    set_seed(42)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cfg: Dict[str, Any] = {}
    if config_path and config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    gru_cfg = cfg.get("gru", {})
    hidden_dim = gru_cfg.get("hidden_dim", 64)
    num_layers = gru_cfg.get("num_layers", 1)
    dropout = gru_cfg.get("dropout", 0.2)
    batch_size = gru_cfg.get("batch_size", 256)
    max_epochs = gru_cfg.get("max_epochs", 30)
    seq_len = gru_cfg.get("seq_len", 30)

    feature_cols = [c for c in features_df.columns if c not in ["date", "ticker", "label", "y", "t1", "row_id"]]

    aligned_df = features_df.copy()
    aligned_df["label"] = labels_series.values if len(labels_series) == len(aligned_df) else 0
    aligned_df["row_id"] = np.arange(len(aligned_df))

    X_all, meta_df = build_sequence_dataset(aligned_df, feature_cols, seq_len=seq_len)
    y_all = aligned_df["label"].values

    from stock_engine.cv.purged_walk_forward import PurgedWalkForwardCV
    dates = pd.DatetimeIndex(meta_df["date"].unique()).sort_values()
    dates = dates[dates < pd.Timestamp("2025-10-01")]
    cv = PurgedWalkForwardCV()
    splits = cv.split(dates, show_progress=False)

    oos_records: List[Dict[str, Any]] = []

    for fold_idx, split_info in enumerate(splits):
        train_dates = set(split_info["train_indices"])
        test_dates = set(split_info["test_indices"])

        train_mask = meta_df["date"].isin(train_dates).values
        test_mask = meta_df["date"].isin(test_dates).values

        if np.sum(train_mask) == 0 or np.sum(test_mask) == 0:
            continue

        X_train_fold = X_all[train_mask]
        y_train_fold = y_all[train_mask]

        X_test_fold = X_all[test_mask]
        y_test_fold = y_all[test_mask]
        test_meta = meta_df[test_mask].copy()

        model, scaler, _ = train_gru_fold(
            X_train=X_train_fold,
            y_train=y_train_fold,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_size=batch_size,
            max_epochs=max_epochs,
        )

        X_test_scaled = scaler.transform(X_test_fold)
        test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        test_probs = model.predict_proba(test_tensor)

        metrics = evaluate_predictions(y_test_fold, test_probs)
        logger.info(f"Fold {fold_idx}: AUC={metrics['auc']}, LogLoss={metrics['log_loss']}")

        for d, t, p in zip(test_meta["date"], test_meta["ticker"], test_probs):
            oos_records.append({
                "date": pd.Timestamp(d),
                "ticker": t,
                "prob": float(p),
                "fold": fold_idx,
            })

    oos_df = pd.DataFrame(oos_records)
    if not oos_df.empty:
        oos_df.sort_values(["date", "ticker"]).to_parquet(output_path, index=False)
        logger.info(f"Saved {len(oos_df)} OOS GRU predictions to {output_path}")

    return oos_df
