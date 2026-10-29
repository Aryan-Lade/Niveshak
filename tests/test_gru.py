"""
Unit tests for Day 15 PyTorch GRU sequence model.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from stock_engine.models.gru import (
    GRUClassifier,
    FoldStandardScaler,
    build_sequences_for_ticker,
    build_sequence_dataset,
    train_gru_fold,
    evaluate_predictions,
    set_seed,
)


def test_sequence_windows_end_at_t():
    """Ensure sequence at index date t uses only information up to t (no look-ahead)."""
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    data = {
        "feat1": np.arange(10, dtype=float),
        "feat2": np.arange(10, dtype=float) * 2,
    }
    df = pd.DataFrame(data, index=dates)

    seq_len = 4
    sequences, ret_dates, _ = build_sequences_for_ticker(df, ["feat1", "feat2"], seq_len=seq_len)

    assert len(sequences) == 10
    # For date at index 5 (6th date), the sequence must end with values from row 5
    # and contain rows [2, 3, 4, 5]
    expected_feat1 = df["feat1"].iloc[2:6].values
    np.testing.assert_array_equal(sequences[5, :, 0], expected_feat1)

    # For any cut date t, deleting rows after t does not change sequence at t
    for cut_idx in [4, 7, 9]:
        df_cut = df.iloc[: cut_idx + 1]
        seqs_cut, _, _ = build_sequences_for_ticker(df_cut, ["feat1", "feat2"], seq_len=seq_len)
        np.testing.assert_array_equal(seqs_cut[-1], sequences[cut_idx])


def test_scaler_fitted_on_train_only():
    """Verify scaler mean and std are derived strictly from training data."""
    train_data = np.array([
        [[10.0, 100.0], [20.0, 200.0]],
        [[30.0, 300.0], [40.0, 400.0]],
    ])  # shape (2, 2, 2)
    test_data = np.array([
        [[900.0, 9000.0], [1000.0, 10000.0]],
    ])

    scaler = FoldStandardScaler()
    scaler.fit(train_data)

    expected_mean = np.array([25.0, 250.0])
    np.testing.assert_allclose(scaler.mean_, expected_mean, rtol=1e-5)

    transformed_test = scaler.transform(test_data)
    # Ensure test data does not affect scaler.mean_
    np.testing.assert_allclose(scaler.mean_, expected_mean, rtol=1e-5)
    assert transformed_test.shape == test_data.shape


def test_tensor_shapes():
    """Verify GRU forward pass outputs correct tensor shapes."""
    batch_size = 16
    seq_len = 30
    input_dim = 12

    model = GRUClassifier(input_dim=input_dim, hidden_dim=64, num_layers=1, dropout=0.2)
    x = torch.randn(batch_size, seq_len, input_dim)
    logits = model(x)

    assert logits.shape == (batch_size,)
    probs = model.predict_proba(x)
    assert probs.shape == (batch_size,)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_model_overfits_tiny_synthetic_dataset():
    """Sanity check: model overfits a tiny synthetic dataset."""
    set_seed(42)
    N = 20
    seq_len = 10
    input_dim = 4

    X = np.random.randn(N, seq_len, input_dim).astype(np.float32)
    # Synthetic rule: if mean of last step > 0, label 1 else 0
    y = (X[:, -1, :].mean(axis=-1) > 0).astype(np.float32)

    model, scaler, info = train_gru_fold(
        X_train=X,
        y_train=y,
        hidden_dim=32,
        num_layers=1,
        dropout=0.0,
        batch_size=N,
        max_epochs=60,
        lr=0.02,
        patience=60,
        seed=42,
    )

    X_scaled = scaler.transform(X)
    probs = model.predict_proba(torch.tensor(X_scaled, dtype=torch.float32))
    preds = (probs >= 0.5).astype(float)
    acc = np.mean(preds == y)
    assert acc >= 0.80, f"Expected overfitting accuracy >= 0.80, got {acc}"


def test_reproducibility():
    """Deterministic seeds give identical predictions across two runs."""
    N = 10
    seq_len = 5
    input_dim = 3
    X = np.random.randn(N, seq_len, input_dim).astype(np.float32)
    y = np.random.randint(0, 2, size=N).astype(np.float32)

    model1, _, _ = train_gru_fold(X, y, hidden_dim=16, max_epochs=5, seed=42)
    p1 = model1.predict_proba(torch.tensor(X, dtype=torch.float32))

    model2, _, _ = train_gru_fold(X, y, hidden_dim=16, max_epochs=5, seed=42)
    p2 = model2.predict_proba(torch.tensor(X, dtype=torch.float32))

    np.testing.assert_allclose(p1, p2, rtol=1e-5, atol=1e-5)
