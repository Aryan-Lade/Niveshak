"""
Unit tests for Day 16 forward-filtered Gaussian HMM regime detector.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from stock_engine.regime.hmm import (
    MarketRegimeHMM,
    forward_filter_probabilities,
)


def test_truncation_test_forward_filter():
    """
    CRITICAL LEAKAGE TEST:
    Probabilities at date t must be strictly identical when future observations
    after t are removed.
    """
    np.random.seed(42)
    T = 50
    dim = 3
    X = np.random.randn(T, dim)

    model = MarketRegimeHMM(n_components=3, random_state=42)
    model.fit(X[:30])

    full_probs = model.filter_probabilities(X)

    # For random cutoff points t, filtering X[:t+1] must equal full_probs[t]
    for cut_t in [5, 15, 25, 40, 49]:
        cut_X = X[: cut_t + 1]
        cut_probs = model.filter_probabilities(cut_X)
        # Probability vector at cut_t must be identical to full_probs[cut_t]
        np.testing.assert_allclose(
            cut_probs[-1],
            full_probs[cut_t],
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"Truncation violation at t={cut_t}",
        )


def test_probabilities_sum_to_one():
    """Verify that regime probabilities sum to 1.0 across all time steps."""
    np.random.seed(42)
    X = np.random.randn(30, 3)

    model = MarketRegimeHMM(n_components=3, random_state=42)
    model.fit(X)
    probs = model.filter_probabilities(X)

    row_sums = np.sum(probs, axis=1)
    np.testing.assert_allclose(row_sums, np.ones(30), rtol=1e-5, atol=1e-5)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_synthetic_two_regime_recovery():
    """Verify that HMM separates distinct synthetic regimes."""
    np.random.seed(42)
    # Regime 0: positive returns (+2%), low vol (0.5)
    r0 = np.column_stack([
        np.random.normal(2.0, 0.2, 50),
        np.random.normal(0.5, 0.1, 50),
        np.random.normal(12.0, 0.5, 50),
    ])
    # Regime 1: negative returns (-2%), high vol (2.0)
    r1 = np.column_stack([
        np.random.normal(-2.0, 0.2, 50),
        np.random.normal(2.0, 0.2, 50),
        np.random.normal(25.0, 1.0, 50),
    ])

    X = np.vstack([r0, r1])

    model = MarketRegimeHMM(n_components=2, random_state=42)
    model.fit(X)
    probs = model.filter_probabilities(X)

    # Bull regime should dominate in first 50 rows
    assert np.mean(probs[:50, 0]) > 0.80
    # Bear regime should dominate in last 50 rows
    assert np.mean(probs[50:, 1]) > 0.80


def test_no_call_to_predict_or_predict_proba():
    """Assert that hmmlearn predict/predict_proba are NEVER invoked on test data."""
    X = np.random.randn(20, 3)
    model = MarketRegimeHMM(n_components=3, random_state=42)
    model.fit(X[:15])

    with patch.object(model.model, "predict") as mock_predict, \
         patch.object(model.model, "predict_proba") as mock_predict_proba:

        _ = model.filter_probabilities(X[15:])
        mock_predict.assert_not_called()
        mock_predict_proba.assert_not_called()
