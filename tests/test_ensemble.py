"""
Unit tests for Day 17 stacking meta-model and ablation framework.
"""

import numpy as np
import pandas as pd
import pytest

from stock_engine.ensemble.stack import (
    MetaStacker,
    compute_logit,
    run_leak_free_stacking,
    simple_average_ensemble,
)
from stock_engine.ensemble.ablation import run_ablation_study, compute_bootstrap_sharpe_ci


def test_fold_order_leak_free():
    """
    CRITICAL LEAKAGE TEST:
    Meta-model for fold k must be trained ONLY on out-of-sample base-model
    predictions from folds strictly before k (< k).
    """
    dates = pd.date_range("2023-01-01", periods=12, freq="D")
    df_lgbm = pd.DataFrame({
        "date": dates,
        "ticker": ["AAPL"] * 12,
        "prob": np.linspace(0.4, 0.7, 12),
        "fold": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
    })
    df_gru = pd.DataFrame({
        "date": dates,
        "ticker": ["AAPL"] * 12,
        "prob": np.linspace(0.45, 0.65, 12),
        "fold": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
    })

    labels_df = pd.DataFrame({
        "date": dates,
        "ticker": ["AAPL"] * 12,
        "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })

    stacked_df = run_leak_free_stacking(
        df_lgbm,
        df_gru,
        labels_df=labels_df,
        min_train_samples=2,
    )

    assert len(stacked_df) == 12
    # Fold 0 must use fallback simple average
    fold0_rows = stacked_df[stacked_df["fold"] == 0]
    assert all(fold0_rows["method"] == "simple_average_fallback")

    # Later folds must use meta_logistic_regression
    later_rows = stacked_df[stacked_df["fold"] > 0]
    assert all(later_rows["method"] == "meta_logistic_regression")


def test_simple_average_ensemble():
    """Verify simple average probability calculation."""
    df1 = pd.DataFrame({
        "date": [pd.Timestamp("2023-01-01")],
        "ticker": ["RELIANCE.NS"],
        "prob": [0.6],
        "fold": [0],
    })
    df2 = pd.DataFrame({
        "date": [pd.Timestamp("2023-01-01")],
        "ticker": ["RELIANCE.NS"],
        "prob": [0.8],
        "fold": [0],
    })

    res = simple_average_ensemble(df1, df2)
    assert len(res) == 1
    assert abs(res["prob"].iloc[0] - 0.7) < 1e-6


def test_compute_logit_bounds():
    """Verify logit calculation handles probabilities near 0 and 1 safely."""
    probs = np.array([0.0, 0.5, 1.0])
    logits = compute_logit(probs)
    assert not np.any(np.isinf(logits))
    assert not np.any(np.isnan(logits))
    assert abs(logits[1]) < 1e-5  # logit(0.5) == 0


def test_ablation_determinism(tmp_path):
    """Verify ablation report generation is reproducible and deterministic."""
    df1 = run_ablation_study(reports_dir=tmp_path)
    df2 = run_ablation_study(reports_dir=tmp_path)

    pd.testing.assert_frame_equal(df1, df2)
    assert (tmp_path / "ablation.md").exists()
