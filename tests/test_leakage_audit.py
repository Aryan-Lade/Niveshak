from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from stock_engine.audit import (
    check_future_return_correlation,
    check_shuffled_label_auc,
    check_truncation,
)
from stock_engine.dataset import (
    build_model_matrix,
    load_holdout_matrix,
    load_model_matrix,
)


def test_holdout_loader_guard(tmp_path):
    dummy_df = pd.DataFrame({"feat": [1, 2], "label": [0, 1]})
    holdout_file = tmp_path / "holdout_matrix.parquet"
    dummy_df.to_parquet(holdout_file)

    # 1. Must refuse without explicit flag
    with pytest.raises(PermissionError, match="Access to final holdout is locked"):
        load_holdout_matrix(path=holdout_file, i_know_this_is_final=False)

    # 2. Must succeed with explicit flag
    loaded = load_holdout_matrix(path=holdout_file, i_know_this_is_final=True)
    assert len(loaded) == 2


def test_build_model_matrix_splits_at_holdout_date(tmp_path):
    dates = pd.bdate_range("2025-09-20", periods=20)
    idx = pd.MultiIndex.from_product([dates, ["ABC.NS"]], names=["date", "ticker"])
    features = pd.DataFrame({"f_ret": np.random.randn(len(idx))}, index=idx)
    labels = pd.DataFrame({"label": np.random.randint(0, 2, len(idx))}, index=idx)

    model_m, holdout_m = build_model_matrix(
        features=features,
        labels=labels,
        holdout_start="2025-10-01",
        output_dir=tmp_path,
    )

    # All model dates must be strictly before holdout_start
    m_dates = model_m.index.get_level_values("date")
    assert (m_dates < pd.Timestamp("2025-10-01")).all()

    # All holdout dates must be on or after holdout_start
    h_dates = holdout_m.index.get_level_values("date")
    assert (h_dates >= pd.Timestamp("2025-10-01")).all()


def test_shuffled_label_auc_near_half():
    rng = np.random.default_rng(42)
    n = 600
    X = pd.DataFrame(rng.normal(0, 1, size=(n, 6)), columns=[f"f_{i}" for i in range(6)])
    y = pd.Series(rng.integers(0, 2, size=n))

    res = check_shuffled_label_auc(X, y, n_splits=3, seed=42)
    assert res["passed"] is True
    assert 0.45 <= res["mean_auc"] <= 0.55


def test_future_return_correlation_check():
    rng = np.random.default_rng(99)
    n = 200
    dates = pd.bdate_range("2024-01-01", periods=n)
    idx = pd.MultiIndex.from_product([dates, ["XYZ.NS"]], names=["date", "ticker"])

    clean_feature = pd.Series(rng.normal(0, 1, n), index=idx)
    future_returns = pd.Series(rng.normal(0, 1, n), index=idx)

    df_clean = pd.DataFrame({"f_clean": clean_feature}, index=idx)
    res_clean = check_future_return_correlation(df_clean, future_returns, threshold=0.50)
    assert res_clean["passed"] is True

    # Artificially inject forward-return leakage
    leaky_feature = future_returns * 0.95 + rng.normal(0, 0.05, n)
    df_leaky = pd.DataFrame({"f_leaky": leaky_feature}, index=idx)
    res_leaky = check_future_return_correlation(df_leaky, future_returns, threshold=0.50)
    assert res_leaky["passed"] is False
    assert len(res_leaky["violations"]) == 1
