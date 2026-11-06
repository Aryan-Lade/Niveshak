from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from stock_engine.cv.purged_walk_forward import PurgedWalkForwardCV
from stock_engine.tuning.optuna_search import (
    _create_inner_splits,
    run_optuna_walk_forward,
    tune_outer_fold,
)


def _make_synthetic_features(n_dates: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    tickers = ["T1.NS", "T2.NS"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])

    cols = ["ret_1d", "ret_5d", "ret_20d", "vol_20d", "nifty_ret_1d"]
    data = rng.normal(0, 1, size=(len(idx), len(cols)))
    return pd.DataFrame(data, index=idx, columns=cols)


def test_inner_splits_disjoint_and_ordered():
    dates = pd.bdate_range("2020-01-01", periods=100)
    splits = _create_inner_splits(dates, n_splits=3, embargo_days=2)
    assert len(splits) >= 2

    for in_tr, in_te in splits:
        s_tr = set(in_tr)
        s_te = set(in_te)
        # Train and test within inner fold must be disjoint
        assert s_tr.isdisjoint(s_te)
        # Time ordering: max train index must precede min test index
        assert max(in_tr) < min(in_te)


def test_tuning_never_touches_outer_test(tmp_path):
    """
    Test index assertion: outer test dates are never present in inner tuning dates.
    """
    dates = pd.bdate_range("2018-01-01", periods=500)
    cv = PurgedWalkForwardCV(
        initial_train_years=1,
        test_window_months=3,
        step_months=3,
        embargo_days=2,
    )
    outer_splits = cv.split(dates)
    first_fold = outer_splits[0]

    tr_dates = set(first_fold["train_timestamps"])
    te_dates = set(first_fold["test_timestamps"])

    # Assert outer train and test disjoint
    assert tr_dates.isdisjoint(te_dates)

    # When inner splits are created from outer train timestamps, none can ever be in outer test
    unique_tr_dates = pd.DatetimeIndex(sorted(list(tr_dates)))
    inner_splits = _create_inner_splits(unique_tr_dates, n_splits=3)

    for in_tr, in_te in inner_splits:
        for idx in in_tr:
            assert unique_tr_dates[idx] not in te_dates
        for idx in in_te:
            assert unique_tr_dates[idx] not in te_dates


def test_optuna_search_synthetic_dry_run(tmp_path):
    feat_df = _make_synthetic_features(n_dates=1200, seed=77)
    feat_path = tmp_path / "features.parquet"
    feat_df.to_parquet(feat_path)

    out_p = tmp_path / "lgbm_tuned.parquet"
    rep_p = tmp_path / "lgbm_walkforward.md"

    res = run_optuna_walk_forward(
        features_path=feat_path,
        config_dir=Path("configs"),
        out_path=out_p,
        report_path=rep_p,
        holdout_start="2025-10-01",
        n_trials=2,  # Quick test
        dry_run=True,
    )

    assert not res.empty
    assert "prob" in res.columns
    assert "fold" in res.columns
    assert (res["prob"] >= 0).all() and (res["prob"] <= 1).all()
    assert out_p.exists()
    assert rep_p.exists()
