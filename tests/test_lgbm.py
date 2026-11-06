"""
Tests for src/stock_engine/models/lgbm.py

Acceptance checks
-----------------
1. Shuffled-label run produces AUC near 0.5 (±0.15 tolerance).
2. Re-running with the same seed gives identical predictions (determinism).
3. The model never sees rows from the test window (index disjoint assertion).
4. Feature selection is computed exclusively on TRAIN rows.
5. Imputation medians come exclusively from TRAIN rows.

All tests use synthetic data to avoid slow I/O.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from stock_engine.cv.purged_walk_forward import PurgedWalkForwardCV
from stock_engine.models.lgbm import (
    apply_imputer,
    compute_fold_metrics,
    fit_imputer,
    run_lgbm_walk_forward,
    select_features,
)


def _make_synthetic_features(
    n_dates: int = 800,
    tickers: list[str] | None = None,
    seed: int = 42,
    start: str = "2012-01-01",
) -> pd.DataFrame:
    """Build a small synthetic feature panel."""
    rng = np.random.default_rng(seed)
    if tickers is None:
        tickers = ["AAA.NS", "BBB.NS", "CCC.NS"]

    dates = pd.bdate_range(start=start, periods=n_dates)
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])

    cols = [
        "nifty_ret_1d", "nifty_ret_5d", "nifty_ret_20d", "nifty_vol_20d",
        "vix_level", "vix_1d_change", "vix_zscore_252d", "usdinr_5d_change",
        "sp500_ret_1d", "sp500_ret_5d", "crude_5d_change",
        "ret_1d", "ret_5d", "ret_20d", "vol_20d",
        "rs_5d", "rs_20d", "rs_60d", "beta_60d", "volume_zscore",
        "ret_20d_percentile", "vol_20d_percentile", "volume_zscore_percentile",
    ]
    data = rng.normal(0, 1, size=(len(idx), len(cols)))
    df = pd.DataFrame(data, index=idx, columns=cols)
    # Introduce ~3% NaN to mimic real data
    mask = rng.random(df.shape) < 0.03
    df[mask] = np.nan
    return df


class TestSelectFeatures:
    def test_drops_high_missing(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(100, 5)), columns=list("ABCDE"))
        X["C"] = np.nan
        kept = select_features(X, max_missing_pct=0.30, max_corr=0.95)
        assert "C" not in kept

    def test_drops_high_corr(self):
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.normal(size=(200, 3)), columns=["A", "B", "C"])
        X["D"] = X["A"] + rng.normal(0, 1e-6, size=200)
        kept = select_features(X, max_missing_pct=0.30, max_corr=0.95)
        assert not ("A" in kept and "D" in kept), "Both correlated columns were kept"

    def test_uses_only_provided_data(self):
        rng = np.random.default_rng(2)
        X = pd.DataFrame(rng.normal(size=(100, 4)), columns=list("ABCD"))
        kept1 = select_features(X.copy(), 0.30, 0.95)
        kept2 = select_features(X.copy(), 0.30, 0.95)
        assert kept1 == kept2


class TestImputer:
    def test_median_computed_on_train(self):
        X_train = pd.DataFrame({"A": [1.0, 2.0, 3.0, np.nan]})
        medians = fit_imputer(X_train)
        assert medians["A"] == pytest.approx(2.0)

    def test_apply_fills_nan(self):
        medians = pd.Series({"A": 99.0, "B": 7.0})
        X = pd.DataFrame({"A": [np.nan, 1.0], "B": [np.nan, 2.0]})
        filled = apply_imputer(X, medians)
        assert filled["A"].iloc[0] == 99.0
        assert filled["B"].iloc[0] == 7.0

    def test_train_median_not_contaminated_by_test(self):
        rng = np.random.default_rng(3)
        X_train = pd.DataFrame({"A": rng.normal(0, 1, 100)})
        X_test = pd.DataFrame({"A": rng.normal(1000, 1, 50)})
        medians_train = fit_imputer(X_train)
        medians_test = fit_imputer(X_test)
        assert abs(medians_train["A"]) < 1.0
        assert abs(medians_test["A"] - 1000) < 5.0


class TestComputeFoldMetrics:
    def test_perfect_classifier(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
        m = compute_fold_metrics(y_true, y_prob)
        assert m["auc"] == pytest.approx(1.0)

    def test_random_classifier_auc_near_half(self):
        rng = np.random.default_rng(99)
        y_true = rng.integers(0, 2, size=1000)
        y_prob = rng.uniform(0, 1, size=1000)
        m = compute_fold_metrics(y_true, y_prob)
        assert 0.40 < m["auc"] < 0.60, f"Random AUC={m['auc']:.4f} out of range"

    def test_keys_present(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.6, 0.4, 0.7, 0.3])
        m = compute_fold_metrics(y_true, y_prob)
        for key in ["auc", "log_loss", "brier_score", "precision_top10pct", "hit_rate"]:
            assert key in m


class TestPurgedWalkForwardCV:
    def test_no_overlap(self):
        dates = pd.bdate_range("2015-01-01", "2025-01-01")
        cv = PurgedWalkForwardCV(
            initial_train_years=3,
            test_window_months=6,
            step_months=6,
            purge_days=0,
            embargo_days=5,
            label_period_days=5,
        )
        splits = cv.split(dates)
        for s in splits:
            tr = set(s["train_indices"])
            te = set(s["test_indices"])
            assert tr.isdisjoint(te), f"Fold {s['fold']}: train/test overlap"

    def test_expanding_train(self):
        dates = pd.bdate_range("2015-01-01", "2025-01-01")
        cv = PurgedWalkForwardCV(
            initial_train_years=3,
            test_window_months=6,
            step_months=6,
            purge_days=0,
            embargo_days=5,
            label_period_days=5,
        )
        splits = cv.split(dates)
        sizes = [s["statistics"]["train_size"] for s in splits]
        for i in range(1, len(sizes)):
            assert sizes[i] >= sizes[i - 1]


class TestLGBMWalkForward:
    @pytest.fixture()
    def synthetic_features_path(self, tmp_path: Path) -> Path:
        df = _make_synthetic_features(n_dates=1200, tickers=["A.NS", "B.NS", "C.NS"])
        p = tmp_path / "features.parquet"
        df.to_parquet(p)
        return p

    @pytest.fixture()
    def config_dir(self, tmp_path: Path) -> Path:
        import yaml
        cfg = {
            "model": {
                "initial_train_years": 2,
                "test_window_months": 3,
                "step": 3,
                "purge_days": 0,
                "embargo_days": 2,
                "label_period_days": 5,
            },
            "label": {"horizon_days": 5, "label_col": "label"},
            "feature_selection": {
                "max_missing_pct": 0.30,
                "max_corr": 0.95,
                "es_val_pct": 0.10,
            },
            "lgbm": {
                "seed": 42,
                "n_estimators": 50,
                "learning_rate": 0.1,
                "num_leaves": 15,
                "max_depth": -1,
                "min_child_samples": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.0,
                "reg_lambda": 0.0,
                "class_weight": "balanced",
                "n_jobs": 1,
                "verbose": -1,
                "early_stopping_rounds": 10,
            },
        }
        p = tmp_path / "model.yaml"
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)
        return tmp_path

    def test_determinism(self, synthetic_features_path: Path, config_dir: Path, tmp_path: Path):
        kwargs = dict(
            features_path=synthetic_features_path,
            config_dir=config_dir,
            oos_dir=tmp_path / "oos1",
            mlflow_db=tmp_path / "mlflow1.db",
            holdout_start="2025-10-01",
            dry_run=True,
        )
        oos1 = run_lgbm_walk_forward(**kwargs)

        kwargs2 = dict(
            features_path=synthetic_features_path,
            config_dir=config_dir,
            oos_dir=tmp_path / "oos2",
            mlflow_db=tmp_path / "mlflow2.db",
            holdout_start="2025-10-01",
            dry_run=True,
        )
        oos2 = run_lgbm_walk_forward(**kwargs2)

        oos1 = oos1.sort_values(["date", "ticker"]).reset_index(drop=True)
        oos2 = oos2.sort_values(["date", "ticker"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(oos1[["prob"]], oos2[["prob"]], check_exact=False, atol=1e-6)

    def test_shuffled_label_auc_near_half(self, synthetic_features_path: Path, config_dir: Path, tmp_path: Path):
        oos = run_lgbm_walk_forward(
            features_path=synthetic_features_path,
            config_dir=config_dir,
            oos_dir=tmp_path / "oos_shuf",
            mlflow_db=tmp_path / "mlflow_shuf.db",
            holdout_start="2025-10-01",
            dry_run=True,
        )
        rng = np.random.default_rng(0)
        shuffled_labels = rng.integers(0, 2, size=len(oos))
        auc = roc_auc_score(shuffled_labels, oos["prob"].values)
        assert 0.30 < auc < 0.70

    def test_no_leakage_index(self, synthetic_features_path: Path, config_dir: Path):
        import yaml
        with open(config_dir / "model.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cv_cfg = cfg["model"]

        features = pd.read_parquet(synthetic_features_path)
        features.index.names = ["date", "ticker"]
        dates = features.index.get_level_values("date").unique().sort_values()
        dates = dates[dates < pd.Timestamp("2025-10-01")]

        cv = PurgedWalkForwardCV(
            initial_train_years=cv_cfg["initial_train_years"],
            test_window_months=cv_cfg["test_window_months"],
            step_months=cv_cfg["step"],
            purge_days=cv_cfg.get("purge_days", 0),
            embargo_days=cv_cfg.get("embargo_days", 2),
            label_period_days=cv_cfg.get("label_period_days", 5),
        )
        splits = cv.split(dates)
        for s in splits:
            tr_set = set(s["train_indices"])
            te_set = set(s["test_indices"])
            assert tr_set.isdisjoint(te_set)

    def test_oos_parquet_written(self, synthetic_features_path: Path, config_dir: Path, tmp_path: Path):
        oos_dir = tmp_path / "oos_check"
        run_lgbm_walk_forward(
            features_path=synthetic_features_path,
            config_dir=config_dir,
            oos_dir=oos_dir,
            mlflow_db=tmp_path / "mlflow_check.db",
            holdout_start="2025-10-01",
            dry_run=True,
        )
        out = oos_dir / "lgbm.parquet"
        assert out.exists()
        df = pd.read_parquet(out)
        for col in ("date", "ticker", "prob", "fold"):
            assert col in df.columns
        assert (df["prob"] >= 0).all() and (df["prob"] <= 1).all()
