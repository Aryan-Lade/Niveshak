"""
LightGBM walk-forward trainer with MLflow experiment tracking.

Workflow (per fold)
-------------------
1. Slice TRAIN / TEST indices from PurgedWalkForwardCV.
2. Feature selection (on TRAIN only):
   a. Drop features with >30% NaN in train.
   b. Drop one of any pair with |Pearson corr| >= 0.95.
3. Impute remaining NaN with column median (fit on TRAIN, apply to TEST).
4. Split last 10% of TRAIN (time-ordered) as early-stopping validation set.
5. Fit LGBMClassifier with sample_weight from the CV splitter.
6. Predict probability on TEST window → store (date, ticker, prob, fold).
7. Log params, fold-level metrics, fold dates, git hash, seed to MLflow.
8. Save each fold's model as an MLflow artifact.

OOS predictions are written to  data_store/oos/lgbm.parquet.
MLflow tracking URI defaults to  mlruns/  (local file store).

Rules obeyed
------------
* No look-ahead: feature selection, median imputation, correlation threshold
  all computed exclusively on TRAIN rows.
* Time order: early-stopping val is the last 10% of TRAIN (chronologically).
* Determinism: seed=42 everywhere; same seed → same predictions.
* AUC > 0.60 triggers a hard stop with a leakage warning.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    precision_score,
    roc_auc_score,
)

# ── project root on sys.path ───────────────────────────────────────────────
_HERE = Path(__file__).resolve()
# stock-engine/src/stock_engine/models/lgbm.py
# parents: [0]=models, [1]=stock_engine, [2]=src, [3]=stock-engine
_PROJECT_ROOT = _HERE.parents[3]  # stock-engine/
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from stock_engine.cv.purged_walk_forward import PurgedWalkForwardCV  # noqa: E402

logger = logging.getLogger(__name__)

# ── constants / defaults ───────────────────────────────────────────────────
_CONFIGS_DIR = _PROJECT_ROOT / "configs"
# OOS and features live one level up (Niveshak/data_store/...)
_DATA_ROOT = _PROJECT_ROOT.parent / "data_store"
_OOS_DIR = _DATA_ROOT / "oos"
_MLFLOW_DB = _PROJECT_ROOT.parent / "mlruns" / "mlflow.db"
_FEATURES_PATH = _DATA_ROOT / "features" / "features.parquet"
_HOLDOUT_START = "2025-10-01"



# ── config helpers ─────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_model_cfg(config_dir: Path = _CONFIGS_DIR) -> Dict[str, Any]:
    return _load_yaml(config_dir / "model.yaml")


def _get_git_hash() -> str:
    """Return the current HEAD commit hash (short), or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ── feature selection (TRAIN-only) ─────────────────────────────────────────

def select_features(
    X_train: pd.DataFrame,
    max_missing_pct: float = 0.30,
    max_corr: float = 0.95,
) -> List[str]:
    """
    Return list of features to keep, computed on X_train only.

    Steps
    -----
    1. Drop features where fraction of NaN > max_missing_pct.
    2. From remaining, drop one of any pair whose |Pearson corr| >= max_corr
       (greedy: iterate sorted column order; if correlated with an already-kept
       feature, drop the new one).
    """
    # Step 1: missing filter
    missing_pct = X_train.isnull().mean()
    keep = missing_pct[missing_pct <= max_missing_pct].index.tolist()
    n_dropped_missing = len(X_train.columns) - len(keep)
    if n_dropped_missing:
        logger.info("Feature selection: dropped %d cols (>%.0f%% NaN)", n_dropped_missing, max_missing_pct * 100)

    if len(keep) < 2:
        return keep

    # Step 2: correlation filter
    corr_mat = X_train[keep].corr().abs()
    retained: List[str] = []
    for col in keep:
        if not retained:
            retained.append(col)
            continue
        # Check against all retained features
        max_c = corr_mat.loc[col, retained].max()
        if max_c < max_corr:
            retained.append(col)

    n_dropped_corr = len(keep) - len(retained)
    if n_dropped_corr:
        logger.info("Feature selection: dropped %d cols (|corr| >= %.2f)", n_dropped_corr, max_corr)

    logger.info("Feature selection: %d features retained", len(retained))
    return retained


# ── median imputation ───────────────────────────────────────────────────────

def fit_imputer(X_train: pd.DataFrame) -> pd.Series:
    """Compute per-column median on train (ignore NaN)."""
    return X_train.median()


def apply_imputer(X: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    """Fill NaN in X using pre-computed train medians."""
    return X.fillna(medians)


# ── per-fold metrics ────────────────────────────────────────────────────────

def compute_fold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    top_pct: float = 0.10,
) -> Dict[str, float]:
    """
    Compute AUC, log-loss, Brier score, top-decile precision, hit rate.

    Parameters
    ----------
    y_true : array-like of {0, 1}
    y_prob : probability of class 1
    top_pct : fraction of rows considered as the "top bucket"
    """
    auc = roc_auc_score(y_true, y_prob)
    ll = log_loss(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    hit_rate = float((y_true == (y_prob >= 0.5)).mean())

    # Top-decile precision
    n_top = max(1, int(len(y_prob) * top_pct))
    top_idx = np.argsort(y_prob)[-n_top:]
    precision_top = precision_score(
        y_true[top_idx],
        np.ones(n_top, dtype=int),
        zero_division=0,
    )

    return {
        "auc": auc,
        "log_loss": ll,
        "brier_score": brier,
        "precision_top10pct": precision_top,
        "hit_rate": hit_rate,
    }


# ── main walk-forward function ──────────────────────────────────────────────

def run_lgbm_walk_forward(
    features_path: Path = _FEATURES_PATH,
    config_dir: Path = _CONFIGS_DIR,
    oos_dir: Path = _OOS_DIR,
    mlflow_db: Optional[Path] = None,
    experiment_name: str = "lgbm_walk_forward",
    holdout_start: str = _HOLDOUT_START,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Full walk-forward training loop.

    Parameters
    ----------
    features_path : Path
        Combined feature parquet with MultiIndex (date, ticker).
    config_dir : Path
        Directory containing model.yaml (and data.yaml for seed / holdout).
    oos_dir : Path
        Directory where lgbm.parquet is written.
    mlflow_db : Path, optional
        Path to SQLite file for MLflow tracking.  Defaults to
        ``<project>/mlruns/mlflow.db``.  Pass ``None`` to use default.
    experiment_name : str
        MLflow experiment name.
    holdout_start : str
        Rows on or after this date are never touched.
    dry_run : bool
        If True, run only the first fold (for quick testing).

    Returns
    -------
    pd.DataFrame
        OOS predictions with columns (date, ticker, prob, fold).
    """
    cfg = _load_model_cfg(config_dir)
    lgbm_cfg = cfg["lgbm"]
    fs_cfg = cfg["feature_selection"]
    cv_cfg = cfg["model"]

    seed: int = lgbm_cfg["seed"]
    np.random.seed(seed)

    git_hash = _get_git_hash()

    # ── load features ───────────────────────────────────────────────────────
    logger.info("Loading features from %s", features_path)
    features = pd.read_parquet(features_path)
    features.index.names = ["date", "ticker"]
    features = features.sort_index()

    # ── build simple binary label (fwd 5-day return sign) ──────────────────
    # We derive labels from ret_5d which is already in the feature file and
    # represents the PAST 5-day return.  We need the FUTURE 5-day return.
    # Build it from ret_1d by compounding, or use shift.
    # The cleanest approach: re-compute from Close prices if available.
    # Since we only have feature parquet here (no Close col), we synthesise
    # the forward return by shifting ret_1d forward within each ticker group.
    # ret_1d[t] = (Close[t] - Close[t-1]) / Close[t-1]
    # fwd_ret[t] = product of (1 + ret_1d[t+1..t+5]) - 1
    # This is look-ahead-free at t because we ONLY use data at t+1..t+5
    # during TRAINING (when ground truth is available).
    horizon = cv_cfg.get("label_period_days", 5)
    logger.info("Building %d-day forward return label", horizon)

    # Pivot ret_1d to (dates x tickers)
    ret1d = features["ret_1d"].unstack("ticker").sort_index()
    # Forward compound return = product of (1+r) over next `horizon` days
    fwd_ret = (1 + ret1d).rolling(horizon).apply(np.prod, raw=True).shift(-horizon) - 1
    label_long = (fwd_ret.stack(future_stack=True) > 0).astype(int)
    label_long.index.names = ["date", "ticker"]
    label_long = label_long.rename("label")

    # Merge label into feature frame
    data = features.join(label_long, how="inner")
    data = data.dropna(subset=["label"])

    # ── exclude holdout ─────────────────────────────────────────────────────
    holdout_dt = pd.Timestamp(holdout_start)
    data = data[data.index.get_level_values("date") < holdout_dt]

    feature_cols = [c for c in data.columns if c != "label"]
    logger.info(
        "Dataset: %d rows, %d features, %.1f%% positive labels",
        len(data),
        len(feature_cols),
        data["label"].mean() * 100,
    )

    # ── walk-forward CV ─────────────────────────────────────────────────────
    dates_idx = data.index.get_level_values("date")
    unique_dates = dates_idx.unique().sort_values()

    cv = PurgedWalkForwardCV(
        initial_train_years=cv_cfg["initial_train_years"],
        test_window_months=cv_cfg["test_window_months"],
        step_months=cv_cfg["step"],
        purge_days=cv_cfg.get("purge_days", 0),
        embargo_days=cv_cfg.get("embargo_days", 5),
        label_period_days=horizon,
        config_dir=config_dir,
    )
    splits = cv.split(unique_dates, show_progress=False)
    if not splits:
        raise RuntimeError("No CV folds generated — check date range in config.")

    if dry_run:
        splits = splits[:1]
        logger.warning("DRY RUN: using only 1 fold")

    # ── MLflow setup ────────────────────────────────────────────────────────
    if mlflow_db is None:
        mlflow_db = _MLFLOW_DB
    mlflow_db.parent.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{mlflow_db.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    oos_records: List[pd.DataFrame] = []
    fold_metrics_list: List[Dict[str, Any]] = []

    # ── fold loop ───────────────────────────────────────────────────────────
    for fold_info in splits:
        fold_num = fold_info["fold"]
        train_dates: pd.DatetimeIndex = fold_info["train_timestamps"]
        test_dates: pd.DatetimeIndex = fold_info["test_timestamps"]

        if len(train_dates) == 0 or len(test_dates) == 0:
            logger.warning("Fold %d: empty train or test, skipping", fold_num)
            continue

        # ── slice train / test ──────────────────────────────────────────────
        train_mask = dates_idx.isin(set(train_dates))
        test_mask = dates_idx.isin(set(test_dates))

        # Leakage guard: assert indices are disjoint
        assert not (train_mask & test_mask).any(), (
            f"Fold {fold_num}: train and test overlap — leakage detected!"
        )

        train_df = data[train_mask]
        test_df = data[test_mask]

        X_train = train_df[feature_cols]
        y_train = train_df["label"].values.astype(int)
        weights_arr = fold_info["sample_weights"]  # from CV splitter
        # weights_arr is aligned to unique_dates[train_indices]; re-expand to rows
        # The CV returns one weight per *date*; expand to (date, ticker) level.
        date_to_weight: Dict[pd.Timestamp, float] = dict(
            zip(train_dates, weights_arr[: len(train_dates)])
        )
        sample_weights = train_df.index.get_level_values("date").map(
            lambda d: date_to_weight.get(d, 1.0)
        ).values.astype(float)

        X_test = test_df[feature_cols]
        y_test = test_df["label"].values.astype(int)

        # ── feature selection (TRAIN only) ──────────────────────────────────
        selected_feats = select_features(
            X_train,
            max_missing_pct=fs_cfg["max_missing_pct"],
            max_corr=fs_cfg["max_corr"],
        )
        X_train = X_train[selected_feats]
        X_test = X_test[selected_feats]

        # ── imputation (median from TRAIN) ──────────────────────────────────
        medians = fit_imputer(X_train)
        X_train = apply_imputer(X_train, medians)
        X_test = apply_imputer(X_test, medians)

        # ── early-stopping validation split (last 10% of TRAIN) ────────────
        es_val_pct = fs_cfg.get("es_val_pct", 0.10)
        n_val = max(1, int(len(X_train) * es_val_pct))
        X_tr = X_train.iloc[:-n_val]
        y_tr = y_train[:-n_val]
        sw_tr = sample_weights[:-n_val]
        X_val = X_train.iloc[-n_val:]
        y_val = y_train[-n_val:]

        # ── build LightGBM model ────────────────────────────────────────────
        model = LGBMClassifier(
            n_estimators=lgbm_cfg["n_estimators"],
            learning_rate=lgbm_cfg["learning_rate"],
            num_leaves=lgbm_cfg["num_leaves"],
            max_depth=lgbm_cfg["max_depth"],
            min_child_samples=lgbm_cfg["min_child_samples"],
            subsample=lgbm_cfg["subsample"],
            colsample_bytree=lgbm_cfg["colsample_bytree"],
            reg_alpha=lgbm_cfg["reg_alpha"],
            reg_lambda=lgbm_cfg["reg_lambda"],
            class_weight=lgbm_cfg["class_weight"],
            n_jobs=lgbm_cfg["n_jobs"],
            verbose=lgbm_cfg["verbose"],
            random_state=seed,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                X_tr,
                y_tr,
                sample_weight=sw_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    early_stopping(lgbm_cfg["early_stopping_rounds"], verbose=False),
                    log_evaluation(-1),
                ],
            )

        # ── OOS predictions ─────────────────────────────────────────────────
        y_prob = model.predict_proba(X_test)[:, 1]

        fold_oos = test_df[[]].copy()  # keep (date, ticker) index
        fold_oos["prob"] = y_prob
        fold_oos["fold"] = fold_num
        oos_records.append(fold_oos.reset_index())

        # ── metrics ─────────────────────────────────────────────────────────
        metrics = compute_fold_metrics(y_test, y_prob)
        metrics["fold"] = fold_num
        metrics["n_train"] = len(X_tr)
        metrics["n_test"] = len(X_test)
        metrics["n_features"] = len(selected_feats)
        metrics["best_iteration"] = model.best_iteration_
        fold_metrics_list.append(metrics)

        # ── print fold row ───────────────────────────────────────────────────
        print(
            f"Fold {fold_num:2d} | "
            f"train {str(train_dates[0].date())} -> {str(train_dates[-1].date())} "
            f"({len(X_tr):5d} rows) | "
            f"test {str(test_dates[0].date())} -> {str(test_dates[-1].date())} "
            f"({len(X_test):5d} rows) | "
            f"AUC {metrics['auc']:.4f} | "
            f"LogLoss {metrics['log_loss']:.4f} | "
            f"Brier {metrics['brier_score']:.4f} | "
            f"TopPrec {metrics['precision_top10pct']:.4f} | "
            f"HitRate {metrics['hit_rate']:.4f} | "
            f"Trees {metrics['best_iteration']}"
        )

        # ── AUC leakage guard ────────────────────────────────────────────────
        if metrics["auc"] > 0.60:
            logger.warning(
                "LEAKAGE ALERT: Fold %d AUC=%.4f > 0.60. "
                "STOP and run leakage audit before believing this.",
                fold_num,
                metrics["auc"],
            )

        # ── MLflow logging ───────────────────────────────────────────────────
        with mlflow.start_run(run_name=f"lgbm_fold_{fold_num}"):
            mlflow.log_param("fold", fold_num)
            mlflow.log_param("seed", seed)
            mlflow.log_param("git_hash", git_hash)
            mlflow.log_param("n_features_selected", len(selected_feats))
            mlflow.log_param("best_iteration", model.best_iteration_)
            mlflow.log_param("train_start", str(train_dates[0].date()))
            mlflow.log_param("train_end", str(train_dates[-1].date()))
            mlflow.log_param("test_start", str(test_dates[0].date()))
            mlflow.log_param("test_end", str(test_dates[-1].date()))
            # LightGBM hyper-params
            for k, v in lgbm_cfg.items():
                mlflow.log_param(f"lgbm_{k}", v)

            # Metrics
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, float(v))

            # Save model artifact
            mlflow.lightgbm.log_model(model.booster_, artifact_path=f"model_fold_{fold_num}")

    # ── summary table ────────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(fold_metrics_list)
    print("\n" + "=" * 100)
    print("PER-FOLD SUMMARY")
    print("=" * 100)
    display_cols = ["fold", "auc", "log_loss", "brier_score", "precision_top10pct", "hit_rate", "n_train", "n_test"]
    print(metrics_df[display_cols].to_string(index=False))
    print("-" * 100)
    print(
        f"Mean AUC : {metrics_df['auc'].mean():.4f}  +/- {metrics_df['auc'].std():.4f}"
    )
    print(
        f"Mean LogL: {metrics_df['log_loss'].mean():.4f}  +/- {metrics_df['log_loss'].std():.4f}"
    )

    if metrics_df["auc"].mean() > 0.60:
        logger.error(
            "LEAKAGE ALERT: Mean OOS AUC = %.4f > 0.60. "
            "HALTING — run leakage audit before proceeding.",
            metrics_df["auc"].mean(),
        )
        raise RuntimeError(
            f"Mean OOS AUC ({metrics_df['auc'].mean():.4f}) > 0.60: "
            "suspected look-ahead leakage. Audit before continuing."
        )

    # ── write OOS parquet ────────────────────────────────────────────────────
    oos_df = pd.concat(oos_records, ignore_index=True)
    oos_dir.mkdir(parents=True, exist_ok=True)
    out_path = oos_dir / "lgbm.parquet"
    oos_df.to_parquet(out_path, index=False)
    logger.info("OOS predictions saved to %s  (%d rows)", out_path, len(oos_df))

    return oos_df


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="LightGBM walk-forward trainer")
    parser.add_argument("--dry-run", action="store_true", help="Run only the first fold")
    parser.add_argument(
        "--features",
        type=Path,
        default=_FEATURES_PATH,
        help="Path to features.parquet",
    )
    parser.add_argument(
        "--mlflow-db",
        type=Path,
        default=None,
        help="Path to SQLite MLflow DB (default: mlruns/mlflow.db)",
    )
    parser.add_argument(
        "--experiment",
        default="lgbm_walk_forward",
        help="MLflow experiment name",
    )
    args = parser.parse_args()

    oos = run_lgbm_walk_forward(
        features_path=args.features,
        mlflow_db=args.mlflow_db,
        experiment_name=args.experiment,
        dry_run=args.dry_run,
    )
    print(f"\nOOS shape: {oos.shape}")
    print(oos.head())
