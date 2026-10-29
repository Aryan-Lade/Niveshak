"""
Leak-free stacking meta-model for combining base model predictions.

NON-NEGOTIABLE STACKING RULES:
1. The meta-model for fold k is trained ONLY on out-of-sample predictions
   from folds strictly before k (< k).
2. For fold 0 (or folds with insufficient prior samples), a simple average
   of base models is used.
3. Meta features:
   - logit(p_lgbm)
   - logit(p_gru)
   - regime probabilities (p_bull, p_sideways, p_bear)
   - vix_zscore
4. Meta learner: L2 regularized Logistic Regression.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def compute_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute numerical logit = log(p / (1 - p)) safely bounded in [eps, 1-eps]."""
    clipped = np.clip(p, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def simple_average_ensemble(
    df_lgbm: pd.DataFrame,
    df_gru: pd.DataFrame,
) -> pd.DataFrame:
    """Compute simple probability average between LightGBM and GRU."""
    merged = pd.merge(
        df_lgbm[["date", "ticker", "prob", "fold"]],
        df_gru[["date", "ticker", "prob"]],
        on=["date", "ticker"],
        suffixes=("_lgbm", "_gru"),
    )
    merged["prob"] = 0.5 * (merged["prob_lgbm"] + merged["prob_gru"])
    merged["method"] = "simple_average"
    return merged[["date", "ticker", "prob", "fold", "method"]]


class MetaStacker:
    """
    Leak-free expanding walk-forward meta-model stacker.
    """

    def __init__(self, C: float = 1.0, random_state: int = 42) -> None:
        self.C = C
        self.random_state = random_state
        self.model = LogisticRegression(
            C=self.C,
            penalty="l2",
            solver="lbfgs",
            random_state=self.random_state,
            max_iter=500,
        )

    def prepare_meta_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Build feature matrix: [logit(p_lgbm), logit(p_gru), regime_p0, regime_p1, regime_p2, vix_zscore].
        """
        feats: List[np.ndarray] = []

        if "prob_lgbm" in df.columns:
            feats.append(compute_logit(df["prob_lgbm"].values))
        elif "prob" in df.columns:
            feats.append(compute_logit(df["prob"].values))

        if "prob_gru" in df.columns:
            feats.append(compute_logit(df["prob_gru"].values))

        for reg_col in ["regime_p0", "regime_p1", "regime_p2"]:
            if reg_col in df.columns:
                feats.append(df[reg_col].fillna(0.33).values)
            else:
                feats.append(np.full(len(df), 0.33))

        if "vix_zscore" in df.columns:
            feats.append(df["vix_zscore"].fillna(0.0).values)
        else:
            feats.append(np.zeros(len(df)))

        X = np.column_stack(feats)
        return np.nan_to_num(X, nan=0.0)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> MetaStacker:
        """Fit regularized logistic regression on prior folds."""
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Predict calibrated ensemble probability for class 1."""
        return self.model.predict_proba(X_test)[:, 1]


def run_leak_free_stacking(
    df_lgbm: pd.DataFrame,
    df_gru: pd.DataFrame,
    regime_df: Optional[pd.DataFrame] = None,
    labels_df: Optional[pd.DataFrame] = None,
    output_path: Path = Path("data_store/oos/stack.parquet"),
    min_train_samples: int = 2,
) -> pd.DataFrame:
    """
    Run leak-free meta stacking across all folds.
    Fold k is trained strictly on out-of-sample data from folds < k.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge predictions
    merged = pd.merge(
        df_lgbm[["date", "ticker", "prob", "fold"]],
        df_gru[["date", "ticker", "prob"]],
        on=["date", "ticker"],
        suffixes=("_lgbm", "_gru"),
    )

    if regime_df is not None:
        merged = pd.merge(merged, regime_df, on="date", how="left")

    if labels_df is not None:
        if "label" in labels_df.columns:
            merged = pd.merge(merged, labels_df[["date", "ticker", "label"]], on=["date", "ticker"], how="left")
        elif "y" in labels_df.columns:
            merged = pd.merge(merged, labels_df[["date", "ticker", "y"]].rename(columns={"y": "label"}), on=["date", "ticker"], how="left")

    # If label missing, derive pseudo-label from relative consensus
    if "label" not in merged.columns or merged["label"].isna().all():
        merged["label"] = ((merged["prob_lgbm"] + merged["prob_gru"]) >= 1.0).astype(int)

    unique_folds = sorted(merged["fold"].unique())
    stacked_records: List[pd.DataFrame] = []

    for fold_k in unique_folds:
        test_df = merged[merged["fold"] == fold_k].copy()

        # LEAK-FREE RULE: Train strictly on folds < fold_k
        train_df = merged[merged["fold"] < fold_k].copy()

        if fold_k == unique_folds[0] or len(train_df) < min_train_samples or len(np.unique(train_df["label"])) < 2:
            # Fall back to simple average for initial fold
            test_df["prob"] = 0.5 * (test_df["prob_lgbm"] + test_df["prob_gru"])
            test_df["method"] = "simple_average_fallback"
        else:
            meta = MetaStacker(C=1.0, random_state=42)
            X_train = meta.prepare_meta_features(train_df)
            y_train = train_df["label"].values.astype(int)

            meta.fit(X_train, y_train)

            X_test = meta.prepare_meta_features(test_df)
            test_df["prob"] = meta.predict_proba(X_test)
            test_df["method"] = "meta_logistic_regression"

        stacked_records.append(test_df[["date", "ticker", "prob", "fold", "method"]])

    result_df = pd.concat(stacked_records, ignore_index=True).sort_values(["date", "ticker"])
    result_df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(result_df)} stacked ensemble predictions to {output_path}")
    return result_df
