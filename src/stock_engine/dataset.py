"""
Dataset builder and loader for stock engine modeling.

Merges features and labels on (date, ticker), removes warm-up periods,
and enforces a strict security guard preventing premature access to the final holdout.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MODEL_MATRIX_PATH = Path("data_store/model_matrix.parquet")
DEFAULT_HOLDOUT_MATRIX_PATH = Path("data_store/holdout_matrix.parquet")


def build_model_matrix(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    holdout_start: str = "2025-10-01",
    output_dir: str | Path = "data_store",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge features and labels on (date, ticker), drop warm-up rows (NaNs),
    and split into train/val model_matrix and sealed holdout_matrix.

    Returns
    -------
    (model_matrix, holdout_matrix)
    """
    # Ensure standard index
    feat = features.copy()
    lab = labels.copy()

    if not isinstance(feat.index, pd.MultiIndex):
        feat = feat.set_index(["date", "ticker"])
    if not isinstance(lab.index, pd.MultiIndex):
        lab = lab.set_index(["date", "ticker"])

    # Ensure date level is Timestamp
    feat.index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(d), t) for d, t in feat.index],
        names=["date", "ticker"],
    )
    lab.index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(d), t) for d, t in lab.index],
        names=["date", "ticker"],
    )

    # Inner join on (date, ticker)
    merged = feat.join(lab, how="inner")

    # Drop warm-up rows with missing features
    feature_cols = [c for c in feat.columns if c in merged.columns]
    merged = merged.dropna(subset=feature_cols)
    merged = merged.sort_index()

    holdout_dt = pd.Timestamp(holdout_start)
    dates = merged.index.get_level_values("date")

    model_matrix = merged[dates < holdout_dt].copy()
    holdout_matrix = merged[dates >= holdout_dt].copy()

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    model_p = out_p / "model_matrix.parquet"
    holdout_p = out_p / "holdout_matrix.parquet"

    model_matrix.to_parquet(model_p)
    holdout_matrix.to_parquet(holdout_p)

    logger.info(
        "Built dataset: model_matrix=%d rows (< %s), holdout_matrix=%d rows (>= %s)",
        len(model_matrix),
        holdout_start,
        len(holdout_matrix),
        holdout_start,
    )

    return model_matrix, holdout_matrix


def load_model_matrix(path: str | Path = DEFAULT_MODEL_MATRIX_PATH) -> pd.DataFrame:
    """Load the pre-holdout modeling dataset."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model matrix not found at {p}. Build it first.")
    df = pd.read_parquet(p)
    return df


def load_holdout_matrix(
    path: str | Path = DEFAULT_HOLDOUT_MATRIX_PATH,
    i_know_this_is_final: bool = False,
) -> pd.DataFrame:
    """
    Load the final sealed holdout dataset.

    Per Rule 2 of the Master Context, this function MUST refuse to run without
    explicit confirmation `i_know_this_is_final=True`.
    """
    if not i_know_this_is_final:
        raise PermissionError(
            "Access to final holdout is locked! Set i_know_this_is_final=True "
            "only during Day 30 final evaluation."
        )

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Holdout matrix not found at {p}.")

    logger.warning("ALERT: Opening final sealed holdout matrix. This evaluation is FINAL.")
    return pd.read_parquet(p)
