"""
Label generation for stock classification.

Binary label: 1 if the *forward* close-to-close return over horizon_days is
strictly positive, 0 otherwise.

NO LOOK-AHEAD guarantee
-----------------------
``make_labels(df, horizon, holdout_start)`` shifts prices *backward* by
``horizon`` trading days using ``.shift(-horizon)`` on the price series.
The value at row t therefore represents the close ``horizon`` days *after* t.

Truncation test (see ``test_no_lookahead``): removing all rows after date t
must not change the label value at date t.  This holds as long as the future
close used is within the training data range.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _load_label_config(config_dir: Path = Path("configs")) -> dict:
    """Load label section from model.yaml."""
    path = config_dir / "model.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("label", {"horizon_days": 5, "label_col": "label"})


def make_labels(
    prices: pd.DataFrame,
    horizon: int,
    holdout_start: str = "2025-10-01",
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Create binary forward-return labels.

    Parameters
    ----------
    prices : pd.DataFrame
        Panel with MultiIndex (date, ticker) and at minimum a ``Close``
        column.  Must be sorted by (date, ticker).
    horizon : int
        Number of trading days to look forward.
    holdout_start : str
        ISO date string.  All rows on or after this date are excluded from
        the returned label frame to prevent contaminating the holdout.
    label_col : str
        Name of the output binary column.

    Returns
    -------
    pd.DataFrame
        Same index as *prices* (excluding holdout rows), with columns:
        [label_col, "fwd_ret"].
    """
    if prices.empty:
        raise ValueError("prices DataFrame is empty")

    required_cols = {"Close"}
    missing = required_cols - set(prices.columns)
    if missing:
        raise ValueError(f"prices is missing columns: {missing}")

    # Work per ticker to apply .shift correctly within each ticker group
    close = prices["Close"].unstack(level="ticker")  # shape: (dates, tickers)
    close = close.sort_index()

    # Forward return: (close[t+h] / close[t]) - 1
    fwd_close = close.shift(-horizon)
    fwd_ret = (fwd_close / close) - 1.0

    # Convert back to long format
    fwd_ret_long = fwd_ret.stack(future_stack=True).rename("fwd_ret")
    fwd_ret_long.index.names = ["date", "ticker"]

    # Binary label
    labels = (fwd_ret_long > 0).astype(int).rename(label_col)

    result = pd.concat([labels, fwd_ret_long], axis=1)

    # Drop rows that are NaN (last `horizon` rows per ticker have no fwd close)
    result = result.dropna(subset=[label_col])

    # Exclude holdout — we NEVER peek at it until Day 30
    holdout_dt = pd.Timestamp(holdout_start)
    result = result[result.index.get_level_values("date") < holdout_dt]

    logger.info(
        "Labels created: %d rows, %.1f%% positive, horizon=%d",
        len(result),
        result[label_col].mean() * 100,
        horizon,
    )
    return result


def load_or_build_labels(
    features_path: Path,
    prices_path: Optional[Path] = None,
    config_dir: Path = Path("configs"),
    holdout_start: str = "2025-10-01",
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper: load cached labels if they exist, else build from
    prices parquet and features index.

    Parameters
    ----------
    features_path : Path
        Path to the features parquet (used only for the index shape info).
    prices_path : Path, optional
        Path to prices parquet with MultiIndex (date, ticker) and Close col.
        If None, tries ``data_store/raw/prices/prices.parquet``.
    config_dir : Path
        Directory containing model.yaml.
    holdout_start : str
        Holdout boundary date.
    cache_path : Path, optional
        If given, save/load labels here.

    Returns
    -------
    pd.DataFrame
        Labels frame with columns [label, fwd_ret].
    """
    cfg = _load_label_config(config_dir)
    horizon = cfg["horizon_days"]
    label_col = cfg["label_col"]

    if cache_path and cache_path.exists():
        logger.info("Loading cached labels from %s", cache_path)
        return pd.read_parquet(cache_path)

    if prices_path is None:
        prices_path = Path("data_store/raw/prices/prices.parquet")

    if not prices_path.exists():
        raise FileNotFoundError(
            f"Prices parquet not found at {prices_path}. "
            "Run the data download step first."
        )

    prices = pd.read_parquet(prices_path)
    prices.index.names = ["date", "ticker"]

    labels = make_labels(prices, horizon=horizon, holdout_start=holdout_start, label_col=label_col)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        labels.to_parquet(cache_path)
        logger.info("Labels cached to %s", cache_path)

    return labels
