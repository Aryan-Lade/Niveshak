"""
Purged walk-forward cross-validation for time series data.

Implements expanding window cross-validation with purge and embargo procedures
to prevent look-ahead bias in financial time series.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _load_model_config(config_dir: Path = Path("configs")) -> Dict:
    """Load model configuration from YAML file."""
    config_path = config_dir / "model.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["model"]


class PurgedWalkForwardCV:
    """
    Purged walk-forward cross-validation for time series data.

    Features:
    - Expanding window training approach
    - Purge: removes training samples whose label periods overlap with test period
    - Embargo: removes training samples within embargo_days before test start
    - Sample weights based on average uniqueness
    """

    def __init__(
        self,
        initial_train_years: Optional[int] = None,
        test_window_months: Optional[int] = None,
        step_months: Optional[int] = None,
        purge_days: Optional[int] = None,
        embargo_days: Optional[int] = None,
        label_period_days: Optional[int] = None,
        config_dir: Path = Path("configs"),
    ):
        """
        Initialize the purged walk-forward CV.

        Args:
            initial_train_years: Initial training window in years.
            test_window_months: Test window size in months.
            step_months: Step size between folds in months.
            purge_days: Extra purge days beyond label_period_days.
            embargo_days: Number of calendar days to embargo before test start.
            label_period_days: Forward label horizon (trading days).
            config_dir: Directory containing model.yaml.
        """
        cfg = _load_model_config(config_dir)

        self.initial_train_years = (
            initial_train_years if initial_train_years is not None else cfg["initial_train_years"]
        )
        self.test_window_months = (
            test_window_months if test_window_months is not None else cfg["test_window_months"]
        )
        self.step_months = step_months if step_months is not None else cfg["step"]
        self.purge_days = purge_days if purge_days is not None else cfg.get("purge_days", 0)
        self.embargo_days = embargo_days if embargo_days is not None else cfg.get("embargo_days", 5)
        self.label_period_days = (
            label_period_days if label_period_days is not None else cfg.get("label_period_days", 5)
        )

        logger.debug(
            "PurgedWalkForwardCV: train_yrs=%d test_mo=%d step_mo=%d "
            "purge=%d embargo=%d label_days=%d",
            self.initial_train_years,
            self.test_window_months,
            self.step_months,
            self.purge_days,
            self.embargo_days,
            self.label_period_days,
        )

    def split(
        self,
        timestamps: pd.DatetimeIndex,
        show_progress: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Generate train/test splits for purged walk-forward cross-validation.

        Args:
            timestamps: DatetimeIndex of unique dates in the dataset.
            show_progress: If True, log progress at INFO level.

        Returns:
            List of dicts with keys:
              fold, train_indices, test_indices,
              train_timestamps, test_timestamps,
              sample_weights, statistics.
        """
        if not isinstance(timestamps, pd.DatetimeIndex):
            raise ValueError("timestamps must be a pandas DatetimeIndex")
        if len(timestamps) == 0:
            raise ValueError("timestamps cannot be empty")

        time_series = pd.Series(range(len(timestamps)), index=timestamps)
        initial_train_end = timestamps[0] + pd.DateOffset(years=self.initial_train_years)

        splits: List[Dict[str, Any]] = []
        fold_num = 0

        while True:
            test_start_date = initial_train_end + pd.DateOffset(
                months=fold_num * self.step_months
            )
            test_end_date = test_start_date + pd.DateOffset(months=self.test_window_months)

            test_start_idx = time_series.index.searchsorted(test_start_date)
            test_end_idx = time_series.index.searchsorted(test_end_date)

            if test_start_idx >= len(time_series):
                break

            test_end_idx = min(test_end_idx, len(time_series))
            if test_end_idx <= test_start_idx:
                break

            test_start_time = timestamps[test_start_idx]
            test_end_time = timestamps[test_end_idx - 1]

            # Purge: exclude the last `label_period_days` before test start
            # to avoid label-period leakage.
            purge_cutoff = test_start_time - pd.DateOffset(days=self.label_period_days)
            embargo_cutoff = test_start_time - pd.DateOffset(days=self.embargo_days)
            actual_train_end_time = min(purge_cutoff, embargo_cutoff)

            train_start_time = timestamps[0]

            # Slice by label
            train_timestamps = time_series.loc[train_start_time:actual_train_end_time].index
            test_timestamps = time_series.loc[test_start_time:test_end_time].index

            if len(train_timestamps) == 0 or len(test_timestamps) == 0:
                fold_num += 1
                continue

            train_pos = [time_series.index.get_loc(t) for t in train_timestamps]
            test_pos = [time_series.index.get_loc(t) for t in test_timestamps]

            sample_weights = self._calculate_sample_weights(train_timestamps, timestamps)

            purge_applied = max(0, int((purge_cutoff - actual_train_end_time).days))
            embargo_applied = max(0, int((embargo_cutoff - actual_train_end_time).days))

            fold_info: Dict[str, Any] = {
                "fold": fold_num,
                "train_indices": train_pos,
                "test_indices": test_pos,
                "train_timestamps": train_timestamps,
                "test_timestamps": test_timestamps,
                "sample_weights": sample_weights,
                "statistics": {
                    "train_size": len(train_pos),
                    "test_size": len(test_pos),
                    "train_start": train_timestamps[0],
                    "train_end": train_timestamps[-1],
                    "test_start": test_timestamps[0],
                    "test_end": test_timestamps[-1],
                    "purge_days_applied": purge_applied,
                    "embargo_days_applied": embargo_applied,
                },
            }
            splits.append(fold_info)

            if show_progress:
                logger.info(
                    "Fold %d: train %s→%s (%d) | test %s→%s (%d)",
                    fold_num,
                    train_timestamps[0].date(),
                    train_timestamps[-1].date(),
                    len(train_pos),
                    test_timestamps[0].date(),
                    test_timestamps[-1].date(),
                    len(test_pos),
                )

            fold_num += 1

        if show_progress:
            logger.info("Generated %d folds", len(splits))

        return splits

    def _calculate_sample_weights(
        self,
        train_timestamps: pd.DatetimeIndex,
        all_timestamps: pd.DatetimeIndex,
    ) -> np.ndarray:
        """
        Calculate sample weights based on average uniqueness.

        Each training date gets weight = 1 / (number of dates whose label
        window overlaps with it).  Weights are then normalised to sum to
        len(train_timestamps) so that the effective sample size is preserved.
        """
        if len(train_timestamps) == 0:
            return np.array([], dtype=float)

        weights = []
        for ts in train_timestamps:
            label_end = ts + pd.DateOffset(days=self.label_period_days)
            overlap = ((all_timestamps >= ts) & (all_timestamps <= label_end)).sum()
            weights.append(1.0 / max(overlap, 1))

        weights_arr = np.array(weights, dtype=float)
        total = weights_arr.sum()
        if total > 0:
            weights_arr = weights_arr / total * len(weights_arr)
        return weights_arr

    def get_split_info(self, splits: List[Dict[str, Any]]) -> pd.DataFrame:
        """Return a summary DataFrame of all splits."""
        if not splits:
            return pd.DataFrame()
        rows = []
        for s in splits:
            st = s["statistics"]
            rows.append(
                {
                    "fold": s["fold"],
                    "train_size": st["train_size"],
                    "test_size": st["test_size"],
                    "train_start": st["train_start"],
                    "train_end": st["train_end"],
                    "test_start": st["test_start"],
                    "test_end": st["test_end"],
                }
            )
        return pd.DataFrame(rows)
