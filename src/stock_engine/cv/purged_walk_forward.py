"""
Purged walk-forward cross-validation for time series data.
Implements expanding window cross-validation with purge and embargo procedures
to prevent look-ahead bias in financial time series.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

CONFIG_DIR = Path("stock-engine/configs")


def load_model_config() -> Dict:
    """Load model configuration from YAML file."""
    config_path = CONFIG_DIR / "model.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['model']


class PurgedWalkForwardCV:
    """
    Purged walk-forward cross-validation for time series data.

    Features:
    - Expanding window training approach
    - Purge: removes training samples whose label periods overlap with test period
    - Embargo: removes training samples within embargo_days before test start
    - Sample weights based on average uniqueness
    """

    def __init__(self,
                 initial_train_years: int = None,
                 test_window_months: int = None,
                 step_months: int = None,
                 purge_days: int = None,
                 embargo_days: int = None,
                 label_period_days: int = None):
        """
        Initialize the purged walk-forward CV.

        Args:
            initial_train_years: Initial training window in years
            test_window_months: Test window size in months
            step_months: Step size between folds in months
            purge_days: Number of days to purge before test period
            embargo_days: Number of days to embargo before test period
            label_period_days: Label period in days for returns calculation
        """
        config = load_model_config()

        self.initial_train_years = initial_train_years if initial_train_years is not None else config['initial_train_years']
        self.test_window_months = test_window_months if test_window_months is not None else config['test_window_months']
        self.step_months = step_months if step_months is not None else config['step']
        self.purge_days = purge_days if purge_days is not None else config.get('purge_days', 0)
        self.embargo_days = embargo_days if embargo_days is not None else config.get('embargo_days', 5)
        self.label_period_days = label_period_days if label_period_days is not None else config.get('label_period_days', 30)

        logger.info(f"Initialized PurgedWalkForwardCV with:")
        logger.info(f"  Initial train years: {self.initial_train_years}")
        logger.info(f"  Test window months: {self.test_window_months}")
        logger.info(f"  Step months: {self.step_months}")
        logger.info(f"  Purge days: {self.purge_days}")
        logger.info(f"  Embargo days: {self.embargo_days}")
        logger.info(f"  Label period days: {self.label_period_days}")

    def split(self,
              timestamps: pd.DatetimeIndex,
              show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Generate train/test splits for purged walk-forward cross-validation.

        Args:
            timestamps: DatetimeIndex of the time series data
            show_progress: Whether to show progress information

        Returns:
            List of dictionaries containing train_indices, test_indices, and sample_weights
        """
        if not isinstance(timestamps, pd.DatetimeIndex):
            raise ValueError("timestamps must be a pandas DatetimeIndex")

        if len(timestamps) == 0:
            raise ValueError("timestamps cannot be empty")

        time_series = pd.Series(index=timestamps, data=range(len(timestamps)))

        initial_train_end = timestamps[0] + pd.DateOffset(years=self.initial_train_years)

        initial_train_end_idx = time_series.index.searchsorted(initial_train_end)
        if initial_train_end_idx >= len(time_series):
            initial_train_end_idx = len(time_series) - 1

        splits = []
        fold_num = 0

        if show_progress:
            logger.info(f"Starting purged walk-forward CV generation...")
            logger.info(f"Total timestamps: {len(timestamps)}")
            logger.info(f"Date range: {timestamps[0]} to {timestamps[-1]}")

        while True:
            test_start_date = initial_train_end + pd.DateOffset(months=fold_num * self.step_months)
            test_end_date = test_start_date + pd.DateOffset(months=self.test_window_months)

            test_start_idx = time_series.index.searchsorted(test_start_date)
            test_end_idx = time_series.index.searchsorted(test_end_date)

            if test_start_idx >= len(time_series):
                if show_progress:
                    logger.info(f"Not enough data for fold {fold_num}. Stopping.")
                break

            if test_end_idx > len(time_series):
                test_end_idx = len(time_series)

            test_start_time = timestamps[test_start_idx]
            test_end_time = timestamps[test_end_idx - 1] if test_end_idx > test_start_idx else test_start_time

            purge_start_time = test_start_time - pd.DateOffset(days=self.purge_days)
            embargo_start_time = test_start_time - pd.DateOffset(days=self.embargo_days)

            train_start_time = timestamps[0]
            train_end_time = test_start_time - pd.DateOffset(days=1)

            purge_cutoff = test_start_time - pd.DateOffset(days=self.label_period_days)
            embargo_cutoff = test_start_time - pd.DateOffset(days=self.embargo_days)

            actual_train_end_time = min(train_end_time, purge_cutoff, embargo_cutoff)

            train_indices = time_series.loc[train_start_time:actual_train_end_time].index
            test_indices = time_series.loc[test_start_time:test_end_time].index

            train_pos_indices = [time_series.index.get_loc(t) for t in train_indices]
            test_pos_indices = [time_series.index.get_loc(t) for t in test_indices]

            sample_weights = self._calculate_sample_weights(
                time_series.loc[train_start_time:actual_train_end_time].index,
                timestamps
            )

            fold_info = {
                'fold': fold_num,
                'train_indices': train_pos_indices,
                'test_indices': test_pos_indices,
                'train_timestamps': train_indices,
                'test_timestamps': test_indices,
                'sample_weights': sample_weights,
                'statistics': {
                    'train_size': len(train_pos_indices),
                    'test_size': len(test_pos_indices),
                    'train_start': train_indices[0] if len(train_indices) > 0 else None,
                    'train_end': train_indices[-1] if len(train_indices) > 0 else None,
                    'test_start': test_indices[0] if len(test_indices) > 0 else None,
                    'test_end': test_indices[-1] if len(test_indices) > 0 else None,
                    'purge_days_applied': max(0, (train_end_time - purge_cutoff).days),
                    'embargo_days_applied': max(0, (train_end_time - embargo_cutoff).days)
                }
            }

            splits.append(fold_info)

            if show_progress:
                logger.info(f"Fold {fold_num}:")
                logger.info(f"  Train: {fold_info['statistics']['train_start']} to {fold_info['statistics']['train_end']} "
                          f"({fold_info['statistics']['train_size']} samples)")
                logger.info(f"  Test: {fold_info['statistics']['test_start']} to {fold_info['statistics']['test_end']} "
                          f"({fold_info['statistics']['test_size']} samples)")
                logger.info(f"  Purge applied: {fold_info['statistics']['purge_days_applied']} days")
                logger.info(f"  Embargo applied: {fold_info['statistics']['embargo_days_applied']} days")

            fold_num += 1

        if show_progress:
            logger.info(f"Generated {len(splits)} folds total")

        return splits

    def _months_to_offset(self, months: int) -> pd.DateOffset:
        """Convert months to pandas DateOffset."""
        return pd.DateOffset(months=months)

    def _calculate_sample_weights(self,
                                 train_timestamps: pd.DatetimeIndex,
                                 all_timestamps: pd.DatetimeIndex) -> np.ndarray:
        """
        Calculate sample weights based on average uniqueness.

        Args:
            train_timestamps: Timestamps in the training set
            all_timestamps: All timestamps in the dataset

        Returns:
            Array of sample weights for training samples
        """
        if len(train_timestamps) == 0:
            return np.array([])

        weights = []

        for ts in train_timestamps:
            label_end = ts + pd.DateOffset(days=self.label_period_days)
            overlap_count = ((all_timestamps >= ts) & (all_timestamps <= label_end)).sum()
            uniqueness = 1.0 / max(overlap_count, 1)
            weights.append(uniqueness)

        weights = np.array(weights)

        if weights.sum() > 0:
            weights = weights / weights.sum() * len(weights)
        else:
            weights = np.ones_like(weights) / len(weights) * len(weights)

        return weights

    def get_split_info(self, splits: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Get a summary DataFrame of all splits.

        Args:
            splits: List of split dictionaries from split() method

        Returns:
            DataFrame with split information
        """
        if not splits:
            return pd.DataFrame()

        rows = []
        for split in splits:
            stats = split['statistics']
            row = {
                'fold': split['fold'],
                'train_size': stats['train_size'],
                'test_size': stats['test_size'],
                'train_start': stats['train_start'],
                'train_end': stats['train_end'],
                'test_start': stats['test_start'],
                'test_end': stats['test_end'],
                'purge_days': stats['purge_days_applied'],
                'embargo_days': stats['embargo_days_applied']
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def plot_folds(self,
                    splits: List[Dict[str, Any]],
                    timestamps: pd.DatetimeIndex,
                    save_path: Optional[Path] = None,
                    figsize: Tuple[int, int] = (15, 8)) -> plt.Figure:
        """
        Plot the cross-validation folds showing train/test periods and purge/embargo.

        Args:
            splits: List of split dictionaries from split() method
            timestamps: Original timestamps DatetimeIndex
            save_path: Optional path to save the figure
            figsize: Figure size as (width, height)

        Returns:
            Matplotlib figure object
        """
        if not splits:
            raise ValueError("No splits to plot")

        fig, ax = plt.subplots(figsize=figsize)

        timeline_y = 0
        ax.hlines(timeline_y, timestamps[0], timestamps[-1],
                  colors='black', linewidth=2, label='Timeline')

        colors = plt.cm.Set3(np.linspace(0, 1, len(splits)))

        for i, split in enumerate(splits):
            color = colors[i]
            fold_num = split['fold']

            train_ts = split['train_timestamps']
            test_ts = split['test_timestamps']
            stats = split['statistics']

            if len(train_ts) > 0:
                ax.hlines(timeline_y - 0.2 - i * 0.1,
                          train_ts[0], train_ts[-1],
                          colors=color, linewidth=6,
                          label=f'Fold {fold_num} Train' if i == 0 else "")

            if len(test_ts) > 0:
                ax.hlines(timeline_y + 0.2 + i * 0.1,
                          test_ts[0], test_ts[-1],
                          colors=color, linewidth=6,
                          label=f'Fold {fold_num} Test' if i == 0 else "")

            purge_end = test_ts[0] if len(test_ts) > 0 else timestamps[0]
            embargo_end = test_ts[0] if len(test_ts) > 0 else timestamps[0]

            purge_start = purge_end - pd.DateOffset(days=stats['purge_days_applied'])
            embargo_start = purge_end - pd.DateOffset(days=stats['embargo_days_applied'])

            if stats['purge_days_applied'] > 0:
                ax.hlines(timeline_y - 0.4,
                          purge_start, purge_end,
                          colors='red', linewidth=2, linestyles='dashed',
                          label='Purge' if i == 0 else "")

            if stats['embargo_days_applied'] > 0:
                ax.hlines(timeline_y - 0.3,
                          embargo_start, embargo_end,
                          colors='orange', linewidth=2, linestyles='dashed',
                          label='Embargo' if i == 0 else "")

        ax.set_yticks([])
        ax.set_xlabel('Date')
        ax.set_title('Purged Walk-Forward Cross-Validation Folds\n'
                     f'Initial Train: {self.initial_train_years} years, '
                     f'Test: {self.test_window_months} months, '
                     f'Step: {self.step_months} months\n'
                     f'Purge: {self.purge_days} days, Embargo: {self.embargo_days} days')
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Fold plot saved to {save_path}")

        return fig


def run_purged_walk_forward_example():
    """Example usage of the PurgedWalkForwardCV class."""
    dates = pd.date_range('2015-01-01', '2025-12-31', freq='D')
    print(f"Created sample data with {len(dates)} days from {dates[0]} to {dates[-1]}")

    cv = PurgedWalkForwardCV(
        initial_train_years=3,
        test_window_months=6,
        step_months=6,
        purge_days=0,
        embargo_days=5,
        label_period_days=30
    )

    splits = cv.split(dates, show_progress=True)

    print("\n" + "="*50)
    print("SPLIT SUMMARY")
    print("="*50)
    summary_df = cv.get_split_info(splits)
    print(summary_df.to_string(index=False))

    try:
        fig = cv.plot_folds(splits, dates,
                           save_path=Path("reports/cv_folds.png"))
        print(f"\nFold plot saved to reports/cv_folds.png")
        plt.close(fig)
    except Exception as e:
        print(f"Could not create plot: {e}")

    return splits


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    splits = run_purged_walk_forward_example()

    print(f"\nSuccessfully generated {len(splits)} folds for purged walk-forward CV")
