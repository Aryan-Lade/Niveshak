"""
Tests for purged walk-forward cross-validation implementation.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.stock_engine.cv.purged_walk_forward import PurgedWalkForwardCV


def test_purged_walk_forward_initialization():
    """Test initialization of PurgedWalkForwardCV."""
    cv = PurgedWalkForwardCV()
    assert cv.initial_train_years == 3
    assert cv.test_window_months == 6
    assert cv.step_months == 6
    assert cv.purge_days == 0
    assert cv.embargo_days == 5
    assert cv.label_period_days == 30

    cv_custom = PurgedWalkForwardCV(
        initial_train_years=2,
        test_window_months=3,
        step_months=2,
        purge_days=2,
        embargo_days=3,
        label_period_days=10
    )
    assert cv_custom.initial_train_years == 2
    assert cv_custom.test_window_months == 3
    assert cv_custom.step_months == 2
    assert cv_custom.purge_days == 2
    assert cv_custom.embargo_days == 3
    assert cv_custom.label_period_days == 10


def test_purged_walk_forward_split_basic():
    """Test basic split functionality."""
    dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')

    cv = PurgedWalkForwardCV(
        initial_train_years=1,
        test_window_months=3,
        step_months=3,
        purge_days=0,
        embargo_days=0,
        label_period_days=0
    )

    splits = cv.split(dates, show_progress=False)

    assert len(splits) > 0

    first_split = splits[0]
    assert 'train_indices' in first_split
    assert 'test_indices' in first_split
    assert 'sample_weights' in first_split
    assert 'statistics' in first_split

    assert all(0 <= idx < len(dates) for idx in first_split['train_indices'])
    assert all(0 <= idx < len(dates) for idx in first_split['test_indices'])

    if len(first_split['train_indices']) > 0 and len(first_split['test_indices']) > 0:
        assert max(first_split['train_indices']) < min(first_split['test_indices'])


def test_purged_walk_forward_with_purge_and_embargo():
    """Test that purge and embargo are correctly applied."""
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')

    cv = PurgedWalkForwardCV(
        initial_train_years=1,
        test_window_months=6,
        step_months=6,
        purge_days=10,
        embargo_days=5,
        label_period_days=30
    )

    splits = cv.split(dates, show_progress=False)

    assert len(splits) > 0

    first_split = splits[0]
    stats = first_split['statistics']

    assert stats['purge_days_applied'] >= 0
    assert stats['embargo_days_applied'] >= 0

    if len(first_split['train_timestamps']) > 0 and len(first_split['test_timestamps']) > 0:
        train_end = first_split['train_timestamps'][-1]
        test_start = first_split['test_timestamps'][0]

        gap_days = (test_start - train_end).days
        assert gap_days >= cv.embargo_days, f"Gap ({gap_days}) should be at least embargo days ({cv.embargo_days})"


def test_purged_walk_forward_sample_weights():
    """Test that sample weights are calculated correctly."""
    dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')

    cv = PurgedWalkForwardCV(
        initial_train_years=1,
        test_window_months=3,
        step_months=3,
        purge_days=0,
        embargo_days=0,
        label_period_days=30
    )

    splits = cv.split(dates, show_progress=False)

    assert len(splits) > 0

    first_split = splits[0]
    weights = first_split['sample_weights']

    assert len(weights) == len(first_split['train_indices'])
    assert all(w >= 0 for w in weights)

    if len(weights) > 0:
        expected_sum = len(weights)
        actual_sum = np.sum(weights)
        assert abs(actual_sum - expected_sum) < expected_sum * 0.1


def test_get_split_info():
    """Test the get_split_info method."""
    dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')

    cv = PurgedWalkForwardCV(
        initial_train_years=1,
        test_window_months=3,
        step_months=3,
        purge_days=0,
        embargo_days=0,
        label_period_days=0
    )

    splits = cv.split(dates, show_progress=False)
    df = cv.get_split_info(splits)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(splits)
    expected_columns = ['fold', 'train_size', 'test_size', 'train_start', 'train_end',
                       'test_start', 'test_end', 'purge_days', 'embargo_days']
    assert all(col in df.columns for col in expected_columns)


def test_edge_cases():
    """Test edge cases."""
    dates = pd.date_range('2020-01-01', '2020-06-30', freq='D')

    cv = PurgedWalkForwardCV(
        initial_train_years=2,
        test_window_months=6,
        step_months=6,
        purge_days=0,
        embargo_days=0,
        label_period_days=0
    )

    splits = cv.split(dates, show_progress=False)
    assert len(splits) == 0

    dates_single = pd.date_range('2020-01-01', '2020-01-01', freq='D')
    cv_single = PurgedWalkForwardCV(
        initial_train_years=0,
        test_window_months=1,
        step_months=1,
        purge_days=0,
        embargo_days=0,
        label_period_days=0
    )

    splits_single = cv_single.split(dates_single, show_progress=False)
    assert isinstance(splits_single, list)


if __name__ == "__main__":
    test_purged_walk_forward_initialization()
    print("✓ Initialization test passed")

    test_purged_walk_forward_split_basic()
    print("✓ Basic split test passed")

    test_purged_walk_forward_with_purge_and_embargo()
    print("✓ Purge and embargo test passed")

    test_purged_walk_forward_sample_weights()
    print("✓ Sample weights test passed")

    test_get_split_info()
    print("✓ Get split info test passed")

    test_edge_cases()
    print("✓ Edge cases test passed")

    print("\n🎉 All tests passed!")