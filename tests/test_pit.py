import pandas as pd
import numpy as np
from src.stock_engine.data.pit import align_asof, get_trading_calendar

def test_align_asof_no_lookahead():
    """Test that a value dated after the as-of date is never used."""
    # Create a simple series with known values
    index = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=index, name='test')

    # Calendar with gaps
    calendar = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-06', '2023-01-07'])

    # Align with lag 0, ffill_limit=1
    result = align_asof(series, calendar, lag_days=0, ffill_limit=1)

    # For 2023-01-06: last known value is 2023-01-05 (value 5.0), gap=1 day -> within limit -> 5.0
    # For 2023-01-07: last known value is 2023-01-05 (value 5.0), gap=2 days -> exceeds limit -> NaN
    expected = pd.Series([1.0, 2.0, 3.0, 5.0, np.nan], index=calendar, name='test')
    pd.testing.assert_series_equal(result, expected)

def test_align_asof_lag_rules():
    """Test lag rules are respected using a hand-built example."""
    # Series representing a foreign index that closes after Indian market
    # We'll use daily data: value on 2023-01-01 is 100, 2023-01-02 is 101, etc.
    index = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'])
    series = pd.Series([100.0, 101.0, 102.0, 103.0], index=index, name='foreign')

    # Indian trading calendar (same as above but we'll use the dates as is)
    calendar = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'])

    # With lag 1: we shift the series forward by 1 day (so that the value from 01 is used for 02)
    result_lag1 = align_asof(series, calendar, lag_days=1, ffill_limit=0)
    # After shifting by 1:
    #   series becomes: index shifted forward by 1 ->
    #       02:100.0, 03:101.0, 04:102.0, 05:103.0 (and 01 becomes NaN because we shift forward and there's no data for 01)
    #   Then we align to calendar [01,02,03,04] with method='ffill' and ffill_limit=0 (so no fill)
    #   For calendar date 01: no value on or before 01 in shifted series -> NaN
    #   For 02: 100.0 (from shifted series at 02)
    #   For 03: 101.0
    #   For 04: 102.0
    expected_lag1 = pd.Series([np.nan, 100.0, 101.0, 102.0], index=calendar, name='foreign')
    pd.testing.assert_series_equal(result_lag1, expected_lag1, check_names=False)

    # Test lag 0: same day
    result_lag0 = align_asof(series, calendar, lag_days=0, ffill_limit=0)
    expected_lag0 = pd.Series([100.0, 101.0, 102.0, 103.0], index=calendar, name='foreign')
    pd.testing.assert_series_equal(result_lag0, expected_lag0, check_names=False)

def test_align_asof_ffill_limit():
    """Test forward-fill limit works."""
    # Series with data only on Mondays
    index = pd.to_datetime(['2023-01-02', '2023-01-09', '2023-01-16'])  # Mondays
    series = pd.Series([1.0, 2.0, 3.0], index=index, name='weekly')

    # Calendar every day from 2023-01-02 to 2023-01-15
    calendar = pd.date_range(start='2023-01-02', end='2023-01-15', freq='D')

    # With ffill_limit=3: should fill up to 3 days after the Monday
    result = align_asof(series, calendar, lag_days=0, ffill_limit=3)

    # We'll compute expected:
    # For each Monday, the value is that Monday's value and we forward fill up to 3 days (so Monday to Thursday)
    #   2023-01-02 (Mon): 1.0 -> gap0 -> 1.0
    #   2023-01-03 (Tue): gap1 -> 1.0
    #   2023-01-04 (Wed): gap2 -> 1.0
    #   2023-01-05 (Thu): gap3 -> 1.0
    #   2023-01-06 (Fri): gap4 -> NaN
    #   2023-01-07 (Sat): gap5 -> NaN
    #   2023-01-08 (Sun): gap6 -> NaN
    #   2023-01-09 (Mon): 2.0 -> gap0 -> 2.0
    #   2023-01-10 (Tue): gap1 -> 2.0
    #   2023-01-11 (Wed): gap2 -> 2.0
    #   2023-01-12 (Thu): gap3 -> 2.0
    #   2023-01-13 (Fri): gap4 -> NaN
    #   2023-01-14 (Sat): gap5 -> NaN
    #   2023-01-15 (Sun): gap6 -> NaN
    expected = pd.Series([
        1.0,  # 02 Mon
        1.0,  # 03 Tue
        1.0,  # 04 Wed
        1.0,  # 05 Thu
        np.nan, # 06 Fri
        np.nan, # 07 Sat
        np.nan, # 08 Sun
        2.0,  # 09 Mon
        2.0,  # 10 Tue
        2.0,  # 11 Wed
        2.0,  # 12 Thu
        np.nan, # 13 Fri
        np.nan, # 14 Sat
        np.nan  # 15 Sun
    ], index=calendar, name='weekly')
    pd.testing.assert_series_equal(result, expected, check_names=False)

if __name__ == "__main__":
    test_align_asof_no_lookahead()
    test_align_asof_lag_rules()
    test_align_asof_ffill_limit()
    print("All tests passed.")