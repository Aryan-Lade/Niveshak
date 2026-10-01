import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from stock_engine.data.prices import download_prices, update_prices, get_cached_data, save_to_cache
from stock_engine.data.validate import validate_prices, drop_nohlc_rows

def test_download_prices_success():
    mock_data = pd.DataFrame({
        'Date': ['2023-01-01', '2023-01-02'],
        'Open': [100.0, 101.0],
        'High': [105.0, 106.0],
        'Low': [99.0, 100.0],
        'Close': [104.0, 105.0],
        'Volume': [1000, 1100]
    })

    with patch('yfinance.download') as mock_download:
        mock_download.return_value = mock_data

        result = download_prices(['TEST.NS'], '2023-01-01', '2023-01-02')

        assert not result.empty
        assert len(result) == 2
        assert list(result.columns) == ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
        assert result['ticker'].iloc[0] == 'TEST.NS'
        assert result['date'].iloc[0] == pd.to_datetime('2023-01-01').date()

def test_download_prices_failure():
    with patch('yfinance.download') as mock_download:
        mock_download.side_effect = Exception("Network error")

        result = download_prices(['FAIL.NS'], '2023-01-01', '2023-01-02')

        assert result.empty
        assert list(result.columns) == ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']

def test_download_prices_mixed_success_failure():
    mock_data_success = pd.DataFrame({
        'Date': ['2023-01-01'],
        'Open': [100.0],
        'High': [105.0],
        'Low': [99.0],
        'Close': [104.0],
        'Volume': [1000]
    })

    def side_effect(ticker, start, end, auto_adjust, progress):
        if ticker == 'SUCCESS.NS':
            return mock_data_success
        else:
            raise Exception("Network error")

    with patch('yfinance.download') as mock_download:
        mock_download.side_effect = side_effect

        result = download_prices(['SUCCESS.NS', 'FAIL.NS'], '2023-01-01', '2023-01-02')

        assert not result.empty
        assert len(result) == 1
        assert result['ticker'].iloc[0] == 'SUCCESS.NS'

def test_get_cached_data_empty():
    result = get_cached_data('NONEXISTENT.NS')
    assert result.empty

def test_save_and_get_cached_data(tmpdir):
    cache_dir = tmpdir.mkdir("data_store").mkdir("raw").mkdir("prices")
    original_dir = os.getcwd()

    try:
        os.chdir(tmpdir)

        test_data = pd.DataFrame({
            'date': [pd.to_datetime('2023-01-01').date(), pd.to_datetime('2023-01-02').date()],
            'ticker': ['TEST.NS', 'TEST.NS'],
            'open': [100.0, 101.0],
            'high': [105.0, 106.0],
            'low': [99.0, 100.0],
            'close': [104.0, 105.0],
            'volume': [1000, 1100]
        })

        save_to_cache('TEST.NS', test_data)

        cached_data = get_cached_data('TEST.NS')

        assert not cached_data.empty
        assert len(cached_data) == 2
        assert list(cached_data.columns) == ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']

    finally:
        os.chdir(original_dir)

def test_update_prices_incremental():
    cache_dir = "data_store/raw/prices"
    os.makedirs(cache_dir, exist_ok=True)

    try:
        initial_data = pd.DataFrame({
            'date': [pd.to_datetime('2023-01-01').date(), pd.to_datetime('2023-01-02').date()],
            'ticker': ['TEST.NS', 'TEST.NS'],
            'open': [100.0, 101.0],
            'high': [105.0, 106.0],
            'low': [99.0, 100.0],
            'close': [104.0, 105.0],
            'volume': [1000, 1100]
        })

        initial_data.to_parquet(os.path.join(cache_dir, "TEST.NS.parquet"), index=False)

        mock_new_data = pd.DataFrame({
            'Date': ['2023-01-03', '2023-01-04'],
            'Open': [105.0, 106.0],
            'High': [110.0, 111.0],
            'Low': [104.0, 105.0],
            'Close': [109.0, 110.0],
            'Volume': [1200, 1300]
        })

        with patch('yfinance.download') as mock_download:
            mock_download.return_value = mock_new_data

            result = update_prices(['TEST.NS'], '2023-01-01', '2023-01-04')

            assert not result.empty
            assert len(result) == 4

            dates = pd.to_datetime(result['date'])
            assert dates.is_monotonic_increasing

    finally:
        test_file = os.path.join(cache_dir, "TEST.NS.parquet")
        if os.path.exists(test_file):
            os.remove(test_file)

def test_validate_prices_duplicate_dates():
    test_data = pd.DataFrame({
        'date': [pd.to_datetime('2023-01-01').date(), pd.to_datetime('2023-01-01').date(), pd.to_datetime('2023-01-02').date()],
        'ticker': ['TEST.NS', 'TEST.NS', 'TEST.NS'],
        'open': [100.0, 101.0, 102.0],
        'high': [105.0, 106.0, 107.0],
        'low': [99.0, 100.0, 101.0],
        'close': [104.0, 105.0, 106.0],
        'volume': [1000, 1100, 1200]
    })

    results = validate_prices(test_data)

    duplicate_check = results[results['check'] == 'duplicate_dates']
    assert not duplicate_check.empty
    assert duplicate_check.iloc[0]['status'] == 'FAIL'
    assert duplicate_check.iloc[0]['count'] == 1

def test_validate_prices_invalid_prices():
    test_data = pd.DataFrame({
        'date': [pd.to_datetime('2023-01-01').date(), pd.to_datetime('2023-01-02').date()],
        'ticker': ['TEST.NS', 'TEST.NS'],
        'open': [100.0, -1.0],
        'high': [105.0, 106.0],
        'low': [99.0, 0.0],
        'close': [104.0, 105.0],
        'volume': [1000, 1100]
    })

    results = validate_prices(test_data)

    invalid_open = results[results['check'] == 'invalid_open_prices']
    invalid_low = results[results['check'] == 'invalid_low_prices']

    assert not invalid_open.empty
    assert invalid_open.iloc[0]['status'] == 'FAIL'
    assert invalid_open.iloc[0]['count'] == 1

    assert not invalid_low.empty
    assert invalid_low.iloc[0]['status'] == 'FAIL'
    assert invalid_low.iloc[0]['count'] == 1

def test_validate_prices_extreme_returns():
    test_data = pd.DataFrame({
        'date': [pd.to_datetime('2023-01-01').date(), pd.to_datetime('2023-01-02').date()],
        'ticker': ['TEST.NS', 'TEST.NS'],
        'open': [100.0, 130.0],
        'high': [105.0, 135.0],
        'low': [99.0, 129.0],
        'close': [104.0, 130.0],
        'volume': [1000, 1100]
    })

    results = validate_prices(test_data)

    extreme_returns = results[results['check'] == 'extreme_returns']
    assert not extreme_returns.empty
    assert extreme_returns.iloc[0]['status'] == 'WARN'
    assert extreme_returns.iloc[0]['count'] == 1

def test_drop_nohlc_rows():
    test_data = pd.DataFrame({
        'date': [pd.to_datetime('2023-01-01').date(), pd.to_datetime('2023-01-02').date(), pd.to_datetime('2023-01-03').date()],
        'ticker': ['TEST.NS', 'TEST.NS', 'TEST.NS'],
        'open': [100.0, None, 102.0],
        'high': [105.0, 106.0, 107.0],
        'low': [99.0, 100.0, None],
        'close': [104.0, 105.0, 106.0],
        'volume': [1000, 0, 1200]
    })

    result = drop_nohlc_rows(test_data)

    assert len(result) == 2
    assert result.iloc[0]['date'] == pd.to_datetime('2023-01-01').date()
    assert result.iloc[1]['date'] == pd.to_datetime('2023-01-03').date()
    assert result.iloc[1]['volume'] == 0

if __name__ == "__main__":
    pytest.main([__file__])