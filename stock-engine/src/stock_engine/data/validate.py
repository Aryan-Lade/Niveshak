import pandas as pd
import os
import logging
from datetime import datetime

def validate_prices(data):
    if data.empty:
        logging.warning("No data to validate")
        return pd.DataFrame()

    df = data.copy()

    df['date'] = pd.to_datetime(df['date'])

    validation_results = []

    tickers = df['ticker'].unique()

    for ticker in tickers:
        ticker_data = df[df['ticker'] == ticker].copy().sort_values('date')

        if ticker_data.empty:
            continue

        duplicate_dates = ticker_data['date'].duplicated().sum()
        if duplicate_dates > 0:
            validation_results.append({
                'ticker': ticker,
                'check': 'duplicate_dates',
                'status': 'FAIL',
                'message': f'Found {duplicate_dates} duplicate dates',
                'count': int(duplicate_dates)
            })

        date_diffs = ticker_data['date'].diff().dt.days
        non_increasing = (date_diffs < 0).sum()
        if non_increasing > 0:
            validation_results.append({
                'ticker': ticker,
                'check': 'non_increasing_dates',
                'status': 'FAIL',
                'message': f'Found {non_increasing} instances of non-increasing dates',
                'count': int(non_increasing)
            })

        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            invalid_prices = (ticker_data[col] <= 0).sum()
            if invalid_prices > 0:
                validation_results.append({
                    'ticker': ticker,
                    'check': f'invalid_{col}_prices',
                    'status': 'FAIL',
                    'message': f'Found {invalid_prices} {col} prices <= 0',
                    'count': int(invalid_prices)
                })

        invalid_high = (ticker_data['high'] < ticker_data[['open', 'close']].max(axis=1)).sum()
        if invalid_high > 0:
            validation_results.append({
                'ticker': ticker,
                'check': 'invalid_high',
                'status': 'FAIL',
                'message': f'Found {invalid_high} instances where high < max(open, close)',
                'count': int(invalid_high)
            })

        invalid_low = (ticker_data['low'] > ticker_data[['open', 'close']].min(axis=1)).sum()
        if invalid_low > 0:
            validation_results.append({
                'ticker': ticker,
                'check': 'invalid_low',
                'status': 'FAIL',
                'message': f'Found {invalid_low} instances where low > min(open, close)',
                'count': int(invalid_low)
            })

        ticker_data = ticker_data.sort_values('date')
        date_series = ticker_data['date']

        date_gaps = date_series.diff().dt.days
        large_gaps = (date_gaps > 6).sum()

        if large_gaps > 0:
            validation_results.append({
                'ticker': ticker,
                'check': 'large_calendar_gaps',
                'status': 'WARN',
                'message': f'Found {large_gaps} gaps longer than 5 business days',
                'count': int(large_gaps)
            })

        ticker_data['returns'] = ticker_data['close'].pct_change()
        extreme_returns = (ticker_data['returns'].abs() > 0.25).sum()
        if extreme_returns > 0:
            validation_results.append({
                'ticker': ticker,
                'check': 'extreme_returns',
                'status': 'WARN',
                'message': f'Found {extreme_returns} days with returns > 25% in absolute value',
                'count': int(extreme_returns)
            })

        zero_volume = (ticker_data['volume'] == 0).sum()
        if zero_volume > 0:
            validation_results.append({
                'ticker': ticker,
                'check': 'zero_volume',
                'status': 'INFO',
                'message': f'Found {zero_volume} zero-volume days',
                'count': int(zero_volume)
            })

    if validation_results:
        results_df = pd.DataFrame(validation_results)
    else:
        results_df = pd.DataFrame(columns=['ticker', 'check', 'status', 'message', 'count'])

    return results_df

def save_validation_report(results_df, report_path="reports/data_quality.csv"):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    results_df.to_csv(report_path, index=False)
    logging.info(f"Validation report saved to {report_path}")

    if not results_df.empty:
        print("\nData Validation Summary:")
        print("=" * 50)

        summary = results_df.groupby(['status', 'check'])['count'].sum().reset_index()
        for status in ['FAIL', 'WARN', 'INFO']:
            status_data = summary[summary['status'] == status]
            if not status_data.empty:
                print(f"\n{status}:")
                for _, row in status_data.iterrows():
                    print(f"  {row['check']}: {row['count']}")
    else:
        print("\nData Validation Summary:")
        print("=" * 50)
        print("No validation issues found")

def drop_nohlc_rows(data):
    if data.empty:
        return data

    ohlc_cols = ['open', 'high', 'low', 'close']
    nan_mask = data[ohlc_cols].isna().any(axis=1)

    if nan_mask.any():
        nan_count = nan_mask.sum()
        logging.info(f"Dropping {nan_count} rows with NaN OHLC values")
        return data[~nan_mask].copy()

    return data.copy()

if __name__ == "__main__":
    import yaml
    import glob

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        with open('../../configs/university.yaml', 'r') as f:
            universe_config = yaml.safe_load(f)
    except FileNotFoundError:
        try:
            with open('../../configs/universe.yaml', 'r') as f:
                universe_config = yaml.safe_load(f)
        except FileNotFoundError:
            logging.error("Could not find universe config file")
            exit(1)

    cache_dir = "data_store/raw/prices"
    if not os.path.exists(cache_dir):
        logging.error(f"Cache directory {cache_dir} does not exist")
        exit(1)

    parquet_files = glob.glob(os.path.join(cache_dir, "*.parquet"))

    if not parquet_files:
        logging.warning("No parquet files found in cache directory")
        exit(1)

    all_data = []
    for parquet_file in parquet_files:
        try:
            ticker = os.path.basename(parquet_file).replace('.parquet', '')
            data = pd.read_parquet(parquet_file)
            if not data.empty:
                all_data.append(data)
                logging.info(f"Loaded {len(data)} rows for {ticker}")
        except Exception as e:
            logging.error(f"Failed to load {parquet_file}: {str(e)}")

    if not all_data:
        logging.error("No data loaded from cache files")
        exit(1)

    combined_data = pd.concat(all_data, ignore_index=True)

    cleaned_data = drop_nohlc_rows(combined_data)

    validation_results = validate_prices(cleaned_data)

    save_validation_report(validation_results)