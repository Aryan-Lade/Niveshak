import pandas as pd
import yfinance as yf
import os
import time
import logging
from datetime import datetime, timedelta

def download_prices(tickers, start, end):
    all_data = []

    for ticker in tickers:
        try:
            data = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False
            )

            if data.empty:
                logging.warning(f"No data found for {ticker}")
                continue

            data = data.reset_index()

            if 'Date' not in data.columns:
                logging.warning(f"Unexpected data format for {ticker}")
                continue

            data = data.rename(columns={'Date': 'date'})

            data['ticker'] = ticker

            data = data[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']]

            data['date'] = pd.to_datetime(data['date']).dt.date

            all_data.append(data)
            logging.info(f"Downloaded {len(data)} rows for {ticker}")

        except Exception as e:
            logging.error(f"Failed to download data for {ticker}: {str(e)}")
            continue

    if not all_data:
        return pd.DataFrame(columns=['date', 'ticker', 'open', 'high', 'low', 'close', 'volume'])

    combined_data = pd.concat(all_data, ignore_index=True)
    return combined_data

def get_cached_data(ticker):
    cache_dir = "data_store/raw/prices"
    cache_file = os.path.join(cache_dir, f"{ticker}.parquet")

    if os.path.exists(cache_file):
        try:
            return pd.read_parquet(cache_file)
        except Exception as e:
            logging.error(f"Failed to read cached data for {ticker}: {str(e)}")
            return pd.DataFrame()
    else:
        return pd.DataFrame()

def save_to_cache(ticker, data):
    cache_dir = "data_store/raw/prices"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker}.parquet")

    try:
        data.to_parquet(cache_file, index=False)
        logging.info(f"Saved {len(data)} rows for {ticker} to cache")
    except Exception as e:
        logging.error(f"Failed to save cache for {ticker}: {str(e)}")

def update_prices(tickers, start, end):
    all_new_data = []

    for ticker in tickers:
        try:
            cached_data = get_cached_data(ticker)

            if not cached_data.empty:
                latest_cached = pd.to_datetime(cached_data['date']).max()
                fetch_start = (latest_cached + timedelta(days=1)).strftime('%Y-%m-%d')

                if fetch_start > end:
                    logging.info(f"Data for {ticker} is already up to date")
                    all_new_data.append(cached_data)
                    continue
            else:
                fetch_start = start

            new_data = yf.download(
                ticker,
                start=fetch_start,
                end=end,
                auto_adjust=True,
                progress=False
            )

            if new_data.empty:
                logging.warning(f"No new data found for {ticker} from {fetch_start} to {end}")
                if not cached_data.empty:
                    all_new_data.append(cached_data)
                continue

            new_data = new_data.reset_index()
            new_data = new_data.rename(columns={'Date': 'date'})
            new_data['ticker'] = ticker
            new_data = new_data[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']]
            new_data['date'] = pd.to_datetime(new_data['date']).dt.date

            if not cached_data.empty:
                combined_data = pd.concat([cached_data, new_data], ignore_index=True)
                combined_data = combined_data.drop_duplicates(subset=['date', 'ticker'], keep='last')
                combined_data = combined_data.sort_values(['ticker', 'date']).reset_index(drop=True)
            else:
                combined_data = new_data

            save_to_cache(ticker, combined_data)
            all_new_data.append(combined_data)

            logging.info(f"Updated {ticker}: {len(new_data)} new rows, {len(combined_data)} total rows")

            time.sleep(0.1)

        except Exception as e:
            logging.error(f"Failed to update prices for {ticker}: {str(e)}")
            cached_data = get_cached_data(ticker)
            if not cached_data.empty:
                all_new_data.append(cached_data)
            continue

    if not all_new_data:
        return pd.DataFrame(columns=['date', 'ticker', 'open', 'high', 'low', 'close', 'volume'])

    final_data = pd.concat(all_new_data, ignore_index=True)
    return final_data

if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='Update stock prices')
    parser.add_argument('--update', action='store_true', help='Update prices from yfinance')
    args = parser.parse_args()

    if args.update:
        with open('../../configs/universe.yaml', 'r') as f:
            universe_config = yaml.safe_load(f)

        with open('../../configs/data.yaml', 'r') as f:
            data_config = yaml.safe_load(f)

        tickers = universe_config['universe']
        start_date = data_config['start_date']
        end_date = datetime.now().strftime('%Y-%m-%d')

        logging.info(f"Starting price update for {len(tickers)} tickers from {start_date} to {end_date}")

        result_df = update_prices(tickers, start_date, end_date)

        if not result_df.empty:
            logging.info(f"Price update completed. Total rows: {len(result_df)}")
        else:
            logging.warning("No data was updated")