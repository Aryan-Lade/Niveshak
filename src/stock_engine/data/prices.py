"""
Price data download, caching, and updating pipeline.

Features:
- Downloads daily OHLCV bars via yfinance with auto_adjust=True.
- Timezone-naive trading dates.
- Output columns: [date, ticker, open, high, low, close, volume]
- Caches per-ticker Parquet files in data_store/raw/prices/{ticker}.parquet.
- Incremental update support: checks last cached date and fetches only missing rows.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml
import yfinance as yf

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data_store/raw/prices")


def download_prices(
    tickers: List[str],
    start: str,
    end: str,
    retries: int = 3,
    backoff_factor: float = 1.5,
) -> pd.DataFrame:
    """
    Download daily OHLCV data for given tickers between start and end.
    Returns a tidy DataFrame with columns:
    [date, ticker, open, high, low, close, volume]
    """
    all_data = []

    for ticker in tickers:
        success = False
        for attempt in range(1, retries + 1):
            try:
                data = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                )

                if data is None or data.empty:
                    logger.warning(f"No data returned for {ticker} from {start} to {end}")
                    break

                data = data.reset_index()

                # Handle MultiIndex columns from newer yfinance
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]

                # Find date column
                date_col = next((c for c in data.columns if str(c).lower() == "date"), None)
                if date_col is None:
                    logger.warning(f"Date column not found for {ticker}")
                    break

                # Rename columns to standard lowercase
                col_map = {
                    date_col: "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
                data = data.rename(columns=col_map)
                # Lowercase any other columns
                data.columns = [str(c).lower() for c in data.columns]

                required = ["open", "high", "low", "close", "volume"]
                if not all(col in data.columns for col in required):
                    logger.warning(f"Missing required OHLCV columns for {ticker}")
                    break

                data["ticker"] = ticker
                data["date"] = pd.to_datetime(data["date"]).dt.date
                data = data[["date", "ticker", "open", "high", "low", "close", "volume"]]

                all_data.append(data)
                logger.info(f"Downloaded {len(data)} rows for {ticker}")
                success = True
                break

            except Exception as e:
                logger.warning(f"Attempt {attempt} failed for {ticker}: {e}")
                if attempt < retries:
                    time.sleep(backoff_factor ** attempt)
                else:
                    logger.error(f"Failed to download data for {ticker} after {retries} attempts: {e}")

    cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
    if not all_data:
        return pd.DataFrame(columns=cols)

    combined = pd.concat(all_data, ignore_index=True)
    return combined


def get_cached_data(ticker: str, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    """Load cached price parquet for ticker, or return empty DataFrame."""
    cache_path = Path(cache_dir) / f"{ticker}.parquet"
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.date
            return df
        except Exception as e:
            logger.error(f"Failed to read cache for {ticker}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


def save_to_cache(
    ticker: str,
    data: pd.DataFrame,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> None:
    """Save ticker DataFrame to parquet cache."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    file_path = cache_path / f"{ticker}.parquet"
    try:
        data.to_parquet(file_path, index=False)
        logger.info(f"Saved {len(data)} rows for {ticker} to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save cache for {ticker}: {e}")


def update_prices(
    tickers: List[str],
    start: str,
    end: str,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """
    Incrementally update cached prices for universe tickers.
    If cached data exists, fetches only dates from (latest_cached + 1 day) to end.
    """
    all_updated = []
    cols = ["date", "ticker", "open", "high", "low", "close", "volume"]

    for ticker in tickers:
        cached = get_cached_data(ticker, cache_dir=cache_dir)
        if not cached.empty:
            latest_cached = pd.to_datetime(cached["date"]).max()
            fetch_start = (latest_cached + timedelta(days=1)).strftime("%Y-%m-%d")
            if fetch_start > end:
                logger.info(f"Ticker {ticker} is already up to date")
                all_updated.append(cached)
                continue
        else:
            fetch_start = start

        new_data = download_prices([ticker], start=fetch_start, end=end)
        if new_data.empty:
            if not cached.empty:
                all_updated.append(cached)
            continue

        if not cached.empty:
            combined = pd.concat([cached, new_data], ignore_index=True)
            combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
            combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
        else:
            combined = new_data.sort_values(["ticker", "date"]).reset_index(drop=True)

        save_to_cache(ticker, combined, cache_dir=cache_dir)
        all_updated.append(combined)

    if not all_updated:
        return pd.DataFrame(columns=cols)

    return pd.concat(all_updated, ignore_index=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Update stock prices")
    parser.add_argument("--update", action="store_true", help="Update prices from yfinance")
    parser.add_argument("--universe", default="configs/universe.yaml", help="Path to universe.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml", help="Path to data.yaml")
    args = parser.parse_args()

    if args.update:
        with open(args.universe, "r", encoding="utf-8") as f:
            universe_cfg = yaml.safe_load(f)
        with open(args.data_config, "r", encoding="utf-8") as f:
            data_cfg = yaml.safe_load(f)

        ticker_list = universe_cfg.get("universe", [])
        start_d = data_cfg.get("data", {}).get("start_date", "2012-01-01")
        end_d = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Updating prices for {len(ticker_list)} tickers from {start_d} to {end_d}")
        df = update_prices(ticker_list, start_d, end_d)
        logger.info(f"Update completed. Total rows: {len(df)}")
