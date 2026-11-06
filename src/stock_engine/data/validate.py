"""
Data validation and cleaning routines for stock OHLCV data.

Validation checks:
1. Duplicate dates for a ticker.
2. Non-increasing / unordered dates.
3. Prices <= 0 (open, high, low, close).
4. High < max(open, close) violations.
5. Low > min(open, close) violations.
6. Large calendar gaps (> 5 business days).
7. Single-day returns > 25% (extreme jumps).
8. Zero volume days.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_prices(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate a tidy OHLCV DataFrame with columns [date, ticker, open, high, low, close, volume].
    Returns a DataFrame of validation findings with columns:
    [ticker, check, status, message, count]
    """
    if data.empty:
        logger.warning("No data to validate")
        return pd.DataFrame(columns=["ticker", "check", "status", "message", "count"])

    df = data.copy()
    df["date"] = pd.to_datetime(df["date"])

    validation_results: List[Dict] = []
    tickers = df["ticker"].unique()

    for ticker in tickers:
        ticker_data = df[df["ticker"] == ticker].copy().sort_values("date")
        if ticker_data.empty:
            continue

        # 1. Duplicate dates
        duplicate_dates = int(ticker_data["date"].duplicated().sum())
        if duplicate_dates > 0:
            validation_results.append({
                "ticker": ticker,
                "check": "duplicate_dates",
                "status": "FAIL",
                "message": f"Found {duplicate_dates} duplicate dates",
                "count": duplicate_dates,
            })

        # 2. Non-increasing dates
        date_diffs = ticker_data["date"].diff().dt.days
        non_increasing = int((date_diffs < 0).sum())
        if non_increasing > 0:
            validation_results.append({
                "ticker": ticker,
                "check": "non_increasing_dates",
                "status": "FAIL",
                "message": f"Found {non_increasing} instances of non-increasing dates",
                "count": non_increasing,
            })

        # 3. Invalid (non-positive) prices
        for col in ["open", "high", "low", "close"]:
            if col in ticker_data.columns:
                invalid_count = int((ticker_data[col] <= 0).sum())
                if invalid_count > 0:
                    validation_results.append({
                        "ticker": ticker,
                        "check": f"invalid_{col}_prices",
                        "status": "FAIL",
                        "message": f"Found {invalid_count} {col} prices <= 0",
                        "count": invalid_count,
                    })

        # 4. High < max(open, close)
        if all(c in ticker_data.columns for c in ["high", "open", "close"]):
            invalid_high = int((ticker_data["high"] < ticker_data[["open", "close"]].max(axis=1)).sum())
            if invalid_high > 0:
                validation_results.append({
                    "ticker": ticker,
                    "check": "invalid_high",
                    "status": "FAIL",
                    "message": f"Found {invalid_high} instances where high < max(open, close)",
                    "count": invalid_high,
                })

        # 5. Low > min(open, close)
        if all(c in ticker_data.columns for c in ["low", "open", "close"]):
            invalid_low = int((ticker_data["low"] > ticker_data[["open", "close"]].min(axis=1)).sum())
            if invalid_low > 0:
                validation_results.append({
                    "ticker": ticker,
                    "check": "invalid_low",
                    "status": "FAIL",
                    "message": f"Found {invalid_low} instances where low > min(open, close)",
                    "count": invalid_low,
                })

        # 6. Large calendar gaps
        large_gaps = int((date_diffs > 6).sum())
        if large_gaps > 0:
            validation_results.append({
                "ticker": ticker,
                "check": "large_calendar_gaps",
                "status": "WARN",
                "message": f"Found {large_gaps} gaps longer than 5 business days",
                "count": large_gaps,
            })

        # 7. Extreme single-day returns (> 20% per Day 2 spec)
        if "close" in ticker_data.columns:
            returns = ticker_data["close"].pct_change()
            extreme_returns = int((returns.abs() > 0.20).sum())
            if extreme_returns > 0:
                validation_results.append({
                    "ticker": ticker,
                    "check": "extreme_returns",
                    "status": "WARN",
                    "message": f"Found {extreme_returns} days with returns > 20% in absolute value",
                    "count": extreme_returns,
                })

        # 8. Zero volume
        if "volume" in ticker_data.columns:
            zero_vol = int((ticker_data["volume"] == 0).sum())
            if zero_vol > 0:
                validation_results.append({
                    "ticker": ticker,
                    "check": "zero_volume",
                    "status": "INFO",
                    "message": f"Found {zero_vol} zero-volume days",
                    "count": zero_vol,
                })

    if validation_results:
        return pd.DataFrame(validation_results)
    return pd.DataFrame(columns=["ticker", "check", "status", "message", "count"])


def drop_nohlc_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Drop any rows where open, high, low, or close is null or NaN."""
    if data.empty:
        return data

    ohlc_cols = [c for c in ["open", "high", "low", "close"] if c in data.columns]
    nan_mask = data[ohlc_cols].isna().any(axis=1)

    if nan_mask.any():
        nan_count = int(nan_mask.sum())
        logger.info(f"Dropping {nan_count} rows with missing OHLC values")
        return data[~nan_mask].copy()

    return data.copy()


def save_validation_report(results_df: pd.DataFrame, report_path: str | Path = "reports/data_quality.csv") -> None:
    """Save the validation report to CSV."""
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(path, index=False)
    logger.info(f"Validation report saved to {path}")
