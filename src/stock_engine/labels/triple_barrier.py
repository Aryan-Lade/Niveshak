"""
Triple-barrier labeling for long-only equity trading.

Definitions (long-only, cash segment):
- Signal generated at close of day t. Entry is at OPEN of day t+1.
- vol_t = EWM standard deviation of daily returns (span 20), known at close of t.
- Upper barrier = entry * (1 + k_up * vol_t)
- Lower barrier = entry * (1 - k_dn * vol_t)
- Scan trading days t+1 ... t+H:
  * If both touched on same day: conservative rule -> lower barrier touched first.
  * If high >= upper barrier: exit at upper barrier, hit = +1, label = 1.
  * Elif low <= lower barrier: exit at lower barrier, hit = -1, label = 0.
  * Elif reached t+H: exit at close of t+H, hit = 0, label = 1 if ret > 0 else 0.
- Sample weight: inverse of overlapping label intervals active at date t (de Prado concurrency).
- Output DataFrame indexed by (date, ticker) with columns:
  [label, hit, ret, exit_date, t1, sample_weight]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def compute_volatility(close: pd.Series, span: int = 20) -> pd.Series:
    """
    Compute daily volatility using exponentially weighted moving standard deviation
    of percentage returns. Computed strictly on past data up to date t.
    """
    ret = close.pct_change()
    vol = ret.ewm(span=span, min_periods=span // 2).std()
    # Fill early warm-up values with expanding std or constant fallback
    vol = vol.fillna(ret.expanding(min_periods=5).std()).bfill().fillna(0.015)
    # Clip at positive floor so zero-vol flat prices do not collapse barriers
    vol = vol.clip(lower=0.005)
    return vol


def compute_sample_weights(events: pd.DataFrame) -> pd.Series:
    """
    Compute de Prado concurrency weights for overlapping label intervals [date, t1].
    Weight at date t is 1 / c_t where c_t is the number of active concurrent labels.
    """
    if events.empty:
        return pd.Series(dtype=float)

    weights = []
    # Work per ticker
    for ticker, group in events.groupby(level="ticker" if isinstance(events.index, pd.MultiIndex) else "ticker"):
        if isinstance(group.index, pd.MultiIndex):
            t_starts = group.index.get_level_values("date")
        else:
            t_starts = pd.to_datetime(group["date"])

        t_ends = pd.to_datetime(group["t1"])
        t_starts = pd.to_datetime(t_starts)

        # For each event, count how many other events overlap with its start date
        concurrency = np.zeros(len(group), dtype=float)
        for i, (t_s, t_e) in enumerate(zip(t_starts, t_ends)):
            # Active events that contain date t_s
            overlapping = ((t_starts <= t_s) & (t_s <= t_ends)).sum()
            concurrency[i] = max(1, overlapping)

        ticker_weights = 1.0 / concurrency
        weights.extend(ticker_weights)

    return pd.Series(weights, index=events.index, name="sample_weight")


def compute_triple_barrier_labels(
    prices: pd.DataFrame,
    k_up: float = 1.5,
    k_dn: float = 1.0,
    horizon: int = 5,
    holdout_start: Optional[str] = "2025-10-01",
) -> pd.DataFrame:
    """
    Compute triple-barrier labels for a DataFrame of daily bars.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Can have MultiIndex (date, ticker) or columns ['date', 'ticker', 'open', 'high', 'low', 'close']
        (casing can be capitalized or lowercase).
    k_up : float
        Upper barrier multiplier on daily volatility.
    k_dn : float
        Lower barrier multiplier on daily volatility.
    horizon : int
        Maximum holding period H in trading days.
    holdout_start : str, optional
        Rows on or after this date are excluded from final output if specified.
    """
    df = prices.copy()

    # Standardize index and columns
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    col_map = {c: str(c).lower() for c in df.columns}
    df = df.rename(columns=col_map)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    records = []

    for ticker, group in df.groupby("ticker"):
        group = group.sort_values("date").reset_index(drop=True)
        n = len(group)
        if n < 2:
            continue

        vol = compute_volatility(group["close"], span=20)
        dates = group["date"].values
        opens = group["open"].values
        highs = group["high"].values
        lows = group["low"].values
        closes = group["close"].values
        vols = vol.values

        # Date t goes from 0 up to n - 2 (must have at least day t+1 for entry)
        for t in range(n - 1):
            date_t = dates[t]
            entry_price = opens[t + 1]
            vol_t = vols[t]

            upper_barrier = entry_price * (1.0 + k_up * vol_t)
            lower_barrier = entry_price * (1.0 - k_dn * vol_t)

            max_h = min(t + 1 + horizon, n)
            exit_date = None
            exit_price = None
            hit = None
            label = None

            for i in range(t + 1, max_h):
                bar_high = highs[i]
                bar_low = lows[i]
                bar_date = dates[i]

                touched_upper = bar_high >= upper_barrier
                touched_lower = bar_low <= lower_barrier

                # Conservative rule: if both touched on the same day, lower touched first
                if touched_upper and touched_lower:
                    exit_date = bar_date
                    exit_price = lower_barrier
                    hit = -1
                    label = 0
                    break
                elif touched_lower:
                    exit_date = bar_date
                    exit_price = lower_barrier
                    hit = -1
                    label = 0
                    break
                elif touched_upper:
                    exit_date = bar_date
                    exit_price = upper_barrier
                    hit = 1
                    label = 1
                    break
                elif i == t + horizon:
                    # Time barrier reached at close of t+H
                    exit_date = bar_date
                    exit_price = closes[i]
                    hit = 0
                    ret = (exit_price / entry_price) - 1.0
                    label = 1 if ret > 0 else 0
                    break

            if exit_date is not None:
                ret = (exit_price / entry_price) - 1.0
                records.append({
                    "date": pd.Timestamp(date_t),
                    "ticker": ticker,
                    "label": int(label),
                    "hit": int(hit),
                    "ret": float(ret),
                    "exit_date": pd.Timestamp(exit_date),
                    "t1": pd.Timestamp(exit_date),
                })

    if not records:
        cols = ["label", "hit", "ret", "exit_date", "t1", "sample_weight"]
        idx = pd.MultiIndex.from_tuples([], names=["date", "ticker"])
        return pd.DataFrame(columns=cols, index=idx)

    res_df = pd.DataFrame(records)
    res_df = res_df.set_index(["date", "ticker"]).sort_index()

    # Compute concurrency sample weights
    res_df["sample_weight"] = compute_sample_weights(res_df)

    if holdout_start:
        holdout_dt = pd.Timestamp(holdout_start)
        res_df = res_df[res_df.index.get_level_values("date") < holdout_dt]

    return res_df


def save_labels(
    labels_df: pd.DataFrame,
    out_path: str | Path = "data_store/labels/labels.parquet",
) -> None:
    """Save labels DataFrame to Parquet."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_parquet(p)
    logger.info("Saved %d labels to %s", len(labels_df), p)
