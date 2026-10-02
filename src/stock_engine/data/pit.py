import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

RAW_MACRO_DIR = Path("data_store/raw/macro")
FEATURES_DIR = Path("data_store/features")

def get_trading_calendar():
    """Build master trading calendar from NIFTY 50 dates."""
    nifty_path = RAW_MACRO_DIR / "NIFTY_50.parquet"
    if not nifty_path.exists():
        raise FileNotFoundError("NIFTY_50.parquet not found. Run macro.py first.")
    nifty = pd.read_parquet(nifty_path)
    nifty.index = pd.to_datetime(nifty.index)
    calendar = nifty.index.sort_values()
    return calendar

def align_asof(series, calendar, lag_days=0, ffill_limit=3):
    """
    Align a series to the trading calendar using only values known at or before each date.

    Parameters:
    series: pd.Series with datetime index (raw data)
    calendar: pd.DatetimeIndex of trading dates
    lag_days: number of days to lag the series (0 for same-day, 1 for previous day)
    ffill_limit: maximum days to forward fill missing values

    Returns:
    pd.Series aligned to calendar with same name as input series
    """
    if series.empty:
        return pd.Series(index=calendar, dtype=series.dtype, name=series.name)

    series = series.copy()
    series.index = pd.to_datetime(series.index)

    if lag_days != 0:
        series = series.shift(lag_days)

    # Get the aligned values and the corresponding index dates from the series
    values_aligned = series.reindex(calendar, method='ffill')
    index_aligned = series.index.to_series().reindex(calendar, method='ffill')

    # Compute gap in days
    gap_days = (calendar - index_aligned).dt.days

    # Valid if gap is within ffill_limit and we have a valid index (not NaT)
    valid = (gap_days <= ffill_limit) & gap_days.notna()

    # Where not valid, set to NaN
    result = values_aligned.where(valid)

    return result

def build_macro_panel():
    """Build macro panel parquet for all macro series."""
    calendar = get_trading_calendar()

    # Define series and their lag rules
    series_info = {
        "NIFTY_50": {"lag": 0, "ffill_limit": 3},
        "INDIA_VIX": {"lag": 0, "ffill_limit": 3},
        "USD_INR": {"lag": 1, "ffill_limit": 3},
        "S&P_500": {"lag": 1, "ffill_limit": 3},
        "CRUDE_OIL": {"lag": 1, "ffill_limit": 3}
    }

    panel_data = {}
    for name, info in series_info.items():
        filepath = RAW_MACRO_DIR / f"{name}.parquet"
        if not filepath.exists():
            logger.warning(f"{filepath} not found, skipping {name}")
            continue
        try:
            df = pd.read_parquet(filepath)
            # The raw data has a single column with the series name
            series = df.iloc[:, 0]
            series.name = name
            aligned = align_asof(series, calendar, lag_days=info["lag"], ffill_limit=info["ffill_limit"])
            panel_data[name] = aligned
        except Exception as e:
            logger.error(f"Failed to process {name}: {e}")

    if not panel_data:
        raise ValueError("No macro data loaded")

    panel = pd.DataFrame(panel_data)
    panel.index.name = "date"

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FEATURES_DIR / "macro_panel.parquet"
    panel.to_parquet(output_path)
    logger.info(f"Macro panel saved to {output_path} with shape {panel.shape}")
    return panel

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_macro_panel()