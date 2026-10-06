"""
Market features calculation module.
Calculates various market-wide and cross-sectional features for the stock engine.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Paths - assuming current working directory is stock-engine/src/stock_engine/features
FEATURES_DIR = Path("../../../../data_store/features")
MACRO_PANEL_PATH = FEATURES_DIR / "macro_panel.parquet"
PRICES_RAW_DIR = Path("../../../../stock-engine/data_store/raw/prices")


def load_macro_panel() -> pd.DataFrame:
    """
    Load macro panel data.

    Returns:
        DataFrame with macro data indexed by date
    """
    if not MACRO_PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Macro panel not found at {MACRO_PANEL_PATH}. "
            "Run macro.py and pit.py first to generate macro data."
        )

    df = pd.read_parquet(MACRO_PANEL_PATH)
    df.index = pd.to_datetime(df.index)
    return df


def calculate_returns(prices: pd.Series, periods: List[int]) -> Dict[str, pd.Series]:
    """
    Calculate returns for specified periods.

    Args:
        prices: Series of prices (typically close prices)
        periods: List of periods for which to calculate returns

    Returns:
        Dictionary mapping period names to return series
    """
    returns = {}
    for period in periods:
        returns[f'ret_{period}d'] = prices.pct_change(periods=period)
    return returns


def calculate_volatility(returns: pd.Series, window: int) -> pd.Series:
    """
    Calculate rolling volatility (standard deviation of returns).

    Args:
        returns: Series of returns
        window: Rolling window size

    Returns:
        Series of rolling volatility
    """
    return returns.rolling(window=window, min_periods=1).std()


def calculate_relative_strength(stock_prices: pd.Series,
                              nifty_prices: pd.Series,
                              periods: List[int]) -> Dict[str, pd.Series]:
    """
    Calculate relative strength of stock vs NIFTY over specified periods.

    Args:
        stock_prices: Series of stock prices
        nifty_prices: Series of NIFTY prices
        periods: List of periods for relative strength calculation

    Returns:
        Dictionary mapping period names to relative strength series
    """
    # Calculate stock and NIFTY returns
    stock_returns = stock_prices.pct_change()
    nifty_returns = nifty_prices.pct_change()

    # Calculate cumulative returns over periods
    rel_strength = {}
    for period in periods:
        stock_cumret = (1 + stock_returns).rolling(window=period, min_periods=1).apply(
            lambda x: np.prod(1 + x) - 1, raw=True
        )
        nifty_cumret = (1 + nifty_returns).rolling(window=period, min_periods=1).apply(
            lambda x: np.prod(1 + x) - 1, raw=True
        )
        # Relative strength = (1 + stock_cumret) / (1 + nifty_cumret) - 1
        rel_strength[f'rs_{period}d'] = (1 + stock_cumret) / (1 + nifty_cumret) - 1

    return rel_strength


def calculate_beta(stock_returns: pd.Series,
                   market_returns: pd.Series,
                   window: int) -> pd.Series:
    """
    Calculate rolling beta of stock vs market.

    Args:
        stock_returns: Series of stock returns
        market_returns: Series of market returns (NIFTY)
        window: Rolling window size for beta calculation

    Returns:
        Series of rolling beta values
    """
    # Calculate rolling covariance and variance
    covariance = stock_returns.rolling(window=window, min_periods=2).cov(market_returns)
    market_variance = market_returns.rolling(window=window, min_periods=2).var()

    # Beta = covariance / variance
    beta = covariance / market_variance
    return beta


def calculate_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate rolling z-score (standardized value).

    Args:
        series: Input series
        window: Rolling window for mean and std calculation

    Returns:
        Series of z-scores
    """
    mean = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std()
    # Avoid division by zero
    zscore = (series - mean) / std.replace(0, np.nan)
    return zscore


def load_price_data(ticker: str) -> Optional[pd.DataFrame]:
    """
    Load price data for a single ticker.

    Args:
        ticker: Stock ticker symbol (e.g., 'RELIANCE.NS')

    Returns:
        DataFrame with OHLCV data or None if not found
    """
    price_file = PRICES_RAW_DIR / f"{ticker}.parquet"
    if not price_file.exists():
        logger.warning(f"Price data not found for {ticker} at {price_file}")
        return None

    try:
        import ast
        df = pd.read_parquet(price_file)
        # Parse string representations of tuples in column names
        new_cols = {}
        for col in df.columns:
            try:
                parsed = ast.literal_eval(col)
                if isinstance(parsed, tuple):
                    # For date and ticker columns, keep as is
                    if parsed[0] in ['date', 'ticker']:
                        new_cols[col] = parsed[0]
                    # For OHLCV columns, remove the ticker suffix and convert to lowercase
                    else:
                        new_cols[col] = parsed[0].lower()  # Just 'open', 'high', etc. in lowercase
                else:
                    new_cols[col] = col
            except:
                new_cols[col] = col
        df = df.rename(columns=new_cols)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        return df
    except Exception as e:
        logger.error(f"Error loading price data for {ticker}: {e}")
        return None


def build_market_features(universe: List[str]) -> pd.DataFrame:
    """
    Build all market features for the given universe of stocks.

    Args:
        universe: List of ticker symbols

    Returns:
        DataFrame with market features indexed by (date, ticker)
    """
    logger.info("Building market features...")

    # Load macro data
    macro_df = load_macro_panel()
    logger.info(f"Loaded macro data with shape {macro_df.shape}")

    # Extract NIFTY data for beta and relative strength calculations
    if 'NIFTY_50' not in macro_df.columns:
        raise ValueError("NIFTY_50 column not found in macro panel")

    nifty_prices = macro_df['NIFTY_50']
    nifty_returns = nifty_prices.pct_change()

    # Initialize list to hold features for each ticker
    ticker_features = []

    # Process each ticker in the universe
    for ticker in universe:
        logger.info(f"Processing {ticker}")

        # Load price data
        price_df = load_price_data(ticker)
        if price_df is None:
            logger.warning(f"Skipping {ticker} due to missing price data")
            continue

        # Ensure we have close prices
        if 'close' not in price_df.columns:
            logger.warning(f"No close price column for {ticker}")
            continue

        close_prices = price_df['close']

        # Initialize features DataFrame for this ticker
        features = pd.DataFrame(index=price_df.index)
        features['ticker'] = ticker

        # 1. MACRO FEATURES (from macro panel)
        # Align macro data with price data dates
        macro_aligned = macro_df.reindex(features.index, method='ffill')

        # NIFTY returns
        features['nifty_ret_1d'] = macro_aligned['NIFTY_50'].pct_change(1)
        features['nifty_ret_5d'] = macro_aligned['NIFTY_50'].pct_change(5)
        features['nifty_ret_20d'] = macro_aligned['NIFTY_50'].pct_change(20)

        # NIFTY 20-day realized volatility (std of 1-day returns)
        nifty_1d_ret = macro_aligned['NIFTY_50'].pct_change(1)
        features['nifty_vol_20d'] = nifty_1d_ret.rolling(window=20, min_periods=1).std()

        # India VIX features
        if 'INDIA_VIX' in macro_aligned.columns:
            features['vix_level'] = macro_aligned['INDIA_VIX']
            features['vix_1d_change'] = macro_aligned['INDIA_VIX'].pct_change(1)
            # 252-day rolling z-score (past only)
            features['vix_zscore_252d'] = calculate_zscore(
                macro_aligned['INDIA_VIX'], window=252
            )
        else:
            logger.warning("INDIA_VIX not found in macro panel")
            features['vix_level'] = np.nan
            features['vix_1d_change'] = np.nan
            features['vix_zscore_252d'] = np.nan

        # USDINR 5d change
        if 'USD_INR' in macro_aligned.columns:
            features['usdinr_5d_change'] = macro_aligned['USD_INR'].pct_change(5)
        else:
            logger.warning("USD_INR not found in macro panel")
            features['usdinr_5d_change'] = np.nan

        # S&P 500 returns (note: already lagged by PIT layer per requirements)
        if 'S&P_500' in macro_aligned.columns:
            features['sp500_ret_1d'] = macro_aligned['S&P_500'].pct_change(1)
            features['sp500_ret_5d'] = macro_aligned['S&P_500'].pct_change(5)
        else:
            logger.warning("S&P_500 not found in macro panel")
            features['sp500_ret_1d'] = np.nan
            features['sp500_ret_5d'] = np.nan

        # Crude 5d change
        if 'CRUDE_OIL' in macro_aligned.columns:
            features['crude_5d_change'] = macro_aligned['CRUDE_OIL'].pct_change(5)
        else:
            logger.warning("CRUDE_OIL not found in macro panel")
            features['crude_5d_change'] = np.nan

        # 2. STOCK-SPECIFIC FEATURES
        # Stock returns
        stock_returns = close_prices.pct_change()
        features['ret_1d'] = stock_returns
        features['ret_5d'] = close_prices.pct_change(5)
        features['ret_20d'] = close_prices.pct_change(20)

        # Stock volatility
        features['vol_20d'] = calculate_volatility(stock_returns, window=20)

        # Relative strength vs NIFTY
        rs_dict = calculate_relative_strength(
            close_prices, nifty_prices, [5, 20, 60]
        )
        for key, value in rs_dict.items():
            features[key] = value

        # Beta to NIFTY (60-day rolling)
        features['beta_60d'] = calculate_beta(
            stock_returns, nifty_returns, window=60
        )

        # 3. Add ticker for cross-sectional calculations (will be done later)
        # We'll keep the ticker column and do cross-sectional ranking after
        # collecting all tickers

        ticker_features.append(features)

    # Combine all ticker features
    if not ticker_features:
        logger.error("No ticker features were generated")
        return pd.DataFrame()

    combined_features = pd.concat(ticker_features, axis=0)

    # 4. CROSS-SECTIONAL PERCENTILE RANKS
    # Calculate percentile ranks for each date across tickers
    logger.info("Calculating cross-sectional percentile ranks...")

    # Reset index to have date and ticker as columns for easier manipulation
    cs_features = combined_features.reset_index()

    # Define features for which to calculate cross-sectional percentiles
    cs_features_list = ['ret_20d', 'vol_20d']

    # For volume z-score, we need volume data
    # Let's add volume to our features first
    volume_data = []
    for ticker in universe:
        price_df = load_price_data(ticker)
        if price_df is not None and 'volume' in price_df.columns:
            vol_series = price_df['volume']
            vol_series.name = ticker
            volume_data.append(vol_series)

    if volume_data:
        volume_df = pd.concat(volume_data, axis=1)
        volume_df.index = pd.to_datetime(volume_df.index)
        # Calculate volume z-score (20-day)
        volume_mean = volume_df.rolling(window=20, min_periods=1).mean()
        volume_std = volume_df.rolling(window=20, min_periods=1).std()
        volume_zscore = (volume_df - volume_mean) / volume_std.replace(0, np.nan)
        # Stack to long format
        volume_zscore_long = volume_zscore.stack()
        volume_zscore_long.name = 'volume_zscore'
        volume_zscore_long = volume_zscore_long.reset_index()
        volume_zscore_long.columns = ['date', 'ticker', 'volume_zscore']

        # Merge with our features
        cs_features = cs_features.merge(
            volume_zscore_long,
            on=['date', 'ticker'],
            how='left'
        )
        cs_features_list.append('volume_zscore')

    # Calculate cross-sectional percentile ranks for each date
    def calculate_percentile_group(group):
        """Calculate percentile ranks for a group (same date)."""
        result = group.copy()
        for feature in cs_features_list:
            if feature in group.columns:
                # Calculate percentile rank (0-100)
                # Using 'rank' method with 'average' to handle ties
                vals = group[feature].dropna()
                if len(vals) > 0:
                    # Rank and convert to percentile
                    ranks = vals.rank(method='average', pct=True) * 100
                    # Map back to original indices
                    result.loc[vals.index, f'{feature}_percentile'] = ranks
                else:
                    result[f'{feature}_percentile'] = np.nan
        return result

    # Apply percentile calculation by date
    cs_features = cs_features.groupby('date', group_keys=False).apply(calculate_percentile_group)

    # Set index back to (date, ticker) and sort
    cs_features = cs_features.set_index(['date', 'ticker']).sort_index()

    logger.info(f"Market features built successfully with shape {cs_features.shape}")
    return cs_features


def save_market_features(features_df: pd.DataFrame) -> None:
    """
    Save market features to parquet file.

    Args:
        features_df: DataFrame with market features to save
    """
    if features_df.empty:
        logger.warning("No market features to save")
        return

    # Ensure features directory exists
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FEATURES_DIR / "market_features.parquet"
    features_df.to_parquet(output_path)
    logger.info(f"Market features saved to {output_path}")


def main() -> None:
    """Main function to build and save market features."""
    # Load universe from config
    import yaml

    try:
        # Path relative to stock-engine/src
        with open("../../configs/universe.yaml", 'r') as f:
            universe_config = yaml.safe_load(f)
        universe = universe_config['universe']
    except FileNotFoundError:
        logger.error("Failed to load universe config")
        # Hardcoded fallback universe
        universe = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
            'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'LT.NS', 'KOTAKBANK.NS',
            'AXISBANK.NS', 'HINDUNILVR.NS', 'BAJFINANCE.NS', 'MARUTI.NS',
            'SUNPHARMA.NS', 'TITAN.NS', 'ASIANPAINT.NS', 'WIPRO.NS',
            'NTPC.NS', 'ONGC.NS'
        ]

    # Build market features
    features_df = build_market_features(universe)

    # Save to file
    save_market_features(features_df)

    # Print summary
    if not features_df.empty:
        print(f"\nMarket features summary:")
        print(f"  Shape: {features_df.shape}")
        print(f"  Date range: {features_df.index.get_level_values(0).min()} to {features_df.index.get_level_values(0).max()}")
        print(f"  Number of tickers: {features_df.index.get_level_values(1).nunique()}")
        print(f"  Features: {list(features_df.columns)}")
    else:
        print("\nNo market features were generated")
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()