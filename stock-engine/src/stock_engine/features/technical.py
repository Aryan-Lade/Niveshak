"""
Technical features calculation module.
Calculates various technical indicators for individual stocks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Paths - assuming current working directory is stock-engine/src/stock_engine/features
PRICES_RAW_DIR = Path("../../../../stock-engine/data_store/raw/prices")

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


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    Args:
        prices: Series of prices (typically close prices)
        window: RSI window size (default 14)

    Returns:
        Series of RSI values
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()

    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices: pd.Series,
                   fast: int = 12,
                   slow: int = 26,
                   signal: int = 9) -> Dict[str, pd.Series]:
    """
    Calculate Moving Average Convergence Divergence (MACD).

    Args:
        prices: Series of prices (typically close prices)
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line EMA period (default 9)

    Returns:
        Dictionary with macd, signal, and histogram series
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return {
        'macd': macd_line,
        'macd_signal': signal_line,
        'macd_hist': histogram
    }


def calculate_bollinger_bands(prices: pd.Series,
                              window: int = 20,
                              num_std: float = 2.0) -> Dict[str, pd.Series]:
    """
    Calculate Bollinger Bands.

    Args:
        prices: Series of prices (typically close prices)
        window: Rolling window size (default 20)
        num_std: Number of standard deviations (default 2.0)

    Returns:
        Dictionary with upper, middle (SMA), and lower band series
    """
    sma = prices.rolling(window=window, min_periods=1).mean()
    std = prices.rolling(window=window, min_periods=1).std()

    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)

    return {
        'bb_upper': upper_band,
        'bb_middle': sma,
        'bb_lower': lower_band
    }


def calculate_atr(high: pd.Series,
                  low: pd.Series,
                  close: pd.Series,
                  window: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).

    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        window: ATR window size (default 14)

    Returns:
        Series of ATR values
    """
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=window, min_periods=1).mean()
    return atr


def build_technical_features(universe: List[str]) -> pd.DataFrame:
    """
    Build all technical features for the given universe of stocks.

    Args:
        universe: List of ticker symbols

    Returns:
        DataFrame with technical features indexed by (date, ticker)
    """
    logger.info("Building technical features...")

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

        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in price_df.columns]
        if missing_cols:
            logger.warning(f"Skipping {ticker} due to missing columns: {missing_cols}")
            continue

        # Initialize features DataFrame for this ticker
        features = pd.DataFrame(index=price_df.index)
        features['ticker'] = ticker

        # Extract price series
        close_prices = price_df['close']
        high_prices = price_df['high']
        low_prices = price_df['low']
        open_prices = price_df['open']
        volume = price_df['volume']

        # 1. RETURNS
        returns_dict = calculate_returns(close_prices, [1, 5, 10, 20])
        for key, value in returns_dict.items():
            features[key] = value

        # 2. VOLATILITY
        returns_1d = close_prices.pct_change()
        features['vol_5d'] = calculate_volatility(returns_1d, window=5)
        features['vol_10d'] = calculate_volatility(returns_1d, window=10)
        features['vol_20d'] = calculate_volatility(returns_1d, window=20)

        # 3. MOMENTUM INDICATORS
        # RSI
        features['rsi_14'] = calculate_rsi(close_prices, window=14)

        # MACD
        macd_dict = calculate_macd(close_prices)
        for key, value in macd_dict.items():
            features[key] = value

        # 4. VOLATILITY INDICATORS
        # Bollinger Bands
        bb_dict = calculate_bollinger_bands(close_prices)
        for key, value in bb_dict.items():
            features[key] = value

        # ATR
        features['atr_14'] = calculate_atr(high_prices, low_prices, close_prices, window=14)

        # 5. PRICE-BASED FEATURES
        # Price ratios
        features['high_low_ratio'] = high_prices / low_prices.replace(0, np.nan)
        features['close_open_ratio'] = close_prices / open_prices.replace(0, np.nan)

        # 6. VOLUME FEATURES
        # Volume changes
        features['volume_change_1d'] = volume.pct_change(1)
        features['volume_change_5d'] = volume.pct_change(5)

        # Volume moving averages
        features['volume_sma_10'] = volume.rolling(window=10, min_periods=1).mean()
        features['volume_sma_20'] = volume.rolling(window=20, min_periods=1).mean()

        # Price-volume trends
        features['price_volume_trend'] = (close_prices * volume).pct_change(1)

        ticker_features.append(features)

    # Combine all ticker features
    if not ticker_features:
        logger.error("No ticker features were generated")
        return pd.DataFrame()

    combined_features = pd.concat(ticker_features, axis=0)
    combined_features = combined_features.reset_index().set_index(['date', 'ticker']).sort_index()

    logger.info(f"Technical features built successfully with shape {combined_features.shape}")
    return combined_features


def save_technical_features(features_df: pd.DataFrame) -> None:
    """
    Save technical features to parquet file.

    Args:
        features_df: DataFrame with technical features to save
    """
    if features_df.empty:
        logger.warning("No technical features to save")
        return

    # Ensure features directory exists
    FEATURES_DIR = Path("../data_store/features")
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FEATURES_DIR / "technical_features.parquet"
    features_df.to_parquet(output_path)
    logger.info(f"Technical features saved to {output_path}")


def main() -> None:
    """Main function to build and save technical features."""
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

    # Build technical features
    features_df = build_technical_features(universe)

    # Save to file
    save_technical_features(features_df)

    # Print summary
    if not features_df.empty:
        print(f"\nTechnical features summary:")
        print(f"  Shape: {features_df.shape}")
        print(f"  Date range: {features_df.index.get_level_values(0).min()} to {features_df.index.get_level_values(0).max()}")
        print(f"  Number of tickers: {features_df.index.get_level_values(1).nunique()}")
        print(f"  Features: {list(features_df.columns)}")
    else:
        print("\nNo technical features were generated")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()