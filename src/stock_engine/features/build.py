"""
Feature matrix builder module.
Merges technical and market features into a unified feature matrix.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Paths
FEATURES_DIR = Path("data_store/features")
TECHNICAL_FEATURES_PATH = FEATURES_DIR / "technical_features.parquet"
MARKET_FEATURES_PATH = FEATURES_DIR / "market_features.parquet"
OUTPUT_FEATURES_PATH = FEATURES_DIR / "features.parquet"


def load_technical_features() -> Optional[pd.DataFrame]:
    """
    Load technical features from parquet file.

    Returns:
        DataFrame with technical features or None if not found
    """
    if not TECHNICAL_FEATURES_PATH.exists():
        logger.warning(f"Technical features not found at {TECHNICAL_FEATURES_PATH}")
        return None

    try:
        df = pd.read_parquet(TECHNICAL_FEATURES_PATH)
        logger.info(f"Loaded technical features from {TECHNICAL_FEATURES_PATH} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading technical features: {e}")
        return None


def load_market_features() -> Optional[pd.DataFrame]:
    """
    Load market features from parquet file.

    Returns:
        DataFrame with market features or None if not found
    """
    if not MARKET_FEATURES_PATH.exists():
        logger.warning(f"Market features not found at {MARKET_FEATURES_PATH}")
        return None

    try:
        df = pd.read_parquet(MARKET_FEATURES_PATH)
        logger.info(f"Loaded market features from {MARKET_FEATURES_PATH} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading market features: {e}")
        return None


def merge_features(technical_df: pd.DataFrame,
                   market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge technical and market features.

    Args:
        technical_df: DataFrame with technical features
        market_df: DataFrame with market features

    Returns:
        Merged DataFrame with all features
    """
    if technical_df is None and market_df is None:
        logger.error("Both technical and market features are None")
        return pd.DataFrame()

    if technical_df is None:
        logger.warning("Only market features available")
        return market_df.copy()

    if market_df is None:
        logger.warning("Only technical features available")
        return technical_df.copy()

    # Ensure both DataFrames have the same index structure
    # They should both be indexed by (date, ticker)
    logger.info(f"Merging technical features {technical_df.shape} with market features {market_df.shape}")

    # Perform outer merge on index (date, ticker)
    merged_df = technical_df.merge(
        market_df,
        left_index=True,
        right_index=True,
        how='outer',
        suffixes=('_tech', '_market')
    )

    # Check for duplicate columns (shouldn't happen with our design, but just in case)
    # Remove any duplicate columns that might have been created
    cols = merged_df.columns.tolist()
    seen = set()
    dup_cols = []
    for col in cols:
        if col in seen:
            dup_cols.append(col)
        else:
            seen.add(col)

    if dup_cols:
        logger.warning(f"Found duplicate columns after merge: {dup_cols}")
        # Keep the first occurrence of each duplicate
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    logger.info(f"Merged features shape: {merged_df.shape}")
    return merged_df


def build_feature_matrix() -> pd.DataFrame:
    """
    Build the complete feature matrix by merging technical and market features.

    Returns:
        DataFrame with complete feature matrix indexed by (date, ticker)
    """
    logger.info("Building feature matrix...")

    # Load technical features
    technical_df = load_technical_features()

    # Load market features
    market_df = load_market_features()

    # Merge features
    features_df = merge_features(technical_df, market_df)

    if features_df.empty:
        logger.error("Failed to build feature matrix - no data available")
        return pd.DataFrame()

    # Sort index for consistency
    features_df = features_df.sort_index()

    logger.info(f"Feature matrix built successfully with shape {features_df.shape}")
    return features_df


def save_feature_matrix(features_df: pd.DataFrame) -> None:
    """
    Save feature matrix to parquet file.

    Args:
        features_df: DataFrame with feature matrix to save
    """
    if features_df.empty:
        logger.warning("No feature matrix to save")
        return

    # Ensure features directory exists
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    features_df.to_parquet(OUTPUT_FEATURES_PATH)
    logger.info(f"Feature matrix saved to {OUTPUT_FEATURES_PATH}")


def main() -> None:
    """Main function to build and save feature matrix."""
    # Build feature matrix
    features_df = build_feature_matrix()

    # Save to file
    save_feature_matrix(features_df)

    # Print summary
    if not features_df.empty:
        # Count technical vs market features
        market_indicators = ['nifty_', 'vix_', 'usdinr_', 'sp500_', 'crude_', '_percentile', 'beta_', 'rs_']
        market_features = [c for c in features_df.columns if any(ind in c for ind in market_indicators)]
        technical_features = [c for c in features_df.columns if not any(ind in c for ind in market_indicators)]

        print(f"\nFeature matrix summary:")
        print(f"  Shape: {features_df.shape}")
        print(f"  Date range: {features_df.index.get_level_values(0).min()} to {features_df.index.get_level_values(0).max()}")
        print(f"  Number of tickers: {features_df.index.get_level_values(1).nunique()}")
        print(f"  Technical features: {len(technical_features)}")
        print(f"  Market features: {len(market_features)}")
        print(f"  Total features: {len(features_df.columns)}")
        print(f"  Features: {list(features_df.columns)}")
    else:
        print("\nNo feature matrix was generated")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()