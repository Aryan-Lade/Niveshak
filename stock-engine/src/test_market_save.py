import sys
import yaml
from stock_engine.features.market import build_market_features, save_market_features
import os

print('Building market features...')
with open('../configs/universe.yaml') as f:
    universe_config = yaml.safe_load(f)
universe = universe_config['universe'][:2]  # Just use first 2 tickers for testing
print(f'Building market features for: {universe}')

market_features_df = build_market_features(universe)
print(f'Market features built. Shape: {market_features_df.shape}')

print('Saving market features...')
save_market_features(market_features_df)
print('Save function completed.')

# Check if file exists
expected_path = "../data_store/features/market_features.parquet"
print(f"Checking for file at: {expected_path}")
full_path = os.path.join(os.getcwd(), expected_path)
print(f"Full path: {full_path}")
print(f"File exists: {os.path.exists(full_path)}")

# Also check what the current directory is
print(f"Current working directory: {os.getcwd()}")

# List the features directory
features_dir = "../data_store/features"
print(f"Contents of {features_dir}:")
if os.path.exists(features_dir):
    for file in os.listdir(features_dir):
        print(f"  {file}")
else:
    print("  Directory does not exist")