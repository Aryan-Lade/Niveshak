import sys
import yaml
from stock_engine.features.market import build_market_features

print('Building market features...')
with open('../configs/universe.yaml') as f:
    universe_config = yaml.safe_load(f)
universe = universe_config['universe'][:2]  # Just use first 2 tickers for testing
print(f'Building market features for: {universe}')

features_df = build_market_features(universe)
print(f'Market features built. Shape: {features_df.shape}')
if not features_df.empty:
    print(f'Columns: {list(features_df.columns)}')
    print(f'Date range: {features_df.index.get_level_values(0).min()} to {features_df.index.get_level_values(0).max()}')
    print(f'Number of tickers: {features_df.index.get_level_values(1).nunique()}')