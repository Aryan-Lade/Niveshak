import sys
import yaml
from stock_engine.features.technical import build_technical_features, save_technical_features
from stock_engine.features.market import build_market_features, save_market_features
from stock_engine.features.build import build_feature_matrix

print('Building and saving technical features...')
with open('../configs/universe.yaml') as f:
    universe_config = yaml.safe_load(f)
universe = universe_config['universe'][:2]  # Just use first 2 tickers for testing
print(f'Building technical features for: {universe}')

tech_features_df = build_technical_features(universe)
save_technical_features(tech_features_df)
print(f'Technical features saved. Shape: {tech_features_df.shape}')

print('Building and saving market features...')
market_features_df = build_market_features(universe)
save_market_features(market_features_df)
print(f'Market features saved. Shape: {market_features_df.shape}')

print('Building feature matrix...')
features_df = build_feature_matrix()
print(f'Feature matrix built. Shape: {features_df.shape}')
if not features_df.empty:
    print(f'Columns: {list(features_df.columns)}')
    print(f'Date range: {features_df.index.get_level_values(0).min()} to {features_df.index.get_level_values(0).max()}')
    print(f'Number of tickers: {features_df.index.get_level_values(1).nunique()}')

    # Count technical vs market features
    market_indicators = ['nifty_', 'vix_', 'usdinr_', 'sp500_', 'crude_', '_percentile', 'beta_', 'rs_']
    market_features = [c for c in features_df.columns if any(ind in c for ind in market_indicators)]
    technical_features = [c for c in features_df.columns if not any(ind in c for ind in market_indicators)]

    print(f'Technical features: {len(technical_features)}')
    print(f'Market features: {len(market_features)}')
    print(f'Total features: {len(features_df.columns)}')