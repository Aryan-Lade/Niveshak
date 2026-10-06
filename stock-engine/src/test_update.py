import sys
import yaml
from datetime import datetime
from stock_engine.data.prices import update_prices

print('Starting update...')
with open('../configs/universe.yaml') as f:
    universe_config = yaml.safe_load(f)
tickers = universe_config['universe'][:2]  # Just update first 2 tickers for testing
print(f'Updating {len(tickers)} tickers: {tickers}')

with open('../configs/data.yaml') as f:
    data_config = yaml.safe_load(f)
start_date = data_config['data']['start_date']
end_date = datetime.now().strftime('%Y-%m-%d')
print(f'Date range: {start_date} to {end_date}')

result = update_prices(tickers, start_date, end_date)
print(f'Update complete. Shape: {result.shape}')