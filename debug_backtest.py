import os
import tempfile
import pandas as pd
import yaml
from src.stock_engine.backtest.engine import Backtester
from src.stock_engine.backtest.costs import CostModel

dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='B')
prices_df = pd.DataFrame({
    'open': [100, 101, 102, 103, 104, 105, 106],
    'high': [101, 102, 103, 104, 105, 106, 107],
    'low': [99, 100, 101, 102, 103, 104, 105],
    'close': [100, 101, 102, 103, 104, 105, 106]
}, index=dates)
prices = {'TEST.NS': prices_df}

signals = pd.DataFrame([
    {'date': pd.Timestamp('2023-01-02'), 'ticker': 'TEST.NS', 'score': 1.0, 'exit_flag': False},
    {'date': pd.Timestamp('2023-01-03'), 'ticker': 'TEST.NS', 'score': 0.0, 'exit_flag': False},
    {'date': pd.Timestamp('2023-01-04'), 'ticker': 'TEST.NS', 'score': 0.0, 'exit_flag': True},
])

config = {
    'brokerage': {'flat_inr': 0.0, 'percentage': 0.0},
    'stt_ctt': {'buy': 0.0, 'sell': 0.0},
    'exchange_transaction_charges': {'buy': 0.0, 'sell': 0.0},
    'sebi_turnover_fee': {'buy': 0.0, 'sell': 0.0},
    'stamp_duty': {'buy': 0.0, 'sell': 0.0},
    'gst': {'rate': 0.0},
    'dp_charge': {'sell': 0.0, 'buy': 0.0},
    'slippage_bps': {'buy': 0.0, 'sell': 0.0},
    'round_trip_cost_bps_fallback': 0
}

with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    yaml.dump(config, f)
    config_path = f.name
try:
    cost_model = CostModel(config_path)
finally:
    os.unlink(config_path)

params = {
    'max_positions': 1,
    'target_weight_per_position': 1.0,
    'holding_horizon': 2,
    'atr_stop': None
}

initial_cash = 100000.0
backtester = Backtester(prices, signals, cost_model, params)
result = backtester.run_backtest(initial_cash)

print("Equity curve:")
print(result.equity_curve)
print("\nTrade log:")
for i, trade in enumerate(result.trade_log):
    print(f"Trade {i}: {trade}")
print("\nExposure series:")
print(result.exposure_series)