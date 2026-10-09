import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import yaml

from src.stock_engine.backtest.engine import Backtester, BacktestResult
from src.stock_engine.backtest.costs import CostModel
from src.stock_engine.backtest.metrics import (
    calculate_all_metrics,
    cagr,
    annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    win_rate,
    profit_factor,
    average_holding_days,
    annual_turnover,
    average_exposure,
    bootstrap_sharpe_ci
)


def _build_cost_model(config_dict: dict) -> CostModel:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_dict, f)
        config_path = f.name
    try:
        return CostModel(config_path)
    finally:
        os.unlink(config_path)


def _zero_cost_model() -> CostModel:
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
    return _build_cost_model(config)


def test_engine_toy_example():
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

    cost_model = _zero_cost_model()

    params = {
        'max_positions': 1,
        'target_weight_per_position': 1.0,
        'holding_horizon': 2,
        'atr_stop': None
    }

    initial_cash = 100000.0
    backtester = Backtester(prices, signals, cost_model, params)
    result = backtester.run_backtest(initial_cash)

    final_equity = result.equity_curve.iloc[-1]
    assert abs(final_equity - 101980.0) < 0.01

    assert len(result.trade_log) == 1
    trade = result.trade_log[0]
    assert trade['ticker'] == 'TEST.NS'
    assert trade['entry_date'] == pd.Timestamp('2023-01-03')
    assert trade['exit_date'] == pd.Timestamp('2023-01-05')
    assert trade['entry_price'] == 101.0
    assert trade['exit_price'] == 103.0
    assert trade['quantity'] == 990
    assert trade['pnl'] == (103 - 101) * 990
    assert trade['reason'] == 'signal'


def test_engine_no_signals():
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='B')
    prices_df = pd.DataFrame({
        'open': [100, 101, 102, 103, 104, 105, 106],
        'high': [101, 102, 103, 104, 105, 106, 107],
        'low': [99, 100, 101, 102, 103, 104, 105],
        'close': [100, 101, 102, 103, 104, 105, 106]
    }, index=dates)
    prices = {'TEST.NS': prices_df}

    signals = pd.DataFrame(columns=['date', 'ticker', 'score', 'exit_flag'])
    cost_model = _zero_cost_model()

    params = {
        'max_positions': 1,
        'target_weight_per_position': 1.0,
        'holding_horizon': 2,
        'atr_stop': None
    }

    initial_cash = 100000.0
    backtester = Backtester(prices, signals, cost_model, params)
    result = backtester.run_backtest(initial_cash)

    expected_equity = pd.Series([initial_cash] * len(dates), index=pd.Index(dates), name='equity')
    expected_equity.index.name = 'date'
    pd.testing.assert_series_equal(result.equity_curve, expected_equity, check_freq=False)
    assert len(result.trade_log) == 0


def test_engine_higher_costs_lower_final_equity():
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

    low_cost_config = {
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
    high_cost_config = low_cost_config.copy()
    high_cost_config['slippage_bps'] = {'buy': 10.0, 'sell': 10.0}

    low_cost_model = _build_cost_model(low_cost_config)
    high_cost_model = _build_cost_model(high_cost_config)

    params = {
        'max_positions': 1,
        'target_weight_per_position': 1.0,
        'holding_horizon': 2,
        'atr_stop': None
    }

    initial_cash = 100000.0

    backtester_low = Backtester(prices, signals, low_cost_model, params)
    result_low = backtester_low.run_backtest(initial_cash)

    backtester_high = Backtester(prices, signals, high_cost_model, params)
    result_high = backtester_high.run_backtest(initial_cash)

    assert result_low.equity_curve.iloc[-1] > result_high.equity_curve.iloc[-1]


def test_engine_no_lookahead():
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
    ])

    cost_model = _zero_cost_model()

    params = {
        'max_positions': 1,
        'target_weight_per_position': 1.0,
        'holding_horizon': 100,
        'atr_stop': None
    }

    initial_cash = 100000.0
    backtester = Backtester(prices, signals, cost_model, params)
    result = backtester.run_backtest(initial_cash)

    assert len(result.trade_log) == 1
    trade = result.trade_log[0]
    assert trade['entry_date'] == pd.Timestamp('2023-01-03')
    assert trade['entry_price'] == 101.0

    signal_date = pd.Timestamp('2023-01-02')
    signal_loc = prices_df.index.get_loc(signal_date)
    expected_open = prices_df.iloc[signal_loc + 1]['open']
    assert trade['entry_price'] == expected_open


def test_metrics_calculation():
    equity = pd.Series([100, 110], index=[pd.Timestamp('2023-01-01'), pd.Timestamp('2024-01-01')])
    assert abs(cagr(equity) - 0.10) < 1e-4

    returns = pd.Series([0.01] * 252)
    vol = annualized_volatility(returns)
    assert abs(vol - 0.0) < 1e-4

    returns = pd.Series([0.01, -0.01] * 126)
    vol = annualized_volatility(returns)
    expected_vol = np.std(returns, ddof=1) * np.sqrt(252)
    assert abs(vol - expected_vol) < 1e-4

    excess_returns = pd.Series([0.001, -0.001] * 126)
    sharpe = sharpe_ratio(excess_returns, risk_free_rate=0.0)
    assert abs(sharpe) < 1e-4

    equity = pd.Series([100, 120, 110, 90, 100], index=pd.date_range('2023-01-01', periods=5, freq='B'))
    mdd, duration = max_drawdown(equity)
    assert abs(mdd - 0.25) < 1e-4
    assert duration == 2

    trade_log = [
        {'pnl': 100},
        {'pnl': -50},
        {'pnl': 200},
        {'pnl': 0}
    ]
    assert win_rate(trade_log) == 0.5

    trade_log = [
        {'pnl': 200},
        {'pnl': 200},
        {'pnl': -100},
        {'pnl': -50}
    ]
    assert profit_factor(trade_log) == pytest.approx(400 / 150)

    trade_log = [
        {'entry_date': pd.Timestamp('2023-01-01'), 'exit_date': pd.Timestamp('2023-01-03')},
        {'entry_date': pd.Timestamp('2023-01-05'), 'exit_date': pd.Timestamp('2023-01-08')}
    ]
    assert average_holding_days(trade_log) == 2.5

    trade_log = [
        {'entry_price': 100, 'quantity': 10, 'exit_price': 110},
        {'entry_price': 100, 'quantity': 10, 'exit_price': 90}
    ]
    initial_cash = 10000
    total_days = 365.25
    assert annual_turnover(trade_log, initial_cash, total_days) == 0.4

    exposure = pd.Series([1000, 2000, 1500], index=pd.Index(pd.date_range('2023-01-01', periods=3, freq='B')))
    exposure.index.name = 'date'
    assert average_exposure(exposure) == (1000 + 2000 + 1500) / 3

    returns = pd.Series(np.random.randn(252) * 0.01)
    ci_lower, ci_upper = bootstrap_sharpe_ci(returns, n_bootstrap=100)
    assert ci_lower <= ci_upper

    equity_curve = pd.Series([100, 110, 120], index=pd.Index(pd.date_range('2023-01-01', periods=3, freq='B')))
    equity_curve.index.name = 'date'
    daily_returns = pd.Series([0.1, 0.0909], index=equity_curve.index[1:])
    trade_log = [
        {'pnl': 10, 'entry_date': pd.Timestamp('2023-01-01'), 'exit_date': pd.Timestamp('2023-01-02'),
         'entry_price': 100, 'exit_price': 110, 'quantity': 10},
        {'pnl': 20, 'entry_date': pd.Timestamp('2023-01-02'), 'exit_date': pd.Timestamp('2023-01-03'),
         'entry_price': 100, 'exit_price': 90, 'quantity': 10}
    ]
    exposure_series = pd.Series([0, 50, 100], index=pd.Index(pd.date_range('2023-01-01', periods=3, freq='B')))
    exposure_series.index.name = 'date'
    backtest_result = {
        'equity_curve': equity_curve,
        'daily_returns': daily_returns,
        'trade_log': trade_log,
        'exposure_series': exposure_series
    }
    metrics = calculate_all_metrics(backtest_result, initial_cash=100)
    expected_keys = [
        'CAGR', 'Annualized Volatility', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown',
        'Max Drawdown Duration', 'Calmar Ratio', 'Win Rate', 'Profit Factor',
        'Average Holding Days', 'Annual Turnover', 'Average Exposure',
        'Sharpe CI Lower', 'Sharpe CI Upper'
    ]
    for key in expected_keys:
        assert key in metrics


if __name__ == "__main__":
    test_engine_toy_example()
    test_engine_no_signals()
    test_engine_higher_costs_lower_final_equity()
    test_engine_no_lookahead()
    test_metrics_calculation()
    print("All tests passed.")