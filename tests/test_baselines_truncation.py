import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.stock_engine.backtest.baselines import (
    create_nifty_buy_hold_signals,
    create_equal_weight_signals,
    create_sma_crossover_signals,
    create_momentum_signals,
    create_rsi_mean_reversion_signals
)


def test_nifty_buy_hold_truncation():
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    df = pd.DataFrame({
        'open': np.ones(10),
        'high': np.ones(10),
        'low': np.ones(10),
        'close': np.ones(10),
        'volume': np.ones(10)
    }, index=dates)
    prices = {'TEST.NS': df}

    signals = create_nifty_buy_hold_signals(prices)
    assert len(signals) == 1
    assert signals.iloc[0]['date'] == dates[0]


def test_equal_weight_truncation():
    dates = pd.date_range('2020-01-01', periods=20, freq='D')
    df1 = pd.DataFrame({
        'open': np.ones(20),
        'high': np.ones(20),
        'low': np.ones(20),
        'close': np.ones(20),
        'volume': np.ones(20)
    }, index=dates)
    df2 = df1.copy()
    prices = {'TICKER1.NS': df1, 'TICKER2.NS': df2}

    signals = create_equal_weight_signals(prices)
    for _, row in signals.iterrows():
        assert row['date'].day == 1


def test_sma_crossover_truncation():
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    close = np.linspace(10, 20, 100)
    df = pd.DataFrame({
        'open': close,
        'high': close,
        'low': close,
        'close': close,
        'volume': np.ones(100)
    }, index=dates)
    prices = {'TEST.NS': df}

    signals = create_sma_crossover_signals(prices)
    for _, row in signals.iterrows():
        assert row['date'] in df.index


def test_momentum_truncation():
    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    close = np.linspace(10, 20, 300)
    df = pd.DataFrame({
        'open': close,
        'high': close,
        'low': close,
        'close': close,
        'volume': np.ones(300)
    }, index=dates)
    prices = {'TEST.NS': df}

    signals = create_momentum_signals(prices)
    for _, row in signals.iterrows():
        assert row['date'].day == 1


def test_rsi_mean_reversion_truncation():
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    close = np.random.uniform(10, 20, 100)
    df = pd.DataFrame({
        'open': close,
        'high': close,
        'low': close,
        'close': close,
        'volume': np.ones(100)
    }, index=dates)
    prices = {'TEST.NS': df}

    signals = create_rsi_mean_reversion_signals(prices)
    for _, row in signals.iterrows():
        assert row['date'] in df.index


if __name__ == "__main__":
    test_nifty_buy_hold_truncation()
    test_equal_weight_truncation()
    test_sma_crossover_truncation()
    test_momentum_truncation()
    test_rsi_mean_reversion_truncation()
    print("All truncation tests passed.")