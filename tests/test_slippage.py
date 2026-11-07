import pytest

from stock_engine.execution.slippage import AlmgrenChrissSlippageModel, ExecutionFill


@pytest.fixture
def model():
    return AlmgrenChrissSlippageModel(
        eta=0.142,
        gamma=0.314,
        base_spread_bps=5.0,
        max_slippage_bps=150.0,
    )


def test_buy_fills_higher(model):
    fill = model.simulate_fill(
        side="BUY",
        decision_price=1000.0,
        quantity=500,
        daily_volume=500_000,
        daily_volatility=0.015,
    )
    assert fill.fill_price > fill.decision_price
    assert fill.impact_cost_inr > 0
    assert fill.slippage_bps >= 5.0


def test_sell_fills_lower(model):
    fill = model.simulate_fill(
        side="SELL",
        decision_price=1000.0,
        quantity=500,
        daily_volume=500_000,
        daily_volatility=0.015,
    )
    assert fill.fill_price < fill.decision_price
    assert fill.impact_cost_inr > 0
    assert fill.slippage_bps >= 5.0


def test_larger_order_higher_slippage(model):
    small_fill = model.simulate_fill(
        side="BUY",
        decision_price=500.0,
        quantity=100,
        daily_volume=1_000_000,
    )
    large_fill = model.simulate_fill(
        side="BUY",
        decision_price=500.0,
        quantity=50_000,
        daily_volume=1_000_000,
    )
    assert large_fill.slippage_bps > small_fill.slippage_bps
    assert large_fill.impact_cost_inr > small_fill.impact_cost_inr


def test_higher_volatility_higher_slippage(model):
    low_vol = model.simulate_fill(
        side="BUY",
        decision_price=500.0,
        quantity=1000,
        daily_volume=500_000,
        daily_volatility=0.01,
    )
    high_vol = model.simulate_fill(
        side="BUY",
        decision_price=500.0,
        quantity=1000,
        daily_volume=500_000,
        daily_volatility=0.04,
    )
    assert high_vol.slippage_bps > low_vol.slippage_bps


def test_slippage_capped_at_max(model):
    extreme_fill = model.simulate_fill(
        side="BUY",
        decision_price=100.0,
        quantity=2_000_000,
        daily_volume=10_000,
        daily_volatility=0.10,
    )
    assert extreme_fill.slippage_bps <= 150.0
