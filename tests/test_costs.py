import os
import tempfile
import pytest
import yaml
from src.stock_engine.backtest.costs import CostModel, OrderCost


def test_order_cost_buy():
    model = CostModel()
    cost = model.order_cost('buy', 1000.0, 10)
    turnover = 1000.0 * 10
    brokerage_flat = 20.0
    brokerage_pct = 0.05 / 100.0 * turnover
    brokerage = min(brokerage_flat, brokerage_pct)
    stt_ctt = 0.1 / 100.0 * turnover
    exchange = 0.00325 / 100.0 * turnover
    sebi = 0.0001 / 100.0 * turnover
    stamp_duty = 0.015 / 100.0 * turnover
    gst = 0.18 * (brokerage + exchange + sebi)
    dp_charge = 0.0
    slippage = (2.0 / 10000.0) * turnover
    expected_total = brokerage + stt_ctt + exchange + sebi + stamp_duty + gst + dp_charge + slippage
    assert cost.brokerage == pytest.approx(brokerage)
    assert cost.stt_ctt == pytest.approx(stt_ctt)
    assert cost.exchange_transaction_charges == pytest.approx(exchange)
    assert cost.sebi_turnover_fee == pytest.approx(sebi)
    assert cost.stamp_duty == pytest.approx(stamp_duty)
    assert cost.gst == pytest.approx(gst)
    assert cost.dp_charge == pytest.approx(dp_charge)
    assert cost.slippage == pytest.approx(slippage)
    assert cost.total == pytest.approx(expected_total)


def test_order_cost_sell():
    model = CostModel()
    cost = model.order_cost('sell', 1000.0, 10)
    turnover = 1000.0 * 10
    brokerage_flat = 20.0
    brokerage_pct = 0.05 / 100.0 * turnover
    brokerage = min(brokerage_flat, brokerage_pct)
    stt_ctt = 0.1 / 100.0 * turnover
    exchange = 0.00325 / 100.0 * turnover
    sebi = 0.0001 / 100.0 * turnover
    stamp_duty = 0.0
    gst = 0.18 * (brokerage + exchange + sebi)
    dp_charge = 13.5
    slippage = -(2.0 / 10000.0) * turnover
    expected_total = brokerage + stt_ctt + exchange + sebi + stamp_duty + gst + dp_charge + slippage
    assert cost.brokerage == pytest.approx(brokerage)
    assert cost.stt_ctt == pytest.approx(stt_ctt)
    assert cost.exchange_transaction_charges == pytest.approx(exchange)
    assert cost.sebi_turnover_fee == pytest.approx(sebi)
    assert cost.stamp_duty == pytest.approx(stamp_duty)
    assert cost.gst == pytest.approx(gst)
    assert cost.dp_charge == pytest.approx(dp_charge)
    assert cost.slippage == pytest.approx(slippage)
    assert cost.total == pytest.approx(expected_total)


def test_round_trip_bps():
    model = CostModel()
    bps = model.round_trip_bps(1000.0, 10)
    buy_cost = model.order_cost('buy', 1000.0, 10)
    sell_cost = model.order_cost('sell', 1000.0, 10)
    round_trip_cost = buy_cost.total + sell_cost.total
    turnover = 1000.0 * 10
    expected_bps = (round_trip_cost / turnover) * 10000
    assert bps == pytest.approx(expected_bps, rel=1e-4)


def test_fallback_when_component_missing():
    config = {
        'brokerage': {'flat_inr': 20.0, 'percentage': 0.05},
        'stt_ctt': {'buy': 0.1, 'sell': 0.1},
        'sebi_turnover_fee': {'buy': 0.0001, 'sell': 0.0001},
        'stamp_duty': {'buy': 0.015, 'sell': 0.0},
        'gst': {'rate': 0.18},
        'dp_charge': {'sell': 13.5, 'buy': 0.0},
        'slippage_bps': {'buy': 2.0, 'sell': 2.0},
        'round_trip_cost_bps_fallback': 25
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    try:
        model = CostModel(config_path)
        bps = model.round_trip_bps(1000.0, 10)
        assert bps == pytest.approx(25.0, rel=1e-4)
    finally:
        os.unlink(config_path)


def test_costs_non_negative():
    model = CostModel()
    buy_cost = model.order_cost('buy', 1000.0, 10)
    sell_cost = model.order_cost('sell', 1000.0, 10)
    assert buy_cost.total >= 0.0
    assert sell_cost.total >= 0.0


def test_slippage_direction():
    model = CostModel()
    buy_cost = model.order_cost('buy', 1000.0, 10)
    sell_cost = model.order_cost('sell', 1000.0, 10)
    assert buy_cost.slippage > 0
    assert sell_cost.slippage < 0